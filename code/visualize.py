import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
import random
import h5py
import os
import gc
from typing import Dict, Any, Tuple, Optional, List

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import RomanMicrolensingGRU, ModelConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
WARMUP_EPOCHS = 5
SEED = 42
AB_ZEROPOINT_JY = 3631.0
MISSION_DURATION_DAYS = 1826.25


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_compat(path):
    path = str(path)
    if path.endswith('.h5') or path.endswith('.hdf5'):
        with h5py.File(path, 'r') as f:
            return {k: f[k][:] for k in f.keys()}
    return np.load(path, allow_pickle=True)


def setup_distributed() -> Tuple[int, int, int, bool]:
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if rank == 0:
            print(f"Distributed: {world_size} processes", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
            dist.init_process_group(
                backend='nccl', 
                init_method='env://',
                timeout=timedelta(seconds=1800)
            )
            
            # Performance optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory optimization
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.empty_cache()
            gc.collect()
            
            return rank, local_rank, world_size, True
    
    return 0, 0, 1, False


def setup_logging(rank: int, output_dir: Path):
    logger = logging.getLogger("TRAIN")
    
    if rank == 0:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(fmt)
        file_handler.setFormatter(fmt)
        
        logger.addHandler(console)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.CRITICAL)
    
    return logger


def set_seed_everywhere(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class MicrolensingDataset(Dataset):
    def __init__(
        self, 
        flux: np.ndarray, 
        delta_t: np.ndarray, 
        labels: np.ndarray, 
        stats: Optional[Dict] = None,
        handle_nans: str = 'zero'
    ):
        # Contiguous arrays for faster GPU transfer
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        self.padding_mask = (~torch.isnan(self.flux)) & (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)
        
        if handle_nans == 'zero':
            self.flux = torch.nan_to_num(self.flux, nan=0.0)
        elif handle_nans == 'median':
            for i in range(len(self.flux)):
                valid = self.flux[i][self.padding_mask[i]]
                if len(valid) > 0:
                    median = valid.median()
                    self.flux[i] = torch.where(
                        torch.isnan(self.flux[i]), 
                        median, 
                        self.flux[i]
                    )
        
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)
        
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr']
            
            if iqr < 1e-6:
                iqr = 1.0
            
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.flux[idx],
            self.delta_t[idx],
            self.lengths[idx],
            self.labels[idx]
        )


def compute_normalization_stats(flux: np.ndarray, sample_size: int = 10000) -> Dict[str, float]:
    sample_flux = flux[:sample_size]
    valid_flux = sample_flux[~np.isnan(sample_flux) & (sample_flux != 0.0)]
    
    if len(valid_flux) == 0:
        return {'median': 0.0, 'iqr': 1.0}
    
    median = float(np.median(valid_flux))
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = float(q75 - q25)
    
    if iqr < 1e-6:
        iqr = 1.0
    
    return {'median': median, 'iqr': iqr}


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    weight_tensor = torch.zeros(3, device=device)
    weight_tensor[unique] = torch.from_numpy(weights).float()
    return weight_tensor


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    epoch: int,
    logger: logging.Logger,
    config: ModelConfig,
    accumulation_steps: int = 1,
    clip_norm: float = CLIP_NORM
) -> Tuple[float, float]:
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    optimizer.zero_grad(set_to_none=True)
    
    # Prefetch to GPU stream
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        # Non-blocking transfer with prefetch
        with torch.cuda.stream(stream) if stream else torch.no_grad():
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        if stream:
            torch.cuda.current_stream().wait_stream(stream)
        
        with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
            output = model(flux, delta_t, lengths=lengths)
            
            if config.hierarchical:
                loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
            else:
                loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
            
            loss = loss / accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            pred = output['probs'].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += (loss.item() * accumulation_steps) * labels.size(0)
        
        if rank == 0 and batch_idx % 20 == 0:
            iterator.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.0 * correct / total:.2f}'
            })
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    config: ModelConfig,
    return_predictions: bool = False
) -> Dict[str, Any]:
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = [] if return_predictions else None
    all_labels = [] if return_predictions else None
    all_probs = [] if return_predictions else None
    
    # Prefetch stream
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    for flux, delta_t, lengths, labels in tqdm(loader, disable=(rank != 0), desc="Eval"):
        with torch.cuda.stream(stream) if stream else torch.no_grad():
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        if stream:
            torch.cuda.current_stream().wait_stream(stream)
        
        with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
            output = model(flux, delta_t, lengths=lengths)
            
            if config.hierarchical:
                loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
            else:
                loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
        
        pred = output['probs'].argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        
        if return_predictions:
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(output['probs'].cpu().numpy())
    
    results = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    if return_predictions:
        results['predictions'] = np.concatenate(all_preds)
        results['labels'] = np.concatenate(all_labels)
        results['probabilities'] = np.concatenate(all_probs)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_acc: float,
    config: ModelConfig,
    stats: Dict,
    output_dir: Path,
    is_best: bool = False
) -> None:
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config.__dict__,
        'stats': stats
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


def main():
    parser = argparse.ArgumentParser(description='Train Roman microlensing classifier')
    
    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='../results')
    parser.add_argument('--experiment-name', type=str, default=None)
    
    # Architecture
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--window-size', type=int, default=5)
    
    # Model variants
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_true')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    parser.set_defaults(hierarchical=True)
    
    parser.add_argument('--feature-extraction', type=str, choices=['conv', 'mlp'], default='conv')
    
    parser.add_argument('--use-attention-pooling', dest='attention_pooling', action='store_true')
    parser.add_argument('--no-attention-pooling', dest='attention_pooling', action='store_false')
    parser.set_defaults(attention_pooling=True)
    
    parser.add_argument('--use-residual', dest='use_residual', action='store_true')
    parser.add_argument('--no-residual', dest='use_residual', action='store_false')
    parser.set_defaults(use_residual=True)
    
    parser.add_argument('--use-layer-norm', dest='use_layer_norm', action='store_true')
    parser.add_argument('--no-layer-norm', dest='use_layer_norm', action='store_false')
    parser.set_defaults(use_layer_norm=True)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    
    # Optimization
    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)
    
    parser.add_argument('--use-gradient-checkpointing', dest='use_gradient_checkpointing', action='store_true')
    parser.add_argument('--no-gradient-checkpointing', dest='use_gradient_checkpointing', action='store_false')
    parser.set_defaults(use_gradient_checkpointing=True)
    
    parser.add_argument('--use-class-weights', action='store_true', default=True)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--compile-mode', type=str, default='default', choices=['default', 'reduce-overhead', 'max-autotune'])
    
    # Evaluation
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    
    # Data handling
    parser.add_argument('--nan-strategy', type=str, default='zero', choices=['zero', 'median'])
    parser.add_argument('--num-workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"d{args.d_model}_l{args.n_layers}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        dist.barrier()
    
    logger = setup_logging(rank, output_dir)
    
    if rank == 0:
        logger.info(f"Seed: {SEED}")
    
    set_seed_everywhere(SEED, rank)
    
    # Load data
    if rank == 0:
        logger.info(f"Loading: {args.data}")
    
    data = load_compat(args.data)
    flux = data['flux']
    delta_t = data['delta_t']
    labels = data['labels']
    
    if rank == 0:
        logger.info(f"Samples: {len(labels)}")
    
    # Split
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    flux_train = flux[train_idx]
    delta_t_train = delta_t[train_idx]
    labels_train = labels[train_idx]
    
    flux_val = flux[val_idx]
    delta_t_val = delta_t[val_idx]
    labels_val = labels[val_idx]
    
    if rank == 0:
        unique_train, counts_train = np.unique(labels_train, return_counts=True)
        logger.info(f"Train: {len(labels_train)}, Val: {len(labels_val)}")
        logger.info(f"Class distribution: {dict(zip(unique_train.tolist(), counts_train.tolist()))}")
    
    # Normalization
    stats = compute_normalization_stats(flux_train)
    if rank == 0:
        logger.info(f"Norm: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
    
    # Datasets
    train_dataset = MicrolensingDataset(flux_train, delta_t_train, labels_train, stats, handle_nans=args.nan_strategy)
    val_dataset = MicrolensingDataset(flux_val, delta_t_val, labels_val, stats, handle_nans=args.nan_strategy)
    
    # Samplers
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Model
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        hierarchical=args.hierarchical,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        feature_extraction=args.feature_extraction,
        use_attention_pooling=args.attention_pooling,
        use_amp=args.use_amp,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    
    model = RomanMicrolensingGRU(config, dtype=torch.float32).to(device)
    
    if rank == 0:
        n_params = count_parameters(model)
        logger.info(f"Parameters: {n_params:,}")
        
        effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
        logger.info(f"Effective batch: {effective_batch}")
    
    # DDP
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=True
        )
    
    # Compile
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info(f"Compiling (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        fused=torch.cuda.is_available()
    )
    
    scaler = None
    
    # Scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(labels_train, device)
        if rank == 0:
            logger.info(f"Weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(3, device=device)
    
    if is_ddp:
        dist.barrier()
    
    # Training
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info("=" * 60)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler,
            class_weights, device, rank, epoch, logger, config,
            accumulation_steps=args.gradient_accumulation_steps,
            clip_norm=args.clip_norm
        )
        
        scheduler.step()
        
        should_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        
        if should_eval:
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, config,
                return_predictions=(epoch == args.epochs and rank == 0)
            )
            
            val_loss, val_acc = val_results['loss'], val_results['accuracy']
            
            if rank == 0:
                logger.info(
                    f"E{epoch:03d}/{args.epochs} | "
                    f"TrL {train_loss:.4f} TrA {train_acc*100:.2f}% | "
                    f"VaL {val_loss:.4f} VaA {val_acc*100:.2f}% | "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=True
                    )
                    logger.info(f"  Best: {best_acc*100:.2f}%")
                else:
                    patience_counter += 1
                
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=False
                    )
        else:
            if rank == 0:
                logger.info(
                    f"E{epoch:03d}/{args.epochs} | "
                    f"TrL {train_loss:.4f} TrA {train_acc*100:.2f}% | "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )
        
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            if rank == 0:
                logger.info(f"Early stop: epoch {epoch}")
            break
    
    # Final evaluation
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Complete | Best: {best_acc*100:.2f}%")
        logger.info("=" * 60)
        
        checkpoint = torch.load(output_dir / 'best_model.pt')
        if is_ddp:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        final_results = evaluate(
            model, val_loader, class_weights, device, rank, config,
            return_predictions=True
        )
        
        np.savez(
            output_dir / 'final_predictions.npz',
            predictions=final_results['predictions'],
            labels=final_results['labels'],
            probabilities=final_results['probabilities']
        )
        
        cm = confusion_matrix(final_results['labels'], final_results['predictions'])
        logger.info(f"Confusion matrix:\n{cm}")
        np.save(output_dir / 'confusion_matrix.npy', cm)
        
        class_names = ['Flat', 'PSPL', 'Binary']
        report = classification_report(
            final_results['labels'], 
            final_results['predictions'],
            target_names=class_names
        )
        logger.info(f"Report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # Save config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved: {output_dir}")
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
