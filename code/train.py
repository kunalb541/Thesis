import os
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
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
import random

# Import God Mode Model
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import GodModeCausalGRU, GRUConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    print("\nCRITICAL: 'model.py' not found. Ensure both files are in the same directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
SEED = 42
PREFETCH_FACTOR = 4

# =============================================================================
# CUSTOM JSON ENCODER
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            return rank, local_rank, world_size, True
        else:
            return 0, 0, 1, False
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank: int, output_dir: Path, debug_mode: bool):
    logger = logging.getLogger("TRAIN")
    level = logging.DEBUG if debug_mode else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / "training.log"))

    logging.basicConfig(
        level=level, 
        format='%(asctime)s | Rank %(rank)s | %(levelname)s | %(message)s' if rank != 0 else '%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers, 
        force=True
    )
    if rank != 0:
        logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    return logger

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# DATASET
# =============================================================================
class MicrolensingDataset(Dataset):
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, stats: dict = None):
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)

        self.padding_mask = (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        if stats is not None:
            median = stats['median']
            iqr = stats['iqr'] if stats['iqr'] > 1e-6 else 1.0
            
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.flux[idx], self.delta_t[idx], self.lengths[idx], self.labels[idx]

def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                scaler, class_weights: torch.Tensor, device: torch.device, 
                rank: int, epoch: int, logger: logging.Logger, config: GRUConfig) -> tuple:
    
    model_to_train = model.module if isinstance(model, DDP) else model
    model_to_train.train()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    nan_count = 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0), leave=False)

    for step, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        # Note: AutoCast is handled inside the GodMode model forward pass
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
        
        # Loss Calculation
        if config.hierarchical:
            # Output is already log_softmax from hierarchical head
            log_probs = out['logits']
            loss = F.nll_loss(log_probs, labels, weight=class_weights, reduction='mean')
        else:
            # Standard raw logits
            final_logits = out['logits']
            loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='mean')

        batch_samples = labels.size(0)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Check for NaN gradients
        all_finite = True
        for param in model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    param.grad = None  # Effectively zero out
                    all_finite = False

        if all_finite:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if torch.isfinite(norm):
                scaler.step(optimizer)
            else:
                nan_count += 1
        else:
            nan_count += 1
        
        scaler.update()
        
        # Metrics
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        
        with torch.no_grad():
            preds = out['probs'].argmax(dim=1)
            correct += (preds == labels).sum().item()

    if dist.is_initialized():
        metrics = torch.tensor([total_loss, float(correct), float(total_samples), float(nan_count)], 
                               dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples, nan_count = metrics.cpu().numpy()
    
    avg_loss = float(total_loss / max(total_samples, 1))
    accuracy = float(correct / max(total_samples, 1))
    
    return avg_loss, accuracy, int(nan_count)

# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor, 
             device: torch.device, rank: int, config: GRUConfig, check_early_detection: bool = False) -> dict:
    
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    
    milestones = [0.2, 0.5, 1.0]
    early_correct = {m: 0 for m in milestones}
    early_total = {m: 0 for m in milestones}

    iterator = tqdm(loader, desc="Audit" if check_early_detection else "Eval", 
                    disable=(rank != 0), leave=False)

    for flux, delta_t, lengths, labels in iterator:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if check_early_detection and not config.hierarchical:
            # Request all timesteps for audit
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            
            # Standard metrics on final
            final_logits = out['logits']
            if config.hierarchical:
                loss = F.nll_loss(final_logits, labels, weight=class_weights, reduction='mean')
            else:
                loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='mean')
            
            # Early Detection Logic
            probs_seq = out['probs_seq'] # (B, T, Classes)
            B = flux.size(0)
            
            for b in range(B):
                seq_len = lengths[b].item()
                true_lbl = labels[b].item()
                for m in milestones:
                    idx = max(0, min(int(seq_len * m) - 1, seq_len - 1))
                    # Check prediction at specific % of lightcurve
                    if probs_seq[b, idx].argmax().item() == true_lbl:
                        early_correct[m] += 1
                    early_total[m] += 1
        else:
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
            final_logits = out['logits']
            if config.hierarchical:
                loss = F.nll_loss(final_logits, labels, weight=class_weights, reduction='mean')
            else:
                loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='mean')
        
        batch_samples = labels.size(0)
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        
        preds = out['probs'].argmax(dim=1)
        correct += (preds == labels).sum().item()

    if dist.is_initialized():
        metrics = torch.tensor([total_loss, float(correct), float(total_samples)], 
                               dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples = metrics.cpu().numpy()
        
        if check_early_detection:
            for m in milestones:
                e_metrics = torch.tensor([float(early_correct[m]), float(early_total[m])], 
                                         dtype=torch.float64, device=device)
                dist.all_reduce(e_metrics, op=dist.ReduceOp.SUM)
                ec, et = e_metrics.cpu().numpy()
                early_correct[m], early_total[m] = int(ec), int(et)

    results = {
        'loss': float(total_loss / max(total_samples, 1)),
        'accuracy': float(correct / max(total_samples, 1))
    }
    
    if check_early_detection:
        results['early_stats'] = {
            m: float(early_correct[m] / max(early_total[m], 1))
            for m in milestones
        }
    
    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="God Mode Causal Training")
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8) # Kept for back-compat, not used in GRU
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_checkpoint', action='store_true')
    parser.add_argument('--compile', action='store_true', help="Use torch.compile")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}') if is_ddp else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{timestamp}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(rank, output_dir, args.debug)
    set_seed(SEED)

    # Determine precision based on hardware
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        if rank == 0: logger.info("⚡ Using BFloat16 Precision")
    else:
        amp_dtype = torch.float16
        if rank == 0: logger.info("⚡ Using Float16 Precision")

    # Load Data
    if rank == 0: logger.info(f"Loading data from {args.data}...")
    try:
        data = np.load(args.data, allow_pickle=True)
        flux = data['flux'].squeeze()
        labels = data['labels']
        if 'delta_t' in data:
            delta_t = data['delta_t'].squeeze()
        else:
            if rank == 0: logger.warning("No delta_t found, using zeros.")
            delta_t = np.zeros_like(flux)
    except Exception as e:
        if rank == 0: logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Stats Calculation
    sample_flux = flux[:10000].flatten()
    valid_flux = sample_flux[sample_flux != 0.0]
    stats = {
        'median': float(np.median(valid_flux)) if len(valid_flux) > 0 else 0.0,
        'iqr': float(np.subtract(*np.percentile(valid_flux, [75, 25]))) if len(valid_flux) > 0 else 1.0
    }

    # Split
    flux_tr, flux_val, dt_tr, dt_val, y_tr, y_val = train_test_split(
        flux, delta_t, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    # Dataset & Loader
    train_ds = MicrolensingDataset(flux_tr, dt_tr, y_tr, stats)
    val_ds = MicrolensingDataset(flux_val, dt_val, y_val, stats)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED, drop_last=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              pin_memory=True, prefetch_factor=PREFETCH_FACTOR if args.num_workers > 0 else None,
                              persistent_workers=(args.num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

    # Model Setup
    n_classes = len(np.unique(labels))
    config = GRUConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        window_size=args.window_size,
        dropout=args.dropout,
        n_classes=n_classes,
        use_gradient_checkpointing=args.grad_checkpoint,
        compile_model=args.compile
    )

    model = GodModeCausalGRU(config, dtype=amp_dtype).to(device)
    
    if args.compile and torch.cuda.is_available():
        if rank == 0: logger.info("🔧 Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            if rank == 0: logger.warning(f"Compilation failed, falling back: {e}")

    if rank == 0:
        logger.info(f"Model Parameters: {count_parameters(model):,}")
        logger.info(f"Configuration: {config}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    class_weights = compute_class_weights(y_tr, n_classes).to(device)

    # Resume Logic
    start_epoch = 1
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if args.resume and os.path.exists(args.resume):
        if rank == 0: logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        state_dict = checkpoint['state_dict']
        # Handle DDP prefix mismatch
        if is_ddp and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_ddp and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('accuracy', 0.0)
        history = checkpoint.get('history', history)

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, device, rank, epoch, logger, config
        )
        
        is_audit = (epoch % 5 == 0) or (epoch == args.epochs)
        val_results = evaluate(
            model, val_loader, class_weights, device, rank, config, check_early_detection=is_audit
        )
        
        scheduler.step(val_results['accuracy'])
        
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            
            logger.info(f"Epoch {epoch:03d} | Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f} | "
                        f"NaNs: {nan_count}")
            
            if is_audit and 'early_stats' in val_results:
                stats_log = val_results['early_stats']
                logger.info(f" >> Early Detection Audit: 20%={stats_log[0.2]:.3f} | 50%={stats_log[0.5]:.3f} | 100%={stats_log[1.0]:.3f}")

            if val_results['accuracy'] > best_acc:
                best_acc = val_results['accuracy']
                state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'state_dict': state_to_save,
                    'config': config.__dict__,
                    'accuracy': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'history': history
                }, output_dir / "best_model.pt")
                logger.info(f" ⭐ Best model saved (Acc: {best_acc:.4f})")
            
            with open(output_dir / "history.json", 'w') as f:
                json.dump(history, f, indent=2, cls=NumpyEncoder)

    cleanup_distributed()

if __name__ == '__main__':
    main()
