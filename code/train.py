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
from contextlib import contextmanager

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import RomanMicrolensingGRU, ModelConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"Error importing model: {e}")
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


def setup_distributed() -> Tuple[int, int, int, bool]:
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if rank == 0:
            print(f"Initializing distributed: {world_size} processes", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
            dist.init_process_group(
                backend='nccl', 
                init_method='env://',
                timeout=timedelta(seconds=1800)
            )
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            torch.cuda.empty_cache()
            gc.collect()
            
            return rank, local_rank, world_size, True
    
    return 0, 0, 1, False


def setup_logging(rank: int, output_dir: Path):
    logger = logging.getLogger("TRAIN")
    
    if rank == 0:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
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
    torch.use_deterministic_algorithms(False)


def fast_load_labels_and_split(
    data_path: Path, 
    logger: Optional[logging.Logger],
    rank: int = 0,
    is_ddp: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    OPTIMIZED: Use memory-mapped arrays and cached splits to avoid full data loading.
    Reduces startup time from minutes to seconds.
    """
    cache_path = data_path.parent / f"{data_path.stem}_split_cache.npz"
    
    # Load cached split if available
    if cache_path.exists():
        if rank == 0 and logger:
            logger.info(f"Loading cached split from {cache_path}")
        cache = np.load(cache_path)
        train_idx = cache['train_idx']
        val_idx = cache['val_idx']
        stats = {'median': float(cache['median']), 'iqr': float(cache['iqr'])}
        
        if rank == 0 and logger:
            logger.info(f"Loaded: train={len(train_idx)}, val={len(val_idx)}")
        
        if is_ddp:
            dist.barrier()
        
        return train_idx, val_idx, stats
    
    # Create new split (only rank 0)
    if rank == 0:
        with h5py.File(data_path, 'r', rdcc_nbytes=1024**3, rdcc_nslots=10007) as f:
            labels = f['labels'][:]
            
            indices = np.arange(len(labels))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, shuffle=True, random_state=SEED, stratify=labels
            )
            
            # Compute normalization stats using random sampling
            flux_data = f['flux']
            sample_size = min(50000, len(train_idx))  # Reduced sample size
            sample_indices = np.sort(np.random.choice(train_idx, size=sample_size, replace=False))
            sample_indices_sorted = np.sort(sample_indices)
            
            # Read in chunks to avoid memory spikes
            chunk_size = 10000
            valid_flux_list = []
            
            for i in range(0, len(sample_indices_sorted), chunk_size):
                chunk_idx = sample_indices_sorted[i:i+chunk_size]
                chunk_flux = flux_data[chunk_idx.tolist()]
                valid_chunk = chunk_flux[~np.isnan(chunk_flux) & (chunk_flux != 0.0)]
                if len(valid_chunk) > 0:
                    valid_flux_list.append(valid_chunk)
            
            if valid_flux_list:
                valid_flux = np.concatenate(valid_flux_list)
                median = float(np.median(valid_flux))
                q75, q25 = np.percentile(valid_flux, [75, 25])
                iqr = float(q75 - q25)
                if iqr < 1e-6:
                    iqr = 1.0
            else:
                median = 0.0
                iqr = 1.0
            
            del valid_flux_list, valid_flux
            gc.collect()

            stats = {'median': median, 'iqr': iqr}
            
            if logger:
                unique, counts = np.unique(labels[train_idx], return_counts=True)
                logger.info("Training class distribution:")
                for cls, cnt in zip(unique, counts):
                    logger.info(f"  Class {cls}: {cnt} ({100*cnt/len(train_idx):.2f}%)")
        
        # Save cache
        np.savez(
            cache_path, 
            train_idx=train_idx, 
            val_idx=val_idx, 
            median=stats['median'], 
            iqr=stats['iqr']
        )
        if logger:
            logger.info(f"Cached split to {cache_path}")
    else:
        # Other ranks wait
        train_idx = None
        val_idx = None
        stats = None
    
    if is_ddp:
        dist.barrier()
        
        # Broadcast split info to all ranks
        if rank != 0:
            cache = np.load(cache_path)
            train_idx = cache['train_idx']
            val_idx = cache['val_idx']
            stats = {'median': float(cache['median']), 'iqr': float(cache['iqr'])}
    
    return train_idx, val_idx, stats


class OptimizedH5Dataset(Dataset):
    """
    OPTIMIZED: Each worker maintains its own file handle and uses chunk caching.
    Eliminates file handle serialization overhead.
    """
    def __init__(
        self, 
        data_path: Path,
        indices: np.ndarray,
        stats: Dict,
        handle_nans: str = 'zero'
    ):
        self.data_path = data_path
        self.indices = indices
        self.stats = stats
        self.handle_nans = handle_nans
        self._h5_file = None
        self._flux_dset = None
        self._delta_t_dset = None
        self._labels_dset = None
        
    def _open_h5_file(self):
        """Open HDF5 file with optimized settings for throughput."""
        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.data_path, 
                'r', 
                rdcc_nbytes=512*1024*1024,  # 512 MB chunk cache per worker
                rdcc_nslots=100003,          # Large prime number for hash slots
                libver='latest',
                swmr=True
            )
            self._flux_dset = self._h5_file['flux']
            self._delta_t_dset = self._h5_file['delta_t']
            self._labels_dset = self._h5_file['labels']
    
    @property
    def dsets(self):
        self._open_h5_file()
        return self._flux_dset, self._delta_t_dset, self._labels_dset
            
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx_in_subset: int) -> Tuple[int, int]:
        self._open_h5_file()
        global_idx = int(self.indices[idx_in_subset])
        label = int(self._labels_dset[global_idx])
        return global_idx, label

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


def optimized_collate_fn(batch: List[Tuple[int, int]], dataset: OptimizedH5Dataset):
    """
    OPTIMIZED: Vectorized batch reading with minimal memory allocations.
    """
    global_indices = np.array([item[0] for item in batch], dtype=np.int64)
    labels_list = np.array([item[1] for item in batch], dtype=np.int64)
    
    # Sort for optimal HDF5 sequential read
    sort_order = np.argsort(global_indices)
    sorted_indices = global_indices[sort_order]
    
    flux_dset, delta_t_dset, _ = dataset.dsets
    
    # Read data in sorted order (HDF5 optimization)
    flux_batch = flux_dset[sorted_indices.tolist()]
    delta_t_batch = delta_t_dset[sorted_indices.tolist()]
    
    # Unsort to match original batch order
    unsort_order = np.argsort(sort_order)
    flux_batch = flux_batch[unsort_order]
    delta_t_batch = delta_t_batch[unsort_order]
    
    # Convert to tensors (direct from numpy for speed)
    flux = torch.from_numpy(flux_batch.astype(np.float32))
    delta_t = torch.from_numpy(delta_t_batch.astype(np.float32))
    labels = torch.from_numpy(labels_list).long()
    
    # Create padding mask
    padding_mask = (~torch.isnan(flux)) & (flux != 0.0)
    
    # Handle NaNs
    if dataset.handle_nans == 'zero':
        flux = torch.nan_to_num(flux, nan=0.0)
    
    delta_t = torch.nan_to_num(delta_t, nan=0.0)
    
    # Normalize flux
    median = dataset.stats['median']
    iqr = dataset.stats['iqr']
    flux = (flux - median) / iqr
    flux = flux * padding_mask.float()
    
    # Compute lengths
    lengths = padding_mask.sum(dim=1).long().clamp(min=1)
    
    return flux, delta_t, lengths, labels


def worker_init_fn(worker_id: int):
    """Initialize random state for each worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    weight_tensor = torch.zeros(3, device=device)
    weight_tensor[unique] = torch.from_numpy(weights).float().to(device)
    return weight_tensor


@contextmanager
def timer(name: str, logger: logging.Logger, rank: int):
    if rank == 0:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        logger.info(f"{name}: {elapsed:.2f}s")
    else:
        yield


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
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
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
            output = model(flux, delta_t, lengths=lengths)
            loss = F.cross_entropy(output, labels, weight=class_weights)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            pred = output.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += (loss.item() * accumulation_steps) * labels.size(0)
        
        if rank == 0 and batch_idx % 10 == 0:
            iterator.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    torch.cuda.empty_cache()
    
    return total_loss / total, correct / total


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
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for flux, delta_t, lengths, labels in tqdm(loader, disable=(rank != 0), desc="Evaluating"):
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
                output = model(flux, delta_t, lengths=lengths)
                loss = F.cross_entropy(output, labels, weight=class_weights)
            
            pred = output.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
            if return_predictions:
                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(output.cpu().numpy())
    
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
    scaler: torch.cuda.amp.GradScaler,
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
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'config': config.__dict__,
        'stats': stats
    }
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


def main():
    parser = argparse.ArgumentParser(description='Train Roman microlensing classifier')
    
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='../results', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    parser.add_argument('--d-model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--window-size', type=int, default=5, help='Causal window size')
    
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_true', 
                        help='Use hierarchical classification')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    parser.set_defaults(hierarchical=True)
    
    parser.add_argument('--feature-extraction', type=str, choices=['conv', 'mlp'], default='conv',
                        help='Feature extraction method')
    
    parser.add_argument('--attention-pooling', dest='attention_pooling', action='store_true',
                        help='Use attention pooling')
    parser.add_argument('--no-attention-pooling', dest='attention_pooling', action='store_false')
    parser.set_defaults(attention_pooling=True)
    
    parser.add_argument('--use-residual', dest='use_residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--no-residual', dest='use_residual', action='store_false')
    parser.set_defaults(use_residual=True)
    
    parser.add_argument('--use-layer-norm', dest='use_layer_norm', action='store_true',
                        help='Use layer normalization')
    parser.add_argument('--no-layer-norm', dest='use_layer_norm', action='store_false')
    parser.set_defaults(use_layer_norm=True)
    
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--clip-norm', type=float, default=1.0, help='Gradient clipping norm')
    
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', 
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)
    
    parser.add_argument('--use-gradient-checkpointing', dest='use_gradient_checkpointing', 
                        action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', dest='use_gradient_checkpointing', 
                        action='store_false')
    parser.set_defaults(use_gradient_checkpointing=False)
    
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weighting')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile')
    
    parser.add_argument('--num-workers', type=int, default=4, 
                        help='DataLoader workers per GPU')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='Batches to prefetch per worker')
    
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=0, 
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
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
        logger.info(f"Random seed: {SEED}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        if torch.cuda.is_available():
            logger.info(f"Device: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"World size: {world_size} GPUs")
    
    set_seed_everywhere(SEED, rank)
    
    if rank == 0:
        logger.info(f"Loading data: {args.data}")
    
    train_idx, val_idx, stats = fast_load_labels_and_split(
        Path(args.data), 
        logger if rank == 0 else None,
        rank=rank,
        is_ddp=is_ddp
    )
    
    if rank == 0:
        logger.info(f"Dataset size: {len(train_idx) + len(val_idx)} samples")
        logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}")
        logger.info(f"Normalization: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
    
    train_dataset = OptimizedH5Dataset(Path(args.data), train_idx, stats)
    val_dataset = OptimizedH5Dataset(Path(args.data), val_idx, stats)
    
    with h5py.File(args.data, 'r') as f:
        all_labels = f['labels'][:]
    labels_train_for_weights = all_labels[train_idx]
    
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_collate = lambda batch: optimized_collate_fn(batch, train_dataset)
    val_collate = lambda batch: optimized_collate_fn(batch, val_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn,
        collate_fn=train_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn,
        collate_fn=val_collate
    )
    
    if rank == 0:
        logger.info(f"DataLoader workers per GPU: {args.num_workers}")
        logger.info(f"Total DataLoader workers: {args.num_workers * world_size}")
        logger.info(f"Optimized HDF5 batch reading enabled")
    
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
    
    amp_dtype = torch.bfloat16 if args.use_amp and torch.cuda.is_available() else torch.float32
    model = RomanMicrolensingGRU(config).to(device)
    
    if rank == 0:
        n_params = count_parameters(model)
        logger.info(f"Model parameters: {n_params:,}")
        logger.info(f"Architecture: {config}")
        
        effective_batch = args.batch_size * args.accumulation_steps * world_size
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Gradient checkpointing: {args.use_gradient_checkpointing}")
        logger.info(f"Mixed precision: {args.use_amp}")
    
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=True,
            static_graph=True
        )
    
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info("Compiling model with torch.compile")
        model = torch.compile(model, mode='max-autotune')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        fused=True
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if args.use_class_weights:
        class_weights = compute_class_weights(labels_train_for_weights, device)
        if rank == 0:
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(3, device=device)
    
    if is_ddp:
        dist.barrier()
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Training started")
        logger.info("=" * 80)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        with timer(f"Epoch {epoch} training", logger, rank):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scaler,
                class_weights, device, rank, epoch, logger, config,
                accumulation_steps=args.accumulation_steps,
                clip_norm=args.clip_norm
            )
        
        scheduler.step()
        
        should_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        
        if should_eval:
            with timer(f"Epoch {epoch} evaluation", logger, rank):
                val_results = evaluate(
                    model, val_loader, class_weights, device, rank, config,
                    return_predictions=(epoch == args.epochs and rank == 0)
                )
            
            val_loss, val_acc = val_results['loss'], val_results['accuracy']
            
            if rank == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=True
                    )
                    logger.info(f"New best: {best_acc*100:.2f}%")
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
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
        
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch}")
            break
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Training complete | Best accuracy: {best_acc*100:.2f}%")
        logger.info("=" * 80)
        
        checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
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
        
        report = classification_report(
            final_results['labels'], 
            final_results['predictions'],
            target_names=['Flat', 'PSPL', 'Binary']
        )
        logger.info(f"Classification report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        config_dict = {
            'model_config': config.__dict__,
            'training_args': vars(args),
            'stats': stats,
            'best_accuracy': float(best_acc),
            'world_size': world_size
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved: {output_dir}")
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
