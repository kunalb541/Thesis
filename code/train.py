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
import math
from typing import Dict, Any, Tuple, Optional, List
from contextlib import contextmanager
from dataclasses import asdict

# Import model
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import RomanMicrolensingGRU, ModelConfig
except ImportError as e:
    print(f"Error importing model: {e}")
    print("Ensure model.py is in the same directory as train.py")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONSTANTS
# =============================================================================
SEED = 42
CLIP_NORM = 1.0
MIN_SEQUENCE_LENGTH = 1


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed() -> Tuple[int, int, int, bool]:
    """
    Initialize distributed training environment.
    
    Returns:
        rank: Global rank of this process
        local_rank: Local rank on this node
        world_size: Total number of processes
        is_ddp: Whether running in distributed mode
    """
    if 'RANK' not in os.environ:
        return 0, 0, 1, False
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if rank == 0:
        print(f"Initializing distributed: {world_size} processes", flush=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for distributed training")
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(seconds=3600)
    )
    
    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return rank, local_rank, world_size, True


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    """
    Set up logging for training.
    
    Only rank 0 logs to console and file; other ranks are silent.
    """
    logger = logging.getLogger("TRAIN")
    logger.handlers.clear()
    logger.propagate = False
    
    if rank == 0:
        logger.setLevel(logging.INFO)
        
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console)
        
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.CRITICAL + 1)
    
    return logger


def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility."""
    seed = seed + rank
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATA LOADING
# =============================================================================
def load_or_create_split(
    data_path: Path,
    logger: Optional[logging.Logger],
    rank: int = 0,
    is_ddp: bool = False,
    val_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Load cached train/val split or create new one.
    
    Only rank 0 creates the split; other ranks wait and load from cache.
    """
    cache_path = data_path.parent / f"{data_path.stem}_split_cache.npz"
    
    if cache_path.exists():
        if rank == 0 and logger:
            logger.info(f"Loading cached split: {cache_path}")
        
        cache = np.load(cache_path)
        train_idx = cache['train_idx']
        val_idx = cache['val_idx']
        stats = {
            'median': float(cache['median']),
            'iqr': float(cache['iqr'])
        }
        
        if rank == 0 and logger:
            logger.info(f"Loaded: train={len(train_idx)}, val={len(val_idx)}")
        
        if is_ddp:
            dist.barrier()
        
        return train_idx, val_idx, stats
    
    if rank == 0:
        if logger:
            logger.info("Creating new train/val split...")
        
        with h5py.File(data_path, 'r', rdcc_nbytes=512*1024*1024) as f:
            labels = f['labels'][:]
            n_samples = len(labels)
            
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                shuffle=True,
                random_state=SEED,
                stratify=labels
            )
            
            flux_data = f['flux']
            sample_size = min(50000, len(train_idx))
            sample_idx = np.sort(np.random.choice(train_idx, sample_size, replace=False))
            
            valid_flux = []
            chunk_size = 10000
            
            for i in range(0, len(sample_idx), chunk_size):
                chunk = sample_idx[i:i + chunk_size]
                flux_chunk = flux_data[chunk.tolist()]
                valid = flux_chunk[~np.isnan(flux_chunk) & (flux_chunk != 0.0)]
                if len(valid) > 0:
                    valid_flux.append(valid)
            
            if valid_flux:
                all_valid = np.concatenate(valid_flux)
                median = float(np.median(all_valid))
                q75, q25 = np.percentile(all_valid, [75, 25])
                iqr = float(max(q75 - q25, 1e-6))
            else:
                median, iqr = 0.0, 1.0
            
            stats = {'median': median, 'iqr': iqr}
            
            if logger:
                unique, counts = np.unique(labels[train_idx], return_counts=True)
                logger.info("Training class distribution:")
                for cls, cnt in zip(unique, counts):
                    pct = 100.0 * cnt / len(train_idx)
                    logger.info(f"  Class {cls}: {cnt:,} ({pct:.2f}%)")
        
        np.savez(
            cache_path,
            train_idx=train_idx,
            val_idx=val_idx,
            median=stats['median'],
            iqr=stats['iqr']
        )
        
        if logger:
            logger.info(f"Cached split: {cache_path}")
    else:
        train_idx, val_idx, stats = None, None, None
    
    if is_ddp:
        dist.barrier()
        
        if rank != 0:
            cache = np.load(cache_path)
            train_idx = cache['train_idx']
            val_idx = cache['val_idx']
            stats = {
                'median': float(cache['median']),
                'iqr': float(cache['iqr'])
            }
    
    return train_idx, val_idx, stats


class MicrolensingDataset(Dataset):
    """
    HDF5 dataset with lazy loading and per-worker file handles.
    """
    
    def __init__(
        self,
        data_path: Path,
        indices: np.ndarray,
        stats: Dict[str, float]
    ):
        self.data_path = str(data_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = stats
        
        self._file = None
        self._flux = None
        self._delta_t = None
        self._labels = None
    
    def _ensure_open(self):
        """Open HDF5 file if not already open."""
        if self._file is None:
            self._file = h5py.File(
                self.data_path,
                'r',
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_nslots=10007,
                libver='latest',
                swmr=True
            )
            self._flux = self._file['flux']
            self._delta_t = self._file['delta_t']
            self._labels = self._file['labels']
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """Return global index and label."""
        self._ensure_open()
        global_idx = int(self.indices[idx])
        label = int(self._labels[global_idx])
        return global_idx, label
    
    def get_batch_data(
        self,
        global_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read flux and delta_t for a batch of global indices."""
        self._ensure_open()
        
        sort_order = np.argsort(global_indices)
        sorted_idx = global_indices[sort_order]
        
        flux = self._flux[sorted_idx.tolist()]
        delta_t = self._delta_t[sorted_idx.tolist()]
        
        unsort = np.argsort(sort_order)
        flux = flux[unsort]
        delta_t = delta_t[unsort]
        
        return flux, delta_t
    
    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except:
                pass


def collate_fn(
    batch: List[Tuple[int, int]],
    dataset: MicrolensingDataset
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function with optimized HDF5 batch reading."""
    global_indices = np.array([x[0] for x in batch], dtype=np.int64)
    labels = np.array([x[1] for x in batch], dtype=np.int64)
    
    flux, delta_t = dataset.get_batch_data(global_indices)
    
    flux = torch.from_numpy(flux.astype(np.float32))
    delta_t = torch.from_numpy(delta_t.astype(np.float32))
    labels = torch.from_numpy(labels).long()
    
    valid_mask = (~torch.isnan(flux)) & (flux != 0.0)
    
    flux = torch.nan_to_num(flux, nan=0.0)
    delta_t = torch.nan_to_num(delta_t, nan=0.0)
    
    median = dataset.stats['median']
    iqr = dataset.stats['iqr']
    flux = (flux - median) / iqr
    flux = flux * valid_mask.float()
    
    lengths = valid_mask.sum(dim=1).long().clamp(min=MIN_SEQUENCE_LENGTH)
    
    return flux, delta_t, lengths, labels


def worker_init_fn(worker_id: int):
    """Initialize random state for each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    data_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: Dict[str, float],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    is_ddp: bool,
    rank: int
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Create training and validation DataLoaders."""
    train_dataset = MicrolensingDataset(data_path, train_idx, stats)
    val_dataset = MicrolensingDataset(data_path, val_idx, stats)
    
    with h5py.File(data_path, 'r') as f:
        all_labels = f['labels'][:]
    train_labels = all_labels[train_idx]
    
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=SEED,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_collate = lambda batch: collate_fn(batch, train_dataset)
    val_collate = lambda batch: collate_fn(batch, val_dataset)
    
    persistent = num_workers > 0
    prefetch = prefetch_factor if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        worker_init_fn=worker_init_fn,
        collate_fn=train_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        worker_init_fn=worker_init_fn,
        collate_fn=val_collate
    )
    
    return train_loader, val_loader, train_labels


# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def compute_class_weights(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device
) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = torch.ones(n_classes, device=device)
    for cls, cnt in zip(unique, counts):
        weights[cls] = total / (n_classes * cnt)
    
    return weights


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


@contextmanager
def cuda_timer(name: str, logger: logging.Logger, rank: int):
    """Context manager for timing CUDA operations."""
    if rank != 0:
        yield
        return
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    
    yield
    
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    logger.info(f"{name}: {elapsed_ms/1000:.2f}s")


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
    config: ModelConfig,
    accumulation_steps: int = 1,
    clip_norm: float = CLIP_NORM
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    
    iterator = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0),
        leave=False
    )
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config.use_amp and scaler is not None
    amp_dtype = torch.bfloat16 if config.use_amp else torch.float32
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=amp_dtype):
            logits = model(flux, delta_t, lengths)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss_scaled = loss / accumulation_steps
        
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum()
            total += batch_size
        
        if rank == 0 and batch_idx % 20 == 0:
            current_loss = (total_loss / total).item()
            current_acc = (correct / total).item() * 100
            iterator.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.1f}%'
            })
    
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / total).item()
    accuracy = (correct / total).item()
    
    return avg_loss, accuracy


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    config: ModelConfig,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    amp_dtype = torch.bfloat16 if config.use_amp else torch.float32
    
    for flux, delta_t, lengths, labels in tqdm(loader, disable=(rank != 0), desc="Eval", leave=False):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=amp_dtype):
            logits = model(flux, delta_t, lengths)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        
        batch_size = labels.size(0)
        preds = logits.argmax(dim=-1)
        
        total_loss += loss * batch_size
        correct += (preds == labels).sum()
        total += batch_size
        
        if return_predictions:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    results = {
        'loss': (total_loss / total).item(),
        'accuracy': (correct / total).item()
    }
    
    if return_predictions:
        results['predictions'] = np.concatenate(all_preds)
        results['labels'] = np.concatenate(all_labels)
        results['probabilities'] = np.concatenate(all_probs)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_acc: float,
    config: ModelConfig,
    stats: Dict,
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__,
        'stats': stats
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train Roman Microlensing Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 training data')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Causal window size')
    parser.add_argument('--hierarchical', action='store_true', default=True,
                        help='Use hierarchical feature extraction')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    parser.add_argument('--attention-pooling', action='store_true', default=True,
                        help='Use attention pooling')
    parser.add_argument('--no-attention-pooling', dest='attention_pooling', action='store_false')
    parser.add_argument('--use-residual', action='store_true', default=True,
                        help='Use residual connections')
    parser.add_argument('--no-residual', dest='use_residual', action='store_false')
    parser.add_argument('--use-layer-norm', action='store_true', default=True,
                        help='Use layer normalization')
    parser.add_argument('--no-layer-norm', dest='use_layer_norm', action='store_false')
    parser.add_argument('--feature-extraction', type=str, default='conv',
                        choices=['conv', 'mlp'], help='Feature extraction method')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size per GPU')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='AdamW weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Learning rate warmup epochs')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Mixed precision
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    
    # Other options
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class-balanced loss weights')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile')
    parser.add_argument('--use-gradient-checkpointing', action='store_true', default=False,
                        help='Use gradient checkpointing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers per GPU')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='Batches to prefetch per worker')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    
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
    set_seed(SEED, rank)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("ROMAN MICROLENSING CLASSIFIER - TRAINING")
        logger.info("=" * 80)
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA: {torch.version.cuda}")
        if torch.cuda.is_available():
            logger.info(f"Device: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"World size: {world_size} GPU(s)")
        logger.info(f"Output: {output_dir}")
    
    # Data
    if rank == 0:
        logger.info("-" * 80)
        logger.info("Loading data...")
    
    train_idx, val_idx, stats = load_or_create_split(
        Path(args.data), logger, rank, is_ddp
    )
    
    if rank == 0:
        logger.info(f"Dataset: {len(train_idx) + len(val_idx):,} samples")
        logger.info(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}")
        logger.info(f"Normalization: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
    
    train_loader, val_loader, train_labels = create_dataloaders(
        Path(args.data), train_idx, val_idx, stats,
        args.batch_size, args.num_workers, args.prefetch_factor,
        is_ddp, rank
    )
    
    if rank == 0:
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps * world_size}")
        logger.info(f"Workers per GPU: {args.num_workers}")
    
    # Model
    if rank == 0:
        logger.info("-" * 80)
        logger.info("Building model...")
    
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
    
    model = RomanMicrolensingGRU(config).to(device)
    
    if rank == 0:
        n_params = model.count_parameters()
        logger.info(f"Parameters: {n_params:,}")
        logger.info(f"Config: {config}")
    
    # DDP wrapper
    # CRITICAL: static_graph=False because AttentionPooling uses arithmetic masking
    # that produces different gradients based on mask content
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=True,
            static_graph=False  # REQUIRED for branch-free attention pooling
        )
    
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info("Compiling model...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        fused=torch.cuda.is_available()
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr
    )
    
    # GradScaler only needed for float16, not bfloat16
    use_scaler = args.use_amp and torch.cuda.is_available()
    # Check if bfloat16 is supported - if so, don't use scaler
    if use_scaler and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        use_scaler = False
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
        if rank == 0:
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
    # Training loop
    if is_ddp:
        dist.barrier()
    
    if rank == 0:
        logger.info("-" * 80)
        logger.info("Starting training...")
        logger.info("-" * 80)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        with cuda_timer(f"Epoch {epoch} train", logger, rank):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scaler,
                class_weights, device, rank, world_size, epoch, config,
                accumulation_steps=args.accumulation_steps,
                clip_norm=args.clip_norm
            )
        
        scheduler.step()
        
        should_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        
        if should_eval:
            with cuda_timer(f"Epoch {epoch} eval", logger, rank):
                val_results = evaluate(
                    model, val_loader, class_weights, device, rank, world_size, config,
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
                        model, optimizer, scheduler, scaler, epoch,
                        best_acc, config, stats, output_dir, is_best=True
                    )
                    logger.info(f"  -> New best: {best_acc*100:.2f}%")
                else:
                    patience_counter += 1
                
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        best_acc, config, stats, output_dir, is_best=False
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
        
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final evaluation
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Training complete | Best accuracy: {best_acc*100:.2f}%")
        logger.info("=" * 80)
        
        checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
        if is_ddp:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Running final evaluation...")
        final_results = evaluate(
            model, val_loader, class_weights, device, rank, world_size, config,
            return_predictions=True
        )
        
        np.savez(
            output_dir / 'final_predictions.npz',
            predictions=final_results['predictions'],
            labels=final_results['labels'],
            probabilities=final_results['probabilities']
        )
        
        cm = confusion_matrix(final_results['labels'], final_results['predictions'])
        logger.info(f"\nConfusion matrix:\n{cm}")
        np.save(output_dir / 'confusion_matrix.npy', cm)
        
        report = classification_report(
            final_results['labels'],
            final_results['predictions'],
            target_names=['Flat', 'PSPL', 'Binary'],
            digits=4
        )
        logger.info(f"\nClassification report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        config_dict = {
            'model_config': asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__,
            'training_args': vars(args),
            'stats': stats,
            'best_accuracy': float(best_acc),
            'world_size': world_size,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"\nResults saved to: {output_dir}")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
