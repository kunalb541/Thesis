from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

__version__: str = "7.1.0"

__all__ = [
    "MicrolensingDataset",
    "load_data_to_shared_memory",
    "load_and_split_data",
    "WarmupCosineScheduler",
    "compute_class_weights",
    "compute_hierarchical_loss",
    "train_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint_for_resume",
    "setup_ddp",
    "cleanup_ddp",
    "cleanup_shared_memory",
    "torch_load_compat",
    "main",
]

# =============================================================================
# TORCH SERIALIZATION COMPATIBILITY
# =============================================================================

try:
    import torch.serialization
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except (AttributeError, TypeError):
        pass
except ImportError:
    pass

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def _configure_environment() -> None:
    """Environment configuration for distributed training."""
    os.environ.setdefault('NCCL_TIMEOUT', '600')
    os.environ.setdefault('NCCL_SOCKET_TIMEOUT', '300')
    os.environ.setdefault('NCCL_IB_TIMEOUT', '20')
    os.environ.setdefault('NCCL_IB_DISABLE', '0')
    os.environ.setdefault('NCCL_NET_GDR_LEVEL', '3')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_P2P_LEVEL', '5')
    os.environ.setdefault('NCCL_MIN_NCHANNELS', '16')
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('KINETO_LOG_LEVEL', '5')
    os.environ.setdefault('TORCH_NCCL_AVOID_RECORD_STREAMS', '1')
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')

_configure_environment()

# Import model after environment setup
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model import ModelConfig, RomanMicrolensingClassifier, HierarchicalOutput

# =============================================================================
# CONSTANTS
# =============================================================================

SEED: int = 42
EPS: float = 1e-8
CLASS_NAMES: Tuple[str, str, str] = ('Flat', 'PSPL', 'Binary')
N_CLASSES: int = 3
LOG_FORMAT: str = '%(asctime)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 50
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_PREFETCH_FACTOR: int = 2
DEFAULT_ACCUMULATION_STEPS: int = 1
DEFAULT_CLIP_NORM: float = 1.0
DEFAULT_WARMUP_EPOCHS: int = 3
DEFAULT_VAL_FRACTION: float = 0.1

DEFAULT_STAGE1_WEIGHT: float = 1.0
DEFAULT_STAGE2_WEIGHT: float = 1.0
DEFAULT_AUX_WEIGHT: float = 0.3
DEFAULT_NLL_WEIGHT: float = 0.5

PROGRESS_UPDATE_FREQ: int = 50
DDP_INIT_TIMEOUT_MINUTES: int = 10

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(rank: int) -> logging.Logger:
    """Configure logging for distributed training."""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)

    return logger

logger = setup_logging(0)

# =============================================================================
# TORCH LOAD COMPATIBILITY WRAPPER
# =============================================================================

def torch_load_compat(
    path: Union[str, Path],
    map_location: Union[str, torch.device, None],
    weights_only: bool = False
) -> Dict[str, Any]:
    """
    Compatibility torch.load wrapper for older PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

# =============================================================================
# UTILITIES
# =============================================================================

def create_experiment_dir(base_dir: Path, args: argparse.Namespace) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = (
        f"d{args.d_model}_"
        f"l{args.n_layers}_"
        f"{'hier' if args.hierarchical else 'flat'}_"
        f"{timestamp}"
    )
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    return exp_dir

def is_main_process(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0 or rank == -1

def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_cuda() -> None:
    """Configure CUDA settings for optimal performance."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# =============================================================================
# MULTI-GPU SHARED MEMORY
# =============================================================================

def get_node_id(rank: int, local_rank: int) -> int:
    """Determine which physical node this process is on."""
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE",
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    ))
    return (rank - local_rank) // max(local_world_size, 1)

def load_data_to_shared_memory(
    file_path: str,
    rank: int,
    local_rank: int,
    is_ddp: bool
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data to /dev/shm RAM-backed filesystem.
    
    Only local_rank=0 per node copies the file. Other ranks wait and then
    load from the same /dev/shm path.
    
    Returns
    -------
    shm_path : str
        Path to shared memory file.
    flux : np.ndarray
        Flux/magnification array [N, T].
    delta_t : np.ndarray
        Time interval array [N, T].
    labels : np.ndarray
        Labels array [N].
    """
    node_id = get_node_id(rank, local_rank)
    
    src = Path(file_path)
    suffix = src.suffix
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    shm_path = f"/dev/shm/{src.stem}_job{job_id}_node{node_id}{suffix}"
    
    if local_rank == 0:
        if is_main_process(rank):
            logger.info("=" * 80)
            logger.info("/dev/shm OPTIMIZATION: RAM-Backed File Loading")
            logger.info("=" * 80)
            logger.info(f"Node {node_id} (Rank {rank}): Setting up /dev/shm...")
        
        if not Path('/dev/shm').exists():
            raise RuntimeError("/dev/shm not available on this system!")
        
        try:
            if not Path(shm_path).exists():
                shutil.copy2(file_path, shm_path)
                file_size_mb = Path(shm_path).stat().st_size / 1e6
                if is_main_process(rank):
                    logger.info(f"  Copied {file_size_mb:.1f} MB to {shm_path}")
            else:
                file_size_mb = Path(shm_path).stat().st_size / 1e6
                if is_main_process(rank):
                    logger.info(f"  Found existing {shm_path} ({file_size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to setup /dev/shm: {e}")
            raise
        
        if is_main_process(rank):
            logger.info("  Loading data from /dev/shm...")
        
        flux, delta_t, labels = _load_arrays_from_file(shm_path)
        
        if is_main_process(rank):
            total_mem = (flux.nbytes + delta_t.nbytes + labels.nbytes) / 1e9
            logger.info(f"  Data loaded: {total_mem:.2f} GB")
            logger.info(f"  Shape: flux={flux.shape}, delta_t={delta_t.shape}, labels={labels.shape}")
        
        if is_ddp:
            dist.barrier()
    else:
        if is_ddp:
            dist.barrier()
        
        flux, delta_t, labels = _load_arrays_from_file(shm_path)
    
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE",
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    ))
    if local_rank == local_world_size - 1 and is_main_process(rank):
        logger.info("=" * 80)
        total_mem = (flux.nbytes + delta_t.nbytes + labels.nbytes) / 1e9
        logger.info(f"Node {node_id}: All {local_world_size} GPUs loaded {total_mem:.2f} GB each")
        logger.info("=" * 80)
    
    return shm_path, flux, delta_t, labels

def _load_arrays_from_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load numpy arrays from HDF5 or NPZ file."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        flux = data.get('flux', data.get('magnification', data.get('mag')))
        delta_t = data['delta_t']
        labels = data.get('labels', data['y'])
    
    elif file_path.suffix in ['.h5', '.hdf5']:
        with h5py.File(str(file_path), 'r') as f:
            if 'flux' in f:
                flux = f['flux'][:]
            elif 'magnification' in f:
                flux = f['magnification'][:]
            else:
                flux = f['mag'][:]
            delta_t = f['delta_t'][:]
            labels = f['labels'][:]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return flux.astype(np.float32), delta_t.astype(np.float32), labels.astype(np.int64)

def cleanup_shared_memory(shm_path: str, rank: int, local_rank: int) -> None:
    """
    Cleanup /dev/shm files (only local_rank=0 deletes).
    
    Caller must ensure dist.barrier() is called BEFORE this function.
    """
    if local_rank == 0:
        try:
            if Path(shm_path).exists():
                Path(shm_path).unlink()
                if is_main_process(rank):
                    node_id = get_node_id(rank, local_rank)
                    logger.info(f"Node {node_id}: Cleaned up /dev/shm")
        except Exception as e:
            logger.warning(f"Failed to cleanup {shm_path}: {e}")

# =============================================================================
# DATASET
# =============================================================================

class MicrolensingDataset(Dataset):
    """
    Dataset for Roman microlensing light curves.
    
    Compatible with model.py v7.1.0:
    - Returns (flux, delta_t, label) tuples
    - Model infers observation mask from flux != 0.0
    - Normalization applied to valid observations only
    """
    
    def __init__(
        self,
        flux: np.ndarray,
        delta_t: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        flux_mean: float,
        flux_std: float,
        delta_t_mean: float,
        delta_t_std: float,
        normalize: bool = True
    ) -> None:
        self.flux = flux
        self.delta_t = delta_t
        self.labels = labels
        self.indices = indices
        self.normalize = normalize
        
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.delta_t_mean = delta_t_mean
        self.delta_t_std = delta_t_std
        
        self._flux_scale = 1.0 / (flux_std + EPS)
        self._dt_scale = 1.0 / (delta_t_std + EPS)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """
        Get a single sample.
        
        Returns
        -------
        flux : Tensor [T]
            Flux values. Missing observations = 0.0.
        delta_t : Tensor [T]
            Time differences.
        label : int
            Class label.
        """
        full_idx = self.indices[idx]
        
        flux_raw = self.flux[full_idx].copy()
        delta_t_raw = self.delta_t[full_idx].copy()
        label = int(self.labels[full_idx])
        
        if self.normalize:
            # Only normalize valid (non-zero) observations
            valid_mask = flux_raw != 0.0
            if valid_mask.any():
                flux_raw[valid_mask] = (flux_raw[valid_mask] - self.flux_mean) * self._flux_scale
                delta_t_raw[valid_mask] = (delta_t_raw[valid_mask] - self.delta_t_mean) * self._dt_scale
        
        flux_tensor = torch.from_numpy(flux_raw).float()
        delta_t_tensor = torch.from_numpy(delta_t_raw).float()
        
        return flux_tensor, delta_t_tensor, label
    
    def __del__(self) -> None:
        """Cleanup references to prevent memory leaks."""
        try:
            self.flux = None
            self.delta_t = None
            self.labels = None
            self.indices = None
        except Exception:
            pass

# =============================================================================
# DATA LOADING
# =============================================================================

def compute_normalization_statistics_from_indices(
    flux: np.ndarray,
    delta_t: np.ndarray,
    train_idx: np.ndarray,
    rank: int = 0
) -> Dict[str, float]:
    """Compute normalization stats using ONLY the train split (valid observations only)."""
    flux_train = flux[train_idx]
    dt_train = delta_t[train_idx]

    flux_valid = flux_train[flux_train != 0.0]
    dt_valid = dt_train[dt_train != 0.0]

    stats = {
        "flux_mean": float(np.mean(flux_valid)),
        "flux_std": float(np.std(flux_valid)),
        "delta_t_mean": float(np.mean(dt_valid)),
        "delta_t_std": float(np.std(dt_valid)),
    }

    if is_main_process(rank):
        logger.info("Normalization statistics (TRAIN-ONLY):")
        logger.info(f"  Flux: mean={stats['flux_mean']:.4f}, std={stats['flux_std']:.4f}")
        logger.info(f"  Delta_t: mean={stats['delta_t_mean']:.4f}, std={stats['delta_t_std']:.4f}")

    return stats
    
def load_and_split_data(
    file_path: str,
    val_fraction: float,
    seed: int,
    rank: int,
    is_ddp: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Load data, compute statistics, and split into train/val.
    
    Rank 0 computes everything and broadcasts to avoid redundant file reads.
    """
    if is_ddp:
        if rank == 0:
            flux, delta_t, labels = _load_arrays_from_file(file_path)
            total_samples = len(labels)
            
            stats = compute_normalization_statistics_from_indices(flux, delta_t, train_idx, rank)
            
            indices = np.arange(total_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                stratify=labels,
                random_state=seed
            )
            
            train_labels = labels[train_idx]
            
            logger.info(f"Dataset: {format_number(total_samples)} samples")
            logger.info(f"  Train: {format_number(len(train_idx))} samples")
            logger.info(f"  Val:   {format_number(len(val_idx))} samples")
            
            unique, counts = np.unique(train_labels, return_counts=True)
            logger.info("Class distribution (train):")
            for cls_idx, count in zip(unique, counts):
                pct = 100 * count / len(train_labels)
                logger.info(f"  {CLASS_NAMES[cls_idx]}: {count:,} ({pct:.1f}%)")
            
            del flux, delta_t, labels
        else:
            train_idx = None
            val_idx = None
            train_labels = None
            stats = None
        
        obj_list = [train_idx, val_idx, train_labels, stats]
        dist.broadcast_object_list(obj_list, src=0)
        train_idx, val_idx, train_labels, stats = obj_list
        
    else:
        flux, delta_t, labels = _load_arrays_from_file(file_path)
        total_samples = len(labels)
        
        stats = compute_normalization_statistics_from_indices(flux, delta_t, train_idx, rank)
        
        indices = np.arange(total_samples)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            stratify=labels,
            random_state=seed
        )
        
        train_labels = labels[train_idx]
        
        if is_main_process(rank):
            logger.info(f"Dataset: {format_number(total_samples)} samples")
            logger.info(f"  Train: {format_number(len(train_idx))} samples")
            logger.info(f"  Val:   {format_number(len(val_idx))} samples")
            
            unique, counts = np.unique(train_labels, return_counts=True)
            logger.info("Class distribution (train):")
            for cls_idx, count in zip(unique, counts):
                pct = 100 * count / len(train_labels)
                logger.info(f"  {CLASS_NAMES[cls_idx]}: {count:,} ({pct:.1f}%)")
        
        del flux, delta_t, labels
    
    return train_idx, val_idx, train_labels, stats


def create_dataloaders(
    file_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: Dict[str, float],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    is_ddp: bool,
    rank: int,
    local_rank: int,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, str]:
    """Create dataloaders with /dev/shm optimization."""
    shm_path, flux, delta_t, labels = load_data_to_shared_memory(
        file_path, rank, local_rank, is_ddp
    )
    
    train_dataset = MicrolensingDataset(
        flux, delta_t, labels, train_idx,
        stats['flux_mean'], stats['flux_std'],
        stats['delta_t_mean'], stats['delta_t_std'],
        normalize=normalize
    )
    
    val_dataset = MicrolensingDataset(
        flux, delta_t, labels, val_idx,
        stats['flux_mean'], stats['flux_std'],
        stats['delta_t_mean'], stats['delta_t_std'],
        normalize=normalize
    )
    
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=SEED, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    if is_main_process(rank):
        logger.info(f"Dataloaders created:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Workers: {num_workers}")
    
    return train_loader, val_loader, shm_path

# =============================================================================
# SCHEDULER
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / max(self.warmup_steps, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device
) -> Tensor:
    """Compute balanced class weights using inverse frequency."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + EPS)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)

# =============================================================================
# HIERARCHICAL LOSS
# =============================================================================

def compute_hierarchical_loss(
    output: HierarchicalOutput,
    labels: Tensor,
    class_weights: Tensor,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.3,
    nll_weight: float = 0.5
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute hierarchical classification loss.
    
    Components:
    1. Stage 1 BCE: P(deviation) - flat vs non-flat
    2. Stage 2 BCE: P(PSPL|deviation) - PSPL vs binary (non-flat only)
    3. Aux CE: Standard cross-entropy on auxiliary head
    4. NLL: Negative log likelihood on final log_probs
    """
    device = labels.device
    
    is_deviation = (labels > 0).float().unsqueeze(1)
    is_pspl = (labels == 1).float().unsqueeze(1)
    non_flat_mask = (labels > 0)
    n_non_flat = non_flat_mask.sum().item()
    
    # Stage 1: Flat vs Deviation
    stage1_pos_weight = (class_weights[1] + class_weights[2]) / 2.0 / (class_weights[0] + EPS)
    stage1_bce = F.binary_cross_entropy_with_logits(
        output.stage1_logit,
        is_deviation,
        pos_weight=stage1_pos_weight.unsqueeze(0),
        reduction='mean'
    )
    
    # Stage 2: PSPL vs Binary (only for non-flat samples)
    if n_non_flat > 0:
        stage2_logit_nonflat = output.stage2_logit[non_flat_mask]
        is_pspl_nonflat = is_pspl[non_flat_mask]
        stage2_pos_weight = class_weights[1] / (class_weights[2] + EPS)
        stage2_bce = F.binary_cross_entropy_with_logits(
            stage2_logit_nonflat,
            is_pspl_nonflat,
            pos_weight=stage2_pos_weight.unsqueeze(0),
            reduction='mean'
        )
    else:
        stage2_bce = torch.tensor(0.0, device=device)
    
    # Auxiliary head cross-entropy
    if output.aux_logits is not None:
        aux_ce = F.cross_entropy(output.aux_logits, labels, weight=class_weights)
    else:
        aux_ce = torch.tensor(0.0, device=device)
    
    # NLL loss on final log probabilities
    nll_loss = F.nll_loss(output.log_probs, labels, weight=class_weights)
    
    # Combined loss
    total_loss = (
        stage1_weight * stage1_bce +
        stage2_weight * stage2_bce +
        aux_weight * aux_ce +
        nll_weight * nll_loss
    )
    
    return total_loss, {
        'stage1_bce': float(stage1_bce.item()),
        'stage2_bce': float(stage2_bce.item()) if n_non_flat > 0 else 0.0,
        'aux_ce': float(aux_ce.item()) if output.aux_logits is not None else 0.0,
        'nll': float(nll_loss.item()),
        'total': float(total_loss.item()),
        'n_non_flat': n_non_flat
    }

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    class_weights: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
    hierarchical: bool,
    use_amp: bool,
    accumulation_steps: int = 1,
    clip_norm: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.3,
    nll_weight: float = 0.5
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_stage1_loss = torch.zeros(1, device=device)
    total_stage2_loss = torch.zeros(1, device=device)
    
    pbar = tqdm(
        loader,
        desc=f'Epoch {epoch} [Train]',
        disable=not is_main_process(rank),
        ncols=120,
        leave=False
    )
    
    if device.type == 'cuda' and use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        flux = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)
        
        with autocast_ctx:
            if hierarchical:
                # model.py v7.1.0: forward(flux, delta_t, return_intermediates=True)
                output = model(flux, delta_t, return_intermediates=True)
                loss, loss_dict = compute_hierarchical_loss(
                    output, labels, class_weights,
                    stage1_weight, stage2_weight, aux_weight, nll_weight
                )
                log_probs = output.log_probs
                bs = labels.size(0)
                total_stage1_loss += torch.tensor(loss_dict['stage1_bce'], device=device) * bs
                total_stage2_loss += torch.tensor(loss_dict['stage2_bce'], device=device) * bs
            else:
                # Non-hierarchical: model returns log_probs directly
                log_probs = model(flux, delta_t)
                loss = F.nll_loss(log_probs, labels, weight=class_weights)
            
            loss = loss / accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        with torch.no_grad():
            
            preds = log_probs.argmax(dim=1)
        

            loss_unscaled = loss.detach() * accumulation_steps         
            total_loss_gpu += loss_unscaled * labels.size(0)            
        
            total_correct_gpu += (preds == labels).sum()
            total_samples_gpu += labels.size(0)
            

        
        if batch_idx % PROGRESS_UPDATE_FREQ == 0:
            current_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
            current_acc = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
            
            if hierarchical:
                n_batches = batch_idx + 1
                n = max(total_samples_gpu.item(), 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{100*current_acc:.2f}%',
                    's1': f'{total_stage1_loss.item()/n:.3f}',
                    's2': f'{total_stage2_loss.item()/n:.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{100*current_acc:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
    
    if world_size > 1:
        dist.all_reduce(total_loss_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_gpu, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
    avg_acc = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
    
    return avg_loss, avg_acc

# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    hierarchical: bool,
    use_amp: bool,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.3,
    nll_weight: float = 0.5
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    class_correct = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    class_total = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    
    if device.type == 'cuda' and use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    for batch in loader:
        flux = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)
        
        with autocast_ctx:
            if hierarchical:
                output = model(flux, delta_t, return_intermediates=True)
                loss, _ = compute_hierarchical_loss(
                    output, labels, class_weights,
                    stage1_weight, stage2_weight, aux_weight, nll_weight
                )
                log_probs = output.log_probs
            else:
                log_probs = model(flux, delta_t)
                loss = F.nll_loss(log_probs, labels, weight=class_weights)
        
        preds = log_probs.argmax(dim=1)
        
        total_loss_gpu += loss * labels.size(0)
        total_correct_gpu += (preds == labels).sum()
        total_samples_gpu += labels.size(0)
        
        for c in range(N_CLASSES):
            mask = (labels == c)
            class_total[c] += mask.sum()
            class_correct[c] += ((preds == c) & mask).sum()
    
    if world_size > 1:
        dist.all_reduce(total_loss_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_total, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
    accuracy = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
    
    per_class_recall = {}
    for c, name in enumerate(CLASS_NAMES):
        if class_total[c].item() > 0:
            per_class_recall[name] = class_correct[c].item() / class_total[c].item()
        else:
            per_class_recall[name] = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        **{f'recall_{k}': v for k, v in per_class_recall.items()}
    }

# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    config: ModelConfig,
    stats: Dict[str, float],
    epoch: int,
    best_acc: float,
    path: Union[str, Path]
) -> None:
    """Save training checkpoint."""
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'model_config': config.to_dict(),
        'stats': stats,
        'best_acc': best_acc,
        'version': __version__
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")

def load_checkpoint_for_resume(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device
) -> Tuple[int, float]:
    """Load checkpoint for resuming training."""
    checkpoint = torch_load_compat(path, map_location=device, weights_only=False)
    
    state = checkpoint['model_state_dict']
    
    # Handle DDP and torch.compile prefixes
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    
    if isinstance(model, DDP):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = int(checkpoint.get('epoch', 0)) + 1
    best_acc = float(checkpoint.get('best_acc', 0.0))
    
    logger.info(f"Resumed from epoch {start_epoch-1} (best acc: {100*best_acc:.2f}%)")
    
    return start_epoch, best_acc

# =============================================================================
# DDP SETUP
# =============================================================================

def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """Setup Distributed Data Parallel."""
    if dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size()
    elif 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        master_addr = os.environ.get('MASTER_ADDR')
        master_port = os.environ.get('MASTER_PORT')
        
        if not master_addr or not master_port:
            raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set!")
        
        if rank == 0:
            logger.info("=" * 80)
            logger.info("DDP Initialization")
            logger.info(f"  RANK: {rank}")
            logger.info(f"  LOCAL_RANK: {local_rank}")
            logger.info(f"  WORLD_SIZE: {world_size}")
            logger.info(f"  MASTER_ADDR: {master_addr}")
            logger.info(f"  MASTER_PORT: {master_port}")
            logger.info("=" * 80)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            if rank == 0:
                logger.info(f"  Set CUDA device to: cuda:{local_rank}")
        
        try:
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=DDP_INIT_TIMEOUT_MINUTES)
            )
            dist.barrier()
            
            if rank == 0:
                logger.info("  DDP initialization complete!")
                logger.info(f"  CUDA device name: {torch.cuda.get_device_name(local_rank)}")
        except Exception as e:
            logger.error(f"DDP initialization failed: {e}")
            raise
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, local_rank, world_size, device

def cleanup_ddp() -> None:
    """Cleanup DDP process group."""
    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during DDP cleanup: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description=f'Train Roman Microlensing Classifier v{__version__}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output', type=str, default='./results/checkpoints', help='Output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Exact output directory (overrides auto-generated name)')
    parser.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION, help='Validation fraction')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=32, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
    parser.add_argument('--window-size', type=int, default=5, help='Conv kernel size')
    parser.add_argument('--hierarchical', action='store_true', default=True, help='Use hierarchical classification')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    parser.add_argument('--use-attention-pooling', action='store_true', default=True)
    parser.add_argument('--no-attention-pooling', dest='use_attention_pooling', action='store_false')
    parser.add_argument('--num-attention-heads', type=int, default=2)
    parser.add_argument('--num-groups', type=int, default=8)
    
    # Hierarchical head
    parser.add_argument('--use-aux-head', action='store_true', default=True)
    parser.add_argument('--no-aux-head', dest='use_aux_head', action='store_false')
    parser.add_argument('--stage2-temperature', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--stage1-weight', type=float, default=DEFAULT_STAGE1_WEIGHT)
    parser.add_argument('--stage2-weight', type=float, default=DEFAULT_STAGE2_WEIGHT)
    parser.add_argument('--aux-weight', type=float, default=DEFAULT_AUX_WEIGHT)
    parser.add_argument('--nll-weight', type=float, default=DEFAULT_NLL_WEIGHT)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument('--accumulation-steps', type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument('--clip-norm', type=float, default=DEFAULT_CLIP_NORM)
    
    # Performance
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    
    # Data loading
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULT_PREFETCH_FACTOR)
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    
    args = parser.parse_args()
    args.use_class_weights = not args.no_class_weights
    args.normalize = not args.no_normalize
    
    set_seed(SEED)
    configure_cuda()
    rank, local_rank, world_size, device = setup_ddp()
    
    global logger
    logger = setup_logging(rank)
    
    is_ddp = world_size > 1
    
    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info(f"Roman Microlensing Classifier Training v{__version__}")
        logger.info("=" * 80)
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        if torch.cuda.is_available():
            logger.info(f"GPUs per node: {torch.cuda.device_count()}")
        if args.hierarchical:
            logger.info("HIERARCHICAL MODE")
            logger.info(f"  Stage 1 weight: {args.stage1_weight}")
            logger.info(f"  Stage 2 weight: {args.stage2_weight}")
            logger.info(f"  Aux weight: {args.aux_weight}")
            logger.info(f"  NLL weight: {args.nll_weight}")
    
    if is_ddp:
        dist.barrier()
    
    base_output_dir = Path(args.output)
    if is_main_process(rank):
        base_output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        dist.barrier()
    
    if is_main_process(rank):
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            exp_name = output_dir.name
        else:
            output_dir = create_experiment_dir(base_output_dir, args)
            exp_name = output_dir.name
        
        with open(base_output_dir / '.current_experiment', 'w') as f:
            f.write(exp_name)
    else:
        exp_name = None
    
    if is_ddp:
        dist.barrier()
        exp_name_list = [exp_name] if is_main_process(rank) else [None]
        dist.broadcast_object_list(exp_name_list, src=0)
        exp_name = exp_name_list[0]
        output_dir = Path(args.output_dir) if args.output_dir else base_output_dir / exp_name
    
    # Load and split data
    train_idx, val_idx, train_labels, stats = load_and_split_data(
        args.data, args.val_fraction, SEED, rank, is_ddp
    )
    
    # Create dataloaders
    train_loader, val_loader, shm_path = create_dataloaders(
        args.data, train_idx, val_idx, stats,
        args.batch_size, args.num_workers, args.prefetch_factor,
        is_ddp, rank, local_rank,
        normalize=args.normalize
    )
    
    # Create model configuration (compatible with model.py v7.1.0)
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        n_classes=N_CLASSES,
        hierarchical=args.hierarchical,
        use_aux_head=args.use_aux_head,
        stage2_temperature=args.stage2_temperature,
        use_residual=True,
        use_layer_norm=True,
        use_attention_pooling=args.use_attention_pooling,
        use_flash_attention=True,
        num_attention_heads=args.num_attention_heads,
        lstm_dropout=args.dropout * 0.5,
        locked_dropout=args.dropout * 0.5,
        num_groups=args.num_groups,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Model Configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")
        logger.info("-" * 80)
    
    model = RomanMicrolensingClassifier(config).to(device)
    
    if is_main_process(rank):
        n_params = model.count_parameters(trainable_only=True)
        logger.info("Model Architecture:")
        logger.info(f"  Trainable parameters: {format_number(n_params)}")
        logger.info(f"  Receptive field: {model.receptive_field} timesteps")
        
        if config.hierarchical:
            logger.info("Hierarchical Head Initialization:")
            logger.info(f"  Stage 1 bias: {model.head_stage1.bias.item():.4f}")
            logger.info(f"  Stage 2 bias: {model.head_stage2.bias.item():.4f}")
        logger.info("-" * 80)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank):
            logger.info(f"Compiling model with mode={args.compile_mode}...")
        try:
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
            if is_main_process(rank):
                logger.info("  Model compiled")
        except Exception as e:
            if is_main_process(rank):
                logger.warning(f"torch.compile failed: {e}")
    
    # Optimizer
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    if fused_available and device.type == 'cuda':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
        )
        if is_main_process(rank):
            logger.info("Using fused AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # AMP scaler
    if args.use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        if is_main_process(rank):
            logger.info("Using AMP with gradient scaler")
    else:
        scaler = None
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, N_CLASSES, device)
        if is_main_process(rank):
            logger.info(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = torch.ones(N_CLASSES, device=device)
    
    # Resume
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint_for_resume(
            args.resume, model, optimizer, scheduler, scaler, device
        )
    
    if is_ddp:
        dist.barrier()
    
    # Training loop
    try:
        if is_main_process(rank):
            logger.info("Starting training...")
            logger.info("-" * 80)
        
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            
            if is_ddp:
                train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                class_weights, device, rank, world_size, epoch,
                args.hierarchical, args.use_amp,
                args.accumulation_steps, args.clip_norm,
                args.stage1_weight, args.stage2_weight, args.aux_weight, args.nll_weight
            )
            
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, world_size,
                args.hierarchical, args.use_amp,
                args.stage1_weight, args.stage2_weight, args.aux_weight, args.nll_weight
            )
            
            epoch_time = time.time() - epoch_start
            
            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                    f"Val Loss: {val_results['loss']:.4f} | Val Acc: {100*val_results['accuracy']:.2f}% | "
                    f"Time: {format_time(epoch_time)}"
                )
                if args.hierarchical:
                    logger.info(
                        f"         Per-class recall: "
                        f"Flat={100*val_results.get('recall_Flat', 0):.1f}% | "
                        f"PSPL={100*val_results.get('recall_PSPL', 0):.1f}% | "
                        f"Binary={100*val_results.get('recall_Binary', 0):.1f}%"
                    )
            
            is_best = val_results['accuracy'] > best_acc
            if is_best:
                best_acc = val_results['accuracy']
            
            if is_main_process(rank):
                checkpoint_dir = output_dir / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                save_checkpoint(
                    model, optimizer, scheduler, scaler, config, stats,
                    epoch, best_acc, checkpoint_dir / 'checkpoint_latest.pt'
                )
                
                if is_best:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, config, stats,
                        epoch, best_acc, output_dir / 'best.pt'
                    )
                
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, config, stats,
                        epoch, best_acc, checkpoint_dir / f'epoch_{epoch:03d}.pt'
                    )
        
        if is_main_process(rank):
            save_checkpoint(
                model, optimizer, scheduler, scaler, config, stats,
                args.epochs, best_acc, output_dir / 'final.pt'
            )
            logger.info("=" * 80)
            logger.info(f"Training complete! Best validation accuracy: {100*best_acc:.2f}%")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("=" * 80)
    
    except KeyboardInterrupt:
        if is_main_process(rank):
            logger.info("\nTraining interrupted by user")
        raise
    
    except Exception as e:
        if is_main_process(rank):
            logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Barrier before cleanup to prevent race condition
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        cleanup_shared_memory(shm_path, rank, local_rank)
        cleanup_ddp()


if __name__ == '__main__':
    main()
