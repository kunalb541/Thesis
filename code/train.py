"""
Roman Microlensing Classifier Training Engine v4.1.0
====================================================

MULTI-GPU OPTIMIZATION v4.1.0 - CLEANUP RACE FIX + BROADCAST OPTIMIZATION
--------------------------------------------------------------------------
**Fixed:** Added dist.barrier() before /dev/shm cleanup (prevents race condition)
**Fixed:** Stats and indices now broadcast from rank0 (avoids redundant file reads)
**Fixed:** Logging uses local_rank instead of rank % 4 (works with any GPU count)
**Fixed:** Documentation now accurately describes what /dev/shm sharing achieves

**What /dev/shm actually provides (corrected from v4.0.0):**
- Fast RAM-backed filesystem reads (no disk I/O)
- Linux page cache sharing for the FILE (reduces disk reads)
- Each process still allocates its own numpy arrays after load
- NOT true shared memory across processes (would need multiprocessing.shared_memory)

All v4.0.0, v3.2.0, v3.1.0 and v3.0.0 fixes preserved:
- Deterministic /dev/shm path (no PID, uses SLURM_JOB_ID)
- Preserves source file suffix (.npz/.h5)
- Model calls use keyword arguments (lengths=lengths)
- lengths tensor moved to device
- torch.load compatibility wrapper
- SharedRAMLensingDataset (no train/val double loading)
- Hierarchical loss with separate BCE stages
- Memory leak prevention
- Complete type hints

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 4.1.0
Date: December 2024
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

__version__: str = "4.1.0"

__all__ = [
    "SharedRAMLensingDataset",
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
# v4.0.0: TORCH SERIALIZATION COMPATIBILITY GUARD
# =============================================================================
# Guard torch.serialization import (prevents some PyTorch crashes on certain versions)
try:
    import torch.serialization  # type: ignore
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

# Import model after path setup
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
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_PREFETCH_FACTOR: int = 2
DEFAULT_ACCUMULATION_STEPS: int = 1
DEFAULT_CLIP_NORM: float = 1.0
DEFAULT_WARMUP_EPOCHS: int = 3
DEFAULT_VAL_FRACTION: float = 0.1

DEFAULT_STAGE1_WEIGHT: float = 1.0
DEFAULT_STAGE2_WEIGHT: float = 1.0
DEFAULT_AUX_WEIGHT: float = 0.5

PROGRESS_UPDATE_FREQ: int = 50
DDP_INIT_TIMEOUT_MINUTES: int = 10
DDP_BARRIER_TIMEOUT_SECONDS: int = 300
INVALID_TIMESTAMP: float = -999.0
MIN_VALID_SEQ_LEN: int = 10

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(rank: int) -> logging.Logger:
    """Configure logging for distributed training."""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)

    return logger

logger = setup_logging(0)

# =============================================================================
# v4.0.0: TORCH LOAD COMPATIBILITY WRAPPER
# =============================================================================

def torch_load_compat(
    path: Union[str, Path],
    map_location: Union[str, torch.device, None],
    weights_only: bool = False
) -> Dict[str, Any]:
    """
    Compatibility torch.load wrapper.
    
    Handles older PyTorch versions that don't support weights_only parameter.
    
    Parameters
    ----------
    path : str or Path
        Path to checkpoint file.
    map_location : str, torch.device, or None
        Device to map tensors to.
    weights_only : bool
        If True, only load weights (not supported on older PyTorch).
        
    Returns
    -------
    checkpoint : dict
        Loaded checkpoint dictionary.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older PyTorch without weights_only parameter
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
# v4.1.0: MULTI-GPU SHARED MEMORY (DOCUMENTATION CORRECTED)
# =============================================================================

def get_node_id(rank: int, local_rank: int) -> int:
    """
    Determine which physical node this process is on.
    
    v4.0.0 FIX: Use LOCAL_WORLD_SIZE env var for robust node calculation.
    Previous version used torch.cuda.device_count() which can be unreliable.
    
    Parameters
    ----------
    rank : int
        Global rank.
    local_rank : int
        Local rank within node.
    
    Returns
    -------
    node_id : int
        Node identifier (0, 1, 2, ...).
    """
    # v4.0.0: Use LOCAL_WORLD_SIZE from SLURM/torchrun for accuracy
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE",
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    ))
    # More robust calculation: (rank - local_rank) gives first rank on this node
    return (rank - local_rank) // max(local_world_size, 1)

def load_data_to_shared_memory(
    file_path: str,
    rank: int,
    local_rank: int,
    is_ddp: bool
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data to /dev/shm RAM-backed filesystem (ONLY local_rank=0 per node copies).
    
    v4.1.0 DOCUMENTATION CORRECTION:
    This function copies data to /dev/shm for fast RAM-backed reads, but each
    process still allocates its own numpy arrays after loading. The Linux page
    cache is shared (reducing disk I/O), but the arrays themselves are duplicated.
    
    For true shared memory across processes, would need multiprocessing.shared_memory
    or memory-mapped .npy files. Current approach provides:
    - Fast RAM-backed reads (no disk I/O after first copy)
    - Page cache sharing at filesystem level
    - Each process still has separate array copies in RAM
    
    v4.0.0 FIXES (preserved):
    - shm_path now deterministic across ranks (removed os.getpid())
    - Preserves source file suffix (.npz/.h5)
    - Uses SLURM_JOB_ID for collision avoidance across jobs
    - Only copies if file doesn't already exist (recovery-friendly)
    
    Parameters
    ----------
    file_path : str
        Path to source HDF5/NPZ file (could be in /tmp already).
    rank : int
        Global process rank.
    local_rank : int
        Local rank within node (0-3 for 4 GPUs).
    is_ddp : bool
        Whether using distributed training.
    
    Returns
    -------
    shm_path : str
        Path to shared memory file in /dev/shm.
    magnification : np.ndarray
        Magnification array loaded from /dev/shm.
    delta_t : np.ndarray
        Time interval array loaded from /dev/shm.
    labels : np.ndarray
        Labels array loaded from /dev/shm.
    """
    node_id = get_node_id(rank, local_rank)
    
    # v4.0.0 FIX: Create deterministic /dev/shm path
    # - Use SLURM_JOB_ID (not PID) for collision avoidance
    # - Preserve original file suffix (.npz or .h5) - critical fix from v3.2.0
    src = Path(file_path)
    suffix = src.suffix  # Keep .h5 or .npz
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    shm_path = f"/dev/shm/{src.stem}_job{job_id}_node{node_id}{suffix}"
    
    if local_rank == 0:
        # =====================================================================
        # FIRST GPU ON NODE: Copy data to /dev/shm
        # =====================================================================
        if is_main_process(rank):
            logger.info("=" * 80)
            logger.info("v4.1.0 /dev/shm OPTIMIZATION: RAM-Backed File Loading")
            logger.info("=" * 80)
            logger.info(f"Node {node_id} (Rank {rank}): Setting up /dev/shm...")
        
        # Check /dev/shm availability and space
        if not Path('/dev/shm').exists():
            raise RuntimeError("/dev/shm not available on this system!")
        
        # v4.0.0 FIX: Only copy if file doesn't already exist (recovery-friendly)
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
        
        # Load data from /dev/shm
        if is_main_process(rank):
            logger.info(f"  Loading data from /dev/shm...")
        
        magnification, delta_t, labels = _load_arrays_from_file(shm_path, rank)
        
        if is_main_process(rank):
            total_mem = (magnification.nbytes + delta_t.nbytes + labels.nbytes) / 1e9
            logger.info(f"  Data loaded: {total_mem:.2f} GB arrays allocated")
            # v4.1.0: Accurate documentation of what's shared
            logger.info(f"  Note: Page cache shared, but each process has own array copy")
        
        # Signal other GPUs on this node that data is ready
        # Note: This barrier is global (all ranks), not per-node
        if is_ddp:
            dist.barrier()
    
    else:
        # =====================================================================
        # OTHER GPUs ON NODE: Wait and load from same /dev/shm file
        # =====================================================================
        
        # Wait for first GPU to finish copying
        if is_ddp:
            dist.barrier()
        
        # Load from SAME /dev/shm file (fast RAM-backed read, but allocates own arrays)
        magnification, delta_t, labels = _load_arrays_from_file(shm_path, rank)
        
        # v4.1.0 FIX: Use local_rank instead of rank % 4 (works with any GPU count)
        if local_rank == 1 and is_main_process(rank):
            logger.info(f"  GPU {local_rank}: Loaded from /dev/shm (fast RAM read)")
    
    # v4.1.0 FIX: Use local_rank instead of rank % 4
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE",
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    ))
    if local_rank == local_world_size - 1 and is_main_process(rank):
        logger.info("=" * 80)
        total_mem = (magnification.nbytes + delta_t.nbytes + labels.nbytes) / 1e9
        logger.info(f"Node {node_id}: All {local_world_size} GPUs loaded {total_mem:.2f} GB each")
        logger.info("=" * 80)
    
    return shm_path, magnification, delta_t, labels

def _load_arrays_from_file(
    file_path: str,
    rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy arrays from HDF5 or NPZ file.
    
    Helper function called by load_data_to_shared_memory.
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        magnification = data.get('flux', data.get('magnification', data.get('mag')))
        delta_t = data['delta_t']
        labels = data.get('labels', data['y'])
    
    elif file_path.suffix in ['.h5', '.hdf5']:
        with h5py.File(str(file_path), 'r') as f:
            if 'flux' in f:
                magnification = f['flux'][:]
            elif 'magnification' in f:
                magnification = f['magnification'][:]
            else:
                magnification = f['mag'][:]
            delta_t = f['delta_t'][:]
            labels = f['labels'][:]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return magnification, delta_t, labels

def cleanup_shared_memory(shm_path: str, rank: int, local_rank: int) -> None:
    """
    Cleanup /dev/shm files (ONLY local_rank=0 deletes).
    
    v4.1.0: Caller must ensure dist.barrier() is called BEFORE this function
    to prevent race conditions where rank0 deletes while others still reading.
    
    Parameters
    ----------
    shm_path : str
        Path to shared memory file.
    rank : int
        Global rank.
    local_rank : int
        Local rank within node.
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
# v3.1.0: SHARED DATASET (preserved from v3.1.0)
# =============================================================================

class SharedRAMLensingDataset(Dataset):
    """
    Lightweight dataset that references shared memory arrays.
    
    From v3.1.0 - eliminates train/val double loading within each process.
    Combined with v4.1.0 /dev/shm, provides fast RAM-backed reads.
    
    Note: Each process has its own copy of the arrays in RAM.
    """
    
    def __init__(
        self,
        magnification: np.ndarray,
        delta_t: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        magnification_mean: float,
        magnification_std: float,
        delta_t_mean: float,
        delta_t_std: float,
        rank: int = 0
    ) -> None:
        self.magnification = magnification
        self.delta_t = delta_t
        self.labels = labels
        self.indices = indices
        self.rank = rank
        
        self._magnification_scale = 1.0 / (magnification_std + EPS)
        self._dt_scale = 1.0 / (delta_t_std + EPS)
        self.magnification_mean = magnification_mean
        self.delta_t_mean = delta_t_mean
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        full_idx = self.indices[idx]
        
        magnification_raw = self.magnification[full_idx].copy()
        delta_t_raw = self.delta_t[full_idx].copy()
        label = int(self.labels[full_idx])
        
        # Sequence compaction
        valid_mask = magnification_raw != 0.0
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            valid_count = 1
            magnification_raw[0] = self.magnification_mean
            delta_t_raw[0] = 0.0
            valid_mask[0] = True
        
        magnification_compacted = np.zeros_like(magnification_raw)
        delta_t_compacted = np.zeros_like(delta_t_raw)
        
        magnification_compacted[:valid_count] = magnification_raw[valid_mask]
        delta_t_compacted[:valid_count] = delta_t_raw[valid_mask]
        
        magnification_norm = (magnification_compacted - self.magnification_mean) * self._magnification_scale
        dt_norm = (delta_t_compacted - self.delta_t_mean) * self._dt_scale
        
        magnification_tensor = torch.from_numpy(magnification_norm).float()
        delta_t_tensor = torch.from_numpy(dt_norm).float()
        # v4.0.0: Return length as tensor for device transfer
        length_tensor = torch.tensor(int(valid_count), dtype=torch.long)
        
        return magnification_tensor, delta_t_tensor, length_tensor, label
    
    def __del__(self) -> None:
        try:
            self.magnification = None
            self.delta_t = None
            self.labels = None
            self.indices = None
        except:
            pass

# =============================================================================
# DATA LOADING (v4.1.0: Broadcast optimization)
# =============================================================================

def compute_robust_statistics(
    file_path: str,
    rank: int = 0
) -> Dict[str, float]:
    """Compute normalization statistics."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        magnification_data = data.get('flux', data.get('magnification', data.get('mag')))
        delta_t_data = data['delta_t']
    else:
        with h5py.File(str(file_path), 'r') as f:
            if 'flux' in f:
                magnification_data = f['flux'][:]
            elif 'magnification' in f:
                magnification_data = f['magnification'][:]
            else:
                magnification_data = f['mag'][:]
            delta_t_data = f['delta_t'][:]
    
    magnification_valid = magnification_data[magnification_data != 0.0]
    delta_t_valid = delta_t_data[delta_t_data != 0.0]
    
    magnification_mean = float(np.mean(magnification_valid))
    magnification_std = float(np.std(magnification_valid))
    delta_t_mean = float(np.mean(delta_t_valid))
    delta_t_std = float(np.std(delta_t_valid))
    
    return {
        'magnification_mean': magnification_mean,
        'magnification_std': magnification_std,
        'delta_t_mean': delta_t_mean,
        'delta_t_std': delta_t_std
    }


def load_and_split_data(
    file_path: str,
    val_fraction: float,
    seed: int,
    rank: int,
    is_ddp: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Load data, compute statistics, and split into train/val.
    
    v4.1.0: Rank 0 computes everything and broadcasts to avoid redundant file reads.
    
    Parameters
    ----------
    file_path : str
        Path to data file.
    val_fraction : float
        Fraction of data for validation.
    seed : int
        Random seed for reproducibility.
    rank : int
        Process rank.
    is_ddp : bool
        Whether using distributed training.
    
    Returns
    -------
    train_idx : np.ndarray
        Training indices.
    val_idx : np.ndarray
        Validation indices.
    train_labels : np.ndarray
        Labels for training samples.
    stats : dict
        Normalization statistics.
    """
    if is_ddp:
        # v4.1.0: Only rank 0 reads file and computes, then broadcasts
        if rank == 0:
            file_path_obj = Path(file_path)
            
            if file_path_obj.suffix == '.npz':
                data = np.load(str(file_path_obj))
                total_samples = len(data.get('labels', data['y']))
                all_labels = data.get('labels', data['y'])
            else:
                with h5py.File(str(file_path_obj), 'r') as f:
                    total_samples = len(f['labels'])
                    all_labels = f['labels'][:]
            
            stats = compute_robust_statistics(file_path, rank)
            
            indices = np.arange(total_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                stratify=all_labels,
                random_state=seed
            )
            
            train_labels = all_labels[train_idx]
            
            logger.info(f"Dataset: {format_number(total_samples)} samples")
            logger.info(f"  Train: {format_number(len(train_idx))} samples")
            logger.info(f"  Val:   {format_number(len(val_idx))} samples")
            
            unique, counts = np.unique(train_labels, return_counts=True)
            logger.info("Class distribution (train):")
            for cls_idx, count in zip(unique, counts):
                pct = 100 * count / len(train_labels)
                logger.info(f"  {CLASS_NAMES[cls_idx]}: {count:,} ({pct:.1f}%)")
        else:
            train_idx = None
            val_idx = None
            train_labels = None
            stats = None
        
        # v4.1.0: Broadcast from rank0 to all other ranks
        obj_list = [train_idx, val_idx, train_labels, stats]
        dist.broadcast_object_list(obj_list, src=0)
        train_idx, val_idx, train_labels, stats = obj_list
        
    else:
        # Non-DDP path: single process does everything
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix == '.npz':
            data = np.load(str(file_path_obj))
            total_samples = len(data.get('labels', data['y']))
            all_labels = data.get('labels', data['y'])
        else:
            with h5py.File(str(file_path_obj), 'r') as f:
                total_samples = len(f['labels'])
                all_labels = f['labels'][:]
        
        stats = compute_robust_statistics(file_path, rank)
        
        indices = np.arange(total_samples)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            stratify=all_labels,
            random_state=seed
        )
        
        train_labels = all_labels[train_idx]
        
        if is_main_process(rank):
            logger.info(f"Dataset: {format_number(total_samples)} samples")
            logger.info(f"  Train: {format_number(len(train_idx))} samples")
            logger.info(f"  Val:   {format_number(len(val_idx))} samples")
            
            unique, counts = np.unique(train_labels, return_counts=True)
            logger.info("Class distribution (train):")
            for cls_idx, count in zip(unique, counts):
                pct = 100 * count / len(train_labels)
                logger.info(f"  {CLASS_NAMES[cls_idx]}: {count:,} ({pct:.1f}%)")
    
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
    local_rank: int
) -> Tuple[DataLoader, DataLoader, str]:
    """
    Create dataloaders with /dev/shm optimization (v4.1.0).
    
    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    shm_path : str
        Path to /dev/shm file (for cleanup later).
    """
    # =========================================================================
    # v4.1.0: Load to /dev/shm (ONLY local_rank=0 per node copies)
    # =========================================================================
    shm_path, magnification, delta_t, labels = load_data_to_shared_memory(
        file_path, rank, local_rank, is_ddp
    )
    
    # =========================================================================
    # v3.1.0: Create shared datasets (no train/val double loading)
    # v4.1.0 FIX: Use correct stats keys (magnification_mean, not flux_mean)
    # =========================================================================
    train_dataset = SharedRAMLensingDataset(
        magnification, delta_t, labels, train_idx,
        stats['magnification_mean'], stats['magnification_std'],
        stats['delta_t_mean'], stats['delta_t_std'],
        rank=rank
    )
    
    val_dataset = SharedRAMLensingDataset(
        magnification, delta_t, labels, val_idx,
        stats['magnification_mean'], stats['magnification_std'],
        stats['delta_t_mean'], stats['delta_t_std'],
        rank=rank
    )
    
    if is_main_process(rank):
        total_mem = (magnification.nbytes + delta_t.nbytes + labels.nbytes) / 1e9
        local_world_size = int(os.environ.get(
            "LOCAL_WORLD_SIZE",
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        ))
        world_size = dist.get_world_size() if is_ddp else 1
        n_nodes = world_size // local_world_size if is_ddp else 1
        logger.info("=" * 80)
        logger.info(f"v4.1.0 MEMORY LAYOUT:")
        logger.info(f"  Array size per process: {total_mem:.2f} GB")
        logger.info(f"  Processes per node: {local_world_size}")
        logger.info(f"  Total nodes: {n_nodes}")
        logger.info(f"  /dev/shm benefit: Fast RAM reads, page cache shared at OS level")
        logger.info("=" * 80)
    
    # Setup samplers
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
    
    # Create dataloaders
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
# SCHEDULER, WEIGHTS, LOSS (Same as v3.1.0)
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing scheduler with linear warmup."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / max(self.warmup_steps, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

def compute_class_weights(labels, n_classes, device):
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + EPS)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)

def compute_hierarchical_loss(output, labels, class_weights, stage1_weight=1.0, stage2_weight=1.0, aux_weight=0.5):
    """Compute hierarchical loss (v3.0.0 fix)."""
    device = labels.device
    B = labels.size(0)
    
    is_deviation = (labels > 0).float().unsqueeze(1)
    is_pspl = (labels == 1).float().unsqueeze(1)
    non_flat_mask = (labels > 0)
    n_non_flat = non_flat_mask.sum().item()
    
    stage1_pos_weight_scalar = (class_weights[1] + class_weights[2]) / 2.0 / (class_weights[0] + EPS)
    stage1_bce = F.binary_cross_entropy_with_logits(
        output.stage1_logit, is_deviation, pos_weight=stage1_pos_weight_scalar, reduction='mean'
    )
    
    if n_non_flat > 0:
        stage2_logit_nonflat = output.stage2_logit[non_flat_mask]
        is_pspl_nonflat = is_pspl[non_flat_mask]
        stage2_pos_weight_scalar = class_weights[1] / (class_weights[2] + EPS)
        stage2_bce = F.binary_cross_entropy_with_logits(
            stage2_logit_nonflat, is_pspl_nonflat, pos_weight=stage2_pos_weight_scalar, reduction='mean'
        )
    else:
        stage2_bce = torch.tensor(0.0, device=device)
    
    if output.aux_logits is not None:
        aux_ce = F.cross_entropy(output.aux_logits, labels, weight=class_weights)
    else:
        aux_ce = torch.tensor(0.0, device=device)
    
    total_loss = stage1_weight * stage1_bce + stage2_weight * stage2_bce + aux_weight * aux_ce
    
    return total_loss, {
        'stage1_bce': float(stage1_bce.item()),
        'stage2_bce': float(stage2_bce.item()) if n_non_flat > 0 else 0.0,
        'aux_ce': float(aux_ce.item()) if output.aux_logits is not None else 0.0,
        'total': float(total_loss.item()),
        'n_non_flat': n_non_flat
    }

# =============================================================================
# TRAINING & EVALUATION (v4.0.0: Fixed model calls and lengths device)
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, class_weights, device, rank, world_size,
                epoch, config, accumulation_steps=1, clip_norm=1.0, stage1_weight=1.0, stage2_weight=1.0, aux_weight=0.5):
    """Execute one training epoch."""
    model.train()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_stage1_loss = torch.zeros(1, device=device)
    total_stage2_loss = torch.zeros(1, device=device)
    total_aux_loss = torch.zeros(1, device=device)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', disable=not is_main_process(rank), ncols=120, leave=False)
    
    if device.type == 'cuda' and config.use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        magnification = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        # v4.0.0 FIX: Move lengths to device (was staying on CPU)
        lengths = batch[2].to(device, non_blocking=True)
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                # v4.0.0 FIX: Use keyword argument for lengths (matches v4 model API)
                output = model(magnification, delta_t, lengths=lengths, return_intermediates=True)
                loss, loss_dict = compute_hierarchical_loss(
                    output, labels, class_weights, stage1_weight, stage2_weight, aux_weight
                )
                logits = output.logits
                total_stage1_loss += loss_dict['stage1_bce']
                total_stage2_loss += loss_dict['stage2_bce']
                total_aux_loss += loss_dict['aux_ce']
            else:
                # v4.0.0 FIX: Use keyword argument for lengths
                logits = model(magnification, delta_t, lengths=lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            
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
            if config.hierarchical:
                probs = torch.exp(logits)
            else:
                probs = F.softmax(logits, dim=1)
            
            preds = probs.argmax(dim=1)
            total_loss_gpu += loss.detach() * accumulation_steps
            total_correct_gpu += (preds == labels).sum()
            total_samples_gpu += labels.size(0)
        
        if batch_idx % PROGRESS_UPDATE_FREQ == 0:
            current_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
            current_acc = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
            
            if config.hierarchical:
                n_batches = batch_idx + 1
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{100*current_acc:.2f}%',
                    's1': f'{total_stage1_loss.item()/n_batches:.3f}',
                    's2': f'{total_stage2_loss.item()/n_batches:.3f}',
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

@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, world_size, config,
             stage1_weight=1.0, stage2_weight=1.0, aux_weight=0.5):
    """Evaluate model."""
    model.eval()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    class_correct = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    class_total = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    
    if device.type == 'cuda' and config.use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    for batch in loader:
        magnification = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        # v4.0.0 FIX: Move lengths to device (was staying on CPU)
        lengths = batch[2].to(device, non_blocking=True)
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                # v4.0.0 FIX: Use keyword argument for lengths (matches v4 model API)
                output = model(magnification, delta_t, lengths=lengths, return_intermediates=True)
                loss, _ = compute_hierarchical_loss(
                    output, labels, class_weights, stage1_weight, stage2_weight, aux_weight
                )
                logits = output.logits
            else:
                # v4.0.0 FIX: Use keyword argument for lengths
                logits = model(magnification, delta_t, lengths=lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        if config.hierarchical:
            probs = torch.exp(logits)
        else:
            probs = F.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        
        if config.hierarchical:
            total_loss_gpu += loss * labels.size(0)
        else:
            total_loss_gpu += loss
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
# CHECKPOINTING (v4.0.0: Fixed torch.load compatibility)
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, scaler, config, stats, epoch, best_acc, path):
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

def load_checkpoint_for_resume(path, model, optimizer, scheduler, scaler, device):
    """
    Load checkpoint for resuming training.
    
    v4.0.0 FIX: Uses torch_load_compat with weights_only=False for full checkpoint.
    """
    # v4.0.0 FIX: Use compat loader with weights_only=False (checkpoint has metadata)
    checkpoint = torch_load_compat(path, map_location=device, weights_only=False)
    
    state = checkpoint['model_state_dict']
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
# DDP SETUP (Same as v3.1.0)
# =============================================================================

def setup_ddp():
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

def cleanup_ddp():
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

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description=f'Train Roman Microlensing Classifier v{__version__}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output', type=str, default='../results/checkpoints', help='Output directory')
    parser.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION, help='Validation fraction')
    
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--window-size', type=int, default=7, help='Conv kernel size')
    parser.add_argument('--hierarchical', action='store_true', help='Use hierarchical classification')
    parser.add_argument('--attention-pooling', action='store_true', help='Use attention pooling')
    
    parser.add_argument('--use-aux-head', action='store_true', default=True, help='Use auxiliary head')
    parser.add_argument('--no-aux-head', dest='use_aux_head', action='store_false', help='Disable aux head')
    parser.add_argument('--stage2-temperature', type=float, default=1.0, help='Stage 2 temperature')
    parser.add_argument('--stage1-weight', type=float, default=DEFAULT_STAGE1_WEIGHT, help='Stage 1 weight')
    parser.add_argument('--stage2-weight', type=float, default=DEFAULT_STAGE2_WEIGHT, help='Stage 2 weight')
    parser.add_argument('--aux-weight', type=float, default=DEFAULT_AUX_WEIGHT, help='Aux weight')
    
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_WARMUP_EPOCHS, help='Warmup epochs')
    parser.add_argument('--accumulation-steps', type=int, default=DEFAULT_ACCUMULATION_STEPS, help='Gradient accumulation')
    parser.add_argument('--clip-norm', type=float, default=DEFAULT_CLIP_NORM, help='Gradient clipping norm')
    
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'], help='Compile mode')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weighting')
    
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS, help='Data workers')
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULT_PREFETCH_FACTOR, help='Prefetch factor')
    
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('--save-every', type=int, default=5, help='Save every N epochs')
    
    args = parser.parse_args()
    args.use_class_weights = not args.no_class_weights
    
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
        logger.info(f"GPUs per node: {torch.cuda.device_count()}")
        logger.info(f"Workers: {args.num_workers}")
        if args.hierarchical:
            logger.info("=" * 80)
            logger.info("HIERARCHICAL MODE (v4.1 FIXES APPLIED)")
            logger.info(f"  Stage 1 weight: {args.stage1_weight}")
            logger.info(f"  Stage 2 weight: {args.stage2_weight}")
            logger.info(f"  Aux weight: {args.aux_weight}")
            logger.info("=" * 80)
    
    if is_ddp:
        if is_main_process(rank):
            logger.info("Synchronizing all processes...")
        dist.barrier()
    
    base_output_dir = Path(args.output)
    if is_main_process(rank):
        base_output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        dist.barrier()
    
    if is_main_process(rank):
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
        output_dir = base_output_dir / exp_name
    
    # v4.1.0: Pass is_ddp to enable broadcast optimization
    train_idx, val_idx, train_labels, stats = load_and_split_data(
        args.data, args.val_fraction, SEED, rank, is_ddp
    )
    
    # v4.1.0: Multi-GPU /dev/shm dataloaders
    train_loader, val_loader, shm_path = create_dataloaders(
        args.data, train_idx, val_idx, stats,
        args.batch_size, args.num_workers, args.prefetch_factor,
        is_ddp, rank, local_rank
    )
    
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        hierarchical=args.hierarchical,
        use_aux_head=args.use_aux_head,
        stage2_temperature=args.stage2_temperature,
        use_attention_pooling=args.attention_pooling,
        use_amp=args.use_amp
    )
    
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Model Configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")
        logger.info("-" * 80)
    
    model = RomanMicrolensingClassifier(config).to(device)
    
    if is_main_process(rank):
        complexity = model.get_complexity_info()
        logger.info("Model Architecture:")
        logger.info(f"  Total parameters: {format_number(complexity['total_parameters'])}")
        logger.info(f"  Trainable parameters: {format_number(complexity['trainable_parameters'])}")
        
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
    
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    if fused_available and device.type == 'cuda':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        if is_main_process(rank):
            logger.info("Using fused AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    use_scaler = args.use_amp and device.type == 'cuda' and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint_for_resume(
            args.resume, model, optimizer, scheduler, scaler, device
        )
    
    if is_ddp:
        dist.barrier()
    
    try:
        if is_main_process(rank):
            logger.info("Starting training...")
            logger.info("-" * 80)
        
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            
            if is_ddp:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                class_weights, device, rank, world_size, epoch, config,
                args.accumulation_steps, args.clip_norm,
                args.stage1_weight, args.stage2_weight, args.aux_weight
            )
            
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, world_size, config,
                args.stage1_weight, args.stage2_weight, args.aux_weight
            )
            
            epoch_time = time.time() - epoch_start
            
            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                    f"Val Loss: {val_results['loss']:.4f} | Val Acc: {100*val_results['accuracy']:.2f}% | "
                    f"Time: {format_time(epoch_time)}"
                )
                if config.hierarchical:
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
                        epoch, best_acc, output_dir / f'epoch_{epoch:03d}.pt'
                    )
        
        if is_main_process(rank):
            save_checkpoint(
                model, optimizer, scheduler, scaler, config, stats,
                args.epochs, best_acc, output_dir / 'final.pt'
            )
            logger.info("=" * 80)
            logger.info(f"Training complete! Best validation accuracy: {100*best_acc:.2f}%")
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
        # =====================================================================
        # v4.1.0 FIX: Barrier BEFORE cleanup to prevent race condition
        # Without this, rank0 can delete /dev/shm while others still reading
        # =====================================================================
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        cleanup_shared_memory(shm_path, rank, local_rank)
        cleanup_ddp()

if __name__ == '__main__':
    main()
