#!/usr/bin/env python3
"""
Roman Microlensing Classifier - Distributed Training Script
============================================================

High-performance distributed training for the Roman Space Telescope
microlensing event classifier using PyTorch DDP with strict causality guarantees.

Features:
    - Multi-node multi-GPU training with NCCL backend (tested up to 48 GPUs)
    - Optimized HDF5 data loading with per-worker file handles
    - Mixed precision training (bfloat16/float16 with automatic detection)
    - Gradient accumulation and clipping for large effective batch sizes
    - Cosine annealing with linear warmup
    - Class-balanced loss weighting for imbalanced datasets
    - Comprehensive logging and checkpointing
    - 100% type hint coverage for thesis-grade code
    - Extensive validation and error handling

Critical Properties:
    - Model uses strictly causal convolutions (no future information leakage)
    - DDP static_graph=False (required for data-dependent attention pooling)
    - Proper GradScaler handling for bfloat16 (disabled when BF16 available)
    - Worker-isolated HDF5 file handles (prevents deadlocks)
    - Correct distributed metric aggregation across all ranks

Usage:
    Single GPU:
        python train.py --data path/to/data.h5
    
    Multi-GPU (single node with torchrun):
        torchrun --nproc-per-node=4 train.py --data path/to/data.h5
    
    Multi-Node (e.g., 48 GPUs across 12 nodes):
        srun torchrun --nnodes=12 --nproc-per-node=4 \\
            --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT \\
            train.py --data path/to/data.h5

    SLURM Example:
        #SBATCH --nodes=12
        #SBATCH --ntasks-per-node=4
        #SBATCH --gpus-per-node=4
        srun python -m torch.distributed.run \\
            --nnodes=$SLURM_NNODES --nproc-per-node=4 \\
            train.py --data path/to/data.h5

Author: Kunal Bhatia
Institution: University of Heidelberg 
Thesis: "From Light Curves to Labels: Machine Learning in Microlensing"
Version: 1.0 
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Import model
try:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from model import ModelConfig, RomanMicrolensingClassifier
except ImportError as e:
    print(f"Error importing model: {e}", file=sys.stderr)
    print("Ensure model.py is in the same directory as train.py", file=sys.stderr)
    sys.exit(1)

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONSTANTS
# =============================================================================

SEED: int = 42
DEFAULT_CLIP_NORM: float = 1.0
MIN_SEQUENCE_LENGTH: int = 1
DEFAULT_VAL_FRACTION: float = 0.2

# Class names for Roman microlensing classification
CLASS_NAMES: Tuple[str, ...] = ('Flat', 'PSPL', 'Binary')


# =============================================================================
# UTILITIES
# =============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and other non-serializable objects."""
    
    def default(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_distributed() -> Tuple[int, int, int, bool]:
    """
    Initialize distributed training environment.
    
    Handles both torchrun and manual process spawning. Sets up NCCL backend
    with optimized settings for multi-GPU training.
    
    Returns:
        Tuple containing:
            - rank: Global rank of this process (0 to world_size-1)
            - local_rank: Local rank on this node (0 to gpus_per_node-1)
            - world_size: Total number of processes across all nodes
            - is_ddp: Whether running in distributed mode
    
    Raises:
        RuntimeError: If CUDA is not available for distributed training
    """
    # Check if running in distributed mode
    if 'RANK' not in os.environ:
        return 0, 0, 1, False
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if rank == 0:
        print(f"Initializing distributed training: {world_size} processes", flush=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for distributed training. "
            "For CPU training, run without torchrun."
        )
    
    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(seconds=3600)  # 1 hour timeout for large jobs
    )
    
    # CUDA optimizations for HPC efficiency
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return rank, local_rank, world_size, True


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    """
    Set up logging for training.
    
    Only rank 0 logs to console and file; other ranks are completely silent
    to avoid interleaved output.
    
    Args:
        rank: Process rank
        output_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ROMAN_TRAIN")
    logger.handlers.clear()
    logger.propagate = False
    
    if is_main_process(rank):
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        # Completely silence non-main processes
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
    
    return logger


def set_global_seeds(seed: int, rank: int = 0) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Base random seed
        rank: Process rank (used as offset for different per-rank seeds)
    """
    effective_seed = seed + rank
    
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    
    # Note: cudnn.benchmark=True sacrifices strict reproducibility for performance
    # For exact reproducibility, set deterministic=True and benchmark=False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_or_create_split(
    data_path: Path,
    logger: logging.Logger,
    rank: int = 0,
    is_ddp: bool = False,
    val_fraction: float = DEFAULT_VAL_FRACTION
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Load cached train/val split or create new one.
    
    Only rank 0 creates the split; other ranks wait and load from cache.
    This ensures consistent splits across all processes.
    
    Args:
        data_path: Path to HDF5 data file
        logger: Logger instance
        rank: Process rank
        is_ddp: Whether running in distributed mode
        val_fraction: Fraction of data for validation (default: 0.2)
        
    Returns:
        Tuple containing:
            - train_idx: Array of training sample indices
            - val_idx: Array of validation sample indices
            - stats: Normalization statistics {'median': float, 'iqr': float}
    """
    cache_path = data_path.parent / f"{data_path.stem}_split_cache.npz"
    
    # Try loading from cache
    if cache_path.exists():
        if is_main_process(rank):
            logger.info(f"Loading cached split from: {cache_path}")
        
        cache = np.load(cache_path)
        train_idx = cache['train_idx']
        val_idx = cache['val_idx']
        stats = {
            'median': float(cache['median']),
            'iqr': float(cache['iqr'])
        }
        
        if is_main_process(rank):
            logger.info(
                f"Loaded split: train={format_number(len(train_idx))}, "
                f"val={format_number(len(val_idx))}"
            )
        
        if is_ddp:
            dist.barrier()
        
        return train_idx, val_idx, stats
    
    # Create new split (only rank 0)
    if is_main_process(rank):
        logger.info("Creating new train/val split...")
        
        with h5py.File(data_path, 'r', rdcc_nbytes=512*1024*1024) as f:
            labels = f['labels'][:]
            n_samples = len(labels)
            
            logger.info(f"Total samples: {format_number(n_samples)}")
            
            # Stratified split
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                shuffle=True,
                random_state=SEED,
                stratify=labels
            )
            
            # Compute normalization statistics from training set
            flux_data = f['flux']
            sample_size = min(50000, len(train_idx))
            sample_idx = np.sort(
                np.random.choice(train_idx, sample_size, replace=False)
            )
            
            valid_flux_chunks: List[np.ndarray] = []
            chunk_size = 10000
            
            for i in range(0, len(sample_idx), chunk_size):
                chunk_indices = sample_idx[i:i + chunk_size]
                flux_chunk = flux_data[chunk_indices.tolist()]
                valid = flux_chunk[~np.isnan(flux_chunk) & (flux_chunk != 0.0)]
                if len(valid) > 0:
                    valid_flux_chunks.append(valid)
            
            if valid_flux_chunks:
                all_valid = np.concatenate(valid_flux_chunks)
                median = float(np.median(all_valid))
                q75, q25 = np.percentile(all_valid, [75, 25])
                iqr = float(max(q75 - q25, 1e-6))
            else:
                logger.warning("No valid flux values found, using defaults")
                median, iqr = 0.0, 1.0
            
            stats = {'median': median, 'iqr': iqr}
            
            # Log class distribution
            unique, counts = np.unique(labels[train_idx], return_counts=True)
            logger.info("Training class distribution:")
            for cls_idx, count in zip(unique, counts):
                pct = 100.0 * count / len(train_idx)
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
                logger.info(f"  {cls_name}: {format_number(count)} ({pct:.2f}%)")
        
        # Save cache
        np.savez(
            cache_path,
            train_idx=train_idx,
            val_idx=val_idx,
            median=stats['median'],
            iqr=stats['iqr']
        )
        
        logger.info(f"Cached split to: {cache_path}")
    else:
        train_idx, val_idx, stats = None, None, None
    
    # Synchronize and load on other ranks
    if is_ddp:
        dist.barrier()
        
        if not is_main_process(rank):
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
    
    CRITICAL: Each DataLoader worker maintains its own HDF5 file handle
    to prevent thread safety issues and potential deadlocks with h5py.
    
    The dataset only returns indices and labels in __getitem__ for efficiency.
    Actual data loading is done in batches via get_batch_data() called from
    the collate function.
    
    Args:
        data_path: Path to HDF5 file
        indices: Array of sample indices to use
        stats: Normalization statistics {'median': float, 'iqr': float}
    """
    
    def __init__(
        self,
        data_path: Path,
        indices: np.ndarray,
        stats: Dict[str, float]
    ) -> None:
        self.data_path: str = str(data_path)
        self.indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        self.stats: Dict[str, float] = stats
        
        # Per-worker HDF5 handles (initialized lazily)
        self._file: Optional[h5py.File] = None
        self._flux: Optional[h5py.Dataset] = None
        self._delta_t: Optional[h5py.Dataset] = None
        self._labels: Optional[h5py.Dataset] = None
    
    def _ensure_open(self) -> None:
        """Open HDF5 file if not already open (lazy initialization)."""
        if self._file is None:
            self._file = h5py.File(
                self.data_path,
                'r',
                rdcc_nbytes=256 * 1024 * 1024,  # 256 MB chunk cache
                rdcc_nslots=10007,  # Prime number for hash table
                libver='latest',
                swmr=True  # Single Writer Multiple Reader mode
            )
            self._flux = self._file['flux']
            self._delta_t = self._file['delta_t']
            self._labels = self._file['labels']
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """
        Return global index and label for a sample.
        
        Args:
            idx: Local index in this dataset
            
        Returns:
            Tuple of (global_index, label)
        """
        self._ensure_open()
        global_idx = int(self.indices[idx])
        label = int(self._labels[global_idx])
        return global_idx, label
    
    def get_batch_data(
        self,
        global_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read flux and delta_t for a batch of global indices.
        
        Optimizes HDF5 reading by sorting indices before reading
        to minimize disk seeks, then restoring original order.
        
        Args:
            global_indices: Array of global indices to read
            
        Returns:
            Tuple of (flux, delta_t) arrays
        """
        self._ensure_open()
        
        # Sort indices for sequential disk access
        sort_order = np.argsort(global_indices)
        sorted_idx = global_indices[sort_order]
        
        # Read from HDF5 (efficient with sorted indices)
        flux = self._flux[sorted_idx.tolist()]
        delta_t = self._delta_t[sorted_idx.tolist()]
        
        # Restore original order
        unsort_order = np.argsort(sort_order)
        flux = flux[unsort_order]
        delta_t = delta_t[unsort_order]
        
        return flux, delta_t
    
    def __del__(self) -> None:
        """Close HDF5 file on deletion."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def collate_fn(
    batch: List[Tuple[int, int]],
    dataset: MicrolensingDataset
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Collate function with optimized HDF5 batch reading and normalization.
    
    Args:
        batch: List of (global_index, label) tuples from __getitem__
        dataset: Dataset instance for reading actual data
        
    Returns:
        Tuple of tensors:
            - flux: Normalized flux (B, T)
            - delta_t: Time differences (B, T)
            - lengths: Valid sequence lengths (B,)
            - labels: Class labels (B,)
    """
    global_indices = np.array([x[0] for x in batch], dtype=np.int64)
    labels = np.array([x[1] for x in batch], dtype=np.int64)
    
    # Batch read from HDF5
    flux, delta_t = dataset.get_batch_data(global_indices)
    
    # Convert to tensors
    flux = torch.from_numpy(flux.astype(np.float32))
    delta_t = torch.from_numpy(delta_t.astype(np.float32))
    labels = torch.from_numpy(labels).long()
    
    # Create validity mask (non-NaN and non-zero)
    valid_mask = (~torch.isnan(flux)) & (flux != 0.0)
    
    # Handle NaN values
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Robust normalization using IQR (less sensitive to outliers than std)
    median = dataset.stats['median']
    iqr = dataset.stats['iqr']
    flux = (flux - median) / iqr
    
    # Zero out invalid positions
    flux = flux * valid_mask.float()
    
    # Compute actual sequence lengths
    lengths = valid_mask.sum(dim=1).long().clamp(min=MIN_SEQUENCE_LENGTH)
    
    return flux, delta_t, lengths, labels


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize random state for each DataLoader worker.
    
    Ensures different workers get different random states while
    maintaining reproducibility.
    
    Args:
        worker_id: Worker ID (0 to num_workers-1)
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


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
    """
    Create training and validation DataLoaders.
    
    Args:
        data_path: Path to HDF5 data file
        train_idx: Training sample indices
        val_idx: Validation sample indices
        stats: Normalization statistics
        batch_size: Batch size per GPU
        num_workers: Number of DataLoader workers
        prefetch_factor: Batches to prefetch per worker
        is_ddp: Whether using distributed training
        rank: Process rank
        
    Returns:
        Tuple of (train_loader, val_loader, train_labels)
    """
    train_dataset = MicrolensingDataset(data_path, train_idx, stats)
    val_dataset = MicrolensingDataset(data_path, val_idx, stats)
    
    # Load labels for class weight computation
    with h5py.File(data_path, 'r') as f:
        all_labels = f['labels'][:]
    train_labels = all_labels[train_idx]
    
    # Create samplers
    if is_ddp:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=SEED,
            drop_last=True
        )
        val_sampler: Optional[DistributedSampler] = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create collate functions with dataset reference
    train_collate: Callable = lambda batch: collate_fn(batch, train_dataset)
    val_collate: Callable = lambda batch: collate_fn(batch, val_dataset)
    
    # DataLoader settings
    use_persistent_workers = num_workers > 0
    use_prefetch = prefetch_factor if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent_workers,
        prefetch_factor=use_prefetch,
        worker_init_fn=worker_init_fn,
        collate_fn=train_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation (no gradients)
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=use_persistent_workers,
        prefetch_factor=use_prefetch,
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
) -> Tensor:
    """
    Compute inverse frequency class weights for balanced loss.
    
    Weights are computed as: w_c = N_total / (n_classes * N_c)
    This upweights minority classes to handle class imbalance.
    
    Args:
        labels: Array of training labels
        n_classes: Number of classes
        device: Device to place weights on
        
    Returns:
        Class weights tensor of shape (n_classes,)
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = torch.ones(n_classes, device=device)
    for cls_idx, count in zip(unique, counts):
        weights[cls_idx] = total / (n_classes * count)
    
    return weights


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    Learning rate schedule:
        - Warmup phase: Linear increase from 0 to base_lr
        - Decay phase: Cosine decay from base_lr to min_lr
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate at end of training
        last_epoch: Last epoch index (for resuming)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs: int = warmup_epochs
        self.total_epochs: int = total_epochs
        self.min_lr: float = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


@contextmanager
def cuda_timer(
    name: str, 
    logger: logging.Logger, 
    rank: int
) -> Generator[None, None, None]:
    """
    Context manager for timing CUDA operations with proper synchronization.
    
    Args:
        name: Timer name for logging
        logger: Logger instance
        rank: Process rank (only rank 0 logs)
        
    Yields:
        None
    """
    if not is_main_process(rank):
        yield
        return
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    yield
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    logger.debug(f"{name}: {elapsed_ms/1000:.2f}s")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    class_weights: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
    config: ModelConfig,
    accumulation_steps: int = 1,
    clip_norm: float = DEFAULT_CLIP_NORM
) -> Tuple[float, float]:
    """
    Train for one epoch with gradient accumulation and mixed precision.
    
    Args:
        model: Model to train
        loader: Training DataLoader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (None if not using)
        class_weights: Class weights for loss computation
        device: Device to train on
        rank: Process rank
        world_size: Number of processes
        epoch: Current epoch number
        config: Model configuration
        accumulation_steps: Steps to accumulate gradients
        clip_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    # Accumulator tensors on device for distributed aggregation
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    
    # Progress bar (only on main process)
    iterator = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        disable=not is_main_process(rank),
        leave=False,
        dynamic_ncols=True
    )
    
    # Clear gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Mixed precision settings
    use_amp = config.use_amp and scaler is not None
    amp_dtype = torch.bfloat16 if config.use_amp else torch.float32
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        # Move to device with non-blocking transfers
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=amp_dtype):
            logits = model(flux, delta_t, lengths)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss_scaled = loss / accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        
        # Optimizer step (with gradient accumulation)
        is_accumulation_step = (
            (batch_idx + 1) % accumulation_steps == 0 or 
            (batch_idx + 1) == len(loader)
        )
        
        if is_accumulation_step:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Accumulate metrics (no gradient needed)
        with torch.no_grad():
            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum()
            total += batch_size
        
        # Update progress bar
        if is_main_process(rank) and batch_idx % 20 == 0:
            current_loss = (total_loss / total).item()
            current_acc = (correct / total).float().item() * 100
            iterator.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.1f}%'
            })
    
    # Aggregate metrics across all processes
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / total).item()
    accuracy = (correct / total).float().item()
    
    return avg_loss, accuracy


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    config: ModelConfig,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        loader: Validation DataLoader
        class_weights: Class weights for loss computation
        device: Device to evaluate on
        rank: Process rank
        world_size: Number of processes
        config: Model configuration
        return_predictions: Whether to return predictions and labels
        
    Returns:
        Dictionary containing:
            - loss: Average validation loss
            - accuracy: Validation accuracy
            - predictions: (optional) Predicted class indices
            - labels: (optional) True labels
            - probabilities: (optional) Class probabilities
    """
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    
    amp_dtype = torch.bfloat16 if config.use_amp else torch.float32
    
    iterator = tqdm(
        loader,
        disable=not is_main_process(rank),
        desc="Eval",
        leave=False,
        dynamic_ncols=True
    )
    
    for flux, delta_t, lengths, labels in iterator:
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
    
    # Aggregate across processes
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    results: Dict[str, Any] = {
        'loss': (total_loss / total).item(),
        'accuracy': (correct / total).float().item()
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
    stats: Dict[str, float],
    output_dir: Path,
    is_best: bool = False
) -> Path:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save (handles DDP wrapper automatically)
        optimizer: Optimizer state
        scheduler: Scheduler state
        scaler: GradScaler state (if using)
        epoch: Current epoch
        best_acc: Best validation accuracy so far
        config: Model configuration
        stats: Normalization statistics
        output_dir: Output directory
        is_best: Whether this is the best model so far
        
    Returns:
        Path to saved checkpoint
    """
    # Extract model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        'stats': stats,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if is_best:
        save_path = output_dir / 'best_model.pt'
    else:
        save_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    
    torch.save(checkpoint, save_path)
    
    return save_path


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Roman Microlensing Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--data', type=str, required=True,
        help='Path to HDF5 training data file'
    )
    data_group.add_argument(
        '--output-dir', type=str, default='../results',
        help='Output directory for results and checkpoints'
    )
    data_group.add_argument(
        '--experiment-name', type=str, default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    data_group.add_argument(
        '--val-fraction', type=float, default=DEFAULT_VAL_FRACTION,
        help='Fraction of data for validation'
    )
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--d-model', type=int, default=64,
        help='Model hidden dimension'
    )
    model_group.add_argument(
        '--n-layers', type=int, default=2,
        help='Number of GRU layers'
    )
    model_group.add_argument(
        '--dropout', type=float, default=0.3,
        help='Dropout probability'
    )
    model_group.add_argument(
        '--window-size', type=int, default=5,
        help='Causal window size'
    )
    model_group.add_argument(
        '--hierarchical', action='store_true', default=True,
        help='Use hierarchical (multi-scale) feature extraction'
    )
    model_group.add_argument(
        '--no-hierarchical', dest='hierarchical', action='store_false'
    )
    model_group.add_argument(
        '--attention-pooling', action='store_true', default=True,
        help='Use attention pooling'
    )
    model_group.add_argument(
        '--no-attention-pooling', dest='attention_pooling', action='store_false'
    )
    model_group.add_argument(
        '--use-residual', action='store_true', default=True,
        help='Use residual connections'
    )
    model_group.add_argument(
        '--no-residual', dest='use_residual', action='store_false'
    )
    model_group.add_argument(
        '--use-layer-norm', action='store_true', default=True,
        help='Use layer normalization'
    )
    model_group.add_argument(
        '--no-layer-norm', dest='use_layer_norm', action='store_false'
    )
    model_group.add_argument(
        '--feature-extraction', type=str, default='conv',
        choices=['conv', 'mlp'],
        help='Feature extraction method'
    )
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size per GPU'
    )
    train_group.add_argument(
        '--accumulation-steps', type=int, default=1,
        help='Gradient accumulation steps'
    )
    train_group.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--lr', type=float, default=1e-3,
        help='Peak learning rate'
    )
    train_group.add_argument(
        '--weight-decay', type=float, default=1e-3,
        help='AdamW weight decay'
    )
    train_group.add_argument(
        '--warmup-epochs', type=int, default=5,
        help='Learning rate warmup epochs'
    )
    train_group.add_argument(
        '--clip-norm', type=float, default=DEFAULT_CLIP_NORM,
        help='Gradient clipping norm'
    )
    train_group.add_argument(
        '--min-lr', type=float, default=1e-6,
        help='Minimum learning rate'
    )
    
    # Mixed precision
    amp_group = parser.add_argument_group('Mixed Precision')
    amp_group.add_argument(
        '--use-amp', action='store_true', default=True,
        help='Use automatic mixed precision'
    )
    amp_group.add_argument(
        '--no-amp', dest='use_amp', action='store_false'
    )
    
    # Other options
    other_group = parser.add_argument_group('Other')
    other_group.add_argument(
        '--use-class-weights', action='store_true', default=True,
        help='Use class-balanced loss weights'
    )
    other_group.add_argument(
        '--no-class-weights', dest='use_class_weights', action='store_false'
    )
    other_group.add_argument(
        '--compile', action='store_true', default=False,
        help='Use torch.compile for optimization'
    )
    other_group.add_argument(
        '--use-gradient-checkpointing', action='store_true', default=False,
        help='Use gradient checkpointing to reduce memory'
    )
    other_group.add_argument(
        '--num-workers', type=int, default=4,
        help='DataLoader workers per GPU'
    )
    other_group.add_argument(
        '--prefetch-factor', type=int, default=4,
        help='Batches to prefetch per worker'
    )
    other_group.add_argument(
        '--eval-every', type=int, default=5,
        help='Evaluate every N epochs'
    )
    other_group.add_argument(
        '--save-every', type=int, default=10,
        help='Save checkpoint every N epochs'
    )
    other_group.add_argument(
        '--early-stopping-patience', type=int, default=0,
        help='Early stopping patience (0 = disabled)'
    )
    other_group.add_argument(
        '--broadcast-buffers', action='store_true', default=True,
        help='Broadcast BatchNorm buffers in DDP'
    )
    other_group.add_argument(
        '--no-broadcast-buffers', dest='broadcast_buffers', action='store_false'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Initialize distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"roman_d{args.d_model}_l{args.n_layers}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        dist.barrier()
    
    # Setup logging and seeds
    logger = setup_logging(rank, output_dir)
    set_global_seeds(SEED, rank)
    
    # Log configuration
    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("ROMAN MICROLENSING CLASSIFIER - DISTRIBUTED TRAINING")
        logger.info("=" * 80)
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        logger.info(f"World size: {world_size} GPU(s)")
        logger.info(f"Rank: {rank}, Local rank: {local_rank}")
        logger.info(f"Output directory: {output_dir}")
    
    # Load data
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Loading data...")
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    train_idx, val_idx, stats = load_or_create_split(
        data_path, logger, rank, is_ddp, args.val_fraction
    )
    
    if is_main_process(rank):
        logger.info(f"Dataset: {format_number(len(train_idx) + len(val_idx))} samples")
        logger.info(f"Train: {format_number(len(train_idx))}, Val: {format_number(len(val_idx))}")
        logger.info(f"Normalization: median={stats['median']:.4f}, IQR={stats['iqr']:.4f}")
    
    # Create data loaders
    train_loader, val_loader, train_labels = create_dataloaders(
        data_path, train_idx, val_idx, stats,
        args.batch_size, args.num_workers, args.prefetch_factor,
        is_ddp, rank
    )
    
    effective_batch_size = args.batch_size * args.accumulation_steps * world_size
    
    if is_main_process(rank):
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Gradient accumulation: {args.accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Workers per GPU: {args.num_workers}")
    
    # Build model
    if is_main_process(rank):
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
    
    model = RomanMicrolensingClassifier(config).to(device)
    
    if is_main_process(rank):
        n_params = model.count_parameters()
        receptive_field = model.receptive_field
        logger.info(f"Parameters: {format_number(n_params)}")
        logger.info(f"Receptive field: {receptive_field} timesteps")
        logger.info(f"Architecture: {'Hierarchical' if config.hierarchical else 'Simple'} CNN + GRU")
        logger.info(f"Pooling: {'Attention' if config.use_attention_pooling else 'Mean'}")
    
    # Wrap with DDP
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=args.broadcast_buffers,
            static_graph=False  # REQUIRED: attention pooling has data-dependent gradients
        )
    
    # Optional: torch.compile
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank):
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        fused=torch.cuda.is_available()  # Use fused optimizer if available
    )
    
    # Scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr
    )
    
    # GradScaler setup
    # Note: GradScaler is only needed for float16, not bfloat16
    use_scaler = args.use_amp and torch.cuda.is_available()
    if use_scaler:
        # Check if bfloat16 is supported - if so, scaler is unnecessary
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_scaler = False
            if is_main_process(rank):
                logger.info("Using bfloat16 (no GradScaler needed)")
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
        if is_main_process(rank):
            logger.info(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
    # Synchronize before training
    if is_ddp:
        dist.barrier()
    
    # Training loop
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Starting training...")
        logger.info("-" * 80)
    
    best_acc: float = 0.0
    patience_counter: int = 0
    training_start_time = datetime.now()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = datetime.now()
        
        # Set epoch for distributed sampler
        if is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler,
            class_weights, device, rank, world_size, epoch, config,
            accumulation_steps=args.accumulation_steps,
            clip_norm=args.clip_norm
        )
        
        # Update scheduler
        scheduler.step()
        
        # Evaluation
        should_eval = (
            epoch % args.eval_every == 0 or 
            epoch == 1 or 
            epoch == args.epochs
        )
        
        if should_eval:
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, world_size, config,
                return_predictions=(epoch == args.epochs and is_main_process(rank))
            )
            
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            if is_main_process(rank):
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"Time={format_time(epoch_time)}"
                )
                
                # Save best model
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
                
                # Periodic checkpoint
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        best_acc, config, stats, output_dir, is_best=False
                    )
        else:
            if is_main_process(rank):
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"Time={format_time(epoch_time)}"
                )
        
        # Early stopping
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            if is_main_process(rank):
                logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Periodic memory cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final evaluation and reporting
    total_time = (datetime.now() - training_start_time).total_seconds()
    
    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info(f"Training complete")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best validation accuracy: {best_acc*100:.2f}%")
        logger.info("=" * 80)
        
        # Load best model
        checkpoint = torch.load(
            output_dir / 'best_model.pt',
            map_location=device,
            weights_only=False
        )
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation with predictions
        logger.info("Running final evaluation...")
        final_results = evaluate(
            model, val_loader, class_weights, device, rank, world_size, config,
            return_predictions=True
        )
        
        # Save predictions
        np.savez(
            output_dir / 'final_predictions.npz',
            predictions=final_results['predictions'],
            labels=final_results['labels'],
            probabilities=final_results['probabilities']
        )
        
        # Confusion matrix
        cm = confusion_matrix(final_results['labels'], final_results['predictions'])
        logger.info(f"\nConfusion matrix:\n{cm}")
        np.save(output_dir / 'confusion_matrix.npy', cm)
        
        # Classification report
        report = classification_report(
            final_results['labels'],
            final_results['predictions'],
            target_names=list(CLASS_NAMES),
            digits=4
        )
        logger.info(f"\nClassification report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # Save configuration
        config_dict = {
            'model_config': asdict(config),
            'training_args': vars(args),
            'stats': stats,
            'best_accuracy': float(best_acc),
            'final_accuracy': final_results['accuracy'],
            'world_size': world_size,
            'total_training_time_seconds': total_time,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"\nAll results saved to: {output_dir}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
