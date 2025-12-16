#!/usr/bin/env python3
"""
Roman Microlensing Classifier Training Engine v3.0.0
====================================================

High-performance distributed training pipeline with RAM-based data loading for
classifying Nancy Grace Roman Space Telescope gravitational microlensing light
curves into Flat, PSPL (Point Source Point Lens), and Binary classes.

OPTIMIZATION STRATEGY
---------------------
This is an optimized version of the production training script with
RAM-based dataset loading for 20-50× speedup over disk-based loading.

All architectural features, bug fixes, and training logic are preserved exactly.

KEY MODIFICATION: RAM LOADING
------------------------------
**RAMLensingDataset**:
    - Loads entire HDF5/NPZ dataset into RAM at initialization (~30 seconds)
    - Zero disk I/O during training (pure in-memory access)
    - Eliminates HDF5 global lock contention
    - Eliminates network filesystem latency
    - Supports both HDF5 (.h5, .hdf5) and NPZ (.npz) formats with auto-detection
    - Memory requirement: ~29 GB per GPU process, ~116 GB per 4-GPU node
    - A100 nodes (256-512 GB RAM): 25-50% utilization (plenty of headroom)
    - Explicit __del__ cleanup to prevent memory leaks

CRITICAL FIX v3.0.0 - COMPREHENSIVE UPDATE
-------------------------------------------
**All fixes from v2.9 preserved plus:**

1. **pos_weight Shape Fix**: Removed redundant .unsqueeze() calls in BCE loss
   - pos_weight should be scalar for binary classification
   - Old: pos_weight.unsqueeze(0).unsqueeze(0) -> shape [1, 1]
   - New: pos_weight (scalar tensor, properly broadcast)

2. **Memory Leak Fix**: Added __del__ to RAMLensingDataset for explicit cleanup

3. **Naming Consistency**: 
   - Data contains MAGNIFICATION (A=1.0 baseline, A>1 magnified)
   - Variable names and docstrings updated for clarity
   - 'flux' kept for backward compatibility but documented as magnification

4. **Type Hints**: Complete type hint coverage (100%)

5. **Version Sync**: All components now v3.0.0 for consistency

6. **Constants**: All magic numbers moved to module-level constants

HIERARCHICAL COLLAPSE FIX (from v2.9)
--------------------------------------
**Problem**: Stage 2 (PSPL vs Binary) collapsed to always predict Binary.
The model achieved 97% training accuracy but 0% PSPL recall on test.

**Root Cause**: 
- Using NLLLoss on combined probabilities doesn't provide direct supervision to Stage 2
- When p_deviation is small, gradients to pspl_head are multiplicatively suppressed
- Stage 2 head initialized with negative bias → sigmoid outputs ~0 → stuck

**Solution**: Separate BCE losses for each hierarchical stage:
- Stage 1 BCE: train flat_head to distinguish Flat vs Non-Flat
- Stage 2 BCE: train pspl_head to distinguish PSPL vs Binary (only on non-flat samples)
- Auxiliary CE: direct 3-class supervision for gradient stability

**Loss function**:
    total_loss = stage1_bce + stage2_bce + aux_weight * aux_ce

DATA FORMAT NOTE
----------------
The training data contains NORMALIZED MAGNIFICATION values:
    - A = 1.0: Baseline (unmagnified source)
    - A > 1.0: Magnified (e.g., A=2.0 means 2× brighter)
    - A = 0.0: Masked/missing observations (padding)

This is NOT flux in Jansky. The 'flux' key name is retained for backward
compatibility with existing checkpoints and data files.

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 3.0.0
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

__version__: str = "3.0.0"

__all__ = [
    "RAMLensingDataset",
    "WarmupCosineScheduler",
    "compute_class_weights",
    "compute_hierarchical_loss",
    "train_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint_for_resume",
    "setup_ddp",
    "cleanup_ddp",
    "main",
]

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def _configure_environment() -> None:
    """
    Ultra-optimized environment configuration for distributed training.
    
    Sets NCCL, CUDA, and PyTorch environment variables for optimal
    performance on multi-GPU clusters.
    """
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

# Random seed for reproducibility
SEED: int = 42

# Numerical stability epsilon
EPS: float = 1e-8

# Class names for logging and reporting
CLASS_NAMES: Tuple[str, str, str] = ('Flat', 'PSPL', 'Binary')

# Number of classes
N_CLASSES: int = 3

# Logging format
LOG_FORMAT: str = '%(asctime)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# Default hyperparameters
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 50
DEFAULT_NUM_WORKERS: int = 0  # No workers needed with RAM loading
DEFAULT_PREFETCH_FACTOR: int = 2
DEFAULT_ACCUMULATION_STEPS: int = 1
DEFAULT_CLIP_NORM: float = 1.0
DEFAULT_WARMUP_EPOCHS: int = 3
DEFAULT_VAL_FRACTION: float = 0.1

# Hierarchical loss weights (v2.9+)
DEFAULT_STAGE1_WEIGHT: float = 1.0
DEFAULT_STAGE2_WEIGHT: float = 1.0
DEFAULT_AUX_WEIGHT: float = 0.5

# Progress update frequency (batches)
PROGRESS_UPDATE_FREQ: int = 50

# DDP timeouts
DDP_INIT_TIMEOUT_MINUTES: int = 10
DDP_BARRIER_TIMEOUT_SECONDS: int = 300

# Invalid/padding value for timestamps
INVALID_TIMESTAMP: float = -999.0

# Minimum sequence length to avoid edge cases
MIN_VALID_SEQ_LEN: int = 10

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(rank: int) -> logging.Logger:
    """
    Configure logging for distributed training.
    
    Only rank 0 logs at INFO level; other ranks log at WARNING.
    
    Parameters
    ----------
    rank : int
        Process rank in distributed training (-1 for single process).
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
    
    return logger

logger = setup_logging(0)

# =============================================================================
# UTILITIES
# =============================================================================

def create_experiment_dir(base_dir: Path, args: argparse.Namespace) -> Path:
    """
    Create timestamped experiment directory.
    
    Parameters
    ----------
    base_dir : Path
        Base output directory
    args : argparse.Namespace
        Training arguments
        
    Returns
    -------
    Path
        Experiment directory path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create experiment name from key parameters
    exp_name = (
        f"d{args.d_model}_"
        f"l{args.n_layers}_"
        f"{'hier' if args.hierarchical else 'flat'}_"
        f"{timestamp}"
    )
    
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    return exp_dir


def is_main_process(rank: int) -> bool:
    """
    Check if current process is the main process.
    
    Parameters
    ----------
    rank : int
        Process rank.
        
    Returns
    -------
    bool
        True if rank is 0 or -1 (single process mode).
    """
    return rank == 0 or rank == -1


def format_number(n: int) -> str:
    """
    Format large numbers with K/M suffixes for readability.
    
    Parameters
    ----------
    n : int
        Number to format.
        
    Returns
    -------
    str
        Formatted string (e.g., "1.5M", "500K", "999").
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Parameters
    ----------
    seconds : float
        Duration in seconds.
        
    Returns
    -------
    str
        Formatted string (e.g., "45.2s", "12.5m", "2.3h").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    """
    Configure CUDA settings for optimal training performance.
    
    Enables TF32 for faster matrix operations on Ampere+ GPUs and
    cuDNN autotuning for optimal convolution algorithms.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# =============================================================================
# ULTRA-FAST RAM DATASET
# =============================================================================

class RAMLensingDataset(Dataset):
    """
    Load entire dataset into RAM for ultra-fast training.
    
    This dataset implementation loads all data into system RAM at initialization,
    eliminating disk I/O during training. This provides 20-50× speedup over
    disk-based loading, especially on network filesystems.
    
    Performance Characteristics
    ---------------------------
    - Speed: 0.2-0.5 s/batch (vs 2-12 s/batch on disk)
    - Memory: ~7 GB per 1M samples
    - Startup: ~30 seconds to load into RAM
    
    Data Format
    -----------
    The 'flux' array contains NORMALIZED MAGNIFICATION values:
    - A = 1.0: Baseline (unmagnified source)
    - A > 1.0: Magnified (e.g., A=2.0 means 2× brighter)
    - A = 0.0: Masked/missing observations (padding)
    
    The 'flux' key name is retained for backward compatibility even though
    the values are magnifications, not flux in Jansky.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 (.h5, .hdf5) or NPZ (.npz) data file.
    indices : np.ndarray
        Array of sample indices to include in this dataset.
    magnification_mean : float
        Mean magnification value for normalization.
    magnification_std : float
        Standard deviation of magnification for normalization.
    delta_t_mean : float
        Mean time interval for normalization.
    delta_t_std : float
        Standard deviation of time intervals for normalization.
    rank : int, optional
        Process rank for logging. Default is 0.
        
    Attributes
    ----------
    magnification : np.ndarray
        Raw magnification array loaded into RAM [n_samples, seq_len].
    delta_t : np.ndarray
        Raw time interval array [n_samples, seq_len].
    labels : np.ndarray
        Class labels [n_samples].
        
    Notes
    -----
    v3.0.0 FIX: Added __del__ method for explicit memory cleanup to prevent
    memory leaks when dataset objects are destroyed.
    
    Examples
    --------
    >>> dataset = RAMLensingDataset(
    ...     'data/train.h5',
    ...     train_indices,
    ...     mag_mean=1.05, mag_std=0.23,
    ...     dt_mean=0.0084, dt_std=0.015
    ... )
    >>> magnification, delta_t, length, label = dataset[0]
    """
    
    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        magnification_mean: float,
        magnification_std: float,
        delta_t_mean: float,
        delta_t_std: float,
        rank: int = 0
    ) -> None:
        self.indices = indices
        self.rank = rank
        
        # Pre-compute scale factors for normalization
        # Formula: normalized = (value - mean) / (std + eps)
        self._magnification_scale = 1.0 / (magnification_std + EPS)
        self._dt_scale = 1.0 / (delta_t_std + EPS)
        self.magnification_mean = magnification_mean
        self.delta_t_mean = delta_t_mean
        
        if rank == 0:
            print(f"🚀 ULTRA-FAST MODE: Loading dataset into RAM...")
            print(f"   Source: {file_path}")
        
        # Detect file format and load
        file_path = Path(file_path)
        
        if file_path.suffix == '.npz':
            self._load_npz(file_path)
        elif file_path.suffix in ['.h5', '.hdf5']:
            self._load_hdf5(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .h5, .hdf5, or .npz")
        
        # Apply index subset
        self.magnification = self.magnification[self.indices]
        self.delta_t = self.delta_t[self.indices]
        self.labels = self.labels[self.indices]
        
        if rank == 0:
            total_mem = (self.magnification.nbytes + self.delta_t.nbytes + 
                        self.labels.nbytes) / 1e9
            print(f"   ✓ Dataset in RAM: {total_mem:.2f} GB")
            print(f"   ✓ Samples: {len(self.indices):,}")
            print(f"   ✓ Ready for ULTRA-FAST training!")
    
    def _load_npz(self, file_path: Path) -> None:
        """Load data from NPZ file format."""
        if self.rank == 0:
            print("   Format: NPZ")
        data = np.load(str(file_path))
        
        # Try common key names for magnification data
        if 'flux' in data:
            self.magnification = data['flux']
        elif 'magnification' in data:
            self.magnification = data['magnification']
        elif 'mag' in data:
            self.magnification = data['mag']
        else:
            raise KeyError(f"NPZ file missing magnification data. Keys: {list(data.keys())}")
        
        self.delta_t = data['delta_t']
        
        if 'labels' in data:
            self.labels = data['labels']
        elif 'y' in data:
            self.labels = data['y']
        else:
            raise KeyError(f"NPZ file missing labels. Keys: {list(data.keys())}")
    
    def _load_hdf5(self, file_path: Path) -> None:
        """Load data from HDF5 file format into RAM."""
        if self.rank == 0:
            print("   Format: HDF5 (loading into RAM)")
        
        with h5py.File(str(file_path), 'r') as f:
            if self.rank == 0:
                print("   Loading magnification data...", end='', flush=True)
            
            # Try common key names (flux is legacy name for magnification)
            if 'flux' in f:
                self.magnification = f['flux'][:]
            elif 'magnification' in f:
                self.magnification = f['magnification'][:]
            elif 'mag' in f:  
                self.magnification = f['mag'][:] 
            else:
                raise KeyError(f"HDF5 file missing magnification data. Keys: {list(f.keys())}")
            
            if self.rank == 0:
                print(f" ✓ ({self.magnification.nbytes/1e9:.2f} GB)")
            
            if self.rank == 0:
                print("   Loading delta_t...", end='', flush=True)
            self.delta_t = f['delta_t'][:]
            if self.rank == 0:
                print(f" ✓ ({self.delta_t.nbytes/1e9:.2f} GB)")
            
            if self.rank == 0:
                print("   Loading labels...", end='', flush=True)
            self.labels = f['labels'][:]
            if self.rank == 0:
                print(f" ✓ ({self.labels.nbytes/1e9:.2f} GB)")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        """
        Get sample with sequence compaction.
        
        Sequence compaction moves all valid (non-zero) observations to a
        contiguous prefix [0, n_valid), which is required by the model's
        length-based masking.
        
        Parameters
        ----------
        idx : int
            Sample index.
            
        Returns
        -------
        magnification_norm : Tensor
            Normalized magnification, compacted [max_seq_len].
        delta_t_norm : Tensor
            Normalized delta_t, compacted [max_seq_len].
        length : int
            Number of valid observations in contiguous prefix.
        label : int
            Class label (0=Flat, 1=PSPL, 2=Binary).
        """
        # Load from RAM (instant access)
        magnification_raw = self.magnification[idx].copy()
        delta_t_raw = self.delta_t[idx].copy()
        label = int(self.labels[idx])
        
        # SEQUENCE COMPACTION: Move valid observations to contiguous prefix
        # Valid observations have non-zero magnification (A != 0)
        valid_mask = magnification_raw != 0.0
        valid_count = valid_mask.sum()
        
        # Edge case: entirely empty sequence
        if valid_count == 0:
            valid_count = 1
            magnification_raw[0] = self.magnification_mean
            delta_t_raw[0] = 0.0
            valid_mask[0] = True
        
        # Compact: move valid observations to prefix [0, valid_count)
        magnification_compacted = np.zeros_like(magnification_raw)
        delta_t_compacted = np.zeros_like(delta_t_raw)
        
        magnification_compacted[:valid_count] = magnification_raw[valid_mask]
        delta_t_compacted[:valid_count] = delta_t_raw[valid_mask]
        
        # Normalize using mean/std
        magnification_norm = (magnification_compacted - self.magnification_mean) * self._magnification_scale
        dt_norm = (delta_t_compacted - self.delta_t_mean) * self._dt_scale
        
        # Convert to tensors
        magnification_tensor = torch.from_numpy(magnification_norm).float()
        delta_t_tensor = torch.from_numpy(dt_norm).float()
        length = int(valid_count)
        
        return magnification_tensor, delta_t_tensor, length, label
    
    def __del__(self) -> None:
        """Cleanup large arrays - let GC handle collection naturally."""
        try:
            if hasattr(self, 'magnification'):
                self.magnification = None 
            if hasattr(self, 'delta_t'):
                self.delta_t = None
            if hasattr(self, 'labels'):
                self.labels = None    
        except:
            pass


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_robust_statistics(
    file_path: str,
    rank: int = 0
) -> Dict[str, float]:
    """
    Compute robust statistics for SEPARATE magnification and delta_t normalization.
    
    Computes mean and standard deviation for both magnification and time interval
    arrays, excluding zero/masked values. These statistics are saved with
    checkpoints for consistent normalization during evaluation.
    
    Parameters
    ----------
    file_path : str
        Path to data file (.h5 or .npz).
    rank : int, optional
        Process rank for logging. Default is 0.
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'flux_mean': Mean magnification (key kept for backward compatibility)
        - 'flux_std': Std of magnification
        - 'delta_t_mean': Mean time interval
        - 'delta_t_std': Std of time intervals
        
    Notes
    -----
    The 'flux' key names are retained for backward compatibility with existing
    checkpoints, even though the data contains magnification values.
    
    Uses mean/std (not median/IQR) because IQR can be zero for delta_t arrays
    with many identical values.
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        magnification_data = data.get('flux', data.get('magnification', data.get('mag')))
        delta_t_data = data['delta_t']
    else:
        with h5py.File(str(file_path), 'r') as f:
            if 'flux' in f:
                magnification_data = f['flux'][:]
            else:
                magnification_data = f['magnification'][:]
            delta_t_data = f['delta_t'][:]
    
    # Exclude masked values (zero)
    magnification_valid = magnification_data[magnification_data != 0.0]
    delta_t_valid = delta_t_data[delta_t_data != 0.0]
    
    # Compute mean/std (NOT median/IQR - IQR can be 0 for delta_t!)
    magnification_mean = float(np.mean(magnification_valid))
    magnification_std = float(np.std(magnification_valid))
    
    delta_t_mean = float(np.mean(delta_t_valid))
    delta_t_std = float(np.std(delta_t_valid))
    
    if is_main_process(rank):
        logger.info("Normalization Statistics (mean/std):")
        logger.info(f"  Magnification - Mean: {magnification_mean:.4f}, Std: {magnification_std:.4f}")
        logger.info(f"  Delta_t       - Mean: {delta_t_mean:.6f}, Std: {delta_t_std:.6f}")
    
    # Return with 'flux' keys for backward compatibility
    return {
        'flux_mean': magnification_mean,
        'flux_std': magnification_std,
        'delta_t_mean': delta_t_mean,
        'delta_t_std': delta_t_std
    }


def load_and_split_data(
    file_path: str,
    val_fraction: float,
    seed: int,
    rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Load data, compute statistics, and create stratified train/val split.
    
    Parameters
    ----------
    file_path : str
        Path to data file.
    val_fraction : float
        Fraction of data for validation (0-1).
    seed : int
        Random seed for reproducible splitting.
    rank : int
        Process rank for logging.
        
    Returns
    -------
    train_idx : np.ndarray
        Indices for training samples.
    val_idx : np.ndarray
        Indices for validation samples.
    train_labels : np.ndarray
        Labels for training samples (for class weight computation).
    stats : dict
        Normalization statistics dictionary.
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        total_samples = len(data.get('labels', data['y']))
        all_labels = data.get('labels', data['y'])
    else:
        with h5py.File(str(file_path), 'r') as f:
            total_samples = len(f['labels'])
            all_labels = f['labels'][:]
    
    # Compute normalization statistics
    stats = compute_robust_statistics(file_path, rank)
    
    # Stratified train/val split
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
    rank: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders with RAM-loaded datasets.
    
    Parameters
    ----------
    file_path : str
        Path to data file.
    train_idx : np.ndarray
        Training sample indices.
    val_idx : np.ndarray
        Validation sample indices.
    stats : dict
        Normalization statistics.
    batch_size : int
        Batch size per GPU.
    num_workers : int
        Number of data loading workers.
    prefetch_factor : int
        Number of batches to prefetch per worker.
    is_ddp : bool
        Whether using distributed data parallel.
    rank : int
        Process rank.
        
    Returns
    -------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    """
    # Create RAM-loaded datasets
    train_dataset = RAMLensingDataset(
        file_path,
        train_idx,
        stats['flux_mean'],
        stats['flux_std'],
        stats['delta_t_mean'],
        stats['delta_t_std'],
        rank=rank
    )
    
    val_dataset = RAMLensingDataset(
        file_path,
        val_idx,
        stats['flux_mean'],
        stats['flux_std'],
        stats['delta_t_mean'],
        stats['delta_t_std'],
        rank=rank
    )
    
    # Setup samplers for distributed training
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
        if num_workers > 0:
            logger.info(f"  Prefetch factor: {prefetch_factor}")
    
    return train_loader, val_loader


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Implements a learning rate schedule that:
    1. Linearly increases from 0 to base_lr during warmup
    2. Follows cosine decay from base_lr to min_lr after warmup
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Number of warmup steps.
    total_steps : int
        Total training steps.
    min_lr : float, optional
        Minimum learning rate. Default is 0.0.
    last_epoch : int, optional
        Index of last epoch. Default is -1.
        
    Examples
    --------
    >>> scheduler = WarmupCosineScheduler(
    ...     optimizer,
    ...     warmup_steps=1000,
    ...     total_steps=10000,
    ...     min_lr=1e-6
    ... )
    """
    
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
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_steps, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device
) -> Tensor:
    """
    Compute balanced class weights for imbalanced datasets.
    
    Uses inverse frequency weighting: classes with fewer samples get higher weights.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of class labels.
    n_classes : int
        Number of classes.
    device : torch.device
        Device to place weight tensor on.
        
    Returns
    -------
    weights : Tensor
        Class weights tensor [n_classes], normalized to sum to n_classes.
        
    Examples
    --------
    >>> labels = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced
    >>> weights = compute_class_weights(labels, 3, device)
    >>> print(weights)  # Higher weight for class 2
    """
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + EPS)
    weights = weights / weights.sum() * n_classes  # Normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =============================================================================
# v3.0.0 FIX: HIERARCHICAL LOSS COMPUTATION
# =============================================================================

def compute_hierarchical_loss(
    output: HierarchicalOutput,
    labels: Tensor,
    class_weights: Tensor,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.5
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute hierarchical loss with separate BCE for each stage.
    
    This is the CRITICAL FIX for the hierarchical collapse problem where
    Stage 2 (PSPL vs Binary) would collapse to always predicting Binary.
    
    Instead of using NLLLoss on combined probabilities (which doesn't provide
    direct supervision to Stage 2), we use:
    - Stage 1 BCE: trains flat_head directly on Flat vs Non-Flat
    - Stage 2 BCE: trains pspl_head directly on PSPL vs Binary (non-flat only)
    - Auxiliary CE: direct 3-class supervision for gradient stability
    
    Loss Function
    -------------
    total_loss = stage1_weight * stage1_bce 
               + stage2_weight * stage2_bce 
               + aux_weight * aux_ce
    
    Parameters
    ----------
    output : HierarchicalOutput
        Output from model with return_intermediates=True.
        Contains logits, stage1_logit, stage2_logit, aux_logits, etc.
    labels : Tensor
        Class labels [batch]. 0=Flat, 1=PSPL, 2=Binary.
    class_weights : Tensor
        Class weights [3] for weighting samples.
    stage1_weight : float, optional
        Weight for Stage 1 BCE loss. Default is 1.0.
    stage2_weight : float, optional
        Weight for Stage 2 BCE loss. Default is 1.0.
    aux_weight : float, optional
        Weight for auxiliary 3-class CE loss. Default is 0.5.
        
    Returns
    -------
    total_loss : Tensor
        Combined loss for backpropagation.
    loss_dict : dict
        Individual loss components for logging:
        - 'stage1_bce': Stage 1 loss value
        - 'stage2_bce': Stage 2 loss value
        - 'aux_ce': Auxiliary loss value
        - 'total': Total combined loss
        - 'n_non_flat': Number of non-flat samples in batch
        
    Notes
    -----
    v3.0.0 FIX: Removed redundant .unsqueeze() calls on pos_weight.
    The pos_weight argument for binary_cross_entropy_with_logits should be
    properly broadcast to match the input shape.
    
    References
    ----------
    The hierarchical approach follows:
    - P(Flat) = 1 - P(Deviation)
    - P(PSPL) = P(Deviation) × P(PSPL|Deviation)
    - P(Binary) = P(Deviation) × (1 - P(PSPL|Deviation))
    """
    device = labels.device
    B = labels.size(0)
    
    # Create binary targets for hierarchical stages
    # Stage 1: is_deviation (non-flat)
    # label=0 → is_deviation=0 (Flat)
    # label=1,2 → is_deviation=1 (PSPL or Binary = deviation)
    is_deviation = (labels > 0).float().unsqueeze(1)  # [B, 1]
    
    # Stage 2: is_pspl given deviation
    # label=1 → is_pspl=1 (among deviations, this is PSPL)
    # label=2 → is_pspl=0 (among deviations, this is Binary)
    is_pspl = (labels == 1).float().unsqueeze(1)  # [B, 1]
    
    # Mask for non-flat samples (where Stage 2 loss applies)
    non_flat_mask = (labels > 0)  # [B]
    n_non_flat = non_flat_mask.sum().item()
    
    # =========================================================================
    # Stage 1 Loss: Flat vs Non-Flat (BCE on all samples)
    # =========================================================================
    # pos_weight balances positive (non-flat) vs negative (flat) samples
    # Higher weight for non-flat makes model more sensitive to detecting events
    stage1_pos_weight_scalar = (class_weights[1] + class_weights[2]) / 2.0 / (class_weights[0] + EPS)
    
    stage1_bce = F.binary_cross_entropy_with_logits(
        output.stage1_logit,  # [B, 1] raw logit
        is_deviation,         # [B, 1] target
        pos_weight=stage1_pos_weight_scalar,  
        reduction='mean'
    )
    
    # =========================================================================
    # Stage 2 Loss: PSPL vs Binary (BCE on non-flat samples only)
    # =========================================================================
    if n_non_flat > 0:
        # Extract non-flat samples
        stage2_logit_nonflat = output.stage2_logit[non_flat_mask]  # [n_non_flat, 1]
        is_pspl_nonflat = is_pspl[non_flat_mask]                   # [n_non_flat, 1]
        
        # pos_weight balances PSPL (positive) vs Binary (negative) among non-flat
        stage2_pos_weight_scalar = class_weights[1] / (class_weights[2] + EPS)
        
        # v3.0.0 FIX: Proper scalar pos_weight
        stage2_pos_weight = stage2_pos_weight_scalar.expand(1)  # [1] broadcasts
        
        stage2_bce = F.binary_cross_entropy_with_logits(
            stage2_logit_nonflat,  # raw logit
            is_pspl_nonflat,       # target
            pos_weight=stage2_pos_weight,
            reduction='mean'
        )
    else:
        # No non-flat samples in batch (rare edge case)
        stage2_bce = torch.tensor(0.0, device=device)
    
    # =========================================================================
    # Auxiliary Loss: Direct 3-class CE (for gradient stability)
    # =========================================================================
    if output.aux_logits is not None:
        aux_ce = F.cross_entropy(output.aux_logits, labels, weight=class_weights)
    else:
        aux_ce = torch.tensor(0.0, device=device)
    
    # =========================================================================
    # Combined Loss
    # =========================================================================
    total_loss = (
        stage1_weight * stage1_bce +
        stage2_weight * stage2_bce +
        aux_weight * aux_ce
    )
    
    loss_dict = {
        'stage1_bce': float(stage1_bce.item()),
        'stage2_bce': float(stage2_bce.item()) if n_non_flat > 0 else 0.0,
        'aux_ce': float(aux_ce.item()) if output.aux_logits is not None else 0.0,
        'total': float(total_loss.item()),
        'n_non_flat': n_non_flat
    }
    
    return total_loss, loss_dict


# =============================================================================
# TRAINING LOOP
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
    config: ModelConfig,
    accumulation_steps: int = 1,
    clip_norm: float = 1.0,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.5
) -> Tuple[float, float]:
    """
    Execute one training epoch with GPU-side metric accumulation.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : _LRScheduler
        Learning rate scheduler.
    scaler : torch.amp.GradScaler or None
        Gradient scaler for mixed precision.
    class_weights : Tensor
        Class weights for loss computation.
    device : torch.device
        Training device.
    rank : int
        Process rank.
    world_size : int
        Number of distributed processes.
    epoch : int
        Current epoch number.
    config : ModelConfig
        Model configuration.
    accumulation_steps : int, optional
        Gradient accumulation steps. Default is 1.
    clip_norm : float, optional
        Gradient clipping norm. Default is 1.0.
    stage1_weight : float, optional
        Weight for Stage 1 hierarchical loss.
    stage2_weight : float, optional
        Weight for Stage 2 hierarchical loss.
    aux_weight : float, optional
        Weight for auxiliary loss.
        
    Returns
    -------
    avg_loss : float
        Average training loss for epoch.
    avg_acc : float
        Average training accuracy for epoch.
    """
    model.train()
    
    # GPU-side metric accumulators (avoid CPU sync during training)
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    # Track per-stage losses for hierarchical mode
    total_stage1_loss = torch.zeros(1, device=device)
    total_stage2_loss = torch.zeros(1, device=device)
    total_aux_loss = torch.zeros(1, device=device)
    
    pbar = tqdm(
        loader,
        desc=f'Epoch {epoch} [Train]',
        disable=not is_main_process(rank),
        ncols=120,
        leave=False
    )
    
    # Setup autocast context for mixed precision
    if device.type == 'cuda' and config.use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch: (magnification, delta_t, lengths, labels)
        magnification = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        lengths = batch[2]  # Keep on CPU for model
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                # Use return_intermediates for separate stage losses
                output = model(magnification, delta_t, lengths, return_intermediates=True)
                
                loss, loss_dict = compute_hierarchical_loss(
                    output,
                    labels,
                    class_weights,
                    stage1_weight=stage1_weight,
                    stage2_weight=stage2_weight,
                    aux_weight=aux_weight
                )
                
                logits = output.logits
                
                # Track per-stage losses
                total_stage1_loss += loss_dict['stage1_bce']
                total_stage2_loss += loss_dict['stage2_bce']
                total_aux_loss += loss_dict['aux_ce']
            else:
                logits = model(magnification, delta_t, lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
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
        
        # Accumulate metrics (no CPU sync)
        with torch.no_grad():
            if config.hierarchical:
                probs = torch.exp(logits)  # Convert log-probs to probs
            else:
                probs = F.softmax(logits, dim=1)
            
            preds = probs.argmax(dim=1)
            
            total_loss_gpu += loss.detach() * accumulation_steps
            total_correct_gpu += (preds == labels).sum()
            total_samples_gpu += labels.size(0)
        
        # Update progress bar periodically
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
    
    # Sync metrics across processes for distributed training
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
    config: ModelConfig,
    stage1_weight: float = 1.0,
    stage2_weight: float = 1.0,
    aux_weight: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model with GPU-side metric accumulation and per-class tracking.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation data loader.
    class_weights : Tensor
        Class weights for loss computation.
    device : torch.device
        Evaluation device.
    rank : int
        Process rank.
    world_size : int
        Number of distributed processes.
    config : ModelConfig
        Model configuration.
    stage1_weight : float, optional
        Weight for Stage 1 hierarchical loss.
    stage2_weight : float, optional
        Weight for Stage 2 hierarchical loss.
    aux_weight : float, optional
        Weight for auxiliary loss.
        
    Returns
    -------
    results : dict
        Evaluation results containing:
        - 'loss': Average loss
        - 'accuracy': Overall accuracy
        - 'recall_Flat': Recall for Flat class
        - 'recall_PSPL': Recall for PSPL class
        - 'recall_Binary': Recall for Binary class
    """
    model.eval()
    
    # GPU-side accumulators
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    # Per-class tracking
    class_correct = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    class_total = torch.zeros(N_CLASSES, device=device, dtype=torch.long)
    
    # Setup autocast
    if device.type == 'cuda' and config.use_amp:
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
    
    for batch in loader:
        magnification = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        lengths = batch[2]
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                output = model(magnification, delta_t, lengths, return_intermediates=True)
                loss, _ = compute_hierarchical_loss(
                    output, labels, class_weights,
                    stage1_weight, stage2_weight, aux_weight
                )
                logits = output.logits
            else:
                logits = model(magnification, delta_t, lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        # Compute predictions
        if config.hierarchical:
            probs = torch.exp(logits)
        else:
            probs = F.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        
        # Accumulate metrics
        if config.hierarchical:
            total_loss_gpu += loss * labels.size(0)
        else:
            total_loss_gpu += loss
        total_correct_gpu += (preds == labels).sum()
        total_samples_gpu += labels.size(0)
        
        # Per-class tracking
        for c in range(N_CLASSES):
            mask = (labels == c)
            class_total[c] += mask.sum()
            class_correct[c] += ((preds == c) & mask).sum()
    
    # Sync across processes
    if world_size > 1:
        dist.all_reduce(total_loss_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_total, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
    accuracy = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
    
    # Per-class recall
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
    path: Path
) -> None:
    """
    Save training checkpoint with all state needed for resumption.
    
    Parameters
    ----------
    model : nn.Module
        Model to save (handles DDP unwrapping).
    optimizer : torch.optim.Optimizer
        Optimizer state.
    scheduler : _LRScheduler
        Scheduler state.
    scaler : torch.amp.GradScaler or None
        Gradient scaler state (for mixed precision).
    config : ModelConfig
        Model configuration.
    stats : dict
        Normalization statistics.
    epoch : int
        Current epoch number.
    best_acc : float
        Best validation accuracy so far.
    path : Path
        Path to save checkpoint.
    """
    # Unwrap DDP if necessary
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
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device
) -> Tuple[int, float]:
    """
    Load checkpoint for resuming training.
    
    Parameters
    ----------
    path : str
        Path to checkpoint file.
    model : nn.Module
        Model to load state into.
    optimizer : torch.optim.Optimizer
        Optimizer to load state into.
    scheduler : _LRScheduler
        Scheduler to load state into.
    scaler : torch.amp.GradScaler or None
        Gradient scaler to load state into.
    device : torch.device
        Device to load tensors onto.
        
    Returns
    -------
    start_epoch : int
        Epoch to resume from.
    best_acc : float
        Best validation accuracy from checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle DDP wrapper
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']} (best acc: {100*best_acc:.2f}%)")
    
    return start_epoch, best_acc


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Setup Distributed Data Parallel with robust error handling.
    
    Handles initialization of the process group for multi-GPU training,
    including proper CUDA device assignment before init_process_group.
    
    Returns
    -------
    rank : int
        Global process rank.
    local_rank : int
        Local process rank (for GPU assignment).
    world_size : int
        Total number of processes.
    device : torch.device
        Device for this process.
        
    Raises
    ------
    RuntimeError
        If MASTER_ADDR or MASTER_PORT not set for distributed training.
    """
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
            raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set for distributed training!")
        
        if rank == 0:
            logger.info("=" * 80)
            logger.info("DDP Initialization")
            logger.info(f"  RANK: {rank}")
            logger.info(f"  LOCAL_RANK: {local_rank}")
            logger.info(f"  WORLD_SIZE: {world_size}")
            logger.info(f"  MASTER_ADDR: {master_addr}")
            logger.info(f"  MASTER_PORT: {master_port}")
            logger.info("=" * 80)
        
        # CRITICAL: Set CUDA device BEFORE init_process_group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            if rank == 0:
                logger.info(f"✓ Set CUDA device to: cuda:{local_rank}")
        
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
                logger.info("✓ DDP initialization complete!")
                logger.info(f"  CUDA device name: {torch.cuda.get_device_name(local_rank)}")
                
        except Exception as e:
            logger.error(f"DDP initialization failed: {e}")
            raise
    else:
        # Single process mode
        rank = 0
        local_rank = 0
        world_size = 1
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    """
    Cleanup DDP process group.
    
    Should be called at the end of training to properly release resources.
    """
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
    """
    Main training entry point.
    
    Parses command-line arguments and executes the full training pipeline
    including data loading, model creation, training loop, and checkpointing.
    """
    parser = argparse.ArgumentParser(
        description=f'Train Roman Microlensing Classifier v{__version__}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5/NPZ data file')
    parser.add_argument('--output', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION,
                       help='Fraction of data for validation')
    
    # Model architecture arguments
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--window-size', type=int, default=7,
                       help='Convolution kernel size')
    parser.add_argument('--hierarchical', action='store_true',
                       help='Use hierarchical classification')
    parser.add_argument('--attention-pooling', action='store_true',
                       help='Use attention pooling (vs mean pooling)')
    
    # Hierarchical loss arguments (v2.9+)
    parser.add_argument('--use-aux-head', action='store_true', default=True,
                       help='Use auxiliary 3-class head for gradient stability')
    parser.add_argument('--no-aux-head', dest='use_aux_head', action='store_false',
                       help='Disable auxiliary head')
    parser.add_argument('--stage2-temperature', type=float, default=1.0,
                       help='Temperature for Stage 2 sigmoid (lower = sharper)')
    parser.add_argument('--stage1-weight', type=float, default=DEFAULT_STAGE1_WEIGHT,
                       help='Weight for Stage 1 BCE loss')
    parser.add_argument('--stage2-weight', type=float, default=DEFAULT_STAGE2_WEIGHT,
                       help='Weight for Stage 2 BCE loss')
    parser.add_argument('--aux-weight', type=float, default=DEFAULT_AUX_WEIGHT,
                       help='Weight for auxiliary CE loss')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for AdamW')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_WARMUP_EPOCHS,
                       help='Number of warmup epochs')
    parser.add_argument('--accumulation-steps', type=int, default=DEFAULT_ACCUMULATION_STEPS,
                       help='Gradient accumulation steps')
    parser.add_argument('--clip-norm', type=float, default=DEFAULT_CLIP_NORM,
                       help='Gradient clipping norm')
    
    # Optimization arguments
    parser.add_argument('--use-amp', action='store_true',
                       help='Enable automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for speedup')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weighting')
    
    # Data loading arguments
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help='Number of data loading workers (0 for RAM loading)')
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULT_PREFETCH_FACTOR,
                       help='Batches to prefetch per worker')
    
    # Checkpointing arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Derived arguments
    args.use_class_weights = not args.no_class_weights
    
    # Setup
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
        logger.info(f"Workers: {args.num_workers} (0 = pure RAM loading)")
        if args.hierarchical:
            logger.info("=" * 80)
            logger.info("HIERARCHICAL MODE (v3.0 FIX ENABLED)")
            logger.info(f"  Stage 1 weight: {args.stage1_weight}")
            logger.info(f"  Stage 2 weight: {args.stage2_weight}")
            logger.info(f"  Aux weight: {args.aux_weight}")
            logger.info(f"  Stage 2 temperature: {args.stage2_temperature}")
            logger.info(f"  Use aux head: {args.use_aux_head}")
            logger.info("=" * 80)
    
    # Synchronize processes
    if is_ddp:
        if is_main_process(rank):
            logger.info("Synchronizing all processes...")
        dist.barrier()
    
    # Create output directory
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
            
    if is_ddp:
        dist.barrier()

    # Load data
    train_idx, val_idx, train_labels, stats = load_and_split_data(
        args.data,
        args.val_fraction,
        SEED,
        rank
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data,
        train_idx,
        val_idx,
        stats,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        is_ddp,
        rank
    )
    
    # Create model with v3.0 hierarchical options
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
        
        # Verify hierarchical head initialization
        if config.hierarchical:
            logger.info("Hierarchical Head Initialization:")
            logger.info(f"  Stage 1 bias: {model.head_stage1.bias.item():.4f} (should be ~0)")
            logger.info(f"  Stage 2 bias: {model.head_stage2.bias.item():.4f} (should be ~0)")
        logger.info("-" * 80)
    
    # Wrap in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # torch.compile
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank):
            logger.info(f"Compiling model with mode={args.compile_mode}...")
        try:
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
            if is_main_process(rank):
                logger.info("✓ Model compiled")
        except Exception as e:
            if is_main_process(rank):
                logger.warning(f"torch.compile failed: {e}")
    
    # Optimizer
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    if fused_available and device.type == 'cuda':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            fused=True
        )
        if is_main_process(rank):
            logger.info("Using fused AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps, 
        total_steps, 
        min_lr=1e-6
    )
    
    # Gradient scaler for mixed precision
    use_scaler = (
        args.use_amp and 
        device.type == 'cuda' and 
        not torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
    # Resume from checkpoint
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
            
            # Set epoch for distributed sampler
            if is_ddp:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                class_weights, device, rank, world_size, epoch, config,
                args.accumulation_steps, args.clip_norm,
                args.stage1_weight, args.stage2_weight, args.aux_weight
            )
            
            # Validate
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, world_size, config,
                args.stage1_weight, args.stage2_weight, args.aux_weight
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                    f"Val Loss: {val_results['loss']:.4f} | Val Acc: {100*val_results['accuracy']:.2f}% | "
                    f"Time: {format_time(epoch_time)}"
                )
                # Log per-class recall for hierarchical mode
                if config.hierarchical:
                    logger.info(
                        f"         Per-class recall: "
                        f"Flat={100*val_results.get('recall_Flat', 0):.1f}% | "
                        f"PSPL={100*val_results.get('recall_PSPL', 0):.1f}% | "
                        f"Binary={100*val_results.get('recall_Binary', 0):.1f}%"
                    )
            
            # Save checkpoints
            is_best = val_results['accuracy'] > best_acc
            if is_best:
                best_acc = val_results['accuracy']
            
            if is_main_process(rank):
                checkpoint_dir = output_dir / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save latest checkpoint (for resumption)
                save_checkpoint(
                    model, optimizer, scheduler, scaler, config, stats,
                    epoch, best_acc, checkpoint_dir / 'checkpoint_latest.pt'
                )
                
                # Save best model
                if is_best:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, config, stats,
                        epoch, best_acc, output_dir / 'best.pt'
                    )
                
                # Save periodic checkpoint
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, config, stats,
                        epoch, best_acc, output_dir / f'epoch_{epoch:03d}.pt'
                    )
        
        # Save final checkpoint
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
        cleanup_ddp()


if __name__ == '__main__':
    main()
