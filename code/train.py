#!/usr/bin/env python3
"""
Roman Microlensing Classifier Training Engine - ULTRA-FAST Edition
===================================================================

High-performance distributed training pipeline with RAM-based data loading for
classifying Nancy Grace Roman Space Telescope gravitational microlensing light
curves into Flat, PSPL (Point Source Point Lens), and Binary classes.

OPTIMIZATION STRATEGY
---------------------
This is an optimized version of the production training script (train.py v2.8.0)
with ONE critical change: dataset loading from disk → RAM for 20-50× speedup.

All architectural features, bug fixes, and training logic are preserved exactly.

KEY MODIFICATION: RAM LOADING
------------------------------
**RAMLensingDataset** (NEW):
    - Loads entire HDF5/NPZ dataset into RAM at initialization (~30 seconds)
    - Zero disk I/O during training (pure in-memory access)
    - Eliminates HDF5 global lock contention
    - Eliminates network filesystem latency
    - Supports both HDF5 (.h5, .hdf5) and NPZ (.npz) formats with auto-detection
    - Memory requirement: ~29 GB per GPU process, ~116 GB per 4-GPU node
    - A100 nodes (256-512 GB RAM): 25-50% utilization (plenty of headroom)

CRITICAL FIX v2.9.0 - HIERARCHICAL COLLAPSE FIX
-------------------------------------------------
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

PRESERVED FEATURES (from train.py v2.8.0)
------------------------------------------
All critical bug fixes and optimizations are preserved.

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 2.9.0
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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

__version__ = "2.9.0"

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def _configure_environment() -> None:
    """Ultra-optimized environment configuration for distributed training."""
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

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from model import ModelConfig, RomanMicrolensingClassifier, HierarchicalOutput

# =============================================================================
# CONSTANTS
# =============================================================================

SEED = 42
EPS = 1e-8
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 50
DEFAULT_NUM_WORKERS = 0  # Changed: No workers needed with RAM loading!
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_ACCUMULATION_STEPS = 1
DEFAULT_CLIP_NORM = 1.0
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_VAL_FRACTION = 0.1

# v2.9: Hierarchical loss weights
DEFAULT_STAGE1_WEIGHT = 1.0
DEFAULT_STAGE2_WEIGHT = 1.0
DEFAULT_AUX_WEIGHT = 0.5

PROGRESS_UPDATE_FREQ = 50
DDP_INIT_TIMEOUT_MINUTES = 10
DDP_BARRIER_TIMEOUT_SECONDS = 300

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
# UTILITIES
# =============================================================================

def is_main_process(rank: int) -> bool:
    return rank == 0 or rank == -1

def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_cuda() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# =============================================================================
# ULTRA-FAST RAM DATASET (NEW)
# =============================================================================

class RAMLensingDataset(Dataset):
    """
    Load entire dataset into RAM for ultra-fast training.
    
    SPEED: 0.2-0.5 s/batch (vs 2-12 s/batch on disk)
    MEMORY: ~7 GB per GPU process
    
    Supports both HDF5 and NPZ formats with automatic detection.
    Applies sequence compaction (S0-NEW-1 fix from train.py).
    """
    
    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        flux_mean: float,
        flux_std: float,
        delta_t_mean: float,
        delta_t_std: float,
        rank: int = 0
    ) -> None:
        self.indices = indices
        self.rank = rank
        
        # Pre-compute scale factors (FIXED: using std instead of IQR)
        self._flux_scale = 1.0 / (flux_std + EPS)
        self._dt_scale = 1.0 / (delta_t_std + EPS)
        self.flux_mean = flux_mean
        self.delta_t_mean = delta_t_mean
        
        if rank == 0:
            print(f"🚀 ULTRA-FAST MODE: Loading dataset into RAM...")
            print(f"   Source: {file_path}")
        
        # Detect file format
        file_path = Path(file_path)
        
        if file_path.suffix == '.npz':
            # Load from NPZ
            if rank == 0:
                print("   Format: NPZ")
            data = np.load(str(file_path))
            self.flux = data['flux']
            self.delta_t = data['delta_t']
            self.labels = data['labels']
            
        elif file_path.suffix in ['.h5', '.hdf5']:
            # Load from HDF5
            if rank == 0:
                print("   Format: HDF5 (loading into RAM)")
            with h5py.File(str(file_path), 'r') as f:
                if rank == 0:
                    print("   Loading flux...", end='', flush=True)
                self.flux = f['flux'][:]
                if rank == 0:
                    print(f" ✓ ({self.flux.nbytes/1e9:.2f} GB)")
                
                if rank == 0:
                    print("   Loading delta_t...", end='', flush=True)
                self.delta_t = f['delta_t'][:]
                if rank == 0:
                    print(f" ✓ ({self.delta_t.nbytes/1e9:.2f} GB)")
                
                if rank == 0:
                    print("   Loading labels...", end='', flush=True)
                self.labels = f['labels'][:]
                if rank == 0:
                    print(f" ✓ ({self.labels.nbytes/1e9:.2f} GB)")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Apply index subset
        self.flux = self.flux[self.indices]
        self.delta_t = self.delta_t[self.indices]
        self.labels = self.labels[self.indices]
        
        if rank == 0:
            total_mem = (self.flux.nbytes + self.delta_t.nbytes + 
                        self.labels.nbytes) / 1e9
            print(f"   ✓ Dataset in RAM: {total_mem:.2f} GB")
            print(f"   ✓ Samples: {len(self.indices):,}")
            print(f"   ✓ Ready for ULTRA-FAST training!")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        """
        Get sample with sequence compaction (S0-NEW-1 fix).
        
        Returns
        -------
        flux : Tensor
            Normalized flux, COMPACTED [max_seq_len]
        delta_t : Tensor
            Normalized delta_t, COMPACTED [max_seq_len]
        length : int
            Number of valid observations in contiguous prefix
        label : int
            Class label
        """
        # Load from RAM (instant!)
        flux_raw = self.flux[idx].copy()
        delta_t_raw = self.delta_t[idx].copy()
        label = int(self.labels[idx])
        
        # S0-NEW-1: SEQUENCE COMPACTION
        valid_mask = flux_raw != 0.0
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            valid_count = 1
            flux_raw[0] = self.flux_mean  # FIXED: use mean instead of median
            delta_t_raw[0] = 0.0
            valid_mask[0] = True
        
        # Compact: move valid observations to contiguous prefix [0, valid_count)
        flux_compacted = np.zeros_like(flux_raw)
        delta_t_compacted = np.zeros_like(delta_t_raw)
        
        flux_compacted[:valid_count] = flux_raw[valid_mask]
        delta_t_compacted[:valid_count] = delta_t_raw[valid_mask]
        
        # Normalize (FIXED: use mean instead of median)
        flux_norm = (flux_compacted - self.flux_mean) * self._flux_scale
        dt_norm = (delta_t_compacted - self.delta_t_mean) * self._dt_scale
        
        # Convert to tensors
        flux = torch.from_numpy(flux_norm).float()
        delta_t = torch.from_numpy(dt_norm).float()
        length = int(valid_count)
        
        return flux, delta_t, length, label


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_robust_statistics(
    file_path: str,
    rank: int = 0
) -> Dict[str, float]:
    """Compute robust statistics for SEPARATE flux and delta_t normalization."""
    
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        flux_data = data['flux']
        delta_t_data = data['delta_t']
    else:
        with h5py.File(str(file_path), 'r') as f:
            flux_data = f['flux'][:]
            delta_t_data = f['delta_t'][:]
    
    flux_valid = flux_data[flux_data != 0.0]
    delta_t_valid = delta_t_data[delta_t_data != 0.0]
    
    # FIXED: Use mean/std instead of median/IQR (IQR=0 for delta_t!)
    flux_mean = float(np.mean(flux_valid))
    flux_std = float(np.std(flux_valid))
    
    delta_t_mean = float(np.mean(delta_t_valid))
    delta_t_std = float(np.std(delta_t_valid))
    
    if is_main_process(rank):
        logger.info("Normalization Statistics (SEPARATE for flux and delta_t):")
        logger.info(f"  Flux    - Mean: {flux_mean:.4f}, Std: {flux_std:.4f}")
        logger.info(f"  Delta_t - Mean: {delta_t_mean:.6f}, Std: {delta_t_std:.6f}")
    
    return {
        'flux_mean': flux_mean,
        'flux_std': flux_std,
        'delta_t_mean': delta_t_mean,
        'delta_t_std': delta_t_std
    }


def load_and_split_data(
    file_path: str,
    val_fraction: float,
    seed: int,
    rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Load data, compute stats, and create train/val split."""
    
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(str(file_path))
        total_samples = len(data['labels'])
        all_labels = data['labels']
    else:
        with h5py.File(str(file_path), 'r') as f:
            total_samples = len(f['flux'])
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
            logger.info(f"  {CLASS_NAMES[cls_idx]}: {count} ({100*count/len(train_labels):.1f}%)")
    
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
    """Create dataloaders with RAM-loaded datasets."""
    
    # Create datasets with RAM loading
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
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
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
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + EPS)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =============================================================================
# v2.9 FIX: HIERARCHICAL LOSS COMPUTATION
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
    
    This is the CRITICAL FIX for the hierarchical collapse problem.
    
    Instead of using NLLLoss on combined probabilities (which doesn't
    provide direct supervision to Stage 2), we use:
    - Stage 1 BCE: trains flat_head directly on Flat vs Non-Flat
    - Stage 2 BCE: trains pspl_head directly on PSPL vs Binary (non-flat only)
    - Auxiliary CE: direct 3-class supervision for gradient stability
    
    Parameters
    ----------
    output : HierarchicalOutput
        Output from model with return_intermediates=True.
    labels : Tensor
        Class labels [batch]. 0=Flat, 1=PSPL, 2=Binary.
    class_weights : Tensor
        Class weights [3] for weighting.
    stage1_weight : float
        Weight for Stage 1 loss.
    stage2_weight : float
        Weight for Stage 2 loss.
    aux_weight : float
        Weight for auxiliary 3-class loss.
        
    Returns
    -------
    total_loss : Tensor
        Combined loss for backpropagation.
    loss_dict : dict
        Individual loss components for logging.
    """
    device = labels.device
    B = labels.size(0)
    
    # Create binary targets for hierarchical stages
    # Stage 1: is_deviation (non-flat)
    # label=0 → is_deviation=0 (Flat)
    # label=1 → is_deviation=1 (PSPL = deviation)
    # label=2 → is_deviation=1 (Binary = deviation)
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
    # Weight: Flat samples get weight[0], Non-Flat get average of weight[1] and weight[2]
    stage1_pos_weight = (class_weights[1] + class_weights[2]) / 2.0 / class_weights[0]
    stage1_bce = F.binary_cross_entropy_with_logits(
        output.stage1_logit,  # [B, 1] raw logit
        is_deviation,  # [B, 1] target
        pos_weight=stage1_pos_weight.unsqueeze(0).unsqueeze(0),
        reduction='mean'
    )
    
    # =========================================================================
    # Stage 2 Loss: PSPL vs Binary (BCE on non-flat samples only)
    # =========================================================================
    if n_non_flat > 0:
        # Extract non-flat samples
        stage2_logit_nonflat = output.stage2_logit[non_flat_mask]  # [n_non_flat, 1]
        is_pspl_nonflat = is_pspl[non_flat_mask]  # [n_non_flat, 1]
        
        # Weight: PSPL samples get weight[1], Binary get weight[2]
        # Among non-flat, pos_weight balances PSPL (positive) vs Binary (negative)
        stage2_pos_weight = class_weights[1] / class_weights[2]
        stage2_bce = F.binary_cross_entropy_with_logits(
            stage2_logit_nonflat,  # raw logit
            is_pspl_nonflat,  # target
            pos_weight=stage2_pos_weight.unsqueeze(0).unsqueeze(0),
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
        'stage1_bce': stage1_bce.item(),
        'stage2_bce': stage2_bce.item() if n_non_flat > 0 else 0.0,
        'aux_ce': aux_ce.item() if output.aux_logits is not None else 0.0,
        'total': total_loss.item(),
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
    """Training epoch with GPU-side metric accumulation."""
    model.train()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    # v2.9: Track per-stage losses
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
    
    autocast_ctx = torch.amp.autocast('cuda', enabled=config.use_amp) if device.type == 'cuda' else nullcontext()
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        flux = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        lengths = batch[2]
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                # v2.9 FIX: Use return_intermediates for separate stage losses
                output = model(flux, delta_t, lengths, return_intermediates=True)
                
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
                logits = model(flux, delta_t, lengths)
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
    """Evaluation with GPU-side metric accumulation and per-class tracking."""
    model.eval()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    # v2.9: Track per-class accuracy
    class_correct = torch.zeros(3, device=device, dtype=torch.long)
    class_total = torch.zeros(3, device=device, dtype=torch.long)
    
    autocast_ctx = torch.amp.autocast('cuda', enabled=config.use_amp) if device.type == 'cuda' else nullcontext()
    
    for batch in loader:
        flux = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        lengths = batch[2]
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            if config.hierarchical:
                output = model(flux, delta_t, lengths, return_intermediates=True)
                loss, _ = compute_hierarchical_loss(
                    output, labels, class_weights,
                    stage1_weight, stage2_weight, aux_weight
                )
                logits = output.logits
            else:
                logits = model(flux, delta_t, lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        if config.hierarchical:
            probs = torch.exp(logits)
        else:
            probs = F.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        
        total_loss_gpu += loss if not config.hierarchical else loss * labels.size(0)
        total_correct_gpu += (preds == labels).sum()
        total_samples_gpu += labels.size(0)
        
        # Per-class tracking
        for c in range(3):
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
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device
) -> Tuple[int, float]:
    """Load checkpoint for resuming training."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']} (best acc: {100*best_acc:.2f}%)")
    
    return start_epoch, best_acc


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """Setup DDP with robust error handling."""
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
        
        # CRITICAL FIX: Set CUDA device BEFORE init_process_group
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
        rank = 0
        local_rank = 0
        world_size = 1
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    """Cleanup DDP."""
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
    parser = argparse.ArgumentParser(description='Train Roman Microlensing Classifier - v2.9 HIERARCHICAL FIX')
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5/NPZ data file')
    parser.add_argument('--output', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION)
    
    # Model
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--hierarchical', action='store_true')
    parser.add_argument('--attention-pooling', action='store_true')
    
    # v2.9: Hierarchical loss options
    parser.add_argument('--use-aux-head', action='store_true', default=True,
                       help='Use auxiliary 3-class head (default: True)')
    parser.add_argument('--no-aux-head', dest='use_aux_head', action='store_false')
    parser.add_argument('--stage2-temperature', type=float, default=1.0,
                       help='Temperature for Stage 2 sigmoid (lower = sharper)')
    parser.add_argument('--stage1-weight', type=float, default=DEFAULT_STAGE1_WEIGHT,
                       help='Weight for Stage 1 BCE loss')
    parser.add_argument('--stage2-weight', type=float, default=DEFAULT_STAGE2_WEIGHT,
                       help='Weight for Stage 2 BCE loss')
    parser.add_argument('--aux-weight', type=float, default=DEFAULT_AUX_WEIGHT,
                       help='Weight for auxiliary CE loss')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument('--accumulation-steps', type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument('--clip-norm', type=float, default=DEFAULT_CLIP_NORM)
    
    # Optimization
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead')
    parser.add_argument('--no-class-weights', action='store_true')
    
    # Data loading
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULT_PREFETCH_FACTOR)
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    
    args = parser.parse_args()
    
    args.use_class_weights = not args.no_class_weights
    
    # Setup
    set_seed(SEED)
    configure_cuda()
    rank, local_rank, world_size, device = setup_ddp()
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
            logger.info("HIERARCHICAL MODE (v2.9 FIX ENABLED)")
            logger.info(f"  Stage 1 weight: {args.stage1_weight}")
            logger.info(f"  Stage 2 weight: {args.stage2_weight}")
            logger.info(f"  Aux weight: {args.aux_weight}")
            logger.info(f"  Stage 2 temperature: {args.stage2_temperature}")
            logger.info(f"  Use aux head: {args.use_aux_head}")
            logger.info("=" * 80)
    
    if is_ddp:
        if is_main_process(rank):
            logger.info("Synchronizing all processes...")
        dist.barrier()
    
    # Create output directory
    output_dir = Path(args.output)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create model with v2.9 hierarchical options
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
        
        # v2.9: Verify hierarchical head initialization
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                      weight_decay=args.weight_decay, fused=True)
        if is_main_process(rank):
            logger.info("Using fused AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # Gradient scaler
    use_scaler = args.use_amp and device.type == 'cuda' and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
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
            
            # Log
            if is_main_process(rank):
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                    f"Val Loss: {val_results['loss']:.4f} | Val Acc: {100*val_results['accuracy']:.2f}% | "
                    f"Time: {format_time(epoch_time)}"
                )
                # v2.9: Log per-class recall
                if config.hierarchical:
                    logger.info(
                        f"         Per-class recall: "
                        f"Flat={100*val_results.get('recall_Flat', 0):.1f}% | "
                        f"PSPL={100*val_results.get('recall_PSPL', 0):.1f}% | "
                        f"Binary={100*val_results.get('recall_Binary', 0):.1f}%"
                    )
            
            # Save
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
        cleanup_ddp()


if __name__ == '__main__':
    main()
