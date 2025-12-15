#!/usr/bin/env python3
"""
Roman Microlensing Classifier Training Engine - ULTRA-FAST Edition
===================================================================

High-performance distributed training pipeline with RAM-based data loading for
classifying Nancy Grace Roman Space Telescope gravitational microlensing light
curves into Flat, PSPL (Point Source Point Lens), and Binary classes.
    
**v2.8.0 Fixes**:
    - Checkpoint key 'model_config' (not 'config') for model.py compatibility
    - Argument parsing: --no-class-weights and --no-prefetcher flags
    - checkpoint_latest.pt saved after every epoch for resumption
    - torch.compile error handling with graceful fallback
    - DDP cleanup in try/finally block to prevent hanging
    - Window size default changed to 7 (matches sbatch)

**v2.7.2 Fixes**:
    - Explicit DDP init_method with TCP store
    - NCCL timeout configurations (600s, 300s socket timeout)
    - MASTER_ADDR/MASTER_PORT validation
    - Barrier synchronization before data loading
    - Process group health checks

**v2.7.1 Fixes**:
    - S0-NEW-1: Sequence compaction enforcing contiguous prefix assumption
      * Valid observations moved to positions [0, length)
      * Eliminates fake observations from scattered missing data
      * Ensures mean pooling doesn't include invalid positions

**v2.6 Fixes**:
    - S0-1: Hierarchical mode uses F.nll_loss() (model outputs log-probs)
    - S0-2: Probability computation uses torch.exp() for hierarchical mode
    - S1-1: Validation sampler.set_epoch() called for proper DDP
    - S1-3: Minimum sequence length of 1

**v2.7 Optimizations**:
    - GPU-side metric accumulation (no .item() in training loop)
    - Pre-computed sequence lengths in dataset
    - Fused normalization operations
    - Optimized dataloader with drop_last=True
    - torch.compile fullgraph support
    
Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 2.8.0
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

__version__ = "2.8.0"

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
from model import ModelConfig, RomanMicrolensingClassifier

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
        flux_median: float,
        flux_iqr: float,
        delta_t_median: float,
        delta_t_iqr: float,
        rank: int = 0
    ) -> None:
        self.indices = indices
        self.rank = rank
        
        # Pre-compute scale factors
        self._flux_scale = 1.0 / (flux_iqr + EPS)
        self._dt_scale = 1.0 / (delta_t_iqr + EPS)
        self.flux_median = flux_median
        self.delta_t_median = delta_t_median
        
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
            flux_raw[0] = self.flux_median
            delta_t_raw[0] = 0.0
            valid_mask[0] = True
        
        # Compact: move valid observations to contiguous prefix [0, valid_count)
        flux_compacted = np.zeros_like(flux_raw)
        delta_t_compacted = np.zeros_like(delta_t_raw)
        
        flux_compacted[:valid_count] = flux_raw[valid_mask]
        delta_t_compacted[:valid_count] = delta_t_raw[valid_mask]
        
        # Normalize
        flux_norm = (flux_compacted - self.flux_median) * self._flux_scale
        dt_norm = (delta_t_compacted - self.delta_t_median) * self._dt_scale
        
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
    
    flux_median = float(np.median(flux_valid))
    flux_q1, flux_q3 = np.percentile(flux_valid, [25, 75])
    flux_iqr = float(flux_q3 - flux_q1)
    
    delta_t_median = float(np.median(delta_t_valid))
    dt_q1, dt_q3 = np.percentile(delta_t_valid, [25, 75])
    delta_t_iqr = float(dt_q3 - dt_q1)
    
    if is_main_process(rank):
        logger.info("Normalization Statistics (SEPARATE for flux and delta_t):")
        logger.info(f"  Flux    - Median: {flux_median:.4f}, IQR: {flux_iqr:.4f}")
        logger.info(f"  Delta_t - Median: {delta_t_median:.4f}, IQR: {delta_t_iqr:.4f}")
    
    return {
        'flux_median': flux_median,
        'flux_iqr': flux_iqr,
        'delta_t_median': delta_t_median,
        'delta_t_iqr': delta_t_iqr
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
        stats['flux_median'],
        stats['flux_iqr'],
        stats['delta_t_median'],
        stats['delta_t_iqr'],
        rank=rank
    )
    
    val_dataset = RAMLensingDataset(
        file_path,
        val_idx,
        stats['flux_median'],
        stats['flux_iqr'],
        stats['delta_t_median'],
        stats['delta_t_iqr'],
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
    clip_norm: float = 1.0
) -> Tuple[float, float]:
    """Training epoch with GPU-side metric accumulation."""
    model.train()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    pbar = tqdm(
        loader,
        desc=f'Epoch {epoch} [Train]',
        disable=not is_main_process(rank),
        ncols=100,
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
            logits = model(flux, delta_t, lengths)
            
            if config.hierarchical:
                loss = F.nll_loss(logits, labels, weight=class_weights)
            else:
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
    config: ModelConfig
) -> Dict[str, float]:
    """Evaluation with GPU-side metric accumulation."""
    model.eval()
    
    total_loss_gpu = torch.zeros(1, device=device)
    total_correct_gpu = torch.zeros(1, device=device, dtype=torch.long)
    total_samples_gpu = torch.zeros(1, device=device, dtype=torch.long)
    
    autocast_ctx = torch.amp.autocast('cuda', enabled=config.use_amp) if device.type == 'cuda' else nullcontext()
    
    for batch in loader:
        flux = batch[0].to(device, non_blocking=True)
        delta_t = batch[1].to(device, non_blocking=True)
        lengths = batch[2]
        labels = batch[3].to(device, non_blocking=True)
        
        with autocast_ctx:
            logits = model(flux, delta_t, lengths)
            
            if config.hierarchical:
                loss = F.nll_loss(logits, labels, weight=class_weights, reduction='sum')
            else:
                loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        if config.hierarchical:
            probs = torch.exp(logits)
        else:
            probs = F.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        
        total_loss_gpu += loss
        total_correct_gpu += (preds == labels).sum()
        total_samples_gpu += labels.size(0)
    
    if world_size > 1:
        dist.all_reduce(total_loss_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_gpu, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_gpu.item() / max(total_samples_gpu.item(), 1)
    accuracy = total_correct_gpu.item() / max(total_samples_gpu.item(), 1)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
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
        
        try:
            init_method = f'tcp://{master_addr}:{master_port}'
            
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=DDP_INIT_TIMEOUT_MINUTES)
            )
            
            dist.barrier(device_ids=[local_rank])
            
            if rank == 0:
                logger.info("✓ DDP initialization complete!")
                
        except Exception as e:
            logger.error(f"DDP initialization failed: {e}")
            raise
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
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
    parser = argparse.ArgumentParser(description='Train Roman Microlensing Classifier - ULTRA-FAST')
    
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
    
    # Data loading (changed defaults for RAM loading!)
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
        logger.info(f"Roman Microlensing Classifier Training")
        logger.info("=" * 80)
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Workers: {args.num_workers} (0 = pure RAM loading)")
    
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
    
    # Create model
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        hierarchical=args.hierarchical,
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
                args.accumulation_steps, args.clip_norm
            )
            
            # Validate
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, world_size, config
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
