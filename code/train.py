#!/usr/bin/env python3
"""
HPC Distributed Training Script (

Features:
- DDP (Distributed Data Parallel) on 40+ GPUs.
- AMP (Automatic Mixed Precision) for 2x speedup.
- RESUME capability (checkpoints).
- Advanced Metrics (F1, Precision, Recall) via distributed reduction.
- Robust Data Normalization (Median/IQR) & NaN handling.

Author: Kunal Bhatia
Version: 1.0
Date: December 2025
"""

import os
import sys
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Dict, Optional, List

# --- Import Refactored Model ---
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    # UPDATED: Importing from the generic 'model.py'
    from model import CausalHybridModel, CausalConfig
except ImportError:
    if int(os.environ.get('RANK', 0)) == 0:
        print("CRITICAL: 'model.py' not found. Please ensure the model file is present.")
    sys.exit(1)

# =============================================================================
# HPC CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
ACCUMULATE_STEPS = 1
PREFETCH_FACTOR = 4
NUM_WORKERS = 8
SEED = 42

# =============================================================================
# DISTRIBUTED UTILS
# =============================================================================

def setup_distributed() -> Tuple[int, int, int, bool]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return rank, local_rank, world_size, True
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

def get_logger(rank: int, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("hpc_trainer")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if logger.hasHandlers(): logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | Rank %(process)d | %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if rank == 0:
        fh = logging.FileHandler(output_dir / "training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# =============================================================================
# ROBUST DATASET & PROCESSING
# =============================================================================

class MicrolensingDataset(Dataset):
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, 
                 stats: Dict[str, float] = None):
        # 1. NaN Handling: Replace NaNs with 0 (assuming pre-padding)
        flux = np.nan_to_num(flux, nan=0.0)
        delta_t = np.nan_to_num(delta_t, nan=0.0)

        # 2. Normalization (Robust Scaler logic)
        # (X - Median) / IQR
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr']
            if iqr < 1e-6: iqr = 1.0 
            flux = (flux - median) / iqr

        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # Calculate valid lengths (assuming 0 is padding)
        self.lengths = (self.flux != 0.0).sum(dim=1).long().clamp(min=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.flux[idx], self.delta_t[idx], self.lengths[idx], self.labels[idx]

def load_and_preprocess_data(path: str, rank: int, logger: logging.Logger):
    """Loads NPZ, calculates stats on TRAIN split only to avoid leakage."""
    if rank == 0: logger.info(f"Loading raw data from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
        flux = data['flux'] if 'flux' in data else data['X']
        labels = data['labels'] if 'labels' in data else data['y']
        dt = data['delta_t'] if 'delta_t' in data else np.zeros_like(flux)
        
        # Split first
        flux_tr, flux_val, dt_tr, dt_val, y_tr, y_val = train_test_split(
            flux, dt, labels, test_size=0.1, stratify=labels, random_state=SEED
        )
        
        # Calculate Stats on Train ONLY (Data Safety)
        flat_flux = flux_tr.flatten()
        valid_mask = flat_flux != 0
        valid_flux = flat_flux[valid_mask]
        
        median = np.median(valid_flux)
        q75, q25 = np.percentile(valid_flux, [75 ,25])
        iqr = q75 - q25
        
        stats = {'median': median, 'iqr': iqr}
        
        if rank == 0:
            logger.info(f"Data Stats (Train): Median={median:.4f}, IQR={iqr:.4f}")
            
        return (flux_tr, dt_tr, y_tr), (flux_val, dt_val, y_val), stats

    except Exception as e:
        if rank == 0: logger.error(f"Data load failed: {e}")
        sys.exit(1)

# =============================================================================
# METRICS & CHECKPOINTING
# =============================================================================

class MetricTracker:
    """Calculates F1/Precision/Recall in a DDP-safe way."""
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.n_classes, device=self.device)
        self.fp = torch.zeros(self.n_classes, device=self.device)
        self.fn = torch.zeros(self.n_classes, device=self.device)

    def update(self, preds, targets):
        for c in range(self.n_classes):
            self.tp[c] += ((preds == c) & (targets == c)).sum()
            self.fp[c] += ((preds == c) & (targets != c)).sum()
            self.fn[c] += ((preds != c) & (targets == c)).sum()

    def compute(self):
        # DDP Reduction: Sum counts across all GPUs
        if dist.is_initialized():
            dist.all_reduce(self.tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fn, op=dist.ReduceOp.SUM)

        # Calculate per-class metrics
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "accuracy": self.tp.sum() / (self.tp.sum() + self.fp.sum() + 1e-8),
            "macro_f1": f1.mean(),
            "macro_prec": precision.mean(),
            "macro_rec": recall.mean()
        }

def save_checkpoint(state, is_best, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "last.pt")
    if is_best:
        torch.save(state, output_dir / "best.pt")

# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, device, rank, epoch, logger):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    
    iterator = tqdm(loader, desc=f"Train Ep {epoch}", disable=not is_main_process(rank))
    optimizer.zero_grad(set_to_none=True)

    for step, (flux, dt, lengths, labels) in enumerate(iterator):
        flux, dt = flux.to(device, non_blocking=True), dt.to(device, non_blocking=True)
        lengths, labels = lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with autocast():
            out = model(flux, dt, lengths=lengths, return_all_timesteps=True)
            
            B = flux.size(0)
            last_idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, out['logits'].size(2))
            final_logits = out['logits'].gather(1, last_idx).squeeze(1)
            
            loss = F.cross_entropy(final_logits, labels) / ACCUMULATE_STEPS

        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUMULATE_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * ACCUMULATE_STEPS
            
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
    avg_loss = total_loss / dist.get_world_size() / len(loader)
    return avg_loss.item()

@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    tracker = MetricTracker(n_classes, device)
    
    for flux, dt, lengths, labels in loader:
        flux, dt = flux.to(device, non_blocking=True), dt.to(device, non_blocking=True)
        lengths, labels = lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with autocast():
            out = model(flux, dt, lengths=lengths, return_all_timesteps=False)
            preds = out['probs'].argmax(dim=1)
            
        tracker.update(preds, labels)
        
    return tracker.compute()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint (last.pt)")
    args = parser.parse_args()
    
    # 1. Setup DDP
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # 2. Logging & Output
    current_script_dir = Path(__file__).resolve().parent
    output_dir = current_script_dir.parent / 'results' / args.experiment_name
    
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        
    if is_ddp: dist.barrier()
    logger = get_logger(rank, output_dir)
    
    if rank == 0:
        logger.info(f"🚀 Starting DDP Training | GPUs: {world_size} | AMP: True")
        
    # 3. Data Loading
    (tr_data, val_data, stats) = load_and_preprocess_data(args.data, rank, logger)
    train_ds = MicrolensingDataset(*tr_data, stats=stats)
    val_ds = MicrolensingDataset(*val_data, stats=stats)
    
    train_sampler = DistributedSampler(train_ds) if is_ddp else None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=PREFETCH_FACTOR, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True
    )

    # 4. Model Setup
    n_classes = len(torch.unique(train_ds.labels))
    config = CausalConfig(
        d_model=args.d_model, 
        n_transformer_layers=args.layers,
        n_conv_layers=4,
        n_classes=n_classes
    )
    model = CausalHybridModel(config).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 5. Resume Logic
    start_epoch = 1
    best_f1 = 0.0
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0: logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        if is_ddp and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        
        if rank == 0: logger.info(f"✅ Loaded. Starting at Epoch {start_epoch}")

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp: train_sampler.set_epoch(epoch)
        
        t_loss = train_epoch(model, train_loader, optimizer, scaler, device, rank, epoch, logger)
        metrics = evaluate(model, val_loader, device, n_classes)
        
        scheduler.step(metrics['macro_f1'])
        
        if is_main_process(rank):
            logger.info(
                f"Ep {epoch}/{args.epochs} | "
                f"Loss: {t_loss:.4f} | "
                f"F1: {metrics['macro_f1']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            is_best = metrics['macro_f1'] > best_f1
            if is_best: best_f1 = metrics['macro_f1']
            
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_f1': best_f1,
                'config': config
            }, is_best, output_dir)

    cleanup_distributed()

if __name__ == '__main__':
    main()
