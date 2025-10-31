#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_distributed.py - Multi-node multi-GPU training with DistributedDataParallel

Usage:
  Single node (8 GPUs):
    torchrun --nproc_per_node=8 train_distributed.py --data data.npz
  
  Multi-node (2 nodes × 8 GPUs = 16 GPUs):
    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
             --master_addr=node01 --master_port=29500 train_distributed.py --data data.npz
    
    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
             --master_addr=node01 --master_port=29500 train_distributed.py --data data.npz

Author: Kunal Bhatia
Date: October 2025
"""

import argparse
import os
import random
import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code.model import TimeDistributedCNN
from code.utils import load_npz_dataset, two_stage_normalize, save_scalers
import code.config as CFG


def setup_distributed():
    """
    Initialize distributed training
    
    Environment variables set by torchrun:
    - RANK: Global rank of this process
    - LOCAL_RANK: Rank within this node
    - WORLD_SIZE: Total number of processes
    - MASTER_ADDR: Address of rank 0
    - MASTER_PORT: Port for communication
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        print("Not using distributed mode")
        return -1, 0, 1
    
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU
        init_method='env://',
    )
    
    # Synchronize
    dist.barrier()
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int):
    """Set seed for reproducibility (different per rank)"""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        X = X.copy()
        X[X == CFG.PAD_VALUE] = 0.0
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, rank, grad_clip=None, scaler=None):
    """Training loop for one epoch"""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc="Training", leave=False)
    else:
        pbar = loader
    
    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with autocast():
                outputs = model(xb)  # [B, L, 2]
                
                # Per-timestep loss
                B, L, C = outputs.shape
                yb_repeated = yb.unsqueeze(1).expand(B, L)
                outputs_flat = outputs.reshape(B * L, C)
                yb_flat = yb_repeated.reshape(B * L)
                loss = crit(outputs_flat, yb_flat)
                
                logits = outputs.mean(dim=1)
            
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(xb)
            
            B, L, C = outputs.shape
            yb_repeated = yb.unsqueeze(1).expand(B, L)
            outputs_flat = outputs.reshape(B * L, C)
            yb_flat = yb_repeated.reshape(B * L)
            loss = crit(outputs_flat, yb_flat)
            
            logits = outputs.mean(dim=1)
            
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb) * bs
        n += bs
        
        if rank == 0 and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy(logits.detach(), yb):.3f}'})
    
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, rank):
    """Evaluation loop"""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    if rank == 0:
        pbar = tqdm(loader, desc="Validating", leave=False)
    else:
        pbar = loader
    
    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        
        outputs = model(xb)
        
        B, L, C = outputs.shape
        yb_repeated = yb.unsqueeze(1).expand(B, L)
        outputs_flat = outputs.reshape(B * L, C)
        yb_flat = yb_repeated.reshape(B * L)
        loss = crit(outputs_flat, yb_flat)
        
        logits = outputs.mean(dim=1)
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    
    return total_loss / n, total_acc / n


def reduce_metrics(loss, acc, world_size):
    """Average metrics across all processes"""
    metrics = torch.tensor([loss, acc], device='cuda')
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= world_size
    return metrics[0].item(), metrics[1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="baseline_distributed")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size PER GPU")
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=CFG.RANDOM_SEED)
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    
    # Set seed (different per rank for data augmentation diversity)
    set_seed(args.seed, rank)
    
    # Create output directory (only rank 0)
    if rank == 0 or rank == -1:
        from datetime import datetime
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(CFG.RESULTS_DIR) / f"{args.experiment_name}_{timestamp}"
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = output_dir / "best_model.pt"
        config_path = output_dir / "config.json"
        log_path = output_dir / "training.log"
        
        print("="*80)
        print("🚀 DISTRIBUTED TRAINING")
        print("="*80)
        print(f"\nWorld size: {world_size} processes")
        print(f"GPUs per node: {torch.cuda.device_count()}")
        print(f"Total GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"\n📁 Output: {output_dir}")
    
    # Synchronize
    if dist.is_initialized():
        dist.barrier()
    
    # Load data (all ranks load independently)
    if rank == 0:
        print(f"\nLoading dataset: {args.data}")
    
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X.shape[1]
    
    if rank == 0:
        print(f"Loaded X: {X.shape}, y: {y.shape}")
    
    # Split data
    n_total = len(X)
    n_train = int(n_total * CFG.TRAIN_SPLIT)
    n_val = int(n_total * CFG.VAL_SPLIT)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    if rank == 0:
        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Normalize (all ranks do this identically)
    if rank == 0:
        print("Applying two-stage normalization...")
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
        X_train, X_val, X_test, pad_value=CFG.PAD_VALUE
    )
    
    # Save scalers (only rank 0)
    if rank == 0:
        save_scalers(scaler_std, scaler_mm, output_dir)
    
    # Create datasets
    train_ds = NumpyDataset(X_train_scaled, y_train)
    val_ds = NumpyDataset(X_val_scaled, y_val)
    test_ds = NumpyDataset(X_test_scaled, y_test)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=world_size if world_size > 1 else 1,
        rank=rank if rank >= 0 else 0,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size if world_size > 1 else 1,
        rank=rank if rank >= 0 else 0,
        shuffle=False
    )
    
    # Create dataloaders (no shuffle - sampler handles it)
    num_workers = min(8, os.cpu_count() // max(1, world_size))
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True
    )
    
    # Create model
    if rank == 0:
        print("\nBuilding model...")
    
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    model = model.to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=CFG.LR_PATIENCE, factor=CFG.LR_FACTOR
    )
    
    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    if rank == 0:
        print(f"Mixed precision: {use_amp}")
        print(f"\nStarting training for {args.epochs} epochs...")
    
    # Training loop
    best_val_acc = -1.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, rank,
            grad_clip=CFG.GRAD_CLIP, scaler=scaler
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, rank)
        
        # Reduce metrics across all processes
        if world_size > 1:
            tr_loss, tr_acc = reduce_metrics(tr_loss, tr_acc, world_size)
            val_loss, val_acc = reduce_metrics(val_loss, val_acc, world_size)
        
        epoch_time = time.time() - epoch_start
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_acc)
        
        # Print and save (only rank 0)
        if rank == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"time {epoch_time:.1f}s"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    "model_state_dict": model_to_save.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, best_model_path)
                print(f"  ↳ saved best model (val_acc={val_acc:.4f})")
        
        # Synchronize
        if dist.is_initialized():
            dist.barrier()
    
    # Final evaluation (only rank 0)
    if rank == 0:
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)
        
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        
        test_loss, test_acc = evaluate(model, test_loader, device, rank)
        
        print(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
        print(f"Best  | val acc {best_val_acc:.4f} (epoch {ckpt['epoch']})")
        
        summary = {
            "final_test_acc": float(test_acc),
            "final_test_loss": float(test_loss),
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(ckpt['epoch']),
            "total_epochs": args.epochs,
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size,
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training complete! Results saved to: {output_dir}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()