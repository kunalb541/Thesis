#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — FIXED VERSION with proper temporal loss
Matches TensorFlow notebook's approach of computing loss at EVERY timestep

KEY FIX: Changed from aggregated loss to per-timestep loss
This allows the model to learn temporal patterns properly!
"""

import argparse
import os
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from pathlib import Path
# Config
import config as CFG
from model import TimeDistributedCNN
from utils import load_npz_dataset


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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, grad_clip=None, scaler=None):
    """
    FIXED: Now computes loss at EVERY timestep like TensorFlow notebook
    """
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with autocast():
                outputs = model(xb)  # [B, L, 2]
                
                # ============================================================
                # 🔥 KEY FIX: Compute loss at EVERY timestep
                # ============================================================
                B, L, C = outputs.shape
                
                # Repeat labels across timesteps (like TensorFlow notebook)
                yb_repeated = yb.unsqueeze(1).expand(B, L)  # [B, L]
                
                # Reshape and compute loss
                outputs_flat = outputs.reshape(B * L, C)  # [B*L, 2]
                yb_flat = yb_repeated.reshape(B * L)  # [B*L]
                
                loss = crit(outputs_flat, yb_flat)
                # ============================================================
                
                # For accuracy, aggregate predictions
                logits = outputs.mean(dim=1)  # [B, 2]
            
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(xb)  # [B, L, 2]
            
            # ============================================================
            # 🔥 KEY FIX: Compute loss at EVERY timestep
            # ============================================================
            B, L, C = outputs.shape
            
            # Repeat labels across timesteps
            yb_repeated = yb.unsqueeze(1).expand(B, L)  # [B, L]
            
            # Reshape and compute loss
            outputs_flat = outputs.reshape(B * L, C)  # [B*L, 2]
            yb_flat = yb_repeated.reshape(B * L)  # [B*L]
            
            loss = crit(outputs_flat, yb_flat)
            # ============================================================
            
            # For accuracy, aggregate predictions
            logits = outputs.mean(dim=1)  # [B, 2]
            
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb) * bs
        n += bs
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy(logits.detach(), yb):.3f}'})
    
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    """
    FIXED: Also uses per-timestep loss for consistency
    """
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    for xb, yb in tqdm(loader, desc="Validating", leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        
        outputs = model(xb)  # [B, L, 2]
        
        # ============================================================
        # 🔥 KEY FIX: Compute loss at EVERY timestep
        # ============================================================
        B, L, C = outputs.shape
        
        # Repeat labels across timesteps
        yb_repeated = yb.unsqueeze(1).expand(B, L)  # [B, L]
        
        # Reshape and compute loss
        outputs_flat = outputs.reshape(B * L, C)  # [B*L, 2]
        yb_flat = yb_repeated.reshape(B * L)  # [B*L]
        
        loss = crit(outputs_flat, yb_flat)
        # ============================================================
        
        # For accuracy, aggregate predictions
        logits = outputs.mean(dim=1)  # [B, 2]
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=CFG.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Create timestamped results directory
    from datetime import datetime
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(CFG.RESULTS_DIR) / f"{args.experiment_name}_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    best_model_path = output_dir / "best_model.pt"
    config_path = output_dir / "config.json"
    log_path = output_dir / "training.log"

    print("="*80)
    print("🔥 FIXED TRAINING WITH TEMPORAL LOSS")
    print("="*80)
    print("\n✅ KEY FIX: Loss computed at EVERY timestep (matches TensorFlow notebook)")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Model: {best_model_path}")
    print(f"   Logs:  {log_path}")
    
    # Save configuration
    config_dict = {
        "experiment_name": args.experiment_name,
        "data_path": os.path.abspath(args.data),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_split": CFG.TRAIN_SPLIT,
        "val_split": CFG.VAL_SPLIT,
        "test_split": CFG.TEST_SPLIT,
        "architecture": {
            "conv1_filters": CFG.CONV1_FILTERS,
            "conv2_filters": CFG.CONV2_FILTERS,
            "conv3_filters": CFG.CONV3_FILTERS,
            "fc1_units": CFG.FC1_UNITS,
            "dropout_rate": CFG.DROPOUT_RATE,
        },
        "timestamp": datetime.now().isoformat(),
        "fix_applied": "temporal_loss_per_timestep",
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Configuration saved to {config_path}")
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs available: {num_gpus}")
    if num_gpus > 1:
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load data
    logger.info(f"Loading dataset: {args.data}")
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    L = X.shape[1]
    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")

    # Deterministic split (data already shuffled by permutation)
    n_total = len(X)
    n_train = int(n_total * CFG.TRAIN_SPLIT)
    n_val = int(n_total * CFG.VAL_SPLIT)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Check class balance
    train_balance = (np.sum(y_train==0), np.sum(y_train==1))
    val_balance = (np.sum(y_val==0), np.sum(y_val==1))
    test_balance = (np.sum(y_test==0), np.sum(y_test==1))
    
    logger.info(f"Train balance: PSPL={train_balance[0]:,}, Binary={train_balance[1]:,}")
    logger.info(f"Val balance:   PSPL={val_balance[0]:,}, Binary={val_balance[1]:,}")
    logger.info(f"Test balance:  PSPL={test_balance[0]:,}, Binary={test_balance[1]:,}")
    
    train_ratio = train_balance[1] / train_balance[0]
    if train_ratio < 0.9 or train_ratio > 1.1:
        logger.warning(f"Class imbalance detected (ratio: {train_ratio:.3f})")

    # Datasets
    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val)
    test_ds = NumpyDataset(X_test, y_test)

    # DataLoaders
    num_workers = min(32, os.cpu_count())
    logger.info(f"DataLoader workers: {num_workers}")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers//2, pin_memory=True
    )

    # Model
    logger.info("Building model...")
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    
    if num_gpus > 1:
        logger.info(f"Wrapping with DataParallel ({num_gpus} GPUs)")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=CFG.LR_PATIENCE, 
        factor=CFG.LR_FACTOR,
    )

    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    logger.info(f"Mixed precision: {use_amp}")

    # Training loop
    best_val_acc = -1.0
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Batch size: {args.batch_size} ({'per GPU' if num_gpus > 1 else 'total'})")
    logger.info(f"Effective batch size: {args.batch_size * num_gpus if num_gpus > 1 else args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, 
            grad_clip=CFG.GRAD_CLIP, scaler=scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        if scheduler is not None:
            scheduler.step(val_acc)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": config_dict,
            }, best_model_path)
            logger.info(f"  ↳ saved best model (val_acc={val_acc:.4f})")

    # Test evaluation
    logger.info("="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(ckpt["model_state_dict"])
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
    logger.info(f"Best  | val acc {best_val_acc:.4f} (epoch {ckpt['epoch']})")
    
    # Save final summary
    summary = {
        "final_test_acc": float(test_acc),
        "final_test_loss": float(test_loss),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(ckpt['epoch']),
        "total_epochs": args.epochs,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("="*80)
    logger.info(f"✓ Training complete! Results saved to: {output_dir}")
    logger.info("="*80)
    
    if test_acc > 0.65:
        logger.info("🎉 SUCCESS! Accuracy > 65% - the temporal loss fix worked!")
    elif test_acc > 0.55:
        logger.info("⚠️  Partial improvement. Try longer training or check data quality.")
    else:
        logger.info("❌ Still low accuracy. Check data generation and class balance.")


if __name__ == "__main__":
    main()
