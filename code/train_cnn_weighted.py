#!/usr/bin/env python3
"""
Training script for CNN with class weighting on standardized data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm

from model import TimeDistributedCNN
from utils import load_npz_dataset


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch with dynamic class weighting"""
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        
        # Dynamic class weighting per batch
        n_pspl = (yb == 0).sum().item()
        n_binary = (yb == 1).sum().item()
        
        if n_pspl > 0 and n_binary > 0:
            # Weight inversely proportional to frequency
            batch_size = len(yb)
            weight_pspl = batch_size / (2.0 * n_pspl)
            weight_binary = batch_size / (2.0 * n_binary)
            
            class_weights = torch.tensor([weight_pspl, weight_binary], 
                                        dtype=torch.float32, device=device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
        else:
            crit = nn.CrossEntropyLoss()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward
        outputs = model(xb)  # (batch, seq, 2)
        logits = outputs.mean(dim=1)  # (batch, 2) - average over time
        
        loss = crit(logits, yb)
        
        # Backward
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Statistics
        bs = xb.size(0)
        total_loss += loss.item() * bs
        
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        n += bs
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{total_correct/n:.4f}'})
    
    return total_loss / n, total_correct / n


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    crit = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            outputs = model(xb)
            logits = outputs.mean(dim=1)
            
            loss = crit(logits, yb)
            
            bs = xb.size(0)
            total_loss += loss.item() * bs
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            n += bs
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    return total_loss / n, total_correct / n, (tp, fp, fn, tn)


def main():
    parser = argparse.ArgumentParser(description='Train CNN on standardized data')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--experiment_name', type=str, default='cnn_standardized')
    args = parser.parse_args()
    
    print("="*80)
    print("TRAINING CNN ON STANDARDIZED DATA")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X, y, _, _ = load_npz_dataset(args.data, apply_perm=True)
    
    # Handle missing values (standardized data uses 0 for missing)
    # No need to replace -1 with 0 if already standardized
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {(y==0).sum():,} PSPL, {(y==1).sum():,} Binary")
    
    # Check if data is standardized
    sample_means = []
    sample_stds = []
    for i in range(min(1000, len(X))):
        valid = (X[i] != 0) & (X[i] != -1)
        if valid.sum() > 10:
            sample_means.append(X[i][valid].mean())
            sample_stds.append(X[i][valid].std())
    
    print(f"\nData statistics (sample):")
    print(f"  Mean: {np.mean(sample_means):.4f} ± {np.std(sample_means):.4f}")
    print(f"  Std:  {np.mean(sample_stds):.4f} ± {np.std(sample_stds):.4f}")
    
    if abs(np.mean(sample_means)) < 0.1 and abs(np.mean(sample_stds) - 1.0) < 0.2:
        print("  ✓ Data appears standardized (mean≈0, std≈1)")
    else:
        print("  ⚠️  Data may not be standardized")
    
    # Split
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} ({(y_train==0).sum():,} PSPL, {(y_train==1).sum():,} Binary)")
    print(f"  Val:   {len(X_val):,} ({(y_val==0).sum():,} PSPL, {(y_val==1).sum():,} Binary)")
    print(f"  Test:  {len(X_test):,} ({(y_test==0).sum():,} PSPL, {(y_test==1).sum():,} Binary)")
    
    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float().unsqueeze(1),  # Add channel dim
        torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float().unsqueeze(1),
        torch.from_numpy(y_val).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float().unsqueeze(1),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TimeDistributedCNN(
        sequence_length=X.shape[1],
        num_channels=1,
        num_classes=2
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TimeDistributedCNN")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, 
                                                device, args.grad_clip)
        
        # Validate
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, device)
        tp, fp, fn, tn = val_metrics
        
        # Scheduler step
        scheduler.step(val_acc)
        
        elapsed = time.time() - start_time
        
        # Print
        print(f"Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"time {elapsed:.1f}s")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
            }, args.output)
            
            print(f"  ↳ saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={max_patience})")
            break
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    ckpt = torch.load(args.output, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc, test_metrics = evaluate(model, test_loader, device)
    tp, fp, fn, tn = test_metrics
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTest  | loss {test_loss:.4f} acc {test_acc:.4f}")
    print(f"Best  | val acc {best_val_acc:.4f} (epoch {ckpt['epoch']})")
    
    print(f"\nConfusion Matrix (Test):")
    print(f"  TP: {tp:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TN: {tn:,}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
