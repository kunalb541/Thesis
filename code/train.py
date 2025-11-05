#!/usr/bin/env python3
"""
Training Script for Transformer-Based Microlensing Classifier with DDP

Full PyTorch DistributedDataParallel support for multi-GPU and multi-node training.

Author: Kunal Bhatia
Date: November 2025
Version: 5.0 - Fixed DDP implementation
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
from datetime import datetime
from pathlib import Path
import json
import pickle
import os

from model import TransformerClassifier, count_parameters


def setup_ddp():
    """Initialize DDP - works for both torchrun and manual setup"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MicrolensingDataset(Dataset):
    """Simple dataset for microlensing light curves"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_data(X_train, X_val, X_test, pad_value=-1.0):
    """
    Two-stage normalization (StandardScaler + MinMaxScaler)
    Handles padded values correctly
    
    Returns:
        Normalized data and fitted scalers
    """
    # Get shapes
    N_train, C, T = X_train.shape
    N_val, N_test = X_val.shape[0], X_test.shape[0]
    F = C * T
    
    # Flatten to 2D
    X_train_flat = X_train.reshape(N_train, F)
    X_val_flat = X_val.reshape(N_val, F)
    X_test_flat = X_test.reshape(N_test, F)
    
    # Remember pad positions
    train_pad = (X_train_flat == pad_value)
    val_pad = (X_val_flat == pad_value)
    test_pad = (X_test_flat == pad_value)
    
    # Compute per-feature means on train (ignoring pads)
    means = np.zeros(F)
    for j in range(F):
        col = X_train_flat[:, j]
        valid = col != pad_value
        if valid.any():
            means[j] = col[valid].mean()
    
    # Fill pads with means
    X_train_filled = np.where(train_pad, means, X_train_flat)
    X_val_filled = np.where(val_pad, means, X_val_flat)
    X_test_filled = np.where(test_pad, means, X_test_flat)
    
    # Stage 1: StandardScaler
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train_filled)
    X_val_std = scaler_std.transform(X_val_filled)
    X_test_std = scaler_std.transform(X_test_filled)
    
    # Stage 2: MinMaxScaler
    scaler_mm = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler_mm.fit_transform(X_train_std)
    X_val_norm = scaler_mm.transform(X_val_std)
    X_test_norm = scaler_mm.transform(X_test_std)
    
    # Restore pads
    X_train_norm[train_pad] = pad_value
    X_val_norm[val_pad] = pad_value
    X_test_norm[test_pad] = pad_value
    
    # Reshape back to 3D
    X_train_norm = X_train_norm.reshape(N_train, C, T)
    X_val_norm = X_val_norm.reshape(N_val, C, T)
    X_test_norm = X_test_norm.reshape(N_test, C, T)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler_std, scaler_mm


def train_epoch(model, loader, criterion, optimizer, device, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(y)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * len(y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(y)
    
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Transformer classifier with DDP")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--experiment_name", default="transformer", help="Experiment name")
    
    # Model args
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="FFN dimension")
    parser.add_argument("--downsample_factor", type=int, default=3, help="Downsample factor")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Set seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    if is_main:
        print("="*80)
        print("TRANSFORMER TRAINING PIPELINE WITH DDP")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}/{world_size-1}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    
    # Load data
    if is_main:
        print(f"\nLoading data from {args.data}...")
    
    data = np.load(args.data)
    X, y = data['X'], data['y']
    
    # Apply permutation if available
    if 'perm' in data:
        if is_main:
            print("✓ Applying permutation")
        perm = data['perm']
        X, y = X[perm], y[perm]
    
    # Ensure 3D
    if X.ndim == 2:
        X = X[:, None, :]
    
    if is_main:
        print(f"Data shape: {X.shape}")
        print(f"Class distribution: PSPL={np.sum(y==0)}, Binary={np.sum(y==1)}")
    
    # Split data
    if is_main:
        print("\nSplitting data (60/20/20)...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    
    if is_main:
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize (only on rank 0, then broadcast - or do on all ranks identically)
    if is_main:
        print("\nNormalizing data...")
    
    X_train, X_val, X_test, scaler_std, scaler_mm = normalize_data(
        X_train, X_val, X_test, pad_value=-1.0
    )
    
    # Create datasets
    train_ds = MicrolensingDataset(X_train, y_train)
    val_ds = MicrolensingDataset(X_val, y_val)
    test_ds = MicrolensingDataset(X_test, y_test)
    
    # Create samplers for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if is_main:
        print("\nCreating Transformer model...")
    
    model = TransformerClassifier(
        in_channels=1,
        n_classes=2,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        downsample_factor=args.downsample_factor,
        dropout=args.dropout
    ).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    n_params = count_parameters(model_without_ddp)
    
    if is_main:
        print(f"✓ Model created: {n_params:,} parameters")
        print(f"✓ Effective batch size: {args.batch_size * world_size}")
    
    # Create experiment directory (only rank 0)
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = vars(args)
        config['n_parameters'] = n_params
        config['device'] = str(device)
        config['world_size'] = world_size
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save scalers
        with open(exp_dir / "scaler_std.pkl", "wb") as f:
            pickle.dump(scaler_std, f)
        with open(exp_dir / "scaler_mm.pkl", "wb") as f:
            pickle.dump(scaler_mm, f)
        
        print(f"✓ Experiment directory: {exp_dir}")
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    if is_main:
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, rank)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, rank)
        
        # Aggregate metrics across all ranks
        if world_size > 1:
            # Convert to tensors
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        # Save best model (only rank 0)
        if is_main:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, exp_dir / "best_model.pt")
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {train_acc:.4f} (loss: {train_loss:.4f}) | "
                  f"Val: {val_acc:.4f} (loss: {val_loss:.4f}) | "
                  f"Best: {best_val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            if is_main:
                print(f"\nEarly stopping triggered (patience={args.patience})")
            break
        
        # Early success stopping
        if val_acc > 0.995:
            if is_main:
                print(f"\nReached {val_acc:.2%} - stopping early!")
            break
    
    if is_main:
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)
        
        # Load best model
        checkpoint = torch.load(exp_dir / "best_model.pt")
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
    
    # Synchronize before final eval
    if world_size > 1:
        dist.barrier()
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank)
    
    # Aggregate test metrics
    if world_size > 1:
        metrics = torch.tensor([test_loss, test_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        test_loss, test_acc = metrics.tolist()
    
    if is_main:
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc:.2%})")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Save results
        results = {
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'epochs_trained': epoch + 1,
            'model_architecture': 'Transformer',
            'n_parameters': n_params,
            'world_size': world_size
        }
        
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {exp_dir}")
        print("="*80)
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()