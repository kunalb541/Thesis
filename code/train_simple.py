#!/usr/bin/env python3
"""
Simple Training Script for Stable Transformer
Uses the SimpleStableTransformer to avoid NaN issues

Author: Kunal Bhatia
Version: 1.0
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from simple_transformer import SimpleStableTransformer, count_parameters
from normalization import CausticPreservingNormalizer


class MicrolensingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def safe_loss(outputs, targets, criterion):
    """Calculate loss with NaN checking"""
    binary_logits = outputs['binary']
    
    # Check for NaN in logits
    if torch.isnan(binary_logits).any():
        print("⚠️ NaN in logits!")
        return None
    
    # Clamp logits for stability
    binary_logits = torch.clamp(binary_logits, min=-10, max=10)
    
    loss = criterion(binary_logits, targets)
    
    if torch.isnan(loss):
        print("⚠️ NaN in loss!")
        return None
    
    return loss


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_grad_norm=100):
    """Train one epoch with stability checks"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    skipped = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        outputs = model(X, return_all_timesteps=False)
        
        # Calculate loss safely
        loss = safe_loss(outputs, y, criterion)
        
        if loss is None:
            skipped += 1
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Aggressive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Check gradient norm
        if grad_norm > 10:
            print(f"⚠️ Large gradient: {grad_norm:.2f}")
            skipped += 1
            continue
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            preds = outputs['binary'].argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress
        if batch_idx % 10 == 0:
            acc = correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*acc:.2f}%',
                'skip': skipped
            })
    
    avg_loss = total_loss / max(len(loader) - skipped, 1)
    accuracy = correct / max(total, 1)
    
    if skipped > 0:
        print(f"  Skipped {skipped} batches due to NaN/large gradients")
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        outputs = model(X, return_all_timesteps=False)
        loss = safe_loss(outputs, y, criterion)
        
        if loss is not None:
            total_loss += loss.item()
            valid_batches += 1
        
        preds = outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train SimpleStableTransformer')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--quick', action='store_true', help='Quick test with 1000 samples')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("SIMPLE STABLE TRANSFORMER TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: d_model={args.d_model}, layers={args.num_layers}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    # Quick mode
    if args.quick:
        print("⚡ Quick mode: Using 1000 samples")
        indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Check for NaN in data
    if np.isnan(X).any():
        print("❌ ERROR: Input data contains NaN!")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    print("\nApplying normalization...")
    normalizer = CausticPreservingNormalizer(pad_value=-1.0)
    normalizer.fit(X_train)
    
    X_train_norm = normalizer.transform(X_train).squeeze(1)
    X_val_norm = normalizer.transform(X_val).squeeze(1)
    X_test_norm = normalizer.transform(X_test).squeeze(1)
    
    # Verify normalization
    valid_mask = X_train_norm != -1.0
    if valid_mask.any():
        data_min = X_train_norm[valid_mask].min()
        data_max = X_train_norm[valid_mask].max()
        print(f"Normalized range: [{data_min:.3f}, {data_max:.3f}]")
        
        if data_min < -10 or data_max > 10:
            print("⚠️ Warning: Large values after normalization")
            # Clip for safety
            X_train_norm = np.clip(X_train_norm, -5, 5)
            X_val_norm = np.clip(X_val_norm, -5, 5)
            X_test_norm = np.clip(X_test_norm, -5, 5)
            print("  Clipped to [-5, 5] for stability")
    
    # Create datasets
    train_dataset = MicrolensingDataset(X_train_norm, y_train)
    val_dataset = MicrolensingDataset(X_val_norm, y_val)
    test_dataset = MicrolensingDataset(X_test_norm, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Create model
    print("\nCreating SimpleStableTransformer...")
    model = SimpleStableTransformer(
        n_points=1500,
        d_model=args.d_model,
        nhead=4,
        num_layers=args.num_layers,
        dim_ff=args.d_model * 4,
        dropout=0.1
    ).to(device)
    
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Test model first
    print("\nTesting model initialization...")
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_X, test_y = test_batch[0][:2].to(device), test_batch[1][:2].to(device)
        test_out = model(test_X, return_all_timesteps=False)
        
        print("Test output shapes:")
        for key, val in test_out.items():
            print(f"  {key}: {val.shape}, NaN={torch.isnan(val).any()}")
        
        if any(torch.isnan(val).any() for val in test_out.values()):
            print("❌ Model produces NaN at initialization!")
            return
        else:
            print("✅ Model initialization OK")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("../results") / f"simple_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Training
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, exp_dir / 'best_model.pt')
            
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping!")
            break
        
        # Check for training failure
        if train_acc < 0.52 and epoch > 5:
            print("⚠️ Model not learning (accuracy near random)")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    if (exp_dir / 'best_model.pt').exists():
        checkpoint = torch.load(exp_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        if test_acc > 0.6:
            print("\n✅ SUCCESS! Model learned successfully")
        else:
            print("\n⚠️ Model performance is low")


if __name__ == "__main__":
    main()
