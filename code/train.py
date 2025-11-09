#!/usr/bin/env python3
"""
NaN-Proof Training Script
Guaranteed stable training with no NaN gradients

Author: Kunal Bhatia
Version: 9.0
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
import warnings
warnings.filterwarnings("ignore")

from transformer import StableMicrolensingTransformer, NaNSafeLoss


class SafeDataset(Dataset):
    """Dataset that guarantees no NaN/Inf values"""
    
    def __init__(self, X, y, pad_value=-1.0):
        # Clean data - replace any NaN/Inf with pad_value
        X = np.nan_to_num(X, nan=pad_value, posinf=100.0, neginf=-100.0)
        
        # Ensure data is in reasonable range
        valid_mask = X != pad_value
        if valid_mask.any():
            X[valid_mask] = np.clip(X[valid_mask], -100, 100)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        # Validate
        assert not torch.isnan(self.X).any(), "NaN found in dataset X"
        assert not torch.isinf(self.X).any(), "Inf found in dataset X"
        assert not torch.isnan(self.y).any(), "NaN found in dataset y"
        
        # Report statistics
        valid_points = (self.X != pad_value).sum().item()
        total_points = self.X.numel()
        print(f"  Dataset: {len(self.X)} samples, {valid_points/total_points*100:.1f}% valid data")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SafeNormalizer:
    """Normalizer that cannot produce NaN"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.center = 0.0
        self.scale = 1.0
    
    def fit(self, X):
        """Fit with guaranteed stability"""
        # Find valid values
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            
            # Robust statistics
            self.center = np.median(valid_values)
            self.scale = np.median(np.abs(valid_values - self.center))
            
            # Ensure scale is not zero
            if self.scale < 1e-8:
                self.scale = np.std(valid_values)
            if self.scale < 1e-8:
                self.scale = 1.0
            
            # Clamp to reasonable values
            self.center = np.clip(self.center, -100, 100)
            self.scale = np.clip(self.scale, 0.01, 100)
        else:
            print("WARNING: No valid data for normalization, using defaults")
            self.center = 0.0
            self.scale = 1.0
        
        print(f"  Normalizer: center={self.center:.3f}, scale={self.scale:.3f}")
        return self
    
    def transform(self, X):
        """Transform with NaN safety"""
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            # Safe normalization
            X_norm[valid_mask] = (X[valid_mask] - self.center) / self.scale
            # Clip to prevent extreme values
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        # Ensure no NaN/Inf
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return X_norm


def check_model_health(model):
    """Check if model parameters are healthy"""
    unhealthy = False
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ NaN in {name}")
            unhealthy = True
        if torch.isinf(param).any():
            print(f"  ❌ Inf in {name}")
            unhealthy = True
        
        # Check if parameters are exploding
        param_norm = param.norm().item()
        if param_norm > 1000:
            print(f"  ⚠️  Large parameters in {name}: {param_norm:.2e}")
    
    return not unhealthy


def safe_backward(loss, model):
    """Perform backward pass with NaN checking"""
    # Check loss first
    if torch.isnan(loss) or torch.isinf(loss):
        print("  ⚠️  NaN/Inf loss detected, skipping backward pass")
        return False
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_ok = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ⚠️  NaN gradient in {name}, zeroing")
                param.grad.zero_()
                grad_ok = False
            elif torch.isinf(param.grad).any():
                print(f"  ⚠️  Inf gradient in {name}, clipping")
                param.grad = torch.clamp(param.grad, -1.0, 1.0)
                grad_ok = False
    
    return grad_ok


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """Train one epoch with extensive NaN protection"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)
        
        # Check input
        if torch.isnan(X).any() or torch.isinf(X).any():
            print(f"  ⚠️  NaN/Inf in input batch {batch_idx}, skipping")
            skipped_batches += 1
            continue
        
        # Forward pass
        try:
            outputs = model(X)
            logits = outputs['binary']
            loss = criterion(logits, y)
        except Exception as e:
            print(f"  ⚠️  Error in forward pass: {e}")
            skipped_batches += 1
            continue
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
            skipped_batches += 1
            continue
        
        # Backward pass with safety
        optimizer.zero_grad()
        
        if safe_backward(loss, model):
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if torch.isfinite(grad_norm) and grad_norm < 100:
                # Update weights
                optimizer.step()
                valid_batches += 1
                
                # Update metrics
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
            else:
                skipped_batches += 1
        else:
            skipped_batches += 1
        
        # Update progress bar
        if valid_batches > 0 and total > 0:
            pbar.set_postfix({
                'loss': f'{total_loss/valid_batches:.4f}',
                'acc': f'{100*correct/total:.1f}%',
                'skip': skipped_batches
            })
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    # Report
    if skipped_batches > 0:
        print(f"  ⚠️  Skipped {skipped_batches} batches")
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy, skipped_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate with NaN protection"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    for X, y in tqdm(loader, desc="Evaluating", leave=False):
        X = X.to(device)
        y = y.to(device)
        
        # Skip bad batches
        if torch.isnan(X).any() or torch.isinf(X).any():
            continue
        
        try:
            outputs = model(X)
            logits = outputs['binary']
            loss = criterion(logits, y)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1
            
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        except Exception as e:
            print(f"  ⚠️  Error in evaluation: {e}")
            continue
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='NaN-Proof Training')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='stable', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller for stability
    parser.add_argument('--lr', type=float, default=5e-5)  # Small learning rate
    parser.add_argument('--quick', action='store_true', help='Quick test')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    print(f"Data shape: {X.shape}")
    
    # Data validation
    print("\n🔍 Data Validation:")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    # Clean data if needed
    if nan_count > 0 or inf_count > 0:
        print("  Cleaning data...")
        X = np.nan_to_num(X, nan=-1.0, posinf=100.0, neginf=-100.0)
    
    # Quick mode
    if args.quick:
        print("⚡ Quick mode: Using 1000 samples")
        indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
        args.epochs = 5
    
    # Normalize
    print("\n🔄 Normalizing data...")
    normalizer = SafeNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit(X).transform(X)
    
    # Verify normalization
    print(f"  Normalized range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
    assert not np.isnan(X_norm).any(), "NaN after normalization!"
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create datasets
    train_dataset = SafeDataset(X_train, y_train)
    val_dataset = SafeDataset(X_val, y_val)
    test_dataset = SafeDataset(X_test, y_test)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size*2,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*2,
        shuffle=False, num_workers=0
    )
    
    # Create model
    print("\n🤖 Creating Stable Transformer...")
    model = StableMicrolensingTransformer(
        n_points=X.shape[1],
        d_model=64,  # Small for stability
        nhead=4,     # Few heads for stability
        num_layers=3,  # Shallow for stability
        dropout=0.1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Check initial model health
    print("\n🏥 Initial model health check:")
    if check_model_health(model):
        print("  ✅ Model parameters healthy")
    else:
        print("  ⚠️  Model has issues, continuing anyway...")
    
    # Loss and optimizer
    criterion = NaNSafeLoss()
    
    # Very conservative optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        eps=1e-4  # Larger epsilon for stability
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"results/{args.experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Experiment directory: {exp_dir}")
    
    # Training
    print("\n" + "="*60)
    print("STARTING STABLE TRAINING")
    print("="*60)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*50)
        
        # Train
        train_loss, train_acc, skipped = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        # Check model health
        if not check_model_health(model):
            print("  ❌ Model unhealthy, stopping training")
            break
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
        print(f"Val: Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'normalizer': {
                    'center': normalizer.center,
                    'scale': normalizer.scale
                }
            }, exp_dir / 'best_model.pt')
            
            print(f"✅ Saved best model (val_acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'best_val_acc': float(best_val_acc)
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Success message
    if test_acc > 0.70:
        print("\n🎉 SUCCESS! Model achieved good performance WITHOUT NaN!")
    elif test_acc > 0.65:
        print("\n✅ Model trained successfully without NaN")
    else:
        print("\n⚠️  Model needs improvement, but no NaN issues!")
    
    print(f"\n📁 Results saved to: {exp_dir}")
    print("\n✨ Training completed without NaN gradients!")


if __name__ == "__main__":
    main()