#!/usr/bin/env python3
"""
Training Script for Masked Transformer
Handles missing data properly during training

Author: Kunal Bhatia
Version: 3.0
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

from transformer import MaskedMicrolensingTransformer, count_parameters
from normalization import CausticPreservingNormalizer
import config as CFG


class MaskedMicrolensingDataset(Dataset):
    """Dataset that provides validity masks"""
    
    def __init__(self, X, y, pad_value=-1.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.pad_value = pad_value
        
        # Pre-compute validity masks
        self.validity_masks = (self.X != pad_value)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.validity_masks[idx], self.y[idx]


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_grad_norm=5.0):
    """Train one epoch with masking"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (X, validity_masks, y) in enumerate(pbar):
        X = X.to(device)
        validity_masks = validity_masks.to(device)
        y = y.to(device)
        
        # Forward pass with masks
        outputs = model(X, validity_masks, return_all_timesteps=False)
        
        # Calculate loss
        logits = outputs['binary']
        logits = torch.clamp(logits, min=-10, max=10)
        loss = criterion(logits, y)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Skipping batch {batch_idx} due to NaN/Inf loss")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Check gradients
        if torch.isnan(grad_norm) or grad_norm > max_grad_norm * 10:
            print(f"Skipping batch {batch_idx} due to gradient explosion")
            continue
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        valid_batches += 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar
        if batch_idx % 10 == 0 and total > 0:
            acc = correct / total
            avg_loss = total_loss / max(valid_batches, 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*acc:.1f}%',
                'grad': f'{grad_norm:.1f}'
            })
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate with masking"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    for X, validity_masks, y in tqdm(loader, desc="Evaluating"):
        X = X.to(device)
        validity_masks = validity_masks.to(device)
        y = y.to(device)
        
        # Forward pass with masks
        outputs = model(X, validity_masks, return_all_timesteps=False)
        
        # Calculate loss
        logits = torch.clamp(outputs['binary'], min=-10, max=10)
        loss = criterion(logits, y)
        
        if not torch.isnan(loss):
            total_loss += loss.item()
            valid_batches += 1
        
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Masked Transformer')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='masked', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("MASKED TRANSFORMER TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Experiment: {args.experiment_name}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    # Quick mode - use subset
    if args.quick:
        print("⚡ Quick mode: Using 2000 samples")
        indices = np.random.choice(len(X), min(2000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: Binary={np.sum(y==1)}, PSPL={np.sum(y==0)}")
    
    # Check validity ratio
    pad_value = -1.0
    validity_ratio = (X != pad_value).mean()
    print(f"Average data validity: {validity_ratio*100:.1f}%")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Normalize (but don't destroy padding info!)
    print("\nApplying normalization...")
    normalizer = CausticPreservingNormalizer(pad_value=pad_value)
    
    # Only fit on non-padded data
    X_train_valid = X_train.copy()
    X_train_valid[X_train_valid == pad_value] = np.nan
    normalizer.fit(X_train_valid)
    
    # Transform but preserve padding markers
    def normalize_with_padding(X_data, normalizer, pad_value=-1.0):
        X_norm = normalizer.transform(X_data).squeeze(1)
        # Restore padding markers
        pad_mask = (X_data.squeeze(1) == pad_value)
        X_norm[pad_mask] = pad_value
        return X_norm
    
    X_train_norm = normalize_with_padding(X_train, normalizer, pad_value)
    X_val_norm = normalize_with_padding(X_val, normalizer, pad_value)
    X_test_norm = normalize_with_padding(X_test, normalizer, pad_value)
    
    # Create datasets with masking
    train_dataset = MaskedMicrolensingDataset(X_train_norm, y_train, pad_value)
    val_dataset = MaskedMicrolensingDataset(X_val_norm, y_val, pad_value)
    test_dataset = MaskedMicrolensingDataset(X_test_norm, y_test, pad_value)
    
    # Create data loaders
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
    print("\nCreating Masked Transformer...")
    model = MaskedMicrolensingTransformer(
        n_points=1500,
        d_model=args.d_model,
        nhead=4,
        num_layers=args.num_layers,
        dim_ff=args.d_model * 4,
        dropout=0.2,
        pad_value=pad_value
    ).to(device)
    
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Loss and optimizer
    # Use weighted loss to handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # Learning rate schedule with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("../results") / f"{args.experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Save configuration
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'grad_clip': args.grad_clip,
        'n_train': len(X_train_norm),
        'n_val': len(X_val_norm),
        'n_test': len(X_test_norm),
        'validity_ratio': float(validity_ratio),
        'class_weights': class_weights.cpu().numpy().tolist()
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save normalizer
    normalizer.save(exp_dir / 'normalizer.pkl')
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.grad_clip
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, exp_dir / 'best_model.pt')
            
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Save final results
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Success indicators
    if test_acc > 0.7:
        print("\n✅ SUCCESS! Model achieved good performance")
    elif test_acc > 0.6:
        print("\n⚠️ Model performance is moderate")
    else:
        print("\n❌ Model performance needs improvement")
    
    print(f"\n📁 Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()