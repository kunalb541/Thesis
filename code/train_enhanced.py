#!/usr/bin/env python3
"""
Enhanced Training Script with All Improvements

Integrates:
1. Physics-informed features (4-channel input)
2. Data augmentation
3. Enhanced Transformer with multi-scale attention
4. Focal Loss
5. Ensemble training support

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0 - Complete enhanced pipeline
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import enhanced modules
from model_enhanced import EnhancedTransformerClassifier, count_parameters
from losses import FocalLoss, compute_sample_weights
from features import extract_batch_features
from augment import augment_dataset


class MicrolensingDataset(Dataset):
    """Dataset wrapper for microlensing light curves."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(
    data_path: str,
    use_physics_features: bool = True,
    use_augmentation: bool = False,
    n_augmentations: int = 3,
    seed: int = 42
):
    """
    Load data and apply feature extraction/augmentation.
    
    Args:
        data_path: Path to .npz file
        use_physics_features: Extract physics features (4-channel)
        use_augmentation: Apply data augmentation
        n_augmentations: Number of augmentations per sample
        seed: Random seed
        
    Returns:
        X: Processed data
        y: Labels
        timestamps: Time array
        meta: Metadata
    """
    print("="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load raw data
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    timestamps = data['timestamps']
    meta = json.loads(str(data['meta_json']))
    
    # Apply permutation if available
    if 'perm' in data.files:
        perm = data['perm']
        X = X[perm]
        y = y[perm]
        print(f"✓ Applied shuffle permutation")
    
    # Ensure 3D
    if X.ndim == 2:
        X = X[:, None, :]
    
    print(f"✓ Loaded data: {X.shape}")
    print(f"  PSPL: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Binary: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # Extract physics features
    if use_physics_features:
        print("\nExtracting physics-informed features...")
        X = extract_batch_features(X, timestamps, meta.get('PAD_VALUE', -1.0))
        print(f"✓ Features extracted: {X.shape}")
    
    # Apply augmentation
    if use_augmentation:
        print(f"\nApplying data augmentation ({n_augmentations}x per sample)...")
        X, y = augment_dataset(
            X, y, timestamps,
            n_augmentations=n_augmentations,
            pad_value=meta.get('PAD_VALUE', -1.0),
            augment_binary_only=False  # Augment both classes
        )
        print(f"✓ Augmentation complete: {X.shape}")
    
    return X, y, timestamps, meta


def train_epoch(model, loader, criterion, optimizer, device, epoch, use_tqdm=True):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch+1} Train") if use_tqdm else loader
    
    for X, y in iterator:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)

    return total_loss, correct, total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tqdm=True):
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    iterator = tqdm(loader, desc="Evaluate", leave=False) if use_tqdm else loader

    for X, y in iterator:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)

    return total_loss, correct, total


def main():
    parser = argparse.ArgumentParser(description="Enhanced training with all improvements")
    
    # Data
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--experiment_name", default="enhanced", help="Experiment name")
    
    # Model architecture
    parser.add_argument("--use_physics_features", action="store_true", 
                       help="Use 4-channel physics features")
    parser.add_argument("--use_multi_scale", action="store_true",
                       help="Use multi-scale attention")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--use_augmentation", action="store_true",
                       help="Apply data augmentation")
    parser.add_argument("--n_augmentations", type=int, default=3)
    parser.add_argument("--use_focal_loss", action="store_true",
                       help="Use Focal Loss instead of CE")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("ENHANCED TRAINING PIPELINE")
    print("="*80)
    print(f"Device: {device}")
    print(f"Physics features: {args.use_physics_features}")
    print(f"Multi-scale attention: {args.use_multi_scale}")
    print(f"Data augmentation: {args.use_augmentation}")
    print(f"Focal Loss: {args.use_focal_loss}")
    
    # Load and preprocess data
    X, y, timestamps, meta = load_and_preprocess_data(
        args.data,
        use_physics_features=args.use_physics_features,
        use_augmentation=args.use_augmentation,
        n_augmentations=args.n_augmentations,
        seed=args.seed
    )
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Create datasets
    train_ds = MicrolensingDataset(X_train, y_train)
    val_ds = MicrolensingDataset(X_val, y_val)
    test_ds = MicrolensingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False)
    
    # Create model
    print("\nBuilding enhanced model...")
    in_channels = 4 if args.use_physics_features else 1
    
    model = EnhancedTransformerClassifier(
        in_channels=in_channels,
        n_classes=2,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_multi_scale=args.use_multi_scale
    ).to(device)
    
    print(f"✓ Model created: {count_parameters(model):,} parameters")
    
    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"✓ Using Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"✓ Using Cross Entropy Loss")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config.update({
        'n_parameters': count_parameters(model),
        'data_shape': list(X_train.shape),
        'timestamp': timestamp,
        'in_channels': in_channels
    })
    
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Experiment directory: {exp_dir}")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_correct, train_total = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_correct, val_total = evaluate(
            model, val_loader, criterion, device
        )
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config
            }, exp_dir / "best_model.pt")
            
            print(f"  ✓ New best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_correct, test_total = evaluate(
        model, test_loader, criterion, device
    )
    
    test_acc = test_correct / test_total
    
    print(f"Best Val Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save results
    results = {
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss / test_total),
        'epochs_trained': epoch + 1,
        'improvements_used': {
            'physics_features': args.use_physics_features,
            'multi_scale_attention': args.use_multi_scale,
            'data_augmentation': args.use_augmentation,
            'focal_loss': args.use_focal_loss
        }
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {exp_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
