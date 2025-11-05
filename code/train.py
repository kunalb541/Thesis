#!/usr/bin/env python3
"""
train.py - Training Script with Transformer Support (v7.0)

NEW: Added Transformer model option. Use --model_type to choose:
  - 'cnn': TimeDistributedCNN (default)
  - 'transformer': TransformerClassifier (new!)

Author: Kunal Bhatia
Date: November 2025
Version: 7.0 - Added Transformer architecture
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from model import TimeDistributedCNN
try:
    from model_transformer_efficient import EfficientTransformerClassifier as TransformerClassifier
except ImportError:
    from model_transformer import TransformerClassifier
import argparse
from datetime import datetime
from pathlib import Path
import json

class SimpleDataset(Dataset):
    """Dataset that transposes [N, T, F] to [N, F, T] for Conv1d"""
    def __init__(self, X, y):
        # X is already [N, C, T] from load_npz_dataset
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    parser = argparse.ArgumentParser(description="Train microlensing classifier with CNN or Transformer")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--experiment_name", default="experiment", help="Experiment name")
    parser.add_argument("--model_type", default="cnn", choices=["cnn", "transformer"], 
                       help="Model architecture: 'cnn' or 'transformer'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    
    # Transformer-specific args
    parser.add_argument("--d_model", type=int, default=64, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="FFN dimension")
    parser.add_argument("--downsample_factor", type=int, default=3, help="Downsample factor (3=500 steps, 5=300 steps)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type.upper()}\n")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("Loading data...")
    data = np.load(args.data)
    X, y = data['X'], data['y']
    
    # Apply permutation if available
    if 'perm' in data:
        print("✓ Applying permutation to shuffle data...")
        perm = data['perm']
        X = X[perm]
        y = y[perm]
    else:
        print("⚠️  No permutation found - shuffling manually...")
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
    
    # Ensure X is 3D [N, C, T]
    if X.ndim == 2:
        X = X[:, None, :]  # [N, T] -> [N, 1, T]
    
    print(f"\nData loaded:")
    print(f"  Shape: {X.shape}")
    print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
    
    # Verify signal
    pspl_mean = X[y == 0][:1000].mean()
    binary_mean = X[y == 1][:1000].mean()
    print(f"\n  PSPL mean flux:   {pspl_mean:.4f}")
    print(f"  Binary mean flux: {binary_mean:.4f}")
    print(f"  Difference:       {abs(binary_mean - pspl_mean):.4f}")
    
    # ========================================================================
    # Split Data
    # ========================================================================
    print(f"\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # ========================================================================
    # Create Datasets and DataLoaders
    # ========================================================================
    train_ds = SimpleDataset(X_train, y_train)
    val_ds = SimpleDataset(X_val, y_val)
    test_ds = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    # ========================================================================
    # Create Model
    # ========================================================================
    print(f"\nCreating {args.model_type.upper()} model...")
    
    in_channels = X.shape[1]  # Should be 1
    
    if args.model_type == 'cnn':
        model = TimeDistributedCNN(
            in_channels=in_channels,
            n_classes=2,
            dropout=args.dropout
        ).to(device)
    else:  # transformer
        model = TransformerClassifier(
            in_channels=in_channels,
            n_classes=2,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            downsample_factor=args.downsample_factor,
            dropout=args.dropout
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # ========================================================================
    # Create Experiment Directory
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['device'] = str(device)
    config['model_architecture'] = args.model_type
    config['n_parameters'] = n_params
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Experiment directory: {exp_dir}\n")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("Training...")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x, return_sequence=False)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += len(y_batch)
            train_loss_sum += loss.item() * len(y_batch)
        
        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total
        
        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                logits, _ = model(x, return_sequence=False)
                loss = criterion(logits, y_batch)
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == y_batch).sum().item()
                val_total += len(y_batch)
                val_loss_sum += loss.item() * len(y_batch)
        
        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train: {train_acc:.4f} (loss: {train_loss:.4f}) | "
              f"Val: {val_acc:.4f} (loss: {val_loss:.4f}) | "
              f"Best: {best_val_acc:.4f} @ Epoch {best_epoch}")
        
        # Early stopping
        if val_acc > 0.995:
            print(f"\n✓ Reached {val_acc:.2%} - stopping early!")
            break
    
    print("="*80)
    
    # ========================================================================
    # Test Evaluation
    # ========================================================================
    print("\nEvaluating on test set...")
    
    model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
    model.eval()
    
    test_correct, test_total = 0, 0
    test_loss_sum = 0.0
    
    with torch.no_grad():
        for x, y_batch in test_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            logits, _ = model(x, return_sequence=False)
            loss = criterion(logits, y_batch)
            
            pred = logits.argmax(dim=1)
            test_correct += (pred == y_batch).sum().item()
            test_total += len(y_batch)
            test_loss_sum += loss.item() * len(y_batch)
    
    test_acc = test_correct / test_total
    test_loss = test_loss_sum / test_total
    
    # ========================================================================
    # Save Results
    # ========================================================================
    results = {
        "model_type": args.model_type,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "n_parameters": n_params
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"{'='*80}")
    print(f"  Model Type:         {args.model_type.upper()}")
    print(f"  Parameters:         {n_params:,}")
    print(f"  Best Val Accuracy:  {best_val_acc:.4f} ({best_val_acc:.2%}) @ Epoch {best_epoch}")
    print(f"  Test Accuracy:      {test_acc:.4f} ({test_acc:.2%})")
    print(f"  Test Loss:          {test_loss:.4f}")
    print(f"{'='*80}")
    print(f"\n✓ Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
