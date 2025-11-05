#!/usr/bin/env python3
"""
train.py - WORKING Training Script for Microlensing Classification

Key fixes applied:
1. NO preprocessing (MinMaxScaler was destroying the signal!)
2. Apply permutation to shuffle data
3. Use global average pooling to capture mean flux
4. num_workers=0 to avoid data loading issues

Author: Kunal Bhatia (fixed by Claude)
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from model import TimeDistributedCNN
import argparse
from datetime import datetime
from pathlib import Path
import json

class SimpleDataset(Dataset):
    """Dataset that transposes [N, T, F] to [N, F, T] for Conv1d"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    parser = argparse.ArgumentParser(description="Train microlensing classifier")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--experiment_name", default="experiment", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("Loading data...")
    data = np.load(args.data)
    X, y = data['X'], data['y']
    
    # CRITICAL: Apply permutation to shuffle data
    if 'perm' in data:
        print("✓ Applying permutation to shuffle data...")
        perm = data['perm']
        X = X[perm]
        y = y[perm]
    else:
        print("⚠️  No permutation found - shuffling manually...")
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
    
    # Reshape to windowed format [N, T, F]
    X = X.reshape(-1, 100, 10)
    
    print(f"\nData loaded:")
    print(f"  Shape: {X.shape}")
    print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
    
    # Verify signal is present
    pspl_mean = X[y == 0][:1000].mean()
    binary_mean = X[y == 1][:1000].mean()
    print(f"\n  PSPL mean flux:   {pspl_mean:.4f}")
    print(f"  Binary mean flux: {binary_mean:.4f}")
    print(f"  Difference:       {abs(binary_mean - pspl_mean):.4f} ({100*abs(binary_mean - pspl_mean)/pspl_mean:.1f}%)")
    
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
    
    print(f"  Train: {X_train.shape} (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
    print(f"  Val:   {X_val.shape} (Class 0: {(y_val==0).sum()}, Class 1: {(y_val==1).sum()})")
    print(f"  Test:  {X_test.shape} (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")
    
    # ========================================================================
    # Create Datasets and DataLoaders
    # ========================================================================
    train_ds = SimpleDataset(X_train, y_train)
    val_ds = SimpleDataset(X_val, y_val)
    test_ds = SimpleDataset(X_test, y_test)
    
    # IMPORTANT: num_workers=0 to avoid data loading issues
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    # ========================================================================
    # Create Model
    # ========================================================================
    print("\nCreating model...")
    model = TimeDistributedCNN(
        in_channels=10,  # 10 features per timestep
        n_classes=2,
        dropout=0.3
    ).to(device)
    
    print(f"Model architecture:\n{model}\n")
    
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
    config['model'] = 'TimeDistributedCNN'
    config['preprocessing'] = 'None (raw data)'
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
        # ====================================================================
        # Train
        # ====================================================================
        model.train()
        train_correct, train_total = 0, 0
        
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # CRITICAL: Use return_sequence=False to get global pooling
            logits, _ = model(x, return_sequence=False)
            
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += len(y_batch)
        
        train_acc = train_correct / train_total
        
        # ====================================================================
        # Validate
        # ====================================================================
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                
                logits, _ = model(x, return_sequence=False)
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == y_batch).sum().item()
                val_total += len(y_batch)
        
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"Best: {best_val_acc:.4f} @ Epoch {best_epoch}")
        
        # Early stopping if we reach very high accuracy
        if val_acc > 0.995:
            print(f"\n✓ Reached {val_acc:.2%} - stopping early!")
            break
    
    print("="*80)
    
    # ========================================================================
    # Test Evaluation
    # ========================================================================
    print("\nEvaluating on test set...")
    
    # Load best model
    model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
    model.eval()
    
    test_correct, test_total = 0, 0
    test_loss = 0.0
    
    with torch.no_grad():
        for x, y_batch in test_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            
            logits, _ = model(x, return_sequence=False)
            loss = criterion(logits, y_batch)
            
            pred = logits.argmax(dim=1)
            test_correct += (pred == y_batch).sum().item()
            test_total += len(y_batch)
            test_loss += loss.item() * len(y_batch)
    
    test_acc = test_correct / test_total
    test_loss = test_loss / test_total
    
    # ========================================================================
    # Save Results
    # ========================================================================
    results = {
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss)
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"{'='*80}")
    print(f"  Best Val Accuracy:  {best_val_acc:.4f} ({best_val_acc:.2%}) @ Epoch {best_epoch}")
    print(f"  Test Accuracy:      {test_acc:.4f} ({test_acc:.2%})")
    print(f"  Test Loss:          {test_loss:.4f}")
    print(f"{'='*80}")
    print(f"\n✓ Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
