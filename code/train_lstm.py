#!/usr/bin/env python3
"""
Training script for LSTM models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from model_lstm import BidirectionalLSTMClassifier, SimpleLSTMClassifier, GRUClassifier
from utils import load_npz_dataset


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    """Train for one epoch with optional class weighting"""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Class-weighted loss
        if class_weights is not None:
            crit = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # Dynamic class weighting per batch
            n_pspl = (yb == 0).sum().item()
            n_binary = (yb == 1).sum().item()
            
            if n_pspl > 0 and n_binary > 0:
                total = len(yb)
                weight_pspl = total / (2.0 * n_pspl)
                weight_binary = total / (2.0 * n_binary)
                weights = torch.tensor([weight_pspl, weight_binary], 
                                     dtype=torch.float32, device=device)
                crit = nn.CrossEntropyLoss(weight=weights)
            else:
                crit = nn.CrossEntropyLoss()
        
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(xb)
        loss = crit(logits, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        
        preds = logits.argmax(dim=1)
        total_acc += (preds == yb).sum().item()
        n += bs
    
    return total_loss / n, total_acc / n


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            loss = crit(logits, yb)
            
            bs = xb.size(0)
            total_loss += loss.item() * bs
            
            preds = logits.argmax(dim=1)
            total_acc += (preds == yb).sum().item()
            n += bs
    
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='bilstm',
                       choices=['bilstm', 'simple', 'gru'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--experiment_name', type=str, default='lstm')
    args = parser.parse_args()
    
    print("="*80)
    print(f"TRAINING {args.model_type.upper()} MODEL")
    print("="*80)
    
    # Load data
    X, y, _, _ = load_npz_dataset(args.data, apply_perm=True)
    
    # Replace -1 with 0 for missing values
    X[X == -1] = 0
    
    # Split
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} ({(y_train==1).sum():,} binary)")
    print(f"  Val:   {len(X_val):,} ({(y_val==1).sum():,} binary)")
    print(f"  Test:  {len(X_test):,} ({(y_test==1).sum():,} binary)")
    
    # Create datasets
    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                            torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(),
                          torch.from_numpy(y_val).long())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(),
                           torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'bilstm':
        model = BidirectionalLSTMClassifier(
            sequence_length=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout,
            use_attention=True
        )
    elif args.model_type == 'simple':
        model = SimpleLSTMClassifier(
            sequence_length=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout
        )
    else:  # gru
        model = GRUClassifier(
            sequence_length=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': args.model_type,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
            }, args.output)
            print(f"  ↳ saved best model")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    ckpt = torch.load(args.output, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    
    print(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
    print(f"Best  | val acc {best_val_acc:.4f} (epoch {ckpt['epoch']})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
