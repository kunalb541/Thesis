#!/usr/bin/env python3
"""
Quick Training Script for Lightweight Model

FIXED VERSION: Handles PyTorch 2.x deprecation warnings

Author: Kunal Bhatia
Version: 7.1
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Use lightweight config
import config_lite as CFG
from streaming_transformer_lite import StreamingTransformerLite, count_parameters
from normalization import CausticPreservingNormalizer
from streaming_losses_lite import CombinedLoss  # Use fixed version


class MicrolensingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        # Fixed: Use torch.amp instead of torch.cuda.amp
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(X, return_all_timesteps=True)
                loss = criterion(outputs, y, X)
        else:
            outputs = model(X, return_all_timesteps=True)
            loss = criterion(outputs, y, X)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        final_outputs = model(X, return_all_timesteps=False)
        preds = final_outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
        
        if (pbar.n + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        outputs = model(X, return_all_timesteps=True)
        loss = criterion(outputs, y, X)
        total_loss += loss.item()
        
        final_outputs = model(X, return_all_timesteps=False)
        preds = final_outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Fast lightweight training')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='lite', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=CFG.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=CFG.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=CFG.LEARNING_RATE)
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--quick_test', action='store_true', help='Use small subset for testing')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("LIGHTWEIGHT FAST TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    # Quick test mode - use small subset
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE - Using 10K samples")
        n_samples = min(10000, len(y))
        indices = np.random.choice(len(y), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    
    # Create splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - CFG.TRAIN_RATIO), 
        stratify=y, random_state=CFG.SEED
    )
    
    val_test_ratio = CFG.VAL_RATIO / (CFG.VAL_RATIO + CFG.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio),
        stratify=y_temp, random_state=CFG.SEED
    )
    
    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    print("\nApplying normalization...")
    normalizer = CausticPreservingNormalizer(pad_value=CFG.PAD_VALUE)
    normalizer.fit(X_train)
    
    X_train_norm = normalizer.transform(X_train).squeeze(1)
    X_val_norm = normalizer.transform(X_val).squeeze(1)
    X_test_norm = normalizer.transform(X_test).squeeze(1)
    
    # Create datasets and loaders
    train_dataset = MicrolensingDataset(X_train_norm, y_train)
    val_dataset = MicrolensingDataset(X_val_norm, y_val)
    test_dataset = MicrolensingDataset(X_test_norm, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create lightweight model
    model = StreamingTransformerLite(
        n_points=CFG.N_POINTS,
        downsample_factor=CFG.DOWNSAMPLE_FACTOR,
        d_model=CFG.D_MODEL,
        nhead=CFG.NHEAD,
        num_layers=CFG.NUM_LAYERS,
        dim_feedforward=CFG.DIM_FEEDFORWARD,
        window_size=CFG.WINDOW_SIZE,
        dropout=CFG.DROPOUT,
        use_multi_head=CFG.USE_MULTI_HEAD
    ).to(device)
    
    print(f"\nLightweight model created with {count_parameters(model):,} parameters")
    print(f"Temporal downsampling: {CFG.N_POINTS} -> {CFG.MAX_SEQ_LEN} points")
    
    # Loss and optimizer
    criterion = CombinedLoss(weights=CFG.LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Fixed: Use torch.amp.GradScaler instead of torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda') if args.amp and torch.cuda.is_available() else None
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(CFG.RESULTS_DIR) / f"{args.experiment_name}_lite_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        'args': vars(args),
        'model_params': count_parameters(model),
        'downsample_factor': CFG.DOWNSAMPLE_FACTOR,
        'model_config': {
            'd_model': CFG.D_MODEL,
            'nhead': CFG.NHEAD,
            'num_layers': CFG.NUM_LAYERS,
            'dim_feedforward': CFG.DIM_FEEDFORWARD
        },
        'data_shape': list(X_train.shape),
        'timestamp': timestamp
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    normalizer.save(exp_dir / 'normalizer.pkl')
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    training_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        epoch_time = time.time() - epoch_start
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, exp_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= CFG.PATIENCE:
            print(f"\nEarly stopping triggered (patience: {CFG.PATIENCE})")
            break
    
    training_time = time.time() - training_start
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(exp_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nTotal training time: {training_time/60:.1f} minutes")
    print(f"Average epoch time: {training_time/(epoch+1):.1f} seconds")
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'final_epoch': epoch + 1,
        'training_time_minutes': training_time / 60,
        'avg_epoch_time_seconds': training_time / (epoch + 1)
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
