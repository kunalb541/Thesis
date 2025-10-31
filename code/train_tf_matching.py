#!/usr/bin/env python3
"""
Training script that EXACTLY matches TensorFlow notebook architecture

Key differences from original train.py:
1. Data reshaping: (N, 256) → (N, 64, 4) to match TensorFlow
2. Conv1D operates on features dimension, not time
3. Double normalization (StandardScaler + MinMaxScaler) like TensorFlow
4. Split before normalization (like TensorFlow)

Author: Kunal Bhatia (corrected architecture)
Date: October 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

def load_and_prepare_data(npz_path):
    """Load and prepare data exactly like TensorFlow notebook"""
    data = np.load(npz_path, allow_pickle=False)
    
    X = data['X']  # (N, original_length)
    y = data['y']
    
    # Replace PAD_VALUE with 0
    X = X.copy()
    X[X == -1.0] = 0.0
    
    # Handle label encoding
    if y.dtype.kind in ('U', 'S', 'O'):
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8)
    
    # Split first (like TensorFlow) - 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Double normalization (StandardScaler + MinMaxScaler)
    print("\nApplying double normalization (TensorFlow style)...")
    scaler_standard = StandardScaler()
    X_train_norm = scaler_standard.fit_transform(X_train)
    X_val_norm = scaler_standard.transform(X_val)
    X_test_norm = scaler_standard.transform(X_test)
    
    scaler_minmax = MinMaxScaler()
    X_train_scaled = scaler_minmax.fit_transform(X_train_norm)
    X_val_scaled = scaler_minmax.transform(X_val_norm)
    X_test_scaled = scaler_minmax.transform(X_test_norm)
    
    # CRITICAL: Reshape to match TensorFlow
    # Original: (N, 256) → Reshape to (N, 64, 4)
    # This treats every 4 consecutive time points as features at a timestep
    sequence_length = 64
    features_per_timestep = X_train_scaled.shape[1] // sequence_length
    
    X_train_reshaped = X_train_scaled.reshape(-1, sequence_length, features_per_timestep)
    X_val_reshaped = X_val_scaled.reshape(-1, sequence_length, features_per_timestep)
    X_test_reshaped = X_test_scaled.reshape(-1, sequence_length, features_per_timestep)
    
    print(f"\n✓ Reshaped data:")
    print(f"  Original shape: {X_train_scaled.shape}")
    print(f"  New shape: {X_train_reshaped.shape}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features per timestep: {features_per_timestep}")
    
    return (X_train_reshaped, y_train), (X_val_reshaped, y_val), (X_test_reshaped, y_test), \
           sequence_length, features_per_timestep


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # (N, seq_len, features)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TensorFlowStyleCNN(nn.Module):
    """CNN that matches TensorFlow architecture"""
    def __init__(self, sequence_length, features_per_timestep, num_classes=2):
        super().__init__()
        
        # Conv1D on features dimension (transpose needed for PyTorch)
        self.conv1 = nn.Conv1d(features_per_timestep, 128, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.3)
        
        self.relu = nn.ReLU()
        
        # TimeDistributed Dense (classification at each timestep)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x: (B, seq_len, features_per_timestep)
        
        # Transpose for Conv1d: (B, features, seq_len)
        x = x.transpose(1, 2)
        
        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout2(x)
        
        x = self.relu(self.conv3(x))
        x = self.dropout3(x)
        
        # Back to (B, seq_len, channels)
        x = x.transpose(1, 2)  # (B, seq_len, 32)
        
        # TimeDistributed classification
        x = self.fc(x)  # (B, seq_len, num_classes)
        
        return x


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(xb)  # (B, seq_len, 2)
        
        # Repeat labels across timesteps (like TensorFlow)
        B, L, C = outputs.shape
        yb_repeated = yb.unsqueeze(1).expand(B, L)  # (B, L)
        
        # Compute loss at every timestep
        loss = criterion(outputs.reshape(B * L, C), yb_repeated.reshape(B * L))
        
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accuracy: aggregate predictions across time
        logits_agg = outputs.mean(dim=1)  # (B, 2)
        preds = logits_agg.argmax(dim=1)
        
        total_loss += loss.item() * B
        total_correct += (preds == yb).sum().item()
        total_samples += B
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        outputs = model(xb)  # (B, seq_len, 2)
        
        # Loss at every timestep
        B, L, C = outputs.shape
        yb_repeated = yb.unsqueeze(1).expand(B, L)
        loss = criterion(outputs.reshape(B * L, C), yb_repeated.reshape(B * L))
        
        # Accuracy
        logits_agg = outputs.mean(dim=1)
        preds = logits_agg.argmax(dim=1)
        
        total_loss += loss.item() * B
        total_correct += (preds == yb).sum().item()
        total_samples += B
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description='Train TensorFlow-matching CNN')
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--experiment_name', default='tf_style', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'../results/{args.experiment_name}_{timestamp}')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("🔥 TENSORFLOW-MATCHING ARCHITECTURE")
    print("="*80)
    print("\n✅ Key changes from original train.py:")
    print("  1. Data reshaping: (N, 256) → (N, 64, 4)")
    print("  2. Conv1D on features dimension (not time)")
    print("  3. Double normalization (StandardScaler + MinMaxScaler)")
    print("  4. Split BEFORE normalization")
    print(f"\n📁 Output directory: {output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPUs: {torch.cuda.device_count()}")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), seq_len, features = \
        load_and_prepare_data(args.data)
    
    # Datasets
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    # Model
    logger.info("Building model...")
    model = TensorFlowStyleCNN(seq_len, features).to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     patience=5, factor=0.5)
    
    # Save config
    config = {
        'experiment_name': args.experiment_name,
        'data_path': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'sequence_length': seq_len,
        'features_per_timestep': features,
        'architecture': 'tensorflow_matching',
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        scheduler.step(val_acc)
        
        logger.info(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                   f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save best model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config,
            }, output_dir / 'best_model.pt')
            logger.info(f"  ↳ saved best model (val_acc={val_acc:.4f})")
    
    # Final test evaluation
    logger.info("="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    
    # Load best model
    ckpt = torch.load(output_dir / 'best_model.pt')
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
    logger.info(f"Best  | val acc {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Save summary
    summary = {
        'final_test_acc': float(test_acc),
        'final_test_loss': float(test_loss),
        'best_val_acc': float(best_val_acc),
        'best_epoch': best_epoch,
        'total_epochs': args.epochs,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("="*80)
    logger.info(f"✓ Training complete! Results saved to: {output_dir}")
    logger.info("="*80)
    
    # Success message
    if test_acc > 0.75:
        logger.info("🎉 EXCELLENT! Architecture is working well!")
    elif test_acc > 0.65:
        logger.info("✓ GOOD! Model is learning. Try more data or epochs.")
    elif test_acc > 0.55:
        logger.info("⚠️  Partial success. Check data quality or try distinct parameters.")
    else:
        logger.info("❌ Still struggling. Verify data generation is working correctly.")


if __name__ == '__main__':
    main()
