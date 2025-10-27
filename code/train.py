"""
PyTorch training script for microlensing classifier - IMPROVED VERSION
Fixes: paths, gradient clipping, mixed precision, logging, checkpoints
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
import joblib
import logging
import sys
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

class TimeDistributedCNN(nn.Module):
    """1D CNN with per-timestep classification"""
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Per-timestep classification
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels)
        x = x.transpose(1, 2)  # (batch, channels, time)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Apply classification at each timestep
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, time, classes)
        
        return x

class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def setup_logging(results_dir):
    """Setup logging to both file and console"""
    log_file = Path(results_dir) / 'training.log'
    
    # Create logger
    logger = logging.getLogger('microlens')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=True, grad_clip=1.0):
    """Train model for one epoch with gradient clipping and mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -1, :], batch_y)
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs[:, -1, :], 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc='Validating'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs[:, -1, :], 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(loader), correct / total

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_acc, results_dir, filename='checkpoint.pt'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_acc': val_acc,
    }
    torch.save(checkpoint, Path(results_dir) / filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_acc']

def main():
    parser = argparse.ArgumentParser(description='Train microlensing classifier')
    parser.add_argument('--data', required=True, help='Path to training data (.npz)')
    parser.add_argument('--output', required=True, help='Path to save model (.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
    args = parser.parse_args()
    
    # Setup paths
    output_path = Path(args.output).resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = output_dir.parent / 'results' / f'{args.experiment_name}_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("="*80)
    logger.info("MICROLENSING BINARY CLASSIFICATION - TRAINING")
    logger.info("="*80)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Log configuration
    logger.info("\nConfiguration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  results_dir: {results_dir}")
    
    # Load and validate data
    logger.info(f"\nLoading data from {args.data}...")
    try:
        data = np.load(args.data)
        X, y = data['X'], data['y']
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        logger.error(f"Generate data first using: python simulate.py --output {args.data}")
        sys.exit(1)
    
    # Validate data
    assert X.shape[0] == len(y), f"Mismatch: X has {X.shape[0]} samples, y has {len(y)}"
    assert np.isfinite(X).all(), "Data contains NaN or inf values"
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    unique, counts = np.unique(y_encoded, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data
    logger.info("\nSplitting data (70% train, 15% val, 15% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    
    # Standardize
    logger.info("\nStandardizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Save scaler
    scaler_path = results_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Create datasets
    train_dataset = LightCurveDataset(X_train, y_train)
    val_dataset = LightCurveDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Build model
    logger.info("\nBuilding model...")
    model = TimeDistributedCNN(X_train.shape[1], X_train.shape[2])
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Mixed precision scaler
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler_amp = GradScaler(enabled=use_amp)
    logger.info(f"Mixed precision training: {use_amp}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    if args.resume and Path(args.resume).exists():
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_val_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler_amp
        )
        logger.info(f"  Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
        start_epoch += 1
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            scaler_amp, use_amp=use_amp, grad_clip=args.grad_clip
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = results_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, scaler_amp, epoch, val_acc, 
                          results_dir, 'best_model.pt')
            logger.info(f"  ★ New best model! Saved to {best_model_path}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = results_dir / f'checkpoint_epoch_{epoch+1}.pt'
            save_checkpoint(model, optimizer, scheduler, scaler_amp, epoch, val_acc,
                          results_dir, f'checkpoint_epoch_{epoch+1}.pt')
            logger.info(f"  Checkpoint saved to {checkpoint_path}")
        
        logger.info("")
    
    # Save final artifacts
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save history
    history_path = results_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save config
    config_path = results_dir / 'config.json'
    config = {
        **vars(args),
        'model_params': {
            'conv1_filters': 128,
            'conv2_filters': 64,
            'conv3_filters': 32,
            'fc1_units': 64,
            'dropout': 0.3,
        },
        'data_shape': list(X_train.shape),
        'num_classes': 2,
        'best_val_acc': float(best_val_acc),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # Copy best model to output location
    if output_path != results_dir / 'best_model.pt':
        import shutil
        shutil.copy(results_dir / 'best_model.pt', output_path)
        logger.info(f"Best model also saved to {output_path}")
    
    logger.info(f"\nAll results saved to: {results_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()