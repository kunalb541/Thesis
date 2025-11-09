#!/usr/bin/env python3
"""
Distributed Training for H100 GPUs
Complete Fixed Version with Multi-Task Learning

Fixes:
1. Division by zero in learning rate scheduler
2. DDP unused parameters (now using return_all=True for multi-task learning)
3. Clean logging with suppressed NCCL warnings
4. Proper auxiliary loss computation

Author: Kunal Bhatia
Version: 11.1 - Production Ready
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Suppress NCCL warnings for cleaner output
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'WARN'


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        if rank == 0:
            print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(local_rank)
        
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(message, rank=0):
    """Print only from rank 0"""
    if rank == 0:
        print(message)


class MicrolensingDataset(Dataset):
    """Dataset for distributed training"""
    
    def __init__(self, X, y, pad_value=-1.0):
        # Clean data
        X = np.nan_to_num(X, nan=pad_value, posinf=100.0, neginf=-100.0)
        valid_mask = X != pad_value
        if valid_mask.any():
            X[valid_mask] = np.clip(X[valid_mask], -100, 100)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.pad_value = pad_value
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StableNormalizer:
    """Normalizer that handles padding and prevents NaN"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, X):
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            self.mean = np.median(valid_values)
            self.std = np.median(np.abs(valid_values - self.mean))
            
            if self.std < 1e-8:
                self.std = 1.0
            
            self.mean = np.clip(self.mean, -100, 100)
            self.std = np.clip(self.std, 0.01, 100)
        
        return self
    
    def transform(self, X):
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            X_norm[valid_mask] = (X[valid_mask] - self.mean) / self.std
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        return np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, 
                device, epoch, rank, world_size, use_amp=True, grad_clip=1.0):
    """
    Train one epoch with multi-task learning
    
    Uses all three model heads:
    - binary_head: Main classification (PSPL vs Binary)
    - anomaly_head: Anomaly detection
    - caustic_head: Caustic crossing detection
    """
    model.train()
    
    total_loss = 0
    binary_loss_total = 0
    anomaly_loss_total = 0
    caustic_loss_total = 0
    correct = 0
    total = 0
    num_batches = 0
    skipped_batches = 0
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
    for batch_idx, (X, y) in enumerate(pbar):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass with multi-task learning
        with autocast(enabled=use_amp):
            # CRITICAL: Use return_all=True to ensure all model parameters receive gradients
            # This prevents DDP unused parameter errors
            outputs = model(X, return_all=True)
            
            # Main task: Binary classification
            logits = outputs['binary']
            binary_loss = criterion(logits, y)
            loss = binary_loss
            
            # Auxiliary task 1: Anomaly detection
            # Binary events should have higher anomaly scores
            if 'anomaly' in outputs:
                anomaly_target = y.float()
                anomaly_loss = nn.functional.mse_loss(outputs['anomaly'], anomaly_target)
                loss = loss + 0.1 * anomaly_loss
            else:
                anomaly_loss = torch.tensor(0.0)
            
            # Auxiliary task 2: Caustic detection
            # Binary events have caustic crossings
            if 'caustic' in outputs:
                caustic_target = y.float()
                # BCE is unsafe with autocast, so temporarily disable it
                with autocast(enabled=False):
                    # Convert to float32 for stable BCE computation
                    caustic_pred = outputs['caustic'].float()
                    caustic_target_f32 = caustic_target.float()
                    # Clamp predictions to prevent numerical issues
                    caustic_pred = torch.clamp(caustic_pred, min=1e-7, max=1-1e-7)
                    caustic_loss = nn.functional.binary_cross_entropy(caustic_pred, caustic_target_f32)
                loss = loss + 0.1 * caustic_loss
            else:
                caustic_loss = torch.tensor(0.0)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0 and skipped_batches < 3:
                print(f"[Warning] NaN/Inf loss at batch {batch_idx}, skipping")
            skipped_batches += 1
            continue
        
        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        binary_loss_total += binary_loss.item()
        if isinstance(anomaly_loss, torch.Tensor):
            anomaly_loss_total += anomaly_loss.item()
        if isinstance(caustic_loss, torch.Tensor):
            caustic_loss_total += caustic_loss.item()
        num_batches += 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar (rank 0 only)
        if rank == 0 and batch_idx % 10 == 0:
            acc = correct / total if total > 0 else 0
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc*100:.1f}%',
                    'grad': f'{grad_norm:.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
    
    # Print skipped batches warning
    if rank == 0 and skipped_batches > 0:
        print(f"  [Warning] Skipped {skipped_batches} batches due to NaN/Inf loss")
    
    # Gather metrics across all GPUs
    if world_size > 1:
        metrics = torch.tensor([total_loss, correct, total, num_batches], 
                              dtype=torch.float32).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, num_batches = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, world_size):
    """Evaluate model with multi-task outputs"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc="Validating", leave=False)
    else:
        pbar = loader
    
    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Use return_all=True for DDP consistency (same parameters in every forward pass)
        outputs = model(X, return_all=True)
        logits = outputs['binary']
        loss = criterion(logits, y)
        
        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1
        
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    # Gather metrics across all GPUs
    if world_size > 1:
        metrics = torch.tensor([total_loss, correct, total, num_batches],
                              dtype=torch.float32).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, num_batches = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='H100 Distributed Training')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='h100_dist', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Validate epochs vs warmup to prevent division by zero
    if args.epochs <= args.warmup_epochs:
        print(f"Warning: epochs ({args.epochs}) <= warmup_epochs ({args.warmup_epochs})")
        print(f"Setting warmup_epochs to {max(1, args.epochs - 1)}")
        args.warmup_epochs = max(1, args.epochs - 1)
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Device setup
    device = torch.device(f'cuda:{local_rank}')
    
    # Print info from rank 0
    if rank == 0:
        print("="*70)
        print("H100 DISTRIBUTED TRAINING WITH MULTI-TASK LEARNING")
        print("="*70)
        print(f"World size: {world_size} GPUs")
        print(f"Device: {device}")
        print(f"Mixed Precision: {'Disabled' if args.no_amp else 'Enabled'}")
        print(f"Batch size: {args.batch_size} per GPU ({args.batch_size * world_size} total)")
        print(f"\nMulti-Task Learning:")
        print(f"  - Binary classification (main, weight=1.0)")
        print(f"  - Anomaly detection (auxiliary, weight=0.1)")
        print(f"  - Caustic detection (auxiliary, weight=0.1)")
    
    # Load data
    print_rank0(f"\nLoading data from {args.data}...", rank)
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    print_rank0(f"Data shape: {X.shape}", rank)
    print_rank0(f"Classes: Binary={np.sum(y==1)}, PSPL={np.sum(y==0)}", rank)
    
    # Quick mode
    if args.quick:
        print_rank0("⚡ Quick mode: Using 10000 samples", rank)
        indices = np.random.choice(len(X), min(10000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
    
    # Data validation
    if rank == 0:
        print("\n🔍 Data Validation:")
        print(f"  NaN values: {np.isnan(X).sum()}")
        print(f"  Inf values: {np.isinf(X).sum()}")
    
    # Normalize data
    print_rank0("\n🔄 Normalizing data...", rank)
    normalizer = StableNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit_transform(X)
    
    if rank == 0:
        print(f"  Normalized range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print_rank0(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", rank)
    
    # Create datasets
    train_dataset = MicrolensingDataset(X_train, y_train)
    val_dataset = MicrolensingDataset(X_val, y_val)
    test_dataset = MicrolensingDataset(X_test, y_test)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Import transformer here to avoid issues
    from transformer import MicrolensingTransformer, count_parameters
    
    # Create model
    print_rank0("\n🤖 Creating MicrolensingTransformer...", rank)
    model = MicrolensingTransformer(
        n_points=X.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        pad_value=-1.0
    ).to(device)
    
    # Wrap in DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # All parameters used with return_all=True
        )
    
    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        print(f"Model parameters: {count_parameters(base_model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Scale learning rate by world size
    effective_batch_size = args.batch_size * world_size
    lr_scale = max(1.0, effective_batch_size / 256)  # Base batch size
    scaled_lr = args.lr * lr_scale
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup (FIXED - no more division by zero)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / max(args.warmup_epochs, 1)
        else:
            # Cosine annealing after warmup
            total_epochs = max(args.epochs - args.warmup_epochs, 1)
            progress = (epoch - args.warmup_epochs) / total_epochs
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if not args.no_amp else None
    
    # Create experiment directory (rank 0 only)
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Experiment directory: {exp_dir}")
        
        # Save config
        config = vars(args)
        config['world_size'] = world_size
        config['effective_batch_size'] = effective_batch_size
        config['scaled_lr'] = scaled_lr
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        import pickle
        normalizer_path = exp_dir / 'normalizer.pkl'
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer, f)
        print(f"💾 Normalizer saved to: {normalizer_path}")
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    # Training loop
    if rank == 0:
        print("\n" + "="*70)
        print("STARTING DISTRIBUTED TRAINING")
        print("="*70)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-"*50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, epoch, rank, world_size, use_amp=not args.no_amp,
            grad_clip=args.grad_clip
        )
        
        # Step scheduler after training
        scheduler.step()
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, rank, world_size
        )
        
        # Print metrics (rank 0 only)
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Get base model
                save_model = model.module if hasattr(model, 'module') else model
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, exp_dir / 'best_model.pt')
                
                print(f"✅ Saved best model (val_acc: {val_acc*100:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(exp_dir / 'best_model.pt')
        load_model = model.module if hasattr(model, 'module') else model
        load_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Synchronize before final evaluation
    if world_size > 1:
        dist.barrier()
    
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, rank, world_size
    )
    
    if rank == 0:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'best_val_acc': float(best_val_acc),
            'world_size': world_size,
            'effective_batch_size': effective_batch_size
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Success message
        print("\n" + "="*70)
        if test_acc > 0.75:
            print("🌟 EXCELLENT! Model achieved great performance!")
        elif test_acc > 0.70:
            print("✅ SUCCESS! Model achieved good performance!")
        else:
            print("⚠️  Model performance: {:.2f}%".format(test_acc * 100))
        
        print(f"\n📁 Results saved to: {exp_dir}")
        print("="*70)
    
    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
