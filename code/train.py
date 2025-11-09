#!/usr/bin/env python3
"""
Distributed Training Script for Masked Transformer - FIXED VERSION
Optimized for multi-node multi-GPU training

Author: Kunal Bhatia
Version: 5.0 - Production ready
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from transformer import MaskedMicrolensingTransformer, count_parameters
from normalization import CausticPreservingNormalizer
import config as CFG


def setup_distributed():
    """Setup distributed training environment"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def is_main_process(rank):
    """Check if this is the main process (rank 0)"""
    return rank == 0


def print_rank0(message, rank):
    """Print only from rank 0"""
    if is_main_process(rank):
        print(message)


class MaskedMicrolensingDataset(Dataset):
    """Dataset that provides validity masks"""
    
    def __init__(self, X, y, pad_value=-1.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.pad_value = pad_value
        self.validity_masks = (self.X != pad_value)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.validity_masks[idx], self.y[idx]


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_grad_norm=5.0, rank=0):
    """Train one epoch with masking"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    skipped_batches = 0
    
    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
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
            skipped_batches += 1
            if is_main_process(rank) and skipped_batches == 1:
                print(f"  ⚠️  Warning: Skipping batches due to NaN/Inf loss")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Check gradients
        if torch.isnan(grad_norm) or grad_norm > max_grad_norm * 10:
            skipped_batches += 1
            continue
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        valid_batches += 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar (only on rank 0)
        if is_main_process(rank) and batch_idx % 10 == 0 and total > 0:
            acc = correct / total
            avg_loss = total_loss / max(valid_batches, 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*acc:.1f}%',
                'grad': f'{grad_norm:.1f}'
            })
    
    # Gather metrics across all processes
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, valid_batches, skipped_batches], 
                              dtype=torch.float32).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, valid_batches, skipped_batches = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    if is_main_process(rank) and skipped_batches > 0:
        print(f"  ⚠️  Skipped {int(skipped_batches)} batches due to NaN/Inf")
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank=0):
    """Evaluate with masking"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(loader, desc="Evaluating", leave=False)
    else:
        pbar = loader
    
    for X, validity_masks, y in pbar:
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
    
    # Gather metrics across all processes
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, valid_batches], 
                              dtype=torch.float32).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, valid_batches = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Masked Transformer (Distributed)')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='masked', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Print header only from rank 0
    if is_main_process(rank):
        print("="*70)
        print("MASKED TRANSFORMER TRAINING (DISTRIBUTED)")
        print("="*70)
        print(f"Device: {device}")
        print(f"World size: {world_size} GPUs ({world_size//4} nodes × 4 GPUs)")
        print(f"Experiment: {args.experiment_name}")
    
    # Load data
    print_rank0(f"\nLoading data from {args.data}...", rank)
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    # Quick mode
    if args.quick:
        print_rank0("⚡ Quick mode: Using 2000 samples", rank)
        indices = np.random.choice(len(X), min(2000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
    
    print_rank0(f"Data shape: {X.shape}", rank)
    print_rank0(f"Classes: Binary={np.sum(y==1)}, PSPL={np.sum(y==0)}", rank)
    
    # Check validity
    pad_value = -1.0
    validity_ratio = (X != pad_value).mean()
    print_rank0(f"Average data validity: {validity_ratio*100:.1f}%", rank)
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print_rank0(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", rank)
    
    # Normalization
    print_rank0("\nApplying normalization...", rank)
    normalizer = CausticPreservingNormalizer(pad_value=pad_value)
    normalizer.fit(X_train)
    
    if is_main_process(rank):
        if normalizer.baseline_median is None or np.isnan(normalizer.baseline_median):
            print("❌ ERROR: Normalization failed!")
            valid_data = X_train[X_train != pad_value]
            if len(valid_data) > 0:
                data_mean = valid_data.mean()
                data_std = valid_data.std()
                print(f"  Using fallback: mean={data_mean:.3f}, std={data_std:.3f}")
        else:
            print(f"  ✓ Normalizer fitted: median={normalizer.baseline_median:.3f}, "
                  f"MAD={normalizer.baseline_mad:.3f}")
    
    # Transform data
    X_train_norm = normalizer.transform(X_train).squeeze(1)
    X_val_norm = normalizer.transform(X_val).squeeze(1)
    X_test_norm = normalizer.transform(X_test).squeeze(1)
    
    # Create datasets
    train_dataset = MaskedMicrolensingDataset(X_train_norm, y_train, pad_value)
    val_dataset = MaskedMicrolensingDataset(X_val_norm, y_val, pad_value)
    test_dataset = MaskedMicrolensingDataset(X_test_norm, y_test, pad_value)
    
    # Create distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print_rank0("\nCreating Masked Transformer...", rank)
    model = MaskedMicrolensingTransformer(
        n_points=1500,
        d_model=args.d_model,
        nhead=4,
        num_layers=args.num_layers,
        dim_ff=args.d_model * 4,
        dropout=0.2,
        pad_value=pad_value
    ).to(device)
    
    # Wrap model in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    print_rank0(f"Model: {count_parameters(model):,} parameters", rank)
    
    # Loss and optimizer
    class_counts = np.bincount(y_train)
    if len(class_counts) == 2:
        class_weights = len(y_train) / (2.0 * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = None
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # Learning rate schedule
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create experiment directory (only on rank 0)
    if is_main_process(rank):
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
            'world_size': world_size,
            'n_train': len(X_train_norm),
            'n_val': len(X_val_norm),
            'n_test': len(X_test_norm),
            'validity_ratio': float(validity_ratio)
        }
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save normalizer
        if normalizer.baseline_median is not None and not np.isnan(normalizer.baseline_median):
            normalizer.save(exp_dir / 'normalizer.pkl')
    else:
        exp_dir = None
    
    # Broadcast exp_dir to all processes
    if world_size > 1:
        if is_main_process(rank):
            exp_dir_str = [str(exp_dir)]
        else:
            exp_dir_str = [None]
        dist.broadcast_object_list(exp_dir_str, src=0)
        if not is_main_process(rank):
            exp_dir = Path(exp_dir_str[0])
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print_rank0("\n" + "="*70, rank)
    print_rank0("STARTING TRAINING", rank)
    print_rank0("="*70, rank)
    
    for epoch in range(args.epochs):
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs}", rank)
        print_rank0("-"*50, rank)
        
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.grad_clip, rank=rank
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, rank)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics (only on rank 0)
        if is_main_process(rank):
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            train_accs.append(float(train_acc))
            val_accs.append(float(val_acc))
            
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Get the actual model (unwrap DDP if needed)
                model_to_save = model.module if hasattr(model, 'module') else model
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
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
    
    # Final evaluation (only on rank 0)
    if is_main_process(rank):
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(exp_dir / 'best_model.pt')
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank)
        
        # Save final results
        results = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'best_val_acc': float(best_val_acc),
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
            print("\n⚠️  Model performance is moderate")
        else:
            print("\n❌ Model performance needs improvement")
        
        print(f"\n📁 Results saved to: {exp_dir}")
        print("="*70)
    
    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
