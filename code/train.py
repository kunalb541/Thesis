#!/usr/bin/env python3
"""
Distributed Training for Microlensing Classification
====================================================

Trains transformer model for three-class classification (Flat/PSPL/Binary).

Supports:
- Single GPU and multi-GPU distributed training (DDP)
- Mixed precision training (AMP)
- Gradient checkpointing for memory efficiency
- AMD (ROCm) and NVIDIA (CUDA) GPUs

Author: Kunal Bhatia
Version: 13.0
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
import torch.nn.functional as F

# Suppress NCCL warnings
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
    """
    Dataset for microlensing light curves
    
    Features:
    - Pre-computed sequence lengths
    - Efficient data loading
    - Handles missing data (pad_value=-1.0)
    """
    
    def __init__(self, X, y, pad_value=-1.0):
        """
        Initialize dataset
        
        Args:
            X: Light curves [N, T]
            y: Labels [N]
            pad_value: Padding value
        """
        # Clean data
        X = np.nan_to_num(X, nan=pad_value, posinf=100.0, neginf=-100.0)
        valid_mask = X != pad_value
        if valid_mask.any():
            X[valid_mask] = np.clip(X[valid_mask], -100, 100)
        
        self.X = X
        self.y = y
        self.pad_value = pad_value
        
        # Pre-compute lengths
        self.lengths = np.sum(X != pad_value, axis=1).astype(np.int32)
        
        print(f"      Dataset: {len(X)} events")
        print(f"      Valid observations: {self.lengths.mean():.0f} ± {self.lengths.std():.0f}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Get single item
        
        Returns:
            (x, y, length)
        """
        return (
            self.X[idx],
            self.y[idx],
            self.lengths[idx]
        )


def collate_fn(batch):
    """Collate batch for DataLoader"""
    # Unpack batch
    xs, ys, lengths = zip(*batch)
    
    # Convert to tensors
    x_tensor = torch.from_numpy(np.stack(xs)).float()
    y_tensor = torch.from_numpy(np.array(ys)).long()
    lengths_tensor = torch.from_numpy(np.array(lengths)).long()
    
    return x_tensor, y_tensor, lengths_tensor


class StableNormalizer:
    """Robust normalizer matching training"""
    
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
    """Training epoch with distributed support"""
    model.train()
    
    total_loss = 0
    classification_loss_total = 0
    correct = 0
    total = 0
    num_batches = 0
    skipped_batches = 0
    
    # Loss tracking for auxiliary tasks
    aux_losses = {
        'flat': 0.0,
        'pspl': 0.0,
        'anomaly': 0.0,
        'caustic': 0.0
    }
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
    for batch_idx, (X, y, lengths) in enumerate(pbar):
        # Move to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(X, return_all=True)
            
            # Main classification loss
            logits = outputs['logits']
            classification_loss = criterion(logits, y)
            loss = classification_loss
            
            # Auxiliary losses with proper weights
            if 'flat' in outputs:
                flat_target = (y == 0).float()
                flat_loss = F.binary_cross_entropy_with_logits(
                    outputs['flat'], flat_target
                )
                loss = loss + 0.5 * flat_loss
                aux_losses['flat'] += flat_loss.item()
            
            if 'pspl' in outputs:
                pspl_target = (y == 1).float()
                pspl_loss = F.binary_cross_entropy_with_logits(
                    outputs['pspl'], pspl_target
                )
                loss = loss + 0.5 * pspl_loss
                aux_losses['pspl'] += pspl_loss.item()
            
            if 'anomaly' in outputs:
                anomaly_target = (y > 0).float()
                anomaly_loss = F.binary_cross_entropy_with_logits(
                    outputs['anomaly'], anomaly_target
                )
                loss = loss + 0.2 * anomaly_loss
                aux_losses['anomaly'] += anomaly_loss.item()
            
            if 'caustic' in outputs:
                caustic_target = (y == 2).float()
                caustic_loss = F.binary_cross_entropy_with_logits(
                    outputs['caustic'], caustic_target
                )
                loss = loss + 0.2 * caustic_loss
                aux_losses['caustic'] += caustic_loss.item()
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0 and skipped_batches < 3:
                print(f"[Warning] NaN/Inf loss at batch {batch_idx}, skipping")
            skipped_batches += 1
            continue
        
        # Backward pass with mixed precision
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        classification_loss_total += classification_loss.item()
        num_batches += 1
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar (only on rank 0)
        if rank == 0 and batch_idx % 10 == 0:
            acc = correct / total if total > 0 else 0
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc*100:.1f}%',
                    'grad': f'{grad_norm:.2f}'
                })
    
    if rank == 0 and skipped_batches > 0:
        print(f"  [Warning] Skipped {skipped_batches} batches due to NaN/Inf")
    
    # Gather metrics across all ranks
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, correct, total, num_batches],
            dtype=torch.float32
        ).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, num_batches = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, world_size):
    """Evaluation with distributed support"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    # Per-class accuracy
    class_correct = torch.zeros(3, device=device)
    class_total = torch.zeros(3, device=device)
    
    if rank == 0:
        pbar = tqdm(loader, desc="Validating", leave=False)
    else:
        pbar = loader
    
    for X, y, lengths in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        outputs = model(X, return_all=False)
        logits = outputs['logits']
        loss = criterion(logits, y)
        
        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1
        
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
        
        # Per-class accuracy
        for c in range(3):
            mask = (y == c)
            if mask.sum() > 0:
                class_correct[c] += (preds[mask] == y[mask]).sum()
                class_total[c] += mask.sum()
    
    # Gather metrics
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, correct, total, num_batches],
            dtype=torch.float32
        ).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_total, op=dist.ReduceOp.SUM)
        
        total_loss, correct, total, num_batches = metrics.cpu().numpy()
        class_correct = class_correct.cpu().numpy()
        class_total = class_total.cpu().numpy()
    else:
        class_correct = class_correct.cpu().numpy()
        class_total = class_total.cpu().numpy()
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    # Print per-class accuracy
    if rank == 0:
        class_names = ['Flat', 'PSPL', 'Binary']
        print(f"\n  Per-Class Accuracy:")
        for c in range(3):
            if class_total[c] > 0:
                class_acc = class_correct[c] / class_total[c]
                print(f"    {class_names[c]:8s}: {class_acc*100:5.2f}%")
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Distributed training for microlensing classification"
    )
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='microlensing_transformer', 
                       help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--no_amp', action='store_true', 
                       help='Disable mixed precision')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training options
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing (saves memory)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Validate warmup epochs
    if args.epochs <= args.warmup_epochs:
        print(f"Warning: epochs ({args.epochs}) <= warmup ({args.warmup_epochs})")
        args.warmup_epochs = max(1, args.epochs - 1)
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Print configuration
    if rank == 0:
        print("="*70)
        print("TRAINING MICROLENSING TRANSFORMER")
        print("="*70)
        print(f"GPUs: {world_size}")
        print(f"Device: {device}")
        print(f"Mixed precision: {'Enabled' if not args.no_amp else 'Disabled'}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    
    # Load data
    print_rank0(f"\nLoading {args.data}...", rank)
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    n_classes = len(np.unique(y))
    if rank == 0:
        print(f"   Classes: {n_classes}")
        if n_classes == 3:
            print(f"   Flat:   {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
            print(f"   PSPL:   {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
            print(f"   Binary: {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")
    
    # Quick mode
    if args.quick:
        print_rank0("Quick mode: Using 10k samples", rank)
        indices = np.random.choice(len(X), min(10000, len(X)), replace=False)
        X, y = X[indices], y[indices]
    
    # Normalize data
    print_rank0("\nNormalizing data...", rank)
    normalizer = StableNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit_transform(X)
    
    if rank == 0:
        print(f"   Mean: {normalizer.mean:.3f}")
        print(f"   Std:  {normalizer.std:.3f}")
    
    # Split data
    print_rank0("\nSplitting data...", rank)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    if rank == 0:
        print(f"   Train: {len(X_train)}")
        print(f"   Val:   {len(X_val)}")
        print(f"   Test:  {len(X_test)}")
    
    # Create datasets
    print_rank0("\nCreating datasets...", rank)
    train_dataset = MicrolensingDataset(X_train, y_train)
    val_dataset = MicrolensingDataset(X_val, y_val)
    test_dataset = MicrolensingDataset(X_test, y_test)
    
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    # Import transformer
    from transformer import MicrolensingTransformer, count_parameters
    
    # Create model
    print_rank0("\nCreating model...", rank)
    model = MicrolensingTransformer(
        n_points=X.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        pad_value=-1.0,
        use_checkpoint=args.gradient_checkpointing
    ).to(device)
    
    # Wrap in DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        print(f"   Parameters: {count_parameters(base_model):,}")
    
    # Setup optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Scale learning rate by world size
    effective_batch_size = args.batch_size * world_size
    lr_scale = max(1.0, effective_batch_size / 256)
    scaled_lr = args.lr * lr_scale
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        else:
            progress = (epoch - args.warmup_epochs) / max(
                args.epochs - args.warmup_epochs, 1
            )
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if not args.no_amp else None
    
    # Create experiment directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExperiment: {exp_dir.name}")
        
        # Save config
        config = vars(args)
        config['world_size'] = world_size
        config['effective_batch_size'] = effective_batch_size
        config['scaled_lr'] = scaled_lr
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save normalizer
        import pickle
        with open(exp_dir / 'normalizer.pkl', 'wb') as f:
            pickle.dump(normalizer, f)
    
    # Broadcast experiment directory to all ranks
    if world_size > 1:
        if rank == 0:
            exp_dir_str = str(exp_dir)
        else:
            exp_dir_str = None
        exp_dir_list = [exp_dir_str]
        dist.broadcast_object_list(exp_dir_list, src=0)
        if rank != 0:
            exp_dir = Path(exp_dir_list[0])
        
        dist.barrier()
    
    # Training loop
    if rank == 0:
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, epoch, rank, world_size,
            use_amp=not args.no_amp,
            grad_clip=args.grad_clip
        )
        
        scheduler.step()
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, rank, world_size
        )
        
        if rank == 0:
            print(f"\nResults:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                save_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, exp_dir / 'best_model.pt')
                
                print(f"   Saved best model (val_acc: {val_acc*100:.2f}%)")
            else:
                patience_counter += 1
                print(f"   Patience: {patience_counter}/{max_patience}")
            
            should_stop = (patience_counter >= max_patience)
        else:
            should_stop = False
        
        # Broadcast early stopping decision
        if world_size > 1:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                if rank == 0:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            if should_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
    
    # Load best model
    if world_size > 1:
        dist.barrier()
        checkpoint = torch.load(
            exp_dir / 'best_model.pt',
            map_location={'cuda:0': f'cuda:{local_rank}'},
            weights_only=False
        )
        load_model = model.module if hasattr(model, 'module') else model
        load_model.load_state_dict(checkpoint['model_state_dict'])
        dist.barrier()
    else:
        checkpoint = torch.load(
            exp_dir / 'best_model.pt',
            weights_only=False
        )
        load_model = model.module if hasattr(model, 'module') else model
        load_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, rank, world_size
    )
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST RESULTS")
        print(f"{'='*70}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
        print(f"{'='*70}")
        
        # Save results
        results = {
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'best_val_acc': float(best_val_acc),
            'total_epochs': epoch + 1
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Results saved to: {exp_dir}")
    
    # Cleanup
    if world_size > 1:
        dist.barrier()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()