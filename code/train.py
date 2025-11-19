#!/usr/bin/env python3
"""
Training script for MicrolensingTransformer

Implements distributed training with:
- Temporal encoding distribution tracking and freezing
- Temperature calibration for uncertainty quantification
- Mixed precision training with gradient clipping
- Proper loss aggregation across variable batch sizes
- Early stopping with patience

Author: Kunal Bhatia
Version: 1.0
Date: November 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'


def setup_distributed():
    """Initialize distributed training environment."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class MicrolensingDataset(Dataset):
    """
    Dataset for microlensing light curves.
    
    Returns flux, delta_t, lengths, and labels for each sample.
    """
    
    def __init__(self, flux, delta_t, labels, pad_value=-1.0):
        self.flux = flux
        self.delta_t = delta_t
        self.labels = labels
        self.pad_value = pad_value
        
        # Compute valid lengths
        self.lengths = np.sum(flux != pad_value, axis=1).astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.flux[idx],
            self.delta_t[idx],
            self.lengths[idx],
            self.labels[idx]
        )


def collate_fn(batch):
    """Collate function for DataLoader."""
    flux, delta_t, lengths, labels = zip(*batch)
    
    flux_tensor = torch.from_numpy(np.stack(flux)).float()
    delta_t_tensor = torch.from_numpy(np.stack(delta_t)).float()
    lengths_tensor = torch.from_numpy(np.array(lengths)).long()
    labels_tensor = torch.from_numpy(np.array(labels)).long()
    
    return flux_tensor, delta_t_tensor, lengths_tensor, labels_tensor


def compute_class_weights(labels, n_classes=3):
    """Compute inverse frequency class weights."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)


def train_epoch(model, loader, optimizer, scaler, class_weights, device, rank, use_amp=True):
    """
    Train for one epoch.
    
    Uses reduction='sum' for proper loss aggregation across variable batch sizes.
    
    Returns:
        avg_loss: Average loss per sample
        accuracy: Classification accuracy
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for flux, delta_t, lengths, labels in loader:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        with autocast(enabled=use_amp):
            outputs = model(flux, delta_t, lengths, return_all_timesteps=False)
            logits = outputs['logits']
            
            # Use reduction='sum' to properly aggregate loss across batches
            loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            continue
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total += labels.size(0)
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    # Aggregate across GPUs
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
    
    # Properly average loss per sample
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank):
    """
    Evaluate model on validation/test set.
    
    Note: DistributedSampler pads the dataset to be divisible by world_size,
    which introduces minor bias in metrics.
    
    Returns:
        avg_loss: Average loss per sample
        accuracy: Overall accuracy
        class_acc: Per-class accuracy
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class metrics
    class_correct = torch.zeros(3, device=device)
    class_total = torch.zeros(3, device=device)
    
    for flux, delta_t, lengths, labels in loader:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(flux, delta_t, lengths, return_all_timesteps=False)
        logits = outputs['logits']
        
        # Use reduction='sum' for proper aggregation
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        if torch.isfinite(loss):
            total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Per-class accuracy
        for c in range(3):
            mask = (labels == c)
            if mask.sum() > 0:
                class_correct[c] += (preds[mask] == labels[mask]).sum()
                class_total[c] += mask.sum()
    
    # Aggregate across GPUs
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total],
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_total, op=dist.ReduceOp.SUM)
        
        total_loss, correct, total = metrics.cpu().numpy()
    
    class_correct = class_correct.cpu().numpy()
    class_total = class_total.cpu().numpy()
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    class_acc = np.zeros(3)
    for c in range(3):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]
    
    return avg_loss, accuracy, class_acc


def main():
    parser = argparse.ArgumentParser(description="MicrolensingTransformer - Training")
    
    # Data
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--flux_key', default='flux', help='Key for flux data in npz')
    parser.add_argument('--delta_t_key', default='delta_t', help='Key for delta_t data in npz')
    parser.add_argument('--label_key', default='labels', help='Key for labels in npz')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_window', type=int, default=64)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='../results')
    parser.add_argument('--experiment_name', default='microlens')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print("MicrolensingTransformer - Training")
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
    
    # Load data
    if rank == 0:
        print(f"Loading data: {args.data}")
    
    data = np.load(args.data)
    flux = data[args.flux_key]
    delta_t = data[args.delta_t_key]
    labels = data[args.label_key]
    
    # Remove channel dimension if present
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    if delta_t.ndim == 3:
        delta_t = delta_t.squeeze(1)
    
    n_samples, n_timesteps = flux.shape
    n_classes = len(np.unique(labels))
    
    if rank == 0:
        print(f"Samples: {n_samples}, Timesteps: {n_timesteps}, Classes: {n_classes}")
        for c in range(n_classes):
            count = (labels == c).sum()
            print(f"  Class {c}: {count} ({100*count/n_samples:.1f}%)")
    
    # Split data
    flux_train, flux_temp, delta_t_train, delta_t_temp, y_train, y_temp = train_test_split(
        flux, delta_t, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    flux_val, flux_test, delta_t_val, delta_t_test, y_val, y_test = train_test_split(
        flux_temp, delta_t_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    if rank == 0:
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Create datasets
    train_dataset = MicrolensingDataset(flux_train, delta_t_train, y_train)
    val_dataset = MicrolensingDataset(flux_val, delta_t_val, y_val)
    test_dataset = MicrolensingDataset(flux_test, delta_t_test, y_test)
    
    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,
                                      rank=rank, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Import model
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import MicrolensingTransformer, ModelConfig, calibrate_temperature
    
    # Create model configuration
    config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attention_window=args.attention_window,
        train_final_only=True,
        use_adaptive_normalization=True
    )
    
    # Create model
    model = MicrolensingTransformer(config).to(device)
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
    
    # Wrap in DDP with find_unused_parameters=True
    # Required because we freeze temporal encoding parameters mid-training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=True)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, n_classes).to(device)
    
    if rank == 0:
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if not args.no_amp else None
    
    # Create output directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_dict = vars(args)
        config_dict['world_size'] = world_size
        config_dict['n_params'] = n_params
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Output directory: {exp_dir}")
    
    # Broadcast output directory
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
        print("\nStarting training")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, class_weights,
            device, rank, use_amp=not args.no_amp
        )
        
        scheduler.step()
        
        # Evaluate
        val_loss, val_acc, val_class_acc = evaluate(
            model, val_loader, class_weights, device, rank
        )
        
        # Freeze temporal encoding after warmup
        if epoch == args.warmup_epochs:
            base_model = model.module if hasattr(model, 'module') else model
            base_model.freeze_temporal_encoding()
            if rank == 0:
                print("Temporal encoding frozen")
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            print(f"  Per-class: flat={val_class_acc[0]:.4f}, "
                  f"pspl={val_class_acc[1]:.4f}, binary={val_class_acc[2]:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                save_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, exp_dir / 'best_model.pt')
                
                print(f"  Saved best model (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{args.patience}")
            
            should_stop = (patience_counter >= args.patience)
        else:
            should_stop = False
        
        # Broadcast early stopping decision
        if world_size > 1:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                if rank == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    if world_size > 1:
        dist.barrier()
    
    checkpoint = torch.load(exp_dir / 'best_model.pt', 
                           map_location={'cuda:0': f'cuda:{local_rank}'},
                           weights_only=False)
    load_model = model.module if hasattr(model, 'module') else model
    load_model.load_state_dict(checkpoint['model_state_dict'])
    
    if world_size > 1:
        dist.barrier()
    
    # Final evaluation on test set
    test_loss, test_acc, test_class_acc = evaluate(
        model, test_loader, class_weights, device, rank
    )
    
    if rank == 0:
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"  Per-class: flat={test_class_acc[0]:.4f}, "
              f"pspl={test_class_acc[1]:.4f}, binary={test_class_acc[2]:.4f}")
    
    # Temperature calibration
    if rank == 0:
        print("Calibrating temperature")
    
    calibration_metrics = calibrate_temperature(
        load_model, val_loader, device=str(device)
    )
    
    if rank == 0:
        print(f"Temperature: {calibration_metrics['temperature']:.4f}")
        print(f"ECE: {calibration_metrics['ece_before']:.4f} -> {calibration_metrics['ece_after']:.4f}")
        
        # Save final model with calibrated temperature
        torch.save({
            'model_state_dict': load_model.state_dict(),
            'config': config,
            'test_acc': test_acc,
            'test_class_acc': test_class_acc.tolist(),
            'calibration': calibration_metrics
        }, exp_dir / 'final_model.pt')
        
        # Save results
        results = {
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'test_class_acc': test_class_acc.tolist(),
            'best_val_acc': float(best_val_acc),
            'calibration': calibration_metrics,
            'total_epochs': epoch + 1
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training complete. Results saved to {exp_dir}")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()