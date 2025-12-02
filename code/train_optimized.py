#!/usr/bin/env python3
"""
Stabilized Training Script for Pure GRU Model
Key Improvements:
- Lower learning rate (1e-4) for stability
- Stricter gradient clipping (1.0)
- Gradual temporal encoding freeze over 3 epochs
- Better NaN/Inf detection and recovery
- Loss spike detection and learning rate reduction
Author: Kunal Bhatia
Version: 2.0 (Stabilized)
Date: December 2025
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

# =============================================================================
# GLOBAL CONSTANTS (TUNED FOR STABILITY)
# =============================================================================
DEFAULT_ACCUMULATE_STEPS = 4
CLIP_NORM = 200.0  # Reduced from 5.0 for better stability
DEFAULT_LR = 1e-4  # Reduced from 5e-4 for stability

def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class MicrolensingDataset(Dataset):
    def __init__(self, flux, delta_t, labels, pad_value=-1.0):
        self.flux = flux
        self.delta_t = delta_t
        self.labels = labels
        self.pad_value = pad_value
        self.lengths = np.sum(flux != pad_value, axis=1).astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.flux[idx], self.delta_t[idx], 
                self.lengths[idx], self.labels[idx])

def collate_fn(batch):
    flux, delta_t, lengths, labels = zip(*batch)
    flux_tensor = torch.from_numpy(np.stack(flux)).float()
    delta_t_tensor = torch.from_numpy(np.stack(delta_t)).float()
    lengths_tensor = torch.from_numpy(np.array(lengths)).long()
    labels_tensor = torch.from_numpy(np.array(labels)).long()
    return flux_tensor, delta_t_tensor, lengths_tensor, labels_tensor

def compute_class_weights(labels, n_classes=3):
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

def detect_loss_spike(current_loss, loss_history, threshold=2.0):
    """Detect if current loss is anomalously high"""
    if len(loss_history) < 3:
        return False
    
    recent_avg = np.mean(loss_history[-3:])
    if current_loss > threshold * recent_avg:
        return True
    return False

def train_epoch(model, loader, optimizer, scaler, class_weights, device, rank,
                use_amp=True, accumulate_steps=DEFAULT_ACCUMULATE_STEPS,
                epoch=0, warmup_epochs=5, freeze_epochs=3):
    """Train for one epoch with enhanced stability checks"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    accumulation_loss = 0.0
    nan_count = 0
    max_nan_tolerance = 5  # Skip up to 5 NaN batches before failing
    
    # Update temporal encoding learning rate for gradual freeze
    if epoch >= warmup_epochs:
        freeze_progress = min(1.0, (epoch - warmup_epochs) / max(freeze_epochs, 1))
        base_model = model.module if hasattr(model, 'module') else model
        base_model.update_freeze_progress(freeze_progress)
        
        # Reduce temporal encoding LR gradually to near-zero
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'temporal':
                # Keep a small minimum LR to avoid division by zero issues
                original_lr = param_group.get('initial_lr', param_group['lr'])
                param_group['lr'] = original_lr * max(0.01, 1 - freeze_progress)
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, (flux, delta_t, lengths, labels) in enumerate(loader):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            outputs = model(flux, delta_t, lengths, return_all_timesteps=False)
            logits = outputs['logits']
            
            # Check for NaN in outputs
            if not torch.isfinite(logits).all():
                nan_count += 1
                if rank == 0:
                    print(f"Warning: Non-finite logits at step {step} (count: {nan_count})")
                
                if nan_count > max_nan_tolerance:
                    raise RuntimeError(f"Too many NaN batches ({nan_count}). Training unstable.")
                
                continue
            
            loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
            normalized_loss = loss / accumulate_steps
        
        # Check loss value
        if not torch.isfinite(loss):
            nan_count += 1
            if rank == 0:
                print(f"Warning: Non-finite loss at step {step} (count: {nan_count}). Skipping.")
            
            if nan_count > max_nan_tolerance:
                raise RuntimeError(f"Too many NaN losses ({nan_count}). Training unstable.")
            
            continue
        
        # Backward pass
        if use_amp:
            scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()
        
        accumulation_loss += loss.item()
        
        # Optimizer step with gradient clipping
        if (step + 1) % accumulate_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            
            # Check for exploding gradients
            if grad_norm > 10.0 * CLIP_NORM:
                if rank == 0:
                    print(f"Warning: Large gradient norm {grad_norm:.2f} at step {step}")
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += accumulation_loss
            accumulation_loss = 0.0
        
        # Track accuracy
        total += labels.size(0)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    # Handle final partial accumulation
    if accumulation_loss != 0.0:
        if rank == 0:
            print("Info: Running final partial optimizer step.")
        
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        total_loss += accumulation_loss
    
    # Aggregate across GPUs
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy, nan_count

@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank):
    """Evaluate model with stability checks"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = torch.zeros(3, device=device)
    class_total = torch.zeros(3, device=device)
    
    for flux, delta_t, lengths, labels in loader:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(flux, delta_t, lengths, return_all_timesteps=False)
        logits = outputs['logits']
        
        # Skip batch if non-finite
        if not torch.isfinite(logits).all():
            continue
        
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        
        if torch.isfinite(loss):
            total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for c in range(3):
            mask = (labels == c)
            if mask.sum() > 0:
                class_correct[c] += (preds[mask] == labels[mask]).sum()
                class_total[c] += mask.sum()
    
    # Aggregate
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
    parser = argparse.ArgumentParser(description="Stabilized GRU Training")
    
    # Data
    parser.add_argument('--data', required=True)
    parser.add_argument('--flux_key', default='flux')
    parser.add_argument('--delta_t_key', default='delta_t')
    parser.add_argument('--label_key', default='labels')
    
    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_window', type=int, default=1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_steps', type=int, default=DEFAULT_ACCUMULATE_STEPS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--freeze_epochs', type=int, default=3, 
                       help='Number of epochs for gradual freeze')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', default='../results')
    parser.add_argument('--experiment_name', default='stable_gru')
    
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print("=" * 60)
        print("STABILIZED PURE GRU MODEL TRAINING")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Accumulation steps: {args.accumulate_steps}")
        print(f"Effective batch size: {args.batch_size * world_size * args.accumulate_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"Gradient clip: {CLIP_NORM}")
        print(f"Gradual freeze epochs: {args.freeze_epochs}")
        print("=" * 60)
    
    # Load data
    if rank == 0:
        print(f"\nLoading data: {args.data}")
    
    data = np.load(args.data)
    flux = data['flux']
    delta_t = data['delta_t']
    labels = data['labels']
    
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    if delta_t.ndim == 3:
        delta_t = delta_t.squeeze(1)
    
    n_samples, n_timesteps = flux.shape
    n_classes = len(np.unique(labels))
    
    if rank == 0:
        print(f"Samples: {n_samples}, Timesteps: {n_timesteps}, Classes: {n_classes}")
        for c in range(n_classes):
            print(f"  Class {c}: {(labels == c).sum()} ({100*(labels == c).mean():.1f}%)")
    
    # Train/val/test split
    flux_train, flux_temp, delta_t_train, delta_t_temp, y_train, y_temp = train_test_split(
        flux, delta_t, labels, test_size=0.3, stratify=labels, random_state=42
    )
    flux_val, flux_test, delta_t_val, delta_t_test, y_val, y_test = train_test_split(
        flux_temp, delta_t_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    if rank == 0:
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Datasets
    train_dataset = MicrolensingDataset(flux_train, delta_t_train, y_train)
    val_dataset = MicrolensingDataset(flux_val, delta_t_val, y_val)
    test_dataset = MicrolensingDataset(flux_test, delta_t_test, y_test)
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, 
                                     rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, 
                                      rank=rank, shuffle=False)
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # Model
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer_gru import MicrolensingTransformer, ModelConfig, calibrate_temperature
    
    config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attention_window=args.attention_window,
        train_final_only=True,
        use_adaptive_normalization=True,
        freeze_gradually=True,
        gradual_freeze_epochs=args.freeze_epochs
    )
    
    model = MicrolensingTransformer(config).to(device)
    
    if rank == 0:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False)
    
    # Class weights
    class_weights = compute_class_weights(y_train, n_classes).to(device)
    if rank == 0:
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Optimizer with parameter groups for gradual freeze
    base_model = model.module if hasattr(model, 'module') else model
    temporal_params = list(base_model.temporal_encoding.parameters())
    other_params = [p for n, p in base_model.named_parameters() 
                    if 'temporal_encoding' not in n]
    
    param_groups = [
        {'params': other_params, 'lr': args.lr, 'name': 'main', 'initial_lr': args.lr},
        {'params': temporal_params, 'lr': args.lr, 'name': 'temporal', 'initial_lr': args.lr}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8
    )
    
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if not args.no_amp else None
    
    # Setup output directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_dict = vars(args)
        config_dict['world_size'] = world_size
        config_dict['n_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config_dict['effective_batch_size'] = args.batch_size * world_size * args.accumulate_steps
        config_dict['clip_norm'] = CLIP_NORM
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\nOutput directory: {exp_dir}")
    
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
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
    
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    
    # Start gradual freeze at warmup end
    if rank == 0:
        print(f"Will start gradual freeze at epoch {args.warmup_epochs + 1}")
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        try:
            train_loss, train_acc, nan_count = train_epoch(
                model, train_loader, optimizer, scaler, class_weights,
                device, rank, use_amp=not args.no_amp,
                accumulate_steps=args.accumulate_steps,
                epoch=epoch, warmup_epochs=args.warmup_epochs,
                freeze_epochs=args.freeze_epochs
            )
        except RuntimeError as e:
            if rank == 0:
                print(f"\nTraining failed at epoch {epoch + 1}: {e}")
                print("Try reducing learning rate or increasing gradient clipping.")
            cleanup_distributed()
            return
        
        scheduler.step()
        
        # Evaluate
        val_loss, val_acc, val_class_acc = evaluate(
            model, val_loader, class_weights, device, rank
        )
        
        loss_history.append(val_loss)
        
        # Detect loss spike
        if detect_loss_spike(val_loss, loss_history):
            if rank == 0:
                print(f"  Warning: Loss spike detected! Reducing LR.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        if rank == 0:
            freeze_status = ""
            if epoch >= args.warmup_epochs:
                base_model = model.module if hasattr(model, 'module') else model
                freeze_progress = min(1.0, (epoch - args.warmup_epochs) / max(args.freeze_epochs, 1))
                freeze_status = f" [Freeze: {100*freeze_progress:.0f}%]"
            
            print(f"\nEpoch {epoch+1}/{args.epochs}{freeze_status}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, NaN={nan_count}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
            print(f"  Per-class: flat={val_class_acc[0]:.4f}, pspl={val_class_acc[1]:.4f}, binary={val_class_acc[2]:.4f}")
            
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
                print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{args.patience}")
            
            should_stop = (patience_counter >= args.patience)
        else:
            should_stop = False
        
        # Early stopping
        if world_size > 1:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                if rank == 0:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        elif should_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    if world_size > 1:
        dist.barrier()
    
    checkpoint = torch.load(
        exp_dir / 'best_model.pt',
        map_location={'cuda:0': f'cuda:{local_rank}'},
        weights_only=False
    )
    load_model = model.module if hasattr(model, 'module') else model
    load_model.load_state_dict(checkpoint['model_state_dict'])
    
    if world_size > 1:
        dist.barrier()
    
    test_loss, test_acc, test_class_acc = evaluate(
        model, test_loader, class_weights, device, rank
    )
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"  Per-class: flat={test_class_acc[0]:.4f}, pspl={test_class_acc[1]:.4f}, binary={test_class_acc[2]:.4f}")
        
        print("\nCalibrating temperature...")
        calibration_metrics = calibrate_temperature(load_model, val_loader, device=str(device))
        print(f"Temperature: {calibration_metrics['temperature']:.4f}")
        print(f"ECE: {calibration_metrics['ece_before']:.4f} -> {calibration_metrics['ece_after']:.4f}")
        
        torch.save({
            'model_state_dict': load_model.state_dict(),
            'config': config,
            'test_acc': test_acc,
            'test_class_acc': test_class_acc.tolist(),
            'calibration': calibration_metrics
        }, exp_dir / 'final_model.pt')
        
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
        
        print(f"\n✓ Training complete. Results saved to {exp_dir}")
        print("=" * 60)
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
