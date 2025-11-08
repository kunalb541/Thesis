#!/usr/bin/env python3
"""
DDP-Enabled Training Script with Auto-Detection
Automatically handles single GPU, multi-GPU, and multi-node setups

Author: Kunal Bhatia
Version: 3.0 - Full DDP with stability fixes
"""

import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Optional: disable specific PyTorch warnings (recommended)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # for transformers
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"        # avoids debug spam
os.environ["NCCL_DEBUG"] = "WARN"               # reduces NCCL verbosity
os.environ["OMP_NUM_THREADS"] = "1"             # avoid excessive threading warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # if TensorFlow is indirectly used

# Suppress tqdm warnings
from tqdm import tqdm
tqdm.disable = lambda *a, **k: None  # disables tqdm warnings, not progress bars

# Now import the rest
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime

from transformer import SimpleStableTransformer, count_parameters
from normalization import CausticPreservingNormalizer

def setup_distributed():
    """
    Setup distributed training with auto-detection
    Returns: rank, local_rank, world_size, device, is_distributed
    """
    
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Torchrun or distributed launch detected
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # CUDA device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        if rank == 0:
            print(f"🚀 Distributed training initialized")
            print(f"   World size: {world_size}")
            print(f"   Backend: nccl")
            
        return rank, local_rank, world_size, device, True
    
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Multiple GPUs detected but not launched with torchrun
        print(f"⚠️ Detected {torch.cuda.device_count()} GPUs but not using distributed launch.")
        print(f"   For multi-GPU training, use:")
        print(f"   torchrun --nproc_per_node={torch.cuda.device_count()} train_ddp.py ...")
        print(f"   Falling back to single GPU (cuda:0)")
        
        return 0, 0, 1, torch.device('cuda:0'), False
    
    elif torch.cuda.is_available():
        # Single GPU
        print(f"🖥️ Single GPU training on cuda:0")
        return 0, 0, 1, torch.device('cuda:0'), False
    
    else:
        # CPU only
        print(f"💻 CPU training (no GPUs detected)")
        return 0, 0, 1, torch.device('cpu'), False


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MicrolensingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loader(dataset, batch_size, is_train=True, 
                       is_distributed=False, num_workers=2):
    """Create data loader with optional distributed sampler"""
    
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=is_train,
            drop_last=is_train
        )
        # Don't shuffle when using DistributedSampler
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train
        )


def safe_loss_and_backward(model, outputs, targets, criterion, optimizer, 
                          max_grad_norm=5.0, scaler=None):
    """Calculate loss and perform backward pass with stability checks"""
    
    binary_logits = outputs['binary']
    
    # Clamp logits for stability
    binary_logits = torch.clamp(binary_logits, min=-10, max=10)
    
    # Calculate loss
    loss = criterion(binary_logits, targets)
    
    # Check for NaN
    if torch.isnan(loss) or torch.isinf(loss):
        return None, None
    
    # Backward pass
    optimizer.zero_grad()
    
    if scaler is not None:
        # Mixed precision training
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Check if gradients are finite
        grad_finite = all(
            torch.isfinite(p.grad).all() 
            for p in model.parameters() 
            if p.grad is not None
        )
        
        if grad_finite:
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = float('inf')
    else:
        # Standard training
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
        
        optimizer.step()
    
    return loss.item(), grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


def train_epoch(model, loader, criterion, optimizer, device, epoch, 
                max_grad_norm=5.0, scaler=None, rank=0, world_size=1):
    """Train one epoch with DDP support"""
    model.train()
    
    total_loss = 0
    total_grad_norm = 0
    correct = 0
    total = 0
    valid_batches = 0
    skipped = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        try:
            outputs = model(X, return_all_timesteps=False)
        except RuntimeError as e:
            if rank == 0:
                print(f"⚠️ Forward pass error: {e}")
            skipped += 1
            continue
        
        # Backward pass with safety
        loss_val, grad_norm = safe_loss_and_backward(
            model, outputs, y, criterion, optimizer, max_grad_norm, scaler
        )
        
        if loss_val is None:
            skipped += 1
            continue
        
        # Update metrics
        total_loss += loss_val
        total_grad_norm += grad_norm
        valid_batches += 1
        
        with torch.no_grad():
            preds = outputs['binary'].argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar (rank 0 only)
        if rank == 0 and batch_idx % 10 == 0 and total > 0:
            acc = correct / total
            avg_loss = total_loss / max(valid_batches, 1)
            avg_grad = total_grad_norm / max(valid_batches, 1)
            
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'acc': f'{100*acc:.1f}%',
                'grad': f'{grad_norm:.1f}',
                'skip': skipped
            })
    
    # Gather statistics across all processes
    if world_size > 1:
        # Convert to tensors for all_reduce
        stats = torch.tensor([total_loss, total_grad_norm, correct, total, valid_batches, skipped], 
                            dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        total_loss = stats[0].item()
        total_grad_norm = stats[1].item()
        correct = int(stats[2].item())
        total = int(stats[3].item())
        valid_batches = int(stats[4].item())
        skipped = int(stats[5].item())
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    avg_grad_norm = total_grad_norm / max(valid_batches, 1)
    
    if rank == 0:
        if skipped > 0:
            print(f"  Skipped {skipped} batches ({100*skipped/(len(loader)*world_size):.1f}%)")
        print(f"  Average gradient norm: {avg_grad_norm:.2f}")
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank=0, world_size=1):
    """Evaluate model with DDP support"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    # Only show progress on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader
    
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        outputs = model(X, return_all_timesteps=False)
        
        # Calculate loss
        logits = torch.clamp(outputs['binary'], min=-10, max=10)
        loss = criterion(logits, y)
        
        if not torch.isnan(loss):
            total_loss += loss.item()
            valid_batches += 1
        
        preds = outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    # Gather statistics across all processes
    if world_size > 1:
        stats = torch.tensor([total_loss, correct, total, valid_batches], 
                           dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
        total_loss = stats[0].item()
        correct = int(stats[1].item())
        total = int(stats[2].item())
        valid_batches = int(stats[3].item())
    
    avg_loss = total_loss / max(valid_batches, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Stable Transformer with DDP')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', type=str, default='ddp', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, device, is_distributed = setup_distributed()
    
    # Only print on rank 0
    if rank == 0:
        print("="*60)
        print("STABLE TRANSFORMER TRAINING WITH DDP v3.0")
        print("="*60)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Distributed: {is_distributed}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Mixed precision: {args.amp}")
    
    # Load data (only on rank 0 for efficiency)
    # Load and preprocess data - ALL RANKS DO THIS INDEPENDENTLY (MUCH FASTER!)
    if rank == 0:
        print(f"\nLoading data from {args.data}...")
    
    # All ranks load the data file independently (from shared filesystem)
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    if args.quick:
        if rank == 0:
            print("⚡ Quick mode: Using 1000 samples")
        # Use same seed for all ranks to get same indices
        np.random.seed(42)
        indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
    
    if rank == 0:
        print(f"Data shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
    
    # All ranks do identical splits (deterministic with same seed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    if rank == 0:
        print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # All ranks normalize (normalizer is deterministic)
    if rank == 0:
        print("\nApplying normalization...")
    normalizer = CausticPreservingNormalizer(pad_value=-1.0)
    normalizer.fit(X_train)
    
    X_train_norm = normalizer.transform(X_train).squeeze(1)
    X_val_norm = normalizer.transform(X_val).squeeze(1)
    X_test_norm = normalizer.transform(X_test).squeeze(1)
    
    # Clip for safety
    X_train_norm = np.clip(X_train_norm, -5, 5)
    X_val_norm = np.clip(X_val_norm, -5, 5)
    X_test_norm = np.clip(X_test_norm, -5, 5)
    
    # Synchronize all ranks before proceeding
    if is_distributed:
        dist.barrier()
        if rank == 0:
            print("✅ All ranks ready with data (loaded independently - no broadcast needed!)")
    
    # Create datasets
    train_dataset = MicrolensingDataset(X_train_norm, y_train)
    val_dataset = MicrolensingDataset(X_val_norm, y_val)
    test_dataset = MicrolensingDataset(X_test_norm, y_test)
    
    # Create data loaders with distributed samplers
    train_loader = create_data_loader(
        train_dataset, args.batch_size, is_train=True,
        is_distributed=is_distributed, num_workers=args.num_workers
    )
    val_loader = create_data_loader(
        val_dataset, args.batch_size * 2, is_train=False,
        is_distributed=is_distributed, num_workers=args.num_workers
    )
    test_loader = create_data_loader(
        test_dataset, args.batch_size * 2, is_train=False,
        is_distributed=is_distributed, num_workers=args.num_workers
    )
    
    # Create model
    if rank == 0:
        print("\nCreating Stable Transformer...")
    
    model = SimpleStableTransformer(
        n_points=1500,
        d_model=args.d_model,
        nhead=4,
        num_layers=args.num_layers,
        dim_ff=args.d_model * 4,
        dropout=0.2
    ).to(device)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    if rank == 0:
        param_count = count_parameters(model.module if is_distributed else model)
        print(f"Model: {param_count:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    
    # Create experiment directory (rank 0 only)
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path("../results") / f"{args.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExperiment directory: {exp_dir}")
        
        # Save configuration
        config = {
            'data': args.data,
            'epochs': args.epochs,
            'batch_size_per_gpu': args.batch_size,
            'total_batch_size': args.batch_size * world_size,
            'lr': args.lr,
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'grad_clip': args.grad_clip,
            'amp': args.amp,
            'world_size': world_size,
            'n_train': len(X_train_norm),
            'n_val': len(X_val_norm),
            'n_test': len(X_test_norm)
        }
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save normalizer
        normalizer.save(exp_dir / 'normalizer.pkl')
    else:
        exp_dir = None
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    if rank == 0:
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-"*40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.grad_clip, scaler=scaler, 
            rank=rank, world_size=world_size
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            rank=rank, world_size=world_size
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model (unwrap DDP if needed)
                model_to_save = model.module if is_distributed else model
                
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
    
    # Final evaluation
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        if (exp_dir / 'best_model.pt').exists():
            checkpoint = torch.load(exp_dir / 'best_model.pt', map_location=device)
            model_to_load = model.module if is_distributed else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device,
                rank=rank, world_size=world_size
            )
            
            results = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'best_val_acc': best_val_acc,
                'final_epoch': epoch + 1,
                'model_params': param_count,
                'world_size': world_size
            }
            
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
            
            if test_acc > 0.65:
                print("\n✅ SUCCESS! Model achieved good performance")
            elif test_acc > 0.55:
                print("\n⚠️ Model performance is moderate")
            else:
                print("\n❌ Model performance is poor")
            
            print(f"\n📁 Results saved to: {exp_dir}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
