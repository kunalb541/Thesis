#!/usr/bin/env python3
"""
Multi-Node Distributed Training for Streaming Transformer

Fixed version with proper normalization to prevent data leakage.

Author: Kunal Bhatia
Version: 6.1
"""

import os
import json
import pickle
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config as CFG
from streaming_transformer import StreamingTransformer, count_parameters
from normalization import CausticPreservingNormalizer
from streaming_losses import CombinedLoss


class MicrolensingDataset(Dataset):
    """Dataset for microlensing light curves"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def setup_ddp():
    """Initialize DDP with flexible configuration"""
    
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Initialize process group if distributed
    if world_size > 1:
        if not dist.is_initialized():
            # Try to get master address
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = str(CFG.MASTER_PORT)
            
            backend = CFG.DDP_BACKEND if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method=CFG.INIT_METHOD)
            
            # Verify initialization
            assert dist.is_initialized(), "Failed to initialize process group"
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    
    return rank, world_size, local_rank, device, world_size > 1


def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_and_normalize_data(data_path, rank, world_size):
    """Load data and apply normalization with proper DDP handling"""
    
    is_main = (rank == 0)
    
    if is_main:
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        timestamps = data['timestamps']
        
        # Parse metadata
        meta = json.loads(str(data['meta_json']))
        
        # Ensure correct shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # Add channel dimension
        
        print(f"Data shape: {X.shape}, Labels: {y.shape}")
        
        # Create splits BEFORE normalization
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
        
        # Apply normalization AFTER splitting
        print("\nApplying caustic-preserving normalization...")
        normalizer = CausticPreservingNormalizer(pad_value=CFG.PAD_VALUE)
        
        # Fit ONLY on training data
        normalizer.fit(X_train)
        
        # Transform all splits
        X_train_norm = normalizer.transform(X_train)
        X_val_norm = normalizer.transform(X_val)
        X_test_norm = normalizer.transform(X_test)
        
        # Remove channel dimension for transformer
        X_train_norm = X_train_norm.squeeze(1)
        X_val_norm = X_val_norm.squeeze(1)
        X_test_norm = X_test_norm.squeeze(1)
        
        # Prepare normalizer parameters for broadcast
        norm_params = {
            'baseline_median': normalizer.baseline_median,
            'baseline_mad': normalizer.baseline_mad,
            'flux_min': normalizer.flux_min,
            'flux_max': normalizer.flux_max,
            'pad_value': normalizer.pad_value,
            'is_fitted': normalizer.is_fitted
        }
    
    # Broadcast data to all processes if distributed
    if world_size > 1:
        if is_main:
            data_dict = {
                'X_train': X_train_norm, 'y_train': y_train,
                'X_val': X_val_norm, 'y_val': y_val,
                'X_test': X_test_norm, 'y_test': y_test,
                'timestamps': timestamps, 'meta': meta,
                'norm_params': norm_params
            }
        else:
            data_dict = None
        
        # Broadcast
        data_list = [data_dict]
        dist.broadcast_object_list(data_list, src=0)
        data_dict = data_list[0]
        
        # Unpack
        X_train_norm = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val_norm = data_dict['X_val']
        y_val = data_dict['y_val']
        X_test_norm = data_dict['X_test']
        y_test = data_dict['y_test']
        timestamps = data_dict['timestamps']
        meta = data_dict['meta']
        norm_params = data_dict['norm_params']
        
        # Recreate normalizer on all ranks
        if not is_main:
            normalizer = CausticPreservingNormalizer()
            normalizer.baseline_median = norm_params['baseline_median']
            normalizer.baseline_mad = norm_params['baseline_mad']
            normalizer.flux_min = norm_params['flux_min']
            normalizer.flux_max = norm_params['flux_max']
            normalizer.pad_value = norm_params['pad_value']
            normalizer.is_fitted = norm_params['is_fitted']
    else:
        X_train_norm = X_train_norm
        y_train = y_train
        X_val_norm = X_val_norm
        y_val = y_val
        X_test_norm = X_test_norm
        y_test = y_test
    
    return (X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test, 
            timestamps, meta, normalizer)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Set epoch for distributed sampler
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    # Progress bar only on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(X, return_all_timesteps=True)
            loss = criterion(outputs, y, X)
        
        # Backward pass
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
        
        # Metrics
        total_loss += loss.item()
        
        # Get final predictions for accuracy
        final_outputs = model(X, return_all_timesteps=False)
        preds = final_outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
        
        # Update progress bar
        if not dist.is_initialized() or dist.get_rank() == 0:
            if batch_idx % 10 == 0:
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
        
        # Final predictions
        final_outputs = model(X, return_all_timesteps=False)
        preds = final_outputs['binary'].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Multi-node DDP Training')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--experiment_name', default='streaming', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=CFG.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=CFG.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=CFG.LEARNING_RATE)
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank, device, is_ddp = setup_ddp()
    is_main = (rank == 0)
    
    if is_main:
        print("="*60)
        print("MULTI-NODE DDP TRAINING v6.1")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
    
    # Load and normalize data
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     timestamps, meta, normalizer) = load_and_normalize_data(
        args.data, rank, world_size
    )
    
    # Create datasets
    train_dataset = MicrolensingDataset(X_train, y_train)
    val_dataset = MicrolensingDataset(X_val, y_val)
    test_dataset = MicrolensingDataset(X_test, y_test)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_ddp else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2,
        sampler=test_sampler, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = StreamingTransformer().to(device)
    
    if is_main:
        print(f"\nModel created with {count_parameters(model):,} parameters")
    
    # Wrap in DDP
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=CFG.FIND_UNUSED_PARAMS,
            broadcast_buffers=CFG.BROADCAST_BUFFERS
        )
    
    # Create loss and optimizer
    criterion = CombinedLoss(weights=CFG.LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=CFG.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    
    # Create experiment directory
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(CFG.RESULTS_DIR) / f"{args.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            'args': vars(args),
            'world_size': world_size,
            'model_params': count_parameters(model.module if is_ddp else model),
            'data_shape': list(X_train.shape),
            'timestamp': timestamp
        }
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save normalizer
        normalizer.save(exp_dir / 'normalizer.pkl')
        
        print(f"\nExperiment directory: {exp_dir}")
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Aggregate metrics across all processes
        if is_ddp:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        if is_main:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': (model.module if is_ddp else model).state_dict(),
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
    
    # Synchronize before test evaluation
    if is_ddp:
        dist.barrier()
    
    # Final evaluation on test set
    if is_main:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(exp_dir / 'best_model.pt', map_location=device)
        (model.module if is_ddp else model).load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    if is_ddp:
        metrics = torch.tensor([test_loss, test_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
        test_loss, test_acc = metrics.tolist()
    
    if is_main:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save results
        results = {
            'best_val_acc': best_val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'final_epoch': epoch + 1
        }
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Training complete! Results saved to {exp_dir}")
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()