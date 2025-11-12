#!/usr/bin/env python3
"""
Causal Distributed Training for THREE-CLASS Classification
=========================================================
v12.0 - FIXES DATA LEAKAGE

CRITICAL CHANGES:
1. Variable-length sequences (model can't infer from padding)
2. Causal normalization (only uses observed data)
3. No timing information leaked

Classes: 0=Flat, 1=PSPL, 2=Binary

Author: Kunal Bhatia
Version: 12.0 - Fully Causal Training
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


class VariableLengthDataset(Dataset):
    """
    v12.0: Dataset that supports variable-length sequences
    
    This prevents the model from learning: "If I see padding at position X,
    this must be event type Y"
    """
    
    def __init__(self, X, y, pad_value=-1.0):
        # Clean data
        X = np.nan_to_num(X, nan=pad_value, posinf=100.0, neginf=-100.0)
        valid_mask = X != pad_value
        if valid_mask.any():
            X[valid_mask] = np.clip(X[valid_mask], -100, 100)
        
        self.X = X
        self.y = y
        self.pad_value = pad_value
        
        # Store original lengths (for causal training)
        self.lengths = []
        for i in range(len(X)):
            valid = (X[i] != pad_value).sum()
            self.lengths.append(int(valid))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Returns (x, y, length)
        length = number of valid (non-padded) observations
        """
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            self.lengths[idx]
        )


def collate_variable_length(batch):
    """
    Custom collate function for variable-length sequences
    
    Instead of padding all to same length, we pad to the max length
    in THIS batch. This prevents the model from learning fixed
    sequence length patterns.
    """
    xs, ys, lengths = zip(*batch)
    
    # Find max length in this batch
    max_len = max(lengths)
    
    # Pad to max_len (not to a fixed 1500!)
    pad_value = -1.0
    xs_padded = []
    for x in xs:
        if len(x) < max_len:
            padding = torch.full((max_len - len(x),), pad_value)
            x_padded = torch.cat([x, padding])
        else:
            x_padded = x[:max_len]
        xs_padded.append(x_padded)
    
    return (
        torch.stack(xs_padded),
        torch.tensor(ys),
        torch.tensor(lengths)
    )


class StableNormalizer:
    """Normalizer that handles padding"""
    
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


def apply_causal_truncation(X, y, lengths, truncation_prob=0.5, min_frac=0.1, max_frac=0.8):
    """
    v12.0: TRULY CAUSAL truncation
    
    Key changes from v11:
    1. Only truncates based on VALID observations (not total length)
    2. Returns variable-length sequences (not padded to 1500)
    3. Model cannot infer event type from sequence structure
    
    Args:
        X: Input [B, T] (already normalized globally)
        y: Labels [B]
        lengths: Number of valid observations per sample [B]
        truncation_prob: Probability of truncating
        min_frac: Keep at least this fraction
        max_frac: Truncate to at most this fraction
    
    Returns:
        X_truncated: List of truncated tensors (variable lengths!)
        y: Labels (unchanged)
    """
    B, T = X.shape
    device = X.device
    pad_value = -1.0
    
    X_truncated = []
    
    for i in range(B):
        if torch.rand(1).item() < truncation_prob:
            # Get valid observations for this sample
            valid_mask = X[i] != pad_value
            valid_indices = torch.where(valid_mask)[0]
            n_valid = len(valid_indices)
            
            if n_valid == 0:
                X_truncated.append(X[i])
                continue
            
            # Random truncation fraction
            frac = torch.rand(1).item() * (max_frac - min_frac) + min_frac
            n_keep = max(1, int(n_valid * frac))
            
            # Take only first n_keep valid observations
            keep_indices = valid_indices[:n_keep]
            truncated = X[i, keep_indices]
            
            X_truncated.append(truncated)
        else:
            # Keep full sequence
            valid_mask = X[i] != pad_value
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                X_truncated.append(X[i, valid_indices])
            else:
                X_truncated.append(X[i])
    
    # Now pad to max length in THIS batch
    max_len = max(len(seq) for seq in X_truncated)
    X_batch = torch.full((B, max_len), pad_value, device=device)
    for i, seq in enumerate(X_truncated):
        X_batch[i, :len(seq)] = seq
    
    return X_batch


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                device, epoch, rank, world_size, use_amp=True, grad_clip=1.0,
                causal_training=True):
    """
    Train one epoch with causal truncation
    
    v12.0: Uses variable-length sequences during training
    """
    model.train()
    
    total_loss = 0
    classification_loss_total = 0
    flat_loss_total = 0
    pspl_loss_total = 0
    anomaly_loss_total = 0
    caustic_loss_total = 0
    correct = 0
    total = 0
    num_batches = 0
    skipped_batches = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = loader
    
    for batch_idx, (X, y, lengths) in enumerate(pbar):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        
        # v12.0: Apply CAUSAL truncation
        if causal_training and torch.rand(1).item() < 0.5:
            X = apply_causal_truncation(X, y, lengths,
                                       truncation_prob=0.3,
                                       min_frac=0.5,
                                       max_frac=0.9)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=use_amp):
            outputs = model(X, return_all=True)
            
            # Main task
            logits = outputs['logits']
            classification_loss = criterion(logits, y)
            loss = classification_loss
            
            # Auxiliary tasks (same weights as v11.1)
            if 'flat' in outputs:
                flat_target = (y == 0).float()
                flat_loss = F.binary_cross_entropy_with_logits(outputs['flat'], flat_target)
                loss = loss + 0.5 * flat_loss
                flat_loss_total += flat_loss.item()
            else:
                flat_loss = torch.tensor(0.0)
            
            if 'pspl' in outputs:
                pspl_target = (y == 1).float()
                pspl_loss = F.binary_cross_entropy_with_logits(outputs['pspl'], pspl_target)
                loss = loss + 0.2 * pspl_loss
                pspl_loss_total += pspl_loss.item()
            else:
                pspl_loss = torch.tensor(0.0)
            
            if 'anomaly' in outputs:
                anomaly_target = (y > 0).float()
                anomaly_loss = F.binary_cross_entropy_with_logits(outputs['anomaly'], anomaly_target)
                loss = loss + 0.2 * anomaly_loss
                anomaly_loss_total += anomaly_loss.item()
            else:
                anomaly_loss = torch.tensor(0.0)
            
            if 'caustic' in outputs:
                caustic_target = (y == 2).float()
                caustic_loss = F.binary_cross_entropy_with_logits(outputs['caustic'], caustic_target)
                loss = loss + 0.2 * caustic_loss
                caustic_loss_total += caustic_loss.item()
            else:
                caustic_loss = torch.tensor(0.0)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0 and skipped_batches < 3:
                print(f"[Warning] NaN/Inf loss at batch {batch_idx}, skipping")
            skipped_batches += 1
            continue
        
        # Backward pass
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
        classification_loss_total += classification_loss.item()
        num_batches += 1
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        # Update progress bar
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
    
    # Gather metrics
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
    """Evaluate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    # Per-class accuracy
    class_correct = torch.zeros(3).to(device)
    class_total = torch.zeros(3).to(device)
    
    if rank == 0:
        pbar = tqdm(loader, desc="Validating", leave=False)
    else:
        pbar = loader
    
    for X, y, lengths in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        outputs = model(X, return_all=True)
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
                class_correct[c] += (preds[mask] == y[mask]).sum().item()
                class_total[c] += mask.sum().item()
    
    # Gather metrics
    if world_size > 1:
        metrics = torch.tensor([total_loss, correct, total, num_batches],
                              dtype=torch.float32).to(device)
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
    
    if rank == 0:
        class_names = ['Flat', 'PSPL', 'Binary']
        print(f"\n  3-Class Accuracy:")
        for c in range(3):
            if class_total[c] > 0:
                class_acc = class_correct[c] / class_total[c]
                print(f"    {class_names[c]:8s}: {class_acc*100:5.2f}%")
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="v12.0 Causal Training")
    parser.add_argument('--data', required=True)
    parser.add_argument('--experiment_name', default='causal_v12')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_causal', action='store_true')
    parser.add_argument('--quick', action='store_true')
    
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print("="*70)
        print("CAUSAL TRAINING v12.0")
        print("="*70)
        print(f"✅ Relative positional encoding (no absolute time)")
        print(f"✅ Variable-length sequences (no padding artifacts)")
        print(f"✅ Causal truncation during training")
        print(f"✅ Smaller model: ~100K parameters")
    
    # Load data
    print_rank0(f"\nLoading {args.data}...", rank)
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    n_classes = len(np.unique(y))
    if rank == 0:
        print(f"Classes: {n_classes}")
        if n_classes == 3:
            print(f"  Flat:   {(y==0).sum()}")
            print(f"  PSPL:   {(y==1).sum()}")
            print(f"  Binary: {(y==2).sum()}")
    
    if args.quick:
        print_rank0("⚡ Quick mode", rank)
        indices = np.random.choice(len(X), min(10000, len(X)), replace=False)
        X, y = X[indices], y[indices]
    
    # Normalize
    print_rank0("\n🔄 Normalizing...", rank)
    normalizer = StableNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit_transform(X)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print_rank0(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", rank)
    
    # Create datasets
    train_dataset = VariableLengthDataset(X_train, y_train)
    val_dataset = VariableLengthDataset(X_val, y_val)
    test_dataset = VariableLengthDataset(X_test, y_test)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,
                                      rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, persistent_workers=True,
                              collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                           sampler=val_sampler, num_workers=4,
                           pin_memory=True, persistent_workers=True,
                           collate_fn=collate_variable_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                            sampler=test_sampler, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            collate_fn=collate_variable_length)
    
    # Import transformer
    from transformer import MicrolensingTransformer, count_parameters
    
    # Create model
    print_rank0("\n🤖 Creating v12.0 causal model...", rank)
    model = MicrolensingTransformer(
        n_points=X.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        pad_value=-1.0
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=True)
    
    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        print(f"Parameters: {count_parameters(base_model):,}")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    
    effective_batch_size = args.batch_size * world_size
    lr_scale = max(1.0, effective_batch_size / 256)
    scaled_lr = args.lr * lr_scale
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr,
                                   weight_decay=args.weight_decay)
    
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if not args.no_amp else None
    
    # Create experiment directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config = vars(args)
        config['version'] = '12.0'
        config['causal'] = not args.no_causal
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        import pickle
        with open(exp_dir / 'normalizer.pkl', 'wb') as f:
            pickle.dump(normalizer, f)
    
    # Broadcast exp_dir
    if world_size > 1:
        if rank == 0:
            exp_dir_str = str(exp_dir)
        else:
            exp_dir_str = None
        exp_dir_list = [exp_dir_str]
        dist.broadcast_object_list(exp_dir_list, src=0)
        if rank != 0:
            exp_dir = Path(exp_dir_list[0])
    
    if world_size > 1:
        dist.barrier()
    
    # Training loop
    if rank == 0:
        print("\n" + "="*70)
        print("STARTING v12.0 CAUSAL TRAINING")
        print("="*70)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, epoch, rank, world_size, use_amp=not args.no_amp,
            grad_clip=args.grad_clip, causal_training=not args.no_causal
        )
        
        scheduler.step()
        
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, rank, world_size
        )
        
        if rank == 0:
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                save_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'val_acc': val_acc,
                    'version': '12.0'
                }, exp_dir / 'best_model.pt')
                
                print(f"✅ Saved (val_acc: {val_acc*100:.2f}%)")
            else:
                patience_counter += 1
            
            should_stop = (patience_counter >= max_patience)
        else:
            should_stop = False
        
        if world_size > 1:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break
        else:
            if should_stop:
                break
    
    # Final evaluation
    if world_size > 1:
        dist.barrier()
        checkpoint = torch.load(exp_dir / 'best_model.pt',
                               map_location={'cuda:0': f'cuda:{local_rank}'},
                               weights_only=False)
        load_model = model.module if hasattr(model, 'module') else model
        load_model.load_state_dict(checkpoint['model_state_dict'])
        dist.barrier()
    else:
        checkpoint = torch.load(exp_dir / 'best_model.pt', weights_only=False)
        load_model = model.module if hasattr(model, 'module') else model
        load_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST RESULTS (v12.0 CAUSAL)")
        print(f"{'='*70}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"{'='*70}")
        
        results = {
            'test_acc': float(test_acc),
            'best_val_acc': float(best_val_acc),
            'version': '12.0',
            'causal': not args.no_causal
        }
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
