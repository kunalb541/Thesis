#!/usr/bin/env python3
"""
Rugged Causal Hybrid Training Script 
ARCHITECTURE: Hybrid GRU-Transformer (Strictly Causal)
FEATURES:
- Rank-0 Only Logging (Cleaner Output)
- Correct AMP Logic (Unscale -> Clip -> Step)
- "Mischief-Proof" Causality Auditing
- Auto-Recovery on Loss Spikes
- Graceful DDP Shutdown

Author: Kunal Bhatia
Version: 1.0 
Date: December 2025
"""

import os
import sys
import argparse
import logging
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
from tqdm import tqdm
import warnings

# --- Import the Hardened Model ---
sys.path.insert(0, str(Path(__file__).parent))
try:
    from transformer import CausalHybridModel, CausalConfig
except ImportError:
    # Only print error on rank 0 to avoid spam
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCRITICAL: 'causal_hybrid_model.py' not found in script directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
CLIP_NORM = 1.0           # Strict clipping prevents Transformer exploding grads
DEFAULT_LR = 1e-4         # Safe starting LR
ACCUMULATE_STEPS = 4      # Effective batch size multiplier

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """Robust DDP initialization."""
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

# =============================================================================
# LOGGING SETUP (CLEAN OUTPUT)
# =============================================================================
def setup_logging(rank, debug_mode):
    """Configures logging to only output from Rank 0."""
    logger = logging.getLogger(__name__)
    
    # If not rank 0, set level to WARNING to suppress info/debug noise
    if rank == 0:
        level = logging.DEBUG if debug_mode else logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True # Overwrite any existing config
    )
    return logger

# =============================================================================
# DATASET
# =============================================================================
class MicrolensingDataset(Dataset):
    def __init__(self, flux, delta_t, labels, pad_value=0.0):
        self.flux = flux
        self.delta_t = delta_t
        self.labels = labels
        self.pad_value = pad_value
        
        # Calculate lengths assuming 0.0 is padding.
        is_data = (flux != pad_value)
        self.lengths = np.sum(is_data, axis=1).astype(np.int64)
        # Prevent zero-length errors (rare artifacts)
        self.lengths = np.maximum(self.lengths, 1) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.flux[idx], self.delta_t[idx], 
                self.lengths[idx], self.labels[idx])

def collate_fn(batch):
    flux, delta_t, lengths, labels = zip(*batch)
    return (torch.from_numpy(np.stack(flux)).float(),
            torch.from_numpy(np.stack(delta_t)).float(),
            torch.from_numpy(np.array(lengths)).long(),
            torch.from_numpy(np.array(labels)).long())

def compute_class_weights(labels, n_classes=3):
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

def detect_loss_spike(current_loss, loss_history, threshold=2.5):
    """Heuristic to detect if loss has exploded relative to recent history."""
    if len(loss_history) < 3: return False
    recent_avg = np.mean(loss_history[-3:])
    # Guard against divide by zero if loss is extremely small
    if recent_avg < 1e-6: return False
    return current_loss > (threshold * recent_avg)

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_epoch(model, loader, optimizer, scaler, class_weights, device, rank, epoch, logger):
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    accum_loss = 0.0
    nan_count = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    # Verbose Progress Bar (Only on Rank 0)
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)
    else:
        pbar = loader

    for step, (flux, delta_t, lengths, labels) in enumerate(pbar):
        flux, delta_t = flux.to(device, non_blocking=True), delta_t.to(device, non_blocking=True)
        lengths, labels = lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # --- Forward ---
        with autocast(enabled=True):
            # return_all_timesteps=False for training (Only want final prediction)
            outputs = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
            logits = outputs['logits']
            
            # Loss calculation
            loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
            norm_loss = loss / ACCUMULATE_STEPS

        # --- Backward ---
        scaler.scale(norm_loss).backward()
        accum_loss += loss.item()

        # --- Optimizer Step ---
        current_grad_norm = 0.0
        if (step + 1) % ACCUMULATE_STEPS == 0:
            # 1. Unscale gradients
            scaler.unscale_(optimizer)
            
            # 2. Check Grad Norm (The "Gate")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            current_grad_norm = grad_norm.item()

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
            else:
                nan_count += 1
                if rank == 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logger.debug(f"⚠️ NaN Gradient detected at step {step}. Skipping update.")

            optimizer.zero_grad(set_to_none=True)
            total_loss += accum_loss
            accum_loss = 0.0
            
            # Update TQDM bar (Only Rank 0)
            if rank == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss / max(total + labels.size(0), 1):.4f}", 
                    'gnorm': f"{current_grad_norm:.2f}",
                    'nans': nan_count
                })

        # --- Metrics ---
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP Synchronization
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, nan_count], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, nan_count = metrics.cpu().numpy()

    return total_loss / max(total, 1), correct / max(total, 1), int(nan_count)

# =============================================================================
# EVALUATION & CAUSALITY AUDIT
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, check_early_detection=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Anti-Cheating Metrics
    milestones = [0.2, 0.5, 0.8, 1.0]
    early_correct = {m: 0 for m in milestones}
    early_total = {m: 0 for m in milestones}

    if rank == 0:
        desc = "Evaluating (Audit)" if check_early_detection else "Evaluating"
        # tqdm iterator only for master
        iter_loader = tqdm(loader, desc=desc, leave=False)
    else:
        iter_loader = loader

    for flux, delta_t, lengths, labels in iter_loader:
        flux, delta_t = flux.to(device), delta_t.to(device)
        lengths, labels = lengths.to(device), labels.to(device)

        # Standard Eval: Only get final timestep
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
        logits = out['logits']
        
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # Causality Check: Get all timesteps
        if check_early_detection:
            out_all = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            all_preds = out_all['predictions'] # [B, SeqLen]

            for b in range(flux.size(0)):
                seq_len = lengths[b].item()
                true_lbl = labels[b].item()
                
                for m in milestones:
                    # SAFETY FIX: Ensure we don't index beyond the sequence
                    target_idx = min(int(seq_len * m), seq_len - 1)
                    target_idx = max(0, target_idx) # Safety floor

                    if all_preds[b, target_idx].item() == true_lbl:
                        early_correct[m] += 1
                    early_total[m] += 1

    # DDP Sync
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
        
        if check_early_detection:
            for m in milestones:
                e_metrics = torch.tensor([early_correct[m], early_total[m]], device=device)
                dist.all_reduce(e_metrics, op=dist.ReduceOp.SUM)
                ec, et = e_metrics.cpu().numpy()
                early_correct[m], early_total[m] = ec, et

    results = {
        'loss': total_loss / max(total, 1),
        'accuracy': correct / max(total, 1)
    }
    
    if check_early_detection:
        results['early_stats'] = {m: early_correct[m]/max(early_total[m], 1) for m in milestones}

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hardened Causal Hybrid Trainer")
    parser.add_argument('--data', required=True, help="Path to .npz file")
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    
    # Model Hyperparams
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    
    args = parser.parse_args()
    
    # Setup Distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Setup Logging (Rank 0 only)
    logger = setup_logging(rank, args.debug)
    
    try:
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"🚀 Starting Training | DDP: {world_size > 1} | Device: {device}")
            logger.info(f"⚙️  Params: d_model={args.d_model}, layers={args.n_layers}, clip={CLIP_NORM}")

        # Load Data
        if rank == 0: logger.info(f"Loading data from {args.data}...")
        try:
            raw_data = np.load(args.data)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

        # Split
        X_train, X_val, dt_train, dt_val, y_train, y_val = train_test_split(
            raw_data['flux'], raw_data['delta_t'], raw_data['labels'], 
            test_size=0.2, stratify=raw_data['labels'], random_state=42
        )

        train_ds = MicrolensingDataset(X_train, dt_train, y_train)
        val_ds = MicrolensingDataset(X_val, dt_val, y_val)

        train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
        
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None), collate_fn=collate_fn, 
            pin_memory=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size*2, collate_fn=collate_fn, 
            pin_memory=True, num_workers=4
        )

        # Initialize Model
        config = CausalConfig(
            d_model=args.d_model, n_heads=args.n_heads, 
            n_transformer_layers=args.n_layers, n_classes=3,
            max_len=X_train.shape[1]
        )
        model = CausalHybridModel(config).to(device)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Optimization
        class_weights = compute_class_weights(y_train).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Loop Vars
        best_acc = 0.0
        patience_counter = 0
        loss_history = []

        # --- MAIN LOOP ---
        for epoch in range(args.epochs):
            if train_sampler: train_sampler.set_epoch(epoch)
            
            # Train
            t_loss, t_acc, nans = train_epoch(
                model, train_loader, optimizer, scaler, class_weights, device, rank, epoch, logger
            )

            # Eval (Detailed check every 5 epochs)
            is_check_epoch = ((epoch + 1) % 5 == 0) or (epoch == args.epochs - 1)
            val_res = evaluate(model, val_loader, class_weights, device, rank, 
                               check_early_detection=is_check_epoch)
            
            loss_history.append(val_res['loss'])

            # Spike Detection & Healing
            if detect_loss_spike(val_res['loss'], loss_history):
                logger.warning("⚠️ Loss Spike Detected! Cutting LR by half.")
                for pg in optimizer.param_groups: pg['lr'] *= 0.5

            scheduler.step(val_res['accuracy'])

            # Logging (Rank 0 Only)
            if rank == 0:
                log_msg = (f"Ep {epoch+1:02d} | T.Loss: {t_loss:.4f} | T.Acc: {t_acc:.4f} "
                           f"| V.Acc: {val_res['accuracy']:.4f} | NaNs: {nans}")
                logger.info(log_msg)

                if is_check_epoch:
                    stats = val_res['early_stats']
                    logger.info(f"   [Audit] 20%: {stats[0.2]:.2f} | 50%: {stats[0.5]:.2f} | 100%: {stats[1.0]:.2f}")
                    
                    # Save Best
                    if val_res['accuracy'] > best_acc:
                        best_acc = val_res['accuracy']
                        patience_counter = 0
                        save_model = model.module if hasattr(model, 'module') else model
                        torch.save(save_model.state_dict(), f"{args.output_dir}/best_causal_model.pt")
                        logger.info("   💾 New Best Model Saved.")
                    else:
                        patience_counter += 1
                
                if patience_counter >= 30:
                    logger.info("🛑 Early stopping triggered.")
                    break

    finally:
        # Graceful Exit
        cleanup_distributed()
        if rank == 0:
            logger.info("👋 Training finished/terminated.")

if __name__ == '__main__':
    main()
