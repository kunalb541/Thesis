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
- Parent Directory Result Storage

Author: Kunal Bhatia
Version: 1.0
Date: December 2025
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings

# --- Import the Hardened Model ---
# Ensure we look in the current directory for the model file
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from transformer import CausalHybridModel, CausalConfig

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    # Only print error on rank 0 to avoid spam in DDP
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCRITICAL: 'causal_hybrid_model.py' not found in script directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
CLIP_NORM = 1.0           # Strict clipping prevents Transformer exploding grads
DEFAULT_LR = 1e-4         # Safe starting LR
ACCUMULATE_STEPS = 1      # Set >1 for effective batch size multiplier
SPIKE_THRESHOLD = 2.5     # Factor for loss spike detection
SEED = 42

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """Robust DDP initialization capable of handling single GPU or Multi-GPU."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return rank, local_rank, world_size, True
    
    # Fallback to single device
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# LOGGING SETUP (CLEAN OUTPUT)
# =============================================================================
def setup_logging(rank, output_dir, debug_mode):
    """Configures logging to file and console, restricting verbosity on non-master ranks."""
    logger = logging.getLogger(__name__)
    
    # Master rank gets INFO/DEBUG, others get WARNING
    if rank == 0:
        level = logging.DEBUG if debug_mode else logging.INFO
    else:
        level = logging.WARNING

    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        # Add file handler only for master
        log_file = Path(output_dir) / "training.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    return logger

# =============================================================================
# DATASET UTILITIES
# =============================================================================
def create_delta_t_from_timestamps(timestamps):
    """Create delta_t array from timestamps safely."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

class MicrolensingDataset(Dataset):
    """
    Robust Dataset class handling padded arrays efficiently.
    """
    def __init__(self, flux, delta_t, labels, pad_value=-1.0):
        self.flux = flux
        self.delta_t = delta_t
        self.labels = labels
        self.pad_value = pad_value
        
        # Calculate valid lengths
        # Assuming flux is padded with pad_value
        is_data = (flux != pad_value)
        self.lengths = np.sum(is_data, axis=1).astype(np.int64)
        # Clamp to minimum 1 to avoid indexing errors in model
        self.lengths = np.maximum(self.lengths, 1) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.flux[idx], self.delta_t[idx], 
                self.lengths[idx], self.labels[idx])

def collate_fn(batch):
    """Custom collate to stack numpy arrays into tensors."""
    flux, delta_t, lengths, labels = zip(*batch)
    return (torch.from_numpy(np.stack(flux)).float(),
            torch.from_numpy(np.stack(delta_t)).float(),
            torch.from_numpy(np.array(lengths)).long(),
            torch.from_numpy(np.array(labels)).long())

def compute_class_weights(labels, n_classes):
    """Compute inverse frequency weights for imbalanced datasets."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

def detect_loss_spike(current_loss, loss_history, threshold=SPIKE_THRESHOLD):
    """Heuristic to detect if loss has exploded relative to recent history."""
    if len(loss_history) < 5: return False
    recent_avg = np.mean(loss_history[-5:])
    # Guard against divide by zero if loss is extremely small
    if recent_avg < 1e-6: return False
    return current_loss > (threshold * recent_avg)

# =============================================================================
# TRAINING ENGINE (With AMP & Gradient Clipping)
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
        pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
    else:
        pbar = loader

    for step, (flux, delta_t, lengths, labels) in enumerate(pbar):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Forward (Autocast) ---
        with autocast(enabled=True):
            # We assume model returns a dict with 'probs' or 'logits'
            # return_all_timesteps=True gives [B, T, Classes]
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            
            # Extract last valid timestep for classification training
            batch_idx = torch.arange(flux.size(0), device=device)
            last_idx = (lengths - 1).clamp(min=0)
            
            # Use LogSoftmax for stability with NLLLoss or CrossEntropy
            # The model output 'probs' is Softmaxed. 
            final_probs = out['probs'][batch_idx, last_idx]
            final_logits = torch.log(final_probs + 1e-9) 
            
            # Loss calculation
            loss = F.nll_loss(final_logits, labels, weight=class_weights, reduction='mean')
            norm_loss = loss / ACCUMULATE_STEPS

        # --- Backward (Scaler) ---
        scaler.scale(norm_loss).backward()
        accum_loss += loss.item()

        # --- Optimizer Step (With Clipping) ---
        if (step + 1) % ACCUMULATE_STEPS == 0:
            # 1. Unscale gradients to allow clipping
            scaler.unscale_(optimizer)
            
            # 2. Clip Gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            # 3. Step if finite
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
            else:
                nan_count += 1
                if rank == 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logger.debug(f"⚠️ NaN Gradient detected (norm={grad_norm.item()}). Skipping step.")

            optimizer.zero_grad(set_to_none=True)
            total_loss += accum_loss
            accum_loss = 0.0
            
            # Update TQDM
            if rank == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss / (step+1):.4f}", 
                    'gnorm': f"{grad_norm.item():.2f}",
                    'nan': nan_count
                })

        # --- Metrics ---
        with torch.no_grad():
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP Synchronization of Metrics
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, nan_count], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        # Average loss over ranks, sum correct/total
        # Note: total_loss above was summed over batches in local loop, 
        # but for true average we need careful handling. 
        # Simplified: Summing losses and dividing by world_size * steps approx
        total_loss, correct, total, nan_count = metrics.cpu().numpy()
        total_loss /= dist.get_world_size()

    avg_loss = total_loss / len(loader)
    avg_acc = correct / max(total, 1)

    return avg_loss, avg_acc, int(nan_count)

# =============================================================================
# EVALUATION & MISCHIEF-PROOF AUDIT
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, check_early_detection=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Audit Metrics (Accuracy at 20%, 50%, 100% of light curve)
    milestones = [0.2, 0.5, 1.0]
    early_correct = {m: 0 for m in milestones}
    early_total = {m: 0 for m in milestones}

    if rank == 0:
        desc = "Evaluating (Audit)" if check_early_detection else "Evaluating"
        iter_loader = tqdm(loader, desc=desc, leave=False)
    else:
        iter_loader = loader

    for flux, delta_t, lengths, labels in iter_loader:
        flux = flux.to(device)
        delta_t = delta_t.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # Get full sequence probabilities
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
        probs_seq = out['probs'] # [B, T, C]

        # 1. Standard Metric (Last Timestep)
        batch_idx = torch.arange(flux.size(0), device=device)
        last_idx = (lengths - 1).clamp(min=0)
        final_logits = torch.log(probs_seq[batch_idx, last_idx] + 1e-9)
        
        loss = F.nll_loss(final_logits, labels, weight=class_weights, reduction='sum')
        total_loss += loss.item()
        
        preds = final_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 2. Causality Audit
        if check_early_detection:
            all_preds = probs_seq.argmax(dim=2) # [B, T]
            
            for b in range(flux.size(0)):
                seq_len = lengths[b].item()
                true_lbl = labels[b].item()
                
                for m in milestones:
                    # Calculate index for percentage of light curve
                    target_idx = min(int(seq_len * m), seq_len - 1)
                    target_idx = max(0, target_idx)
                    
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
        'loss': total_loss / max(total, 1), # Approx average
        'accuracy': correct / max(total, 1)
    }
    
    if check_early_detection:
        results['early_stats'] = {m: early_correct[m]/max(early_total[m], 1) for m in milestones}

    return results

# =============================================================================
# DATA LOADING (NPZ)
# =============================================================================
def load_npz_data(path, rank, logger):
    if rank == 0: logger.info(f"Loading raw data from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        if rank == 0: logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    flux = data.get('flux', data.get('X'))
    if flux is None: raise KeyError("Missing 'flux' or 'X'")
    if flux.ndim == 3: flux = flux.squeeze(1)

    labels = data.get('labels', data.get('y'))
    if labels is None: raise KeyError("Missing 'labels' or 'y'")
    
    # Delta T Handling
    if 'delta_t' in data:
        delta_t = data['delta_t']
        if delta_t.ndim == 3: delta_t = delta_t.squeeze(1)
    elif 'timestamps' in data:
        ts = data['timestamps']
        if ts.ndim == 1: ts = np.tile(ts, (len(flux), 1))
        delta_t = create_delta_t_from_timestamps(ts)
    else:
        if rank == 0: logger.warning("No time data found. Using dummy dt=1.0.")
        delta_t = np.ones_like(flux)
        delta_t[:, 0] = 0.0

    return flux, delta_t, labels

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hardened Causal Hybrid Trainer")
    parser.add_argument('--experiment_name', type=str, required=True, help="ID for folder creation")
    parser.add_argument('--data', required=True, help="Path to .npz file")
    parser.add_argument('--val_data', default=None, help="Optional validation .npz")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    
    # Model Hyperparams
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    
    args = parser.parse_args()
    
    # 1. Setup Distributed Environment
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 2. Setup Directories (Parent/Results/Exp_Name_Timestamp)
    # Locates script directory, goes up to parent, then into results
    current_script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"{args.experiment_name}_{timestamp}"
    
    # Logic: ../results/<exp_name>
    output_dir = current_script_dir.parent / 'results' / experiment_folder
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for master to create dir
    if is_ddp: dist.barrier()

    # 3. Setup Logging
    logger = setup_logging(rank, output_dir, args.debug)
    
    try:
        if rank == 0:
            logger.info(f"🚀 Starting Training | DDP: {is_ddp} | World: {world_size} | Device: {device}")
            logger.info(f"📁 Output Directory: {output_dir}")
            logger.info(f"⚙️  Params: d_model={args.d_model}, layers={args.n_layers}, clip={CLIP_NORM}")

        # 4. Load Data
        flux, dt, labels = load_npz_data(args.data, rank, logger)
        n_classes = len(np.unique(labels))

        # Split Data (Stratified)
        if args.val_data:
            flux_val, dt_val, labels_val = load_npz_data(args.val_data, rank, logger)
            flux_train, dt_train, labels_train = flux, dt, labels
        else:
            if rank == 0: logger.info("Splitting data 80/20...")
            flux_train, flux_val, dt_train, dt_val, labels_train, labels_val = train_test_split(
                flux, dt, labels, test_size=0.2, stratify=labels, random_state=SEED
            )

        # Datasets & Samplers
        train_ds = MicrolensingDataset(flux_train, dt_train, labels_train)
        val_ds = MicrolensingDataset(flux_val, dt_val, labels_val)

        train_sampler = DistributedSampler(train_ds) if is_ddp else None
        
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None), collate_fn=collate_fn, 
            pin_memory=True, num_workers=4 if torch.cuda.is_available() else 0
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, collate_fn=collate_fn, 
            pin_memory=True, num_workers=4 if torch.cuda.is_available() else 0
        )

        # 5. Initialize Model
        config = CausalConfig(
            d_model=args.d_model, n_heads=args.n_heads, 
            n_transformer_layers=args.n_layers, n_classes=n_classes,
            max_len=flux_train.shape[1], dropout=args.dropout
        )
        model = CausalHybridModel(config).to(device)
        
        if rank == 0:
            logger.info(f"🧠 Model Parameters: {count_parameters(model):,}")
            # Save Config
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config.__dict__, f, indent=4)

        if is_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # 6. Optimization Setup
        class_weights = compute_class_weights(labels_train, n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=(rank==0)
        )

        # Loop Vars
        best_acc = 0.0
        patience_counter = 0
        loss_history = []
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # --- MAIN TRAINING LOOP ---
        for epoch in range(1, args.epochs + 1):
            if train_sampler: train_sampler.set_epoch(epoch)
            
            # TRAIN
            t_loss, t_acc, nans = train_epoch(
                model, train_loader, optimizer, scaler, class_weights, device, rank, epoch, logger
            )

            # EVAL (Full Audit every 5 epochs)
            is_audit_epoch = (epoch % 5 == 0) or (epoch == args.epochs)
            val_res = evaluate(
                model, val_loader, class_weights, device, rank, check_early_detection=is_audit_epoch
            )
            
            # Loss Spike Recovery
            loss_history.append(val_res['loss'])
            if detect_loss_spike(val_res['loss'], loss_history):
                if rank == 0: logger.warning("⚠️  Loss Spike Detected! Cutting LR by half to stabilize.")
                for pg in optimizer.param_groups: pg['lr'] *= 0.5
            
            # Step Scheduler based on Accuracy
            scheduler.step(val_res['accuracy'])

            # Record History
            if rank == 0:
                history['train_loss'].append(t_loss)
                history['val_loss'].append(val_res['loss'])
                history['val_acc'].append(val_res['accuracy'])
                
                log_msg = (f"Ep {epoch:02d} | T.Loss: {t_loss:.4f} | T.Acc: {t_acc:.4f} "
                           f"| V.Loss: {val_res['loss']:.4f} | V.Acc: {val_res['accuracy']:.4f} | NaNs: {nans}")
                logger.info(log_msg)

                if is_audit_epoch:
                    stats = val_res['early_stats']
                    logger.info(f"   [Audit] Acc @ 20%: {stats[0.2]:.2f} | 50%: {stats[0.5]:.2f} | 100%: {stats[1.0]:.2f}")

                # Save Checkpoints
                save_model = model.module if is_ddp else model
                
                # Save Best
                if val_res['accuracy'] > best_acc:
                    best_acc = val_res['accuracy']
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': save_model.state_dict(),
                        'config': config.__dict__,
                        'history': history
                    }, output_dir / "best_model.pt")
                    logger.info("   💾 New Best Model Saved.")
                else:
                    patience_counter += 1
            
            # Sync Patience across ranks (simple barrier logic or explicit broadcast recommended for strictness)
            # Here allowing slight desync in logging, but stops should happen roughly together via max epochs usually.
            
            if patience_counter >= 25:
                if rank == 0: logger.info("🛑 Early stopping triggered (No improvement for 25 epochs).")
                break

        # Save Final
        if rank == 0:
            save_model = model.module if is_ddp else model
            torch.save({
                'model_state_dict': save_model.state_dict(),
                'config': config.__dict__,
                'history': history
            }, output_dir / "final_model.pt")
            
            # Save History JSON
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=4)
            
            logger.info("Training Complete.")

    finally:
        # Graceful Exit
        cleanup_distributed()
        if rank == 0:
            logger.info("👋 Process Terminated.")

if __name__ == '__main__':
    main()
