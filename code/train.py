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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings

# --- Import the Hardened Model ---
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import CausalHybridModel, CausalConfig

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCRITICAL: 'transformer.py' not found in script directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4 # Slightly higher for AMP
SPIKE_THRESHOLD = 2.5
SEED = 42

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return rank, local_rank, world_size, True
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

def setup_logging(rank, output_dir, debug_mode):
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug_mode else logging.INFO
    if rank != 0: level = logging.WARNING
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(Path(output_dir) / "training.log"))

    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s', handlers=handlers, force=True)
    return logger

# =============================================================================
# ROBUST DATASET (GOD MODE FIX)
# =============================================================================
def create_delta_t_from_timestamps(timestamps):
    if timestamps.ndim == 1: timestamps = timestamps[np.newaxis, :]
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

class MicrolensingDataset(Dataset):
    """
    Robust Dataset: Handles Normalization internally to protect Padding.
    """
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, 
                 stats: dict = None):
        
        # 1. Convert to Tensor
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # 2. Sanitize
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)

        # 3. Calculate Lengths (Strict 0.0 is padding)
        # We use a float mask to avoid boolean issues later
        self.padding_mask = (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        # 4. Robust Normalization (The Anti-Ghost Fix)
        # Only normalize VALID data. Leave padding as strict 0.0.
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr'] if stats['iqr'] > 1e-6 else 1.0
            
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, device=self.flux.device)
            )

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.flux[idx], self.delta_t[idx], self.lengths[idx], self.labels[idx]

def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

def detect_loss_spike(current_loss, loss_history, threshold=SPIKE_THRESHOLD):
    min_history = 10
    if len(loss_history) < min_history: return False
    recent_avg = np.mean(loss_history[-min_history:])
    if recent_avg < 1e-6: return False
    return current_loss > (threshold * recent_avg)

# =============================================================================
# TRAINING ENGINE (AMP ENABLED)
# =============================================================================
def train_epoch(model, loader, optimizer, scaler, class_weights, device, rank, epoch, logger):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0
    
    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc=f"Ep {epoch}", disable=(rank!=0), leave=False)

    for step, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux, delta_t = flux.to(device, non_blocking=True), delta_t.to(device, non_blocking=True)
        lengths, labels = lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # AMP Context
        with autocast():
            # return_all_timesteps=True gives [B, T, C] logits
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            
            # Extract last valid timestep logits
            B = flux.size(0)
            last_idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, out['logits'].size(2))
            final_logits = out['logits'].gather(1, last_idx).squeeze(1)
            
            # Stable Cross Entropy on Logits
            loss = F.cross_entropy(final_logits, labels, weight=class_weights)

        # Scaler Logic
        scaler.scale(loss).backward()
        
        # Unscale & Clip
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            scaler.update()
        else:
            nan_count += 1
        
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP Reduction
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, nan_count], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, nan_count = metrics.cpu().numpy()
        total_loss /= dist.get_world_size() # Avg loss

    return total_loss / len(loader), correct / max(total, 1), int(nan_count)

# =============================================================================
# AUDIT EVALUATION (Early Detection Checks)
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, check_early_detection=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    milestones = [0.2, 0.5, 1.0]
    early_correct = {m: 0 for m in milestones}
    early_total = {m: 0 for m in milestones}

    iterator = tqdm(loader, desc="Audit" if check_early_detection else "Eval", disable=(rank!=0), leave=False)

    for flux, delta_t, lengths, labels in iterator:
        flux, delta_t = flux.to(device), delta_t.to(device)
        lengths, labels = lengths.to(device), labels.to(device)

        with autocast():
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            # Use Logits for Loss
            probs_seq = out['probs'] 
            logits_seq = out['logits']

            # Standard Metric (Last Timestep)
            B = flux.size(0)
            last_idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, logits_seq.size(2))
            final_logits = logits_seq.gather(1, last_idx).squeeze(1)
            
            loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='sum')
            total_loss += loss.item()
            
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Audit: Check Early Timesteps
            if check_early_detection:
                all_preds = probs_seq.argmax(dim=2) # [B, T]
                for b in range(B):
                    seq_len = lengths[b].item()
                    true_lbl = labels[b].item()
                    for m in milestones:
                        idx = max(0, min(int(seq_len * m), seq_len - 1))
                        if all_preds[b, idx].item() == true_lbl:
                            early_correct[m] += 1
                        early_total[m] += 1

    # DDP Reduction
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

    results = {'loss': total_loss / max(total, 1), 'accuracy': correct / max(total, 1)}
    if check_early_detection:
        results['early_stats'] = {m: early_correct[m]/max(early_total[m], 1) for m in milestones}
    return results

# =============================================================================
# DATA LOADING
# =============================================================================
def load_npz_data(path, rank, logger):
    if rank == 0: logger.info(f"Loading data from {path}...")
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        if rank == 0: logger.error(f"Failed to load: {e}"); sys.exit(1)

    flux = data.get('flux', data.get('X'))
    labels = data.get('labels', data.get('y'))
    if 'delta_t' in data: delta_t = data['delta_t']
    elif 'timestamps' in data: 
        ts = data['timestamps']
        delta_t = create_delta_t_from_timestamps(ts if ts.ndim > 1 else np.tile(ts, (len(flux), 1)))
    else: delta_t = np.zeros_like(flux)

    if flux.ndim == 3: flux = flux.squeeze(1)
    if delta_t.ndim == 3: delta_t = delta_t.squeeze(1)

    # Calculate Stats for Robust Normalization (Train Set Estimate)
    # We do a quick estimate on the first 10k samples to save time
    if rank == 0: logger.info("Calculating normalization stats...")
    sample_flux = flux[:10000].flatten()
    valid_flux = sample_flux[sample_flux != 0] # Ignore padding
    median = np.median(valid_flux)
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = q75 - q25
    stats = {'median': median, 'iqr': iqr}
    if rank == 0: logger.info(f"Stats: Median={median:.4f}, IQR={iqr:.4f}")
    
    return flux, delta_t, labels, stats

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Directory Setup
    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0: output_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp: dist.barrier()
    
    logger = setup_logging(rank, output_dir, args.debug)
    if rank == 0: logger.info(f"🚀 God Mode Training | GPUs: {world_size} | AMP: True")

    # Data Loading
    flux, dt, labels, stats = load_npz_data(args.data, rank, logger)
    
    flux_tr, flux_val, dt_tr, dt_val, y_tr, y_val = train_test_split(
        flux, dt, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    
    # Robust Datasets
    train_ds = MicrolensingDataset(flux_tr, dt_tr, y_tr, stats)
    val_ds = MicrolensingDataset(flux_val, dt_val, y_val, stats)
    
    train_sampler = DistributedSampler(train_ds) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    n_classes = len(np.unique(labels))
    config = CausalConfig(d_model=args.d_model, n_heads=args.n_heads, n_transformer_layers=args.n_layers, n_classes=n_classes)
    model = CausalHybridModel(config).to(device)
    
    if is_ddp: model = DDP(model, device_ids=[local_rank])
    
    # Optimizer & Scaler
    class_weights = compute_class_weights(y_tr, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
    scaler = GradScaler() # AMP
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, args.epochs + 1):
        if is_ddp: train_sampler.set_epoch(epoch)
        
        t_loss, t_acc, nans = train_epoch(model, train_loader, optimizer, scaler, class_weights, device, rank, epoch, logger)
        
        is_audit = (epoch % 5 == 0) or (epoch == args.epochs)
        val_res = evaluate(model, val_loader, class_weights, device, rank, check_early_detection=is_audit)
        
        scheduler.step(val_res['accuracy'])
        
        if rank == 0:
            history['train_loss'].append(t_loss)
            logger.info(f"Ep {epoch:02d} | T.Loss: {t_loss:.4f} | V.Acc: {val_res['accuracy']:.4f} | NaNs: {nans}")
            
            if is_audit: 
                s = val_res['early_stats']
                logger.info(f"   [Audit] Acc @ 20%: {s[0.2]:.2f} | 100%: {s[1.0]:.2f}")
            
            if val_res['accuracy'] > best_acc:
                best_acc = val_res['accuracy']
                save_model = model.module if is_ddp else model
                torch.save({'state_dict': save_model.state_dict(), 'config': config.__dict__}, output_dir / "best_model.pt")
                logger.info("   💾 Best Model Saved.")

    cleanup_distributed()

if __name__ == '__main__':
    main()
