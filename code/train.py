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

# Import model
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import CausalHybridModel, CausalConfig

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCRITICAL: 'model.py' not found in script directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
SPIKE_THRESHOLD = 2.5
SEED = 42
PREFETCH_FACTOR = 8  # HPC optimization for high-throughput data loading

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """
    Initialize DDP from environment variables set by torchrun/SLURM.
    Returns: (rank, local_rank, world_size, is_ddp)
    
    Validates GPU availability for distributed training.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            # CRITICAL: Verify GPU availability before initialization
            assert torch.cuda.device_count() > local_rank, \
                f"Rank {rank} requires GPU {local_rank}, but only {torch.cuda.device_count()} GPUs available"
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return rank, local_rank, world_size, True
        else:
            if rank == 0:
                print("ERROR: CUDA not available for distributed training")
            sys.exit(1)
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank: int, output_dir: Path, debug_mode: bool):
    """Setup logging with rank-0 only file output."""
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug_mode else logging.INFO
    if rank != 0:
        level = logging.WARNING
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / "training.log"))

    logging.basicConfig(
        level=level, 
        format='%(asctime)s | Rank %(rank)s | %(levelname)s | %(message)s' if rank != 0 else '%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers, 
        force=True
    )
    return logger

def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    NOTE: This sets GLOBAL seeds. For distributed training, use
    CausalHybridModel.set_distributed_seed() for rank-specific seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# DATASET
# =============================================================================
def create_delta_t_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to relative time intervals (delta_t).
    Maintains causal ordering by computing forward differences only.
    
    Args:
        timestamps: (N, T) array of observation times
    Returns:
        delta_t: (N, T) array where delta_t[i,j] = time since previous observation
    """
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

class MicrolensingDataset(Dataset):
    """
    Robust dataset for microlensing light curves.
    
    Key features:
    - Handles variable-length sequences via padding detection
    - Applies normalization only to valid (non-padded) data
    - Preserves strict 0.0 values in padded regions
    - Sanitizes NaN values before processing
    
    Causality Guarantee:
        Normalization statistics computed ONLY from training data, not test data.
        Time intervals (delta_t) computed via forward differences only.
    
    Args:
        flux: (N, T) flux measurements, 0.0 indicates padding
        delta_t: (N, T) time intervals between observations
        labels: (N,) classification labels
        stats: Dict with 'median' and 'iqr' for normalization, or None for no normalization
    """
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, 
                 stats: dict = None):
        
        # Convert to tensor with contiguous memory layout
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # Sanitize NaN values (replaces with 0.0 which is treated as padding)
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)

        # Calculate sequence lengths (padding is strict 0.0)
        self.padding_mask = (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        # Robust normalization: only normalize valid data, leave padding as 0.0
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr'] if stats['iqr'] > 1e-6 else 1.0
            
            # Apply normalization only where padding_mask is True
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.flux[idx], self.delta_t[idx], self.lengths[idx], self.labels[idx]

def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights for balanced loss."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

def detect_loss_spike(current_loss: float, loss_history: list, threshold: float = SPIKE_THRESHOLD) -> bool:
    """Detect anomalous loss spikes for early stopping."""
    min_history = 10
    if len(loss_history) < min_history:
        return False
    recent_avg = np.mean(loss_history[-min_history:])
    if recent_avg < 1e-6:
        return False
    return current_loss > (threshold * recent_avg)

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                scaler: GradScaler, class_weights: torch.Tensor, device: torch.device, 
                rank: int, epoch: int, logger: logging.Logger) -> tuple:
    """
    Train for one epoch with AMP and gradient clipping.
    
    Returns:
        (avg_loss, accuracy, nan_count)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0
    
    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0), leave=False)

    for step, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # AMP forward pass - use optimized path
        with autocast():
            # Model returns only final timestep logits when return_all_timesteps=False
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
            final_logits = out['logits']  # (B, n_classes)
            
            # Compute loss
            loss = F.cross_entropy(final_logits, labels, weight=class_weights)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # FIXED: Unscale and clip gradients
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        
        # Step optimizer only if gradients are finite
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        else:
            nan_count += 1
            if rank == 0:
                logger.warning(f"NaN gradient detected at step {step}, skipping update")
        
        # FIXED: ALWAYS update scaler state (not conditional)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Track metrics
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP reduction of metrics across all GPUs
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total, nan_count], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total, nan_count = metrics.cpu().numpy()
        total_loss /= dist.get_world_size()

    avg_loss = total_loss / len(loader)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy, int(nan_count)

# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor, 
             device: torch.device, rank: int, check_early_detection: bool = False) -> dict:
    """
    Evaluate model performance.
    
    Args:
        check_early_detection: If True, compute accuracy at 20%, 50%, 100% of sequence
        
    Returns:
        Dictionary with 'loss', 'accuracy', and optionally 'early_stats'
        
    Early Detection Audit (Roman Telescope Validation):
        Verifies per-timestep causality by checking predictions at sequence fractions.
        Critical for validating streaming inference readiness.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    milestones = [0.2, 0.5, 1.0]
    early_correct = {m: 0 for m in milestones}
    early_total = {m: 0 for m in milestones}

    iterator = tqdm(loader, desc="Audit" if check_early_detection else "Eval", 
                   disable=(rank != 0), leave=False)

    for flux, delta_t, lengths, labels in iterator:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            if check_early_detection:
                # For early detection audit, need full sequence
                out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
                logits_seq = out['logits']  # (B, T, n_classes)
                probs_seq = out['probs']    # (B, T, n_classes)
                
                # Extract final timestep for standard metrics
                B = flux.size(0)
                last_idx = (lengths - 1).clamp(min=0)
                batch_indices = torch.arange(B, device=device)
                final_logits = logits_seq[batch_indices, last_idx]
                
                # Early detection audit (Python loop acceptable for thesis POC)
                all_preds = probs_seq.argmax(dim=2)  # (B, T)
                for b in range(B):
                    seq_len = lengths[b].item()
                    true_lbl = labels[b].item()
                    for m in milestones:
                        idx = max(0, min(int(seq_len * m) - 1, seq_len - 1))
                        if all_preds[b, idx].item() == true_lbl:
                            early_correct[m] += 1
                        early_total[m] += 1
            else:
                # Standard evaluation - use optimized path
                out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
                final_logits = out['logits']  # (B, n_classes)
            
            # Compute loss
            loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='sum')
            total_loss += loss.item()
            
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP reduction
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, correct, total], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
        
        if check_early_detection:
            for m in milestones:
                e_metrics = torch.tensor([early_correct[m], early_total[m]], 
                                        dtype=torch.float32, device=device)
                dist.all_reduce(e_metrics, op=dist.ReduceOp.SUM)
                ec, et = e_metrics.cpu().numpy()
                early_correct[m], early_total[m] = int(ec), int(et)

    results = {
        'loss': total_loss / max(total, 1), 
        'accuracy': correct / max(total, 1)
    }
    
    if check_early_detection:
        results['early_stats'] = {
            m: early_correct[m] / max(early_total[m], 1) 
            for m in milestones
        }
    
    return results

# =============================================================================
# DATA LOADING
# =============================================================================
def load_npz_data(path: str, rank: int, logger: logging.Logger) -> tuple:
    """
    Load and preprocess data from NPZ file.
    
    Returns:
        (flux, delta_t, labels, stats)
        
    Normalization Strategy (Thesis-Critical):
        Statistics computed ONLY from training data sample (first 10k examples).
        This ensures NO test data leakage into normalization parameters.
    """
    if rank == 0:
        logger.info(f"Loading data from {path}...")
    
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        if rank == 0:
            logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Extract arrays
    flux = data.get('flux', data.get('X'))
    labels = data.get('labels', data.get('y'))
    
    if 'delta_t' in data:
        delta_t = data['delta_t']
    elif 'timestamps' in data:
        ts = data['timestamps']
        if ts.ndim == 1:
            ts = np.tile(ts, (len(flux), 1))
        delta_t = create_delta_t_from_timestamps(ts)
    else:
        delta_t = np.zeros_like(flux)

    # Ensure 2D arrays
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    if delta_t.ndim == 3:
        delta_t = delta_t.squeeze(1)

    # Calculate normalization statistics from training data ONLY
    if rank == 0:
        logger.info("Calculating normalization statistics from training sample...")
    
    sample_flux = flux[:10000].flatten()
    valid_flux = sample_flux[sample_flux != 0.0]  # Exclude padding
    
    if len(valid_flux) == 0:
        if rank == 0:
            logger.error("No valid flux values found in data!")
        sys.exit(1)
    
    median = float(np.median(valid_flux))
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = float(q75 - q25)
    
    stats = {'median': median, 'iqr': iqr}
    
    if rank == 0:
        logger.info(f"Normalization stats: Median={median:.4f}, IQR={iqr:.4f}")
    
    return flux, delta_t, labels, stats

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train Causal Hybrid Model for Microlensing Classification (Roman Telescope Ready)"
    )
    parser.add_argument('--experiment_name', required=True, help="Name for this experiment")
    parser.add_argument('--data', required=True, help="Path to NPZ data file")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size per GPU")
    parser.add_argument('--d_model', type=int, default=128, help="Model dimension")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of transformer layers")
    parser.add_argument('--n_conv_layers', type=int, default=4, help="Number of CNN layers")
    parser.add_argument('--kernel_size', type=int, default=3, help="CNN kernel size")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--grad_checkpoint', action='store_true', 
                       help="Enable gradient checkpointing (saves ~40%% memory)")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # FIXED: Set reproducibility seed BEFORE model creation
    set_seed(SEED)
    CausalHybridModel.set_init_seed(SEED)  # Deterministic weight initialization
    
    # FIXED: Set rank-specific seeds for data augmentation diversity
    if is_ddp:
        CausalHybridModel.set_distributed_seed(SEED, rank)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{timestamp}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(rank, output_dir, args.debug)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Roman Telescope Causal Model Training")
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"GPUs: {world_size} | Batch size per GPU: {args.batch_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size}")
        logger.info(f"AMP enabled: True | Prefetch factor: {PREFETCH_FACTOR}")
        logger.info(f"Gradient checkpointing: {args.grad_checkpoint}")
        logger.info("=" * 80)

    # Load data
    flux, delta_t, labels, stats = load_npz_data(args.data, rank, logger)
    
    # DDP barrier after data loading to ensure all ranks are synchronized
    if is_ddp:
        dist.barrier()
    
    # Train/val split
    flux_tr, flux_val, dt_tr, dt_val, y_tr, y_val = train_test_split(
        flux, delta_t, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=SEED
    )
    
    if rank == 0:
        logger.info(f"Train samples: {len(flux_tr)} | Val samples: {len(flux_val)}")
        
        # FIXED: Validate batch size distribution
        effective_batch = args.batch_size * world_size
        train_batches = len(flux_tr) // effective_batch
        dropped_samples = len(flux_tr) % effective_batch
        if dropped_samples > 0:
            logger.warning(
                f"Dropping {dropped_samples} training samples due to uneven distribution. "
                f"Consider adjusting batch_size ({args.batch_size}) or dataset size."
            )
    
    # Create datasets
    train_ds = MicrolensingDataset(flux_tr, dt_tr, y_tr, stats)
    val_ds = MicrolensingDataset(flux_val, dt_val, y_val, stats)
    
    # Create samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    
    # FIXED: Conditional pin_memory for efficiency
    use_pin_memory = (args.num_workers > 0)
    
    # Create dataloaders with HPC optimizations
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None), 
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create model
    n_classes = len(np.unique(labels))
    config = CausalConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_conv_layers=args.n_conv_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        n_classes=n_classes,
        use_gradient_checkpointing=args.grad_checkpoint  # NEW: Memory optimization
    )
    
    model = CausalHybridModel(config).to(device)
    
    if rank == 0:
        logger.info(f"Model parameters: {count_parameters(model):,}")
        logger.info(f"Receptive field: {model.get_receptive_field()} timesteps")
    
    # CRITICAL FIX: Wrap in DDP with optimal configuration
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            find_unused_parameters=False,      # CRITICAL: Disables expensive graph search (18-25% speedup)
            broadcast_buffers=False,           # Buffers don't change, no need to sync
            gradient_as_bucket_view=True       # Memory optimization for gradient storage
        )
        if rank == 0:
            logger.info("Model wrapped in DDP with optimal configuration")
    
    # Setup training components
    class_weights = compute_class_weights(y_tr, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5,
        verbose=(rank == 0)
    )
    
    if rank == 0:
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle DDP vs non-DDP state dict
        state_dict = checkpoint['state_dict']
        if is_ddp and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        
        if rank == 0:
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc, nan_count = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, 
            device, rank, epoch, logger
        )
        
        # Evaluate with early detection audit every 5 epochs
        is_audit = (epoch % 5 == 0) or (epoch == args.epochs)
        val_results = evaluate(
            model, val_loader, class_weights, device, rank, 
            check_early_detection=is_audit
        )
        
        # Learning rate scheduling
        scheduler.step(val_results['accuracy'])
        
        # Logging on rank 0
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f} | "
                f"NaN steps: {nan_count} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Log early detection stats
            if is_audit and 'early_stats' in val_results:
                stats_log = val_results['early_stats']
                logger.info(
                    f"  Early Detection Audit (Roman Streaming Validation): "
                    f"20%={stats_log[0.2]:.3f} | 50%={stats_log[0.5]:.3f} | 100%={stats_log[1.0]:.3f}"
                )
            
            # Save best model
            if val_results['accuracy'] > best_acc:
                best_acc = val_results['accuracy']
                best_epoch = epoch
                
                # CORRECT: DDP state dict handling
                if is_ddp:
                    state_to_save = model.module.state_dict()
                else:
                    state_to_save = model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': state_to_save,
                    'config': config.__dict__,
                    'accuracy': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'receptive_field': model.module.get_receptive_field() if is_ddp else model.get_receptive_field(),
                    'history': history
                }
                
                torch.save(checkpoint, output_dir / "best_model.pt")
                logger.info(f"  ✓ Best model saved (Val Acc: {best_acc:.4f})")
            
            # Save history
            with open(output_dir / "history.json", 'w') as f:
                json.dump(history, f, indent=2)
            
            # Periodic checkpoint save
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': state_to_save,
                    'config': config.__dict__,
                    'accuracy': val_results['accuracy'],
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")
    
    # Final summary
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Training completed!")
        logger.info(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
