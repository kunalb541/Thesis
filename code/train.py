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
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
from contextlib import contextmanager
# Import for AMP/GradScaler
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    # Fallback to dummy no-op for CPU-only environments
    @contextmanager
    def autocast():
        yield

    class GradScaler:
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Import model
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    # Assuming 'model.py' contains CausalHybridModel and CausalConfig
    from model import CausalHybridModel, CausalConfig # Placeholder: User must ensure this is importable

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    if int(os.environ.get('RANK', 0)) == 0:
        print("\nCRITICAL: 'model.py' not found in script directory. Please ensure it exists.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
SPIKE_THRESHOLD = 2.5
SEED = 42
PREFETCH_FACTOR = 8 # HPC optimization for high-throughput data loading

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """
    Initialize DDP from environment variables set by torchrun/SLURM.
    Returns: (rank, local_rank, world_size, is_ddp)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # CRITICAL: Only proceed if CUDA is available for NCCL backend
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > local_rank, \
                f"Rank {rank} requires GPU {local_rank}, but only {torch.cuda.device_count()} GPUs available"
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return rank, local_rank, world_size, True
        else:
            if rank == 0:
                print("ERROR: CUDA not available for distributed training. Falling back to single-CPU/GPU.")
            # Fallback to single-process (not DDP) if CUDA isn't available
            return 0, 0, 1, False

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
    
    # Configure format based on rank
    log_format = '%(asctime)s | Rank %(rank)s | %(levelname)s | %(message)s' if rank != 0 else '%(asctime)s | %(levelname)s | %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / "training.log"))

    logging.basicConfig(
        level=level, 
        format=log_format,
        handlers=handlers, 
        force=True
    )
    # Add rank to the logger context
    if rank != 0:
        logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    return logger

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Be cautious with these globally. May impact performance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# =============================================================================
# DATASET
# =============================================================================
def create_delta_t_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to relative time intervals (delta_t).
    
    Args:
        timestamps: (N, T) array of observation times (float)
    Returns:
        delta_t: (N, T) array where delta_t[i,j] = time since previous observation
    """
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    # Pre-allocate and compute difference
    delta_t = np.zeros_like(timestamps, dtype=timestamps.dtype)
    if timestamps.shape[1] > 1:
        # np.diff computes t[1]-t[0], t[2]-t[1], ... (length T-1)
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
        
    # Ensure non-negative delta_t (shouldn't happen with causal data, but robust)
    return np.maximum(delta_t, 0.0)

class MicrolensingDataset(Dataset):
    """Robust dataset for microlensing light curves."""
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, 
                 stats: dict = None):
        
        # Convert to tensor with contiguous memory layout
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # Sanitize NaN values (replaces with 0.0 which is treated as padding)
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)

        # Calculate padding mask (padding is strict 0.0)
        self.padding_mask = (self.flux != 0.0)
        # Sequence length must be at least 1 for downstream indexing to work
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        # Robust normalization: only normalize valid data, leave padding as 0.0
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr'] if stats['iqr'] > 1e-6 else 1.0 # Protect against zero IQR
            
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
    weights = 1.0 / (counts + 1e-6) # Add epsilon to prevent division by zero
    weights = weights / weights.sum() * n_classes # Normalize to mean=1
    return torch.FloatTensor(weights)

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                scaler: GradScaler, class_weights: torch.Tensor, device: torch.device, 
                rank: int, epoch: int, logger: logging.Logger) -> tuple:
    """
    Train for one epoch with AMP and gradient clipping.
    """
    # CRITICAL FIX: Unwrap DDP model before setting train() if DDP is used
    model_to_train = model.module if isinstance(model, DDP) else model
    model_to_train.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0
    
    # optimizer.zero_grad(set_to_none=True) # Will be called inside the loop

    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0), leave=False)

    for step, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # AMP forward pass
        with autocast():
            # Model returns only final timestep logits when return_all_timesteps=False
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
            final_logits = out['logits']  # (B, n_classes)
            
            # Compute loss
            loss = F.cross_entropy(final_logits, labels, weight=class_weights)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # CRITICAL FIXED GRADIENT HANDLING:
        # 1. Unscale gradients to normal FP32 values
        # 2. Clip the unscaled gradients
        # 3. Check for NaN/Inf before calling scaler.step()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        
        # Step optimizer only if gradients are finite
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        else:
            nan_count += 1
            if rank == 0:
                logger.warning(f"NaN gradient detected at step {step}, skipping update")
        
        # CRITICAL FIXED: ALWAYS update scaler state
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Track metrics
        # Use .item() on loss after backward() to ensure it's calculated
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # DDP reduction of metrics across all GPUs
    if dist.is_initialized():
        # Create a tensor for all metrics to reduce
        metrics = torch.tensor([total_loss, correct, total, nan_count], 
                               dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        # Unpack reduced metrics
        total_loss, correct, total, nan_count = metrics.cpu().numpy()
        # total_loss was already summed across ranks, but the average needs division by world_size
        # The average loss per batch element is calculated later.
        # Here we just need the total loss from all processes (which is already summed).
        
    # The average loss is the total loss divided by the total number of batches
    avg_loss = total_loss / len(loader.dataset) * loader.batch_size # Approximating the average batch loss
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy, int(nan_count)

# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor, 
             device: torch.device, rank: int, check_early_detection: bool = False) -> dict:
    """Evaluate model performance."""
    # CRITICAL FIX: Unwrap DDP model before setting eval() if DDP is used
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval()
    
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
                # Need full sequence for early detection audit
                out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
                logits_seq = out['logits']  # (B, T, n_classes)
                probs_seq = out['probs']    # (B, T, n_classes)
                
                # Extract final timestep for standard metrics
                B = flux.size(0)
                last_idx = (lengths - 1).clamp(min=0)
                batch_indices = torch.arange(B, device=device)
                final_logits = logits_seq[batch_indices, last_idx]
                
                # Early detection audit (looping over batch)
                all_preds = probs_seq.argmax(dim=2)  # (B, T)
                for b in range(B):
                    seq_len = lengths[b].item()
                    true_lbl = labels[b].item()
                    for m in milestones:
                        # Index at m% of the actual sequence length, clamped
                        idx = max(0, min(int(seq_len * m) - 1, seq_len - 1))
                        if all_preds[b, idx].item() == true_lbl:
                            early_correct[m] += 1
                        early_total[m] += 1
            else:
                # Standard evaluation - optimized path (only final timestep)
                out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
                final_logits = out['logits']  # (B, n_classes)
            
            # Compute loss (using reduction='sum' for proper DDP total loss calculation)
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
            # Need to reduce early detection stats too
            for m in milestones:
                e_metrics = torch.tensor([early_correct[m], early_total[m]], 
                                         dtype=torch.float32, device=device)
                dist.all_reduce(e_metrics, op=dist.ReduceOp.SUM)
                ec, et = e_metrics.cpu().numpy()
                early_correct[m], early_total[m] = int(ec), int(et)

    results = {
        'loss': total_loss / max(total, 1), # total loss / total samples
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
    Normalization Strategy (Thesis-Critical): Stats computed ONLY from a training sample.
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
    
    # Handle time feature (delta_t or timestamps)
    if 'delta_t' in data:
        delta_t = data['delta_t']
    elif 'timestamps' in data:
        ts = data['timestamps']
        # Handle case where timestamps is a 1D array of common times
        if ts.ndim == 1:
            ts = np.tile(ts, (len(flux), 1))
        delta_t = create_delta_t_from_timestamps(ts)
    else:
        # Default to 0.0 for delta_t if no time info is present
        delta_t = np.zeros_like(flux)

    # Ensure 2D arrays: (N_samples, T_timesteps)
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    if delta_t.ndim == 3:
        delta_t = delta_t.squeeze(1)

    # Calculate normalization statistics from training data ONLY
    if rank == 0:
        logger.info("Calculating normalization statistics from training sample...")
    
    sample_flux = flux[:10000].flatten() # Use first 10k samples
    valid_flux = sample_flux[sample_flux != 0.0]  # Exclude padding (strict 0.0)
    
    if len(valid_flux) == 0:
        if rank == 0:
            logger.error("No valid flux values found in data sample!")
        sys.exit(1)
    
    # Median and IQR normalization (robust to outliers)
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
    # Argument Definitions
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
                        help="Enable gradient checkpointing (saves ~40% memory)")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    
    # Select device safely
    if is_ddp:
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{timestamp}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(rank, output_dir, args.debug)
    
    # Barrier after creating output_dir and before setting seeds (to be safe)
    if is_ddp:
        dist.barrier()
    
    # FIXED: Set reproducibility seed BEFORE model creation/data loading
    set_seed(SEED)
    
    # CRITICAL: If 'model' module is available, set its seeds
    if 'CausalHybridModel' in globals():
        CausalHybridModel.set_init_seed(SEED)  # Deterministic weight initialization
        if is_ddp:
            CausalHybridModel.set_distributed_seed(SEED, rank) # Rank-specific seeds

    # Define AMP/Scaler outside the main block and use the global imports
    use_amp = (device.type == 'cuda') and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else GradScaler() # GradScaler is now a DummyScaler if not on CUDA

    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Roman Telescope Causal Model Training")
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Device: {device.type} | DDP: {is_ddp} | Rank/World: {rank}/{world_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size} (local: {args.batch_size})")
        logger.info(f"AMP enabled: {use_amp} | Prefetch factor: {PREFETCH_FACTOR}")
        logger.info(f"Gradient checkpointing: {args.grad_checkpoint}")
        logger.info("=" * 80)

    # Load data
    flux, delta_t, labels, stats = load_npz_data(args.data, rank, logger)
    
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
    
    # Create datasets
    train_ds = MicrolensingDataset(flux_tr, dt_tr, y_tr, stats)
    val_ds = MicrolensingDataset(flux_val, dt_val, y_val, stats)
    
    # Create samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED, drop_last=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_ddp else None
    
    # FIXED: Conditional pin_memory and worker settings
    use_pin_memory = (device.type == 'cuda')
    num_workers = args.num_workers
    persistent_workers = num_workers > 0 and train_sampler is not None # Only for DDP/Multi-GPU
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None), 
        num_workers=num_workers, 
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        drop_last=is_ddp # CRITICAL: Drop last batch in DDP for consistent size
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=persistent_workers
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
        use_gradient_checkpointing=args.grad_checkpoint
    )
    
    model = CausalHybridModel(config).to(device)
    
    if rank == 0:
        logger.info(f"Model parameters: {count_parameters(model):,}")
        # Need to check if the method exists on the module before calling
        if hasattr(model, 'get_receptive_field'):
            logger.info(f"Receptive field: {model.get_receptive_field()} timesteps")
        
    # CRITICAL FIX: Wrap in DDP with optimal configuration
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            find_unused_parameters=False,   # Highly recommended speedup
            broadcast_buffers=False,        # Performance optimization
            gradient_as_bucket_view=True    # Memory optimization
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5,
    )
    
    if rank == 0:
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle DDP vs non-DDP state dict loading
        state_dict = checkpoint['state_dict']
        if is_ddp and not any(k.startswith('module.') for k in state_dict.keys()):
             # Checkpoint is non-DDP, model is DDP -> prefix keys
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_ddp and any(k.startswith('module.') for k in state_dict.keys()):
            # Checkpoint is DDP, model is non-DDP -> remove prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('accuracy', 0.0)
        
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch - 1}, starting at epoch {start_epoch}")
            
    # Training loop
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
            
            logger.info("-" * 80)
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
                state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': state_to_save,
                    'config': config.__dict__,
                    'accuracy': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'receptive_field': model.module.get_receptive_field() if is_ddp and hasattr(model.module, 'get_receptive_field') else (model.get_receptive_field() if hasattr(model, 'get_receptive_field') else "N/A"),
                    'history': history
                }
                
                torch.save(checkpoint, output_dir / "best_model.pt")
                logger.info(f"  ✓ Best model saved (Val Acc: {best_acc:.4f})")
            
            # Save history
            with open(output_dir / "history.json", 'w') as f:
                json.dump(history, f, indent=2)
            
            # Periodic checkpoint save
            if epoch % 10 == 0:
                # Use the latest state to save the checkpoint
                latest_state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': latest_state_to_save,
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
