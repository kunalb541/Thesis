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

# =============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
SEED = 42
PREFETCH_FACTOR = 8

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """
    Initialize distributed training environment.
    
    CRITICAL FIX: Added deterministic settings for reproducibility.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > local_rank, \
                f"Rank {rank} requires GPU {local_rank}, but only {torch.cuda.device_count()} GPUs available"
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            
            # CRITICAL FIX: Enable deterministic operations for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            return rank, local_rank, world_size, True
        else:
            if rank == 0:
                print("ERROR: CUDA not available for distributed training.")
            return 0, 0, 1, False
    
    # Single GPU mode - also set deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank: int, output_dir: Path, debug_mode: bool):
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug_mode else logging.INFO
    if rank != 0:
        level = logging.WARNING
    
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
    if rank != 0:
        logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    return logger

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# =============================================================================
# DATASET
# =============================================================================
def create_delta_t_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Create time intervals from timestamps.
    
    NOTE: For strict causality, use pre-computed delta_t from simulate.py instead.
    This function is only for backwards compatibility.
    """
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    delta_t = np.zeros_like(timestamps, dtype=timestamps.dtype)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

class MicrolensingDataset(Dataset):
    """
    PyTorch Dataset for microlensing light curves.
    
    Handles:
    - NaN replacement with 0.0
    - Robust normalization (median/IQR)
    - Sequence length computation from padding
    
    Args:
        flux: (N, T) flux measurements
        delta_t: (N, T) time intervals
        labels: (N,) class labels
        stats: Optional normalization statistics dict
    """
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, 
                 stats: dict = None):
        
        # Convert to contiguous tensors
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # Replace NaNs with 0.0
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)

        # Compute padding mask and lengths
        self.padding_mask = (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        # Apply normalization if stats provided
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr'] if stats['iqr'] > 1e-6 else 1.0
            
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
    """Compute inverse frequency class weights for balanced training."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                scaler, class_weights: torch.Tensor, device: torch.device, 
                rank: int, epoch: int, logger: logging.Logger) -> tuple:
    """
    Train for one epoch.
    
    CRITICAL FIX: Using reduction='mean' instead of 'sum' for proper DDP averaging.
    """
    model_to_train = model.module if isinstance(model, DDP) else model
    model_to_train.train()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    nan_count = 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0), leave=False)

    for step, (flux, delta_t, lengths, labels) in enumerate(iterator):
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
        final_logits = out['logits']
        
        # CRITICAL FIX: Use reduction='mean' for proper DDP averaging
        # Previously used 'sum' which caused loss scaling by world_size
        loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='mean')
        batch_samples = labels.size(0)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # CRITICAL FIX: Unscale gradients BEFORE checking/clipping
        scaler.unscale_(optimizer)
        
        # Check for NaN gradients (catastrophic for 40-GPU training)
        all_finite = True
        for param in model.parameters():
            if param.grad is not None:
                grad_finite = torch.isfinite(param.grad).all()
                if not grad_finite:
                    param.grad.zero_()  # Zero out NaN gradients
                    all_finite = False
        
        # Clip gradients if they're finite
        if all_finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                nan_count += 1
        else:
            nan_count += 1
        
        # CRITICAL: Update scaler even if optimizer wasn't stepped
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Track metrics - multiply by batch_samples since we used mean reduction
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        
        with torch.no_grad():
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    # DDP reduction of metrics across all GPUs
    if dist.is_initialized():
        # CRITICAL FIX: Ensure tensor is created on correct device
        metrics = torch.tensor([total_loss, float(correct), float(total_samples), float(nan_count)], 
                               dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples, nan_count = metrics.cpu().numpy()
    
    # Convert to Python float for JSON serialization
    avg_loss = float(total_loss / max(total_samples, 1))
    accuracy = float(correct / max(total_samples, 1))
    
    return avg_loss, accuracy, int(nan_count)

# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor, 
             device: torch.device, rank: int, check_early_detection: bool = False) -> dict:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        class_weights: Class weights for loss computation
        device: Device to use
        rank: DDP rank
        check_early_detection: If True, compute accuracy at different observation fractions
        
    Returns:
        Dictionary with loss, accuracy, and optionally early_stats
    """
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    
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

        if check_early_detection:
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
            logits_seq = out['logits']
            probs_seq = out['probs']
            
            B = flux.size(0)
            last_idx = (lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=device)
            final_logits = logits_seq[batch_indices, last_idx]
            
            all_preds = probs_seq.argmax(dim=2)
            for b in range(B):
                seq_len = lengths[b].item()
                true_lbl = labels[b].item()
                for m in milestones:
                    idx = max(0, min(int(seq_len * m) - 1, seq_len - 1))
                    if all_preds[b, idx].item() == true_lbl:
                        early_correct[m] += 1
                    early_total[m] += 1
        else:
            out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
            final_logits = out['logits']
        
        # CRITICAL FIX: Use mean reduction consistent with training
        loss = F.cross_entropy(final_logits, labels, weight=class_weights, reduction='mean')
        batch_samples = labels.size(0)
        
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        
        preds = final_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    # DDP reduction
    if dist.is_initialized():
        # CRITICAL FIX: Ensure tensor is created on correct device with correct dtype
        metrics = torch.tensor([total_loss, float(correct), float(total_samples)], 
                               dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples = metrics.cpu().numpy()
        
        if check_early_detection:
            for m in milestones:
                e_metrics = torch.tensor([float(early_correct[m]), float(early_total[m])], 
                                         dtype=torch.float64, device=device)
                dist.all_reduce(e_metrics, op=dist.ReduceOp.SUM)
                ec, et = e_metrics.cpu().numpy()
                early_correct[m], early_total[m] = int(ec), int(et)

    results = {
        'loss': float(total_loss / max(total_samples, 1)),
        'accuracy': float(correct / max(total_samples, 1))
    }
    
    if check_early_detection:
        results['early_stats'] = {
            m: float(early_correct[m] / max(early_total[m], 1))
            for m in milestones
        }
    
    return results

# =============================================================================
# DATA LOADING
# =============================================================================
def load_npz_data(path: str, rank: int, logger: logging.Logger) -> tuple:
    """
    Load and validate NPZ data file.
    
    CRITICAL: Expects pre-computed causal delta_t from simulate.py.
    """
    if rank == 0:
        logger.info(f"Loading data from {path}...")
    
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        if rank == 0:
            logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    flux = data.get('flux', data.get('X'))
    labels = data.get('labels', data.get('y'))
    
    if 'delta_t' in data:
        delta_t = data['delta_t']
        if rank == 0:
            logger.info("Using pre-computed causal delta_t from simulation")
    elif 'timestamps' in data:
        ts = data['timestamps']
        if ts.ndim == 1:
            ts = np.tile(ts, (len(flux), 1))
        delta_t = create_delta_t_from_timestamps(ts)
        if rank == 0:
            logger.warning("Computing delta_t from timestamps - consider using simulate.py for strict causality")
    else:
        delta_t = np.zeros_like(flux)
        if rank == 0:
            logger.warning("No temporal information found - using zero delta_t")

    if flux.ndim == 3:
        flux = flux.squeeze(1)
    if delta_t.ndim == 3:
        delta_t = delta_t.squeeze(1)

    if rank == 0:
        logger.info("Calculating normalization statistics from training sample...")
    
    sample_flux = flux[:10000].flatten()
    valid_flux = sample_flux[sample_flux != 0.0]
    
    if len(valid_flux) == 0:
        if rank == 0:
            logger.error("No valid flux values found in data sample!")
        sys.exit(1)
    
    median = float(np.median(valid_flux))
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = float(q75 - q25)
    
    stats = {'median': median, 'iqr': iqr}
    
    if rank == 0:
        logger.info(f"Normalization stats: Median={median:.4f}, IQR={iqr:.4f}")
        logger.info(f"Data shape: flux={flux.shape}, delta_t={delta_t.shape}, labels={labels.shape}")
    
    return flux, delta_t, labels, stats

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train Causal Hybrid Model for Microlensing Classification"
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
    # CRITICAL FIX: Increased default dropout from 0.1 to 0.3 for better regularization
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning rate")
    # CRITICAL FIX: Increased default weight_decay from 1e-4 to 1e-2 for better regularization
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--grad_checkpoint', action='store_true', 
                        help="Enable gradient checkpointing")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    
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
    
    if is_ddp:
        dist.barrier()
    
    set_seed(SEED)
    
    if 'CausalHybridModel' in globals():
        CausalHybridModel.set_init_seed(SEED)
        if is_ddp:
            CausalHybridModel.set_distributed_seed(SEED, rank)

    # Initialize AMP scaler
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()

    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Roman Telescope Causal Model Training")
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Device: {device.type} | DDP: {is_ddp} | Rank/World: {rank}/{world_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size} (local: {args.batch_size})")
        logger.info(f"Regularization: dropout={args.dropout}, weight_decay={args.weight_decay}")
        logger.info(f"Deterministic: cudnn.deterministic={torch.backends.cudnn.deterministic}")
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
    
    use_pin_memory = (device.type == 'cuda')
    num_workers = args.num_workers
    persistent_workers = num_workers > 0 and train_sampler is not None
    
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
        drop_last=is_ddp
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
        if hasattr(model, 'get_receptive_field'):
            logger.info(f"Receptive field: {model.get_receptive_field()} timesteps")
    
    # Wrap in DDP
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
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
        
        state_dict = checkpoint['state_dict']
        if is_ddp and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not is_ddp and any(k.startswith('module.') for k in state_dict.keys()):
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
            # Convert to Python float for JSON serialization
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float(val_results['loss']))
            history['val_acc'].append(float(val_results['accuracy']))
            
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
                
                state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': state_to_save,
                    'config': config.__dict__,
                    'accuracy': float(best_acc),
                    'optimizer': optimizer.state_dict(),
                    'receptive_field': model.module.get_receptive_field() if is_ddp and hasattr(model.module, 'get_receptive_field') else (model.get_receptive_field() if hasattr(model, 'get_receptive_field') else "N/A"),
                    'history': history
                }
                
                torch.save(checkpoint, output_dir / "best_model.pt")
                logger.info(f"  ✓ Best model saved (Val Acc: {best_acc:.4f})")
            
            # Save history with custom encoder
            with open(output_dir / "history.json", 'w') as f:
                json.dump(history, f, indent=2, cls=NumpyEncoder)
            
            # Periodic checkpoint save
            if epoch % 10 == 0:
                latest_state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': latest_state_to_save,
                    'config': config.__dict__,
                    'accuracy': float(val_results['accuracy']),
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
