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
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
import random
import h5py

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import RomanMicrolensingGRU, ModelConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"Error importing model: {e}")
    def count_parameters(model): return 0

warnings.filterwarnings("ignore")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
WARMUP_EPOCHS = 5
SEED = 42

# Physical constants from simulate.py
AB_ZEROPOINT_JY = 3631.0
MISSION_DURATION_DAYS = 1826.25


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# DISTRIBUTED TRAINING SETUP
# =============================================================================
def setup_distributed():
    """Initialize distributed training if environment variables are set."""
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            
            # Performance optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            return rank, local_rank, world_size, True
    
    return 0, 0, 1, False


def setup_logging(rank: int, output_dir: Path):
    """Setup logging to both console and file."""
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if rank == 0:
        # File handler only for rank 0
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s | Rank %(rank)s | %(levelname)s | %(message)s')
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if rank != 0:
        logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    return logger


def set_seed_everywhere(seed: int, rank: int = 0):
    """
    GOD MODE: Set all random seeds for perfect reproducibility.
    
    Args:
        seed: Base random seed
        rank: Process rank (for rank-specific seeds if needed)
    """
    import random
    
    # Python
    random.seed(seed + rank)
    
    # NumPy
    np.random.seed(seed + rank)
    
    # PyTorch CPU
    torch.manual_seed(seed + rank)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    
    # CuDNN
    torch.backends.cudnn.deterministic = False  # Keep False for performance
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    
    if rank == 0:
        print(f"🎲 GOD MODE: All random seeds set to {seed}")



# =============================================================================
# DATASET WITH PHYSICAL REALISM
# =============================================================================
class MicrolensingDataset(Dataset):
    """
    Dataset for Roman Space Telescope microlensing events.
    
    Handles:
    - Physical AB magnitude system
    - NaN values from flux-to-magnitude conversion
    - Causal delta_t encoding
    - Proper normalization (Median/IQR)
    """
    
    def __init__(
        self, 
        flux: np.ndarray, 
        delta_t: np.ndarray, 
        labels: np.ndarray, 
        stats: dict = None,
        handle_nans: str = 'zero'  # 'zero', 'mask', or 'median'
    ):
        """
        Args:
            flux: (N, T) magnitude array (AB system)
            delta_t: (N, T) causal time differences
            labels: (N,) class labels
            stats: Dict with 'median' and 'iqr' for normalization
            handle_nans: Strategy for NaN handling
        """
        # Convert to tensors
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        
        # Identify valid observations (non-zero, non-NaN)
        self.padding_mask = (~torch.isnan(self.flux)) & (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)
        
        # Handle NaN values (from negative flux in mag conversion)
        if handle_nans == 'zero':
            self.flux = torch.nan_to_num(self.flux, nan=0.0)
        elif handle_nans == 'median':
            # Replace NaNs with median of each light curve
            for i in range(len(self.flux)):
                valid = self.flux[i][self.padding_mask[i]]
                if len(valid) > 0:
                    median = valid.median()
                    self.flux[i] = torch.where(
                        torch.isnan(self.flux[i]), 
                        median, 
                        self.flux[i]
                    )
        
        # Handle delta_t NaNs
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)
        
        # Apply normalization if statistics provided
        if stats is not None:
            median = stats['median']
            iqr = stats['iqr']
            
            if iqr < 1e-6:
                iqr = 1.0  # Prevent division by zero
            
            # Normalize only valid (non-padded) observations
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.flux[idx],
            self.delta_t[idx],
            self.lengths[idx],
            self.labels[idx]
        )


def compute_normalization_stats(flux: np.ndarray, sample_size: int = 10000) -> dict:
    """
    Compute median and IQR for flux normalization.
    Matches the normalization in simulate.py.
    
    Args:
        flux: (N, T) flux array
        sample_size: Number of events to sample for statistics
        
    Returns:
        Dict with 'median' and 'iqr'
    """
    # Sample subset for efficiency
    sample_flux = flux[:sample_size]
    
    # Extract valid (non-zero, non-NaN) values
    valid_flux = sample_flux[sample_flux != 0]
    valid_flux = valid_flux[~np.isnan(valid_flux)]
    
    if len(valid_flux) == 0:
        print("Warning: No valid flux values found. Using default normalization.")
        return {'median': 0.0, 'iqr': 1.0}
    
    median = float(np.median(valid_flux))
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = float(q75 - q25)
    
    if iqr < 1e-6:
        iqr = 1.0
    
    return {'median': median, 'iqr': iqr}


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(
    model, 
    loader, 
    optimizer, 
    scaler, 
    class_weights, 
    device, 
    rank, 
    epoch, 
    logger, 
    config
):
    """Train for one epoch."""
    model.train()
    
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    
    # Only show progress bar on rank 0
    iterator = loader if rank != 0 else tqdm(
        loader, 
        desc=f"Epoch {epoch}", 
        leave=False,
        dynamic_ncols=True
    )

    for flux, delta_t, lengths, labels in iterator:
        # Move to device with non-blocking transfer
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        out = model(flux, delta_t, lengths=lengths)
        
        # Compute loss (FIXED: Added auxiliary hierarchical losses)
        if config.hierarchical:
            # Main loss (joint probability)
            loss_main = F.nll_loss(out['logits'], labels, weight=class_weights)
            
            # Auxiliary loss 1: Deviation detection (Flat vs Deviation)
            dev_labels = (labels > 0).long()  # 0=Flat, 1=Deviation
            loss_dev = F.cross_entropy(out['aux_dev'], dev_labels)
            
            # Auxiliary loss 2: Event type (PSPL vs Binary, only for deviation events)
            dev_mask = labels > 0
            if dev_mask.sum() > 0:
                type_labels = (labels[dev_mask] - 1)  # 0=PSPL, 1=Binary
                loss_type = F.cross_entropy(out['aux_type'][dev_mask], type_labels)
            else:
                loss_type = torch.tensor(0.0, device=device)
            
            # Combined loss with auxiliary weights
            loss = loss_main + 0.3 * loss_dev + 0.3 * loss_type
        else:
            loss = F.cross_entropy(out['logits'], labels, weight=class_weights)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate metrics
        with torch.no_grad():
            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size
            correct += (out['probs'].argmax(dim=1) == labels).sum()
            total_samples += batch_size

    # Synchronize across GPUs (FIXED: Batched all_reduce)
    if dist.is_initialized():
        metrics_tensor = torch.stack([total_loss, correct, total_samples])
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples = metrics_tensor
    
    avg_loss = float(total_loss / total_samples)
    accuracy = float(correct / total_samples)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, config, return_predictions=False):
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    
    all_predictions = []
    all_labels = []

    for flux, delta_t, lengths, labels in loader:
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        out = model(flux, delta_t, lengths=lengths)
        
        # Compute loss (FIXED: Added auxiliary hierarchical losses)
        if config.hierarchical:
            # Main loss (joint probability)
            loss_main = F.nll_loss(out['logits'], labels, weight=class_weights)
            
            # Auxiliary loss 1: Deviation detection (Flat vs Deviation)
            dev_labels = (labels > 0).long()  # 0=Flat, 1=Deviation
            loss_dev = F.cross_entropy(out['aux_dev'], dev_labels)
            
            # Auxiliary loss 2: Event type (PSPL vs Binary, only for deviation events)
            dev_mask = labels > 0
            if dev_mask.sum() > 0:
                type_labels = (labels[dev_mask] - 1)  # 0=PSPL, 1=Binary
                loss_type = F.cross_entropy(out['aux_type'][dev_mask], type_labels)
            else:
                loss_type = torch.tensor(0.0, device=device)
            
            # Combined loss with auxiliary weights
            loss = loss_main + 0.3 * loss_dev + 0.3 * loss_type
        else:
            loss = F.cross_entropy(out['logits'], labels, weight=class_weights)
        
        predictions = out['probs'].argmax(dim=1)
        
        # Accumulate metrics
        batch_size = labels.size(0)
        total_loss += loss * batch_size
        correct += (predictions == labels).sum()
        total_samples += batch_size
        
        if return_predictions:
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Synchronize across GPUs (FIXED: Batched all_reduce)
    if dist.is_initialized():
        metrics_tensor = torch.stack([total_loss, correct, total_samples])
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, correct, total_samples = metrics_tensor
    
    avg_loss = float(total_loss / total_samples)
    accuracy = float(correct / total_samples)
    
    results = {'loss': avg_loss, 'accuracy': accuracy}
    
    if return_predictions:
        results['predictions'] = torch.cat(all_predictions).numpy()
        results['labels'] = torch.cat(all_labels).numpy()
    
    return results


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights.
    
    Args:
        labels: (N,) array of class labels
        device: torch device
        
    Returns:
        Tensor of shape (n_classes,) with weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    
    # Inverse frequency weighting
    total = len(labels)
    weights = np.zeros(n_classes)
    
    for cls, count in zip(unique, counts):
        weights[cls] = total / (n_classes * count)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    best_acc,
    config,
    stats,
    output_dir,
    is_best=False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'config': config.__dict__,
        'normalization_stats': stats,
    }
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
    
    return checkpoint_path


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train Roman Space Telescope microlensing classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment_name', required=True, 
                        help='Experiment name for output directory')
    parser.add_argument('--data', required=True, 
                        help='Path to training data (.npz file)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS,
                        help='Number of warmup epochs for learning rate')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--window_size', type=int, default=7,
                        help='Causal window size')
    parser.add_argument('--feature_extraction', type=str, default='mlp',
                        choices=['mlp', 'conv'],
                        help='Feature extraction method')
    parser.add_argument('--use_attention_pooling', action='store_true',
                        help='Use attention pooling instead of last-step')
    parser.add_argument('--non_hierarchical', action='store_true',
                        help='Use flat classification instead of hierarchical')
    
    # Data handling
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--handle_nans', type=str, default='zero',
                        choices=['zero', 'mask', 'median'],
                        help='Strategy for handling NaN values')
    
    # Training options
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use balanced class weights')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience (0 to disable)')
    
    # System
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{timestamp}"
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arguments
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Setup logging
    logger = setup_logging(rank, output_dir)
    set_seed(args.seed)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("ROMAN SPACE TELESCOPE MICROLENSING CLASSIFIER TRAINING")
        logger.info("=" * 80)
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Random seed: {args.seed}")
    
    
    # GOD MODE: Validate hyperparameters for DDP
    if is_ddp:
        if args.batch_size % world_size != 0:
            if rank == 0:
                logger.error(f"Batch size {args.batch_size} not divisible by world_size {world_size}")
            sys.exit(1)
        
        effective_batch = args.batch_size // world_size
        if rank == 0:
            logger.info(f"DDP Mode: {world_size} GPUs × {effective_batch} batch = {args.batch_size} global batch")

    # Load data
    if rank == 0:
        logger.info(f"\nLoading data from {args.data}...")
    
    data = np.load(args.data, allow_pickle=True)
    
    # Extract arrays
    flux = data['flux']
    if flux.ndim == 3 and flux.shape[-1] == 1:
        flux = flux.squeeze(-1)
    
    delta_t = data['delta_t'] if 'delta_t' in data else np.zeros_like(flux)
    if delta_t.ndim == 3 and delta_t.shape[-1] == 1:
        delta_t = delta_t.squeeze(-1)
    
    labels = data['labels']
    
    # Verify physical realism flags
    if rank == 0:
        logger.info("\nDataset Information:")
        logger.info(f"  Shape: {flux.shape}")
        logger.info(f"  Classes: {np.unique(labels)}")
        logger.info(f"  Class distribution: {np.bincount(labels)}")
        
        if 'physical_realism' in data:
            logger.info(f"  Physical realism: {data['physical_realism']}")
        else:
            logger.warning("  Warning: Dataset may not have physical realism flag")
        
        if 'ab_zeropoint_jy' in data:
            # GOD MODE: Rank 0 logging
            if rank == 0:
                logger.info(f"  AB zero point: {data['ab_zeropoint_jy']} Jy")
        
        if 'mission_duration_days' in data:
            # GOD MODE: Rank 0 logging
            if rank == 0:
                logger.info(f"  Mission duration: {data['mission_duration_days']} days")
    
    # Compute normalization statistics
    if rank == 0:
        logger.info("\nComputing normalization statistics...")
    
    stats = compute_normalization_stats(flux, sample_size=10000)
    
    if rank == 0:
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  IQR: {stats['iqr']:.4f}")
    
    # Split data
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed
    )
    
    # Create datasets
    train_ds = MicrolensingDataset(
        flux[train_indices],
        delta_t[train_indices],
        labels[train_indices],
        stats=stats,
        handle_nans=args.handle_nans
    )
    
    val_ds = MicrolensingDataset(
        flux[val_indices],
        delta_t[val_indices],
        labels[val_indices],
        stats=stats,
        handle_nans=args.handle_nans
    )
    
    if rank == 0:
        logger.info(f"\nDataset splits:")
        logger.info(f"  Training: {len(train_ds):,} samples")
        logger.info(f"  Validation: {len(val_ds):,} samples")
    
    # Create data loaders
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    # Create model
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        feature_extraction=args.feature_extraction,
        use_attention_pooling=args.use_attention_pooling,
        hierarchical=not args.non_hierarchical,
        use_amp=True,
        use_gradient_checkpointing=False
    )
    
    model = RomanMicrolensingGRU(config, dtype=torch.bfloat16).to(device)
    
    if rank == 0:
        n_params = count_parameters(model)
        logger.info(f"\nModel: RomanMicrolensingGRU")
        logger.info(f"  Parameters: {n_params:,}")
        logger.info(f"  d_model: {config.d_model}")
        logger.info(f"  n_layers: {config.n_layers}")
        logger.info(f"  Hierarchical: {config.hierarchical}")
        logger.info(f"  Feature extraction: {config.feature_extraction}")
        logger.info(f"  Attention pooling: {config.use_attention_pooling}")
    
    # Wrap with DDP
    if is_ddp:
        # GOD MODE: Optimal DDP configuration for 40 GPUs
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,  # Buffers don't need syncing (no running stats)
            gradient_as_bucket_view=True,  # Memory efficiency
            find_unused_parameters=False  # All parameters used
        )
    
    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info("  Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # FIXED: GradScaler only needed for FP16 (not BF16)
    # FIXED: GradScaler only needed for FP16 (not BF16)
    use_fp16 = (torch.bfloat16 != torch.bfloat16)  # Always False for BF16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(labels, device)
        if rank == 0:
            logger.info(f"\nClass weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(3, device=device)
    
    # Training loop
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler,
            class_weights, device, rank, epoch, logger, config
        )
        
        # Step scheduler
        scheduler.step()
        
        # Evaluate
        should_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        
        if should_eval:
            val_results = evaluate(
                model, val_loader, class_weights, device, rank, config,
                return_predictions=(epoch == args.epochs and rank == 0)
            )
            
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            if rank == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=True
                    )
                    
                    # GOD MODE: Rank 0 logging
                    if rank == 0:
                        logger.info(f"  New best accuracy: {best_acc:.4f}")
                else:
                    patience_counter += 1
                
                # Regular checkpoint
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=False
                    )
        else:
            if rank == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
        
        # Early stopping
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            if rank == 0:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation with detailed metrics
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)
        
        # Load best model
        if is_ddp:
            dist.barrier()  # GOD MODE: Ensure all ranks finish
        
        best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(best_checkpoint['model_state_dict'])
        
        val_results = evaluate(
            model, val_loader, class_weights, device, rank, config,
            return_predictions=True
        )
        
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info(f"Best Validation Accuracy: {val_results['accuracy']:.4f}")
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info(f"Best Validation Loss: {val_results['loss']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(val_results['labels'], val_results['predictions'])
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info("\nConfusion Matrix:")
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info("                Predicted")
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info("             Flat  PSPL  Binary")
        for i, row in enumerate(cm):
            class_name = ['Flat', 'PSPL', 'Binary'][i]
            # GOD MODE: Rank 0 logging
            if rank == 0:
                logger.info(f"  {class_name:6s}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")
        
        # Classification report
        class_names = ['Flat', 'PSPL', 'Binary']
        report = classification_report(
            val_results['labels'],
            val_results['predictions'],
            target_names=class_names,
            digits=4
        )
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info("\nClassification Report:")
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info(report)
        
        # Save metrics
        metrics = {
            'best_val_accuracy': float(val_results['accuracy']),
            'best_val_loss': float(val_results['loss']),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_epochs': epoch,
        }
        
        with open(output_dir / 'final_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info(f"\nTraining complete. Results saved to {output_dir}")
        # GOD MODE: Rank 0 logging
        if rank == 0:
            logger.info("=" * 80)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
