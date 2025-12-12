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
from datetime import datetime, timedelta
import warnings
import random
import h5py
import os
import gc
from typing import Dict, Any, Tuple, Optional
from contextlib import contextmanager

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import RomanMicrolensingGRU, ModelConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
WARMUP_EPOCHS = 5
SEED = 42
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
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed() -> Tuple[int, int, int, bool]:
    """Initialize distributed training environment with optimal settings."""
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if rank == 0:
            print(f"Initializing distributed: {world_size} processes", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
            dist.init_process_group(
                backend='nccl', 
                init_method='env://',
                timeout=timedelta(seconds=1800)
            )
            
            # A100-optimized performance settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable TensorFloat-32 for A100
            torch.set_float32_matmul_precision('high')
            
            # Memory optimizations
            torch.cuda.empty_cache()
            gc.collect()
            
            return rank, local_rank, world_size, True
    
    return 0, 0, 1, False


def setup_logging(rank: int, output_dir: Path):
    """Configure logging system."""
    logger = logging.getLogger("TRAIN")
    
    if rank == 0:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.CRITICAL)
    
    return logger


def set_seed_everywhere(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    
    # For maximum speed on A100, use non-deterministic algorithms
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)


# =============================================================================
# DATASET WITH OPTIMIZED HDF5 ACCESS
# =============================================================================

def load_labels_and_split(data_path: Path, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Loads only the 'labels' eagerly to perform train/val split.
    Computes normalization stats from a sample of training data.
    """
    with h5py.File(data_path, 'r', libver='latest', swmr=True) as f:
        # Load labels (small, fast)
        labels = f['labels'][:]
        
        # Perform train/val split on indices
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, shuffle=True, random_state=SEED, stratify=labels
        )
        
        # Compute normalization stats from training data sample
        flux_data = f['flux']
        sample_size = min(100000, len(train_idx))
        
        # Sample random indices and SORT them for HDF5 fancy indexing
        sample_indices_unsorted = np.random.choice(train_idx, size=sample_size, replace=False)
        sample_indices_sorted = np.sort(sample_indices_unsorted)
        
        # Lazy load only the sample
        sample_flux = flux_data[sample_indices_sorted]
        
        # Compute robust statistics
        valid_flux = sample_flux[~np.isnan(sample_flux) & (sample_flux != 0.0)]
        
        if len(valid_flux) == 0:
            median = 0.0
            iqr = 1.0
        else:
            median = float(np.median(valid_flux))
            q75, q25 = np.percentile(valid_flux, [75, 25])
            iqr = float(q75 - q25)
            if iqr < 1e-6:
                iqr = 1.0

        stats = {'median': median, 'iqr': iqr}
        
        # Log class distribution
        if logger:
            unique, counts = np.unique(labels[train_idx], return_counts=True)
            logger.info("Training class distribution:")
            for cls, cnt in zip(unique, counts):
                logger.info(f"  Class {cls}: {cnt} ({100*cnt/len(train_idx):.2f}%)")
        
    return train_idx, val_idx, stats


class MicrolensingLazyDataset(Dataset):
    """
    Roman Space Telescope microlensing dataset with optimized lazy HDF5 loading.
    
    Thread-safe design: Each DataLoader worker opens its own file handle.
    """
    
    def __init__(
        self, 
        data_path: Path,
        indices: np.ndarray,
        stats: Dict,
        handle_nans: str = 'zero'
    ):
        self.data_path = data_path
        self.indices = indices
        self.stats = stats
        self.handle_nans = handle_nans
        self._h5_file = None
        self._flux_dset = None
        self._delta_t_dset = None
        self._labels_dset = None
        
    def _open_h5_file(self):
        """Open HDF5 file handle with optimal settings for parallel access."""
        if self._h5_file is None:
            # Use libver='latest' and swmr=True for best performance with concurrent access
            self._h5_file = h5py.File(self.data_path, 'r', libver='latest', swmr=True)
            # Cache dataset handles to avoid repeated lookups
            self._flux_dset = self._h5_file['flux']
            self._delta_t_dset = self._h5_file['delta_t']
            self._labels_dset = self._h5_file['labels']
            
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx_in_subset: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch single sample with optimized HDF5 access."""
        # Open file handle lazily in worker thread
        self._open_h5_file()
        
        # Get global index
        global_idx = self.indices[idx_in_subset]
        
        # Direct slice access (fastest method)
        flux_np = self._flux_dset[global_idx, :]
        delta_t_np = self._delta_t_dset[global_idx, :]
        label_np = self._labels_dset[global_idx]
        
        # Convert to tensors (contiguous for GPU transfer)
        flux = torch.from_numpy(flux_np).float().contiguous()
        delta_t = torch.from_numpy(delta_t_np).float().contiguous()
        label = torch.tensor(label_np, dtype=torch.long)
        
        # Preprocessing
        padding_mask = (~torch.isnan(flux)) & (flux != 0.0)
        
        if self.handle_nans == 'zero':
            flux = torch.nan_to_num(flux, nan=0.0)
        
        delta_t = torch.nan_to_num(delta_t, nan=0.0)
        
        # Apply normalization
        median = self.stats['median']
        iqr = self.stats['iqr']
        flux = (flux - median) / iqr
        
        # Re-apply padding mask
        flux = flux * padding_mask.float()
        
        return (
            flux,
            delta_t,
            padding_mask.sum().long().clamp(min=1),
            label
        )

    def __del__(self):
        """Close HDF5 file handle on cleanup."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


def worker_init_fn(worker_id: int):
    """Initialize each DataLoader worker with unique seed and optimal settings."""
    # Set unique seed per worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Force HDF5 to use low-level driver for better multiprocessing
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights for balanced training."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    weight_tensor = torch.zeros(3, device=device)
    weight_tensor[unique] = torch.from_numpy(weights).float().to(device)
    return weight_tensor


# =============================================================================
# TRAINING LOOP WITH OPTIMIZATIONS
# =============================================================================
@contextmanager
def timer(name: str, logger: logging.Logger, rank: int):
    """Context manager for timing code blocks."""
    if rank == 0:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        logger.info(f"{name}: {elapsed:.2f}s")
    else:
        yield


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    epoch: int,
    logger: logging.Logger,
    config: ModelConfig,
    accumulation_steps: int = 1,
    clip_norm: float = CLIP_NORM
) -> Tuple[float, float]:
    """Execute one training epoch with gradient accumulation and AMP."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        # Non-blocking transfer for overlap with computation
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # AMP context with bfloat16 for A100
        with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
            output = model(flux, delta_t, lengths=lengths)
            loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
            loss = loss / accumulation_steps
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Compute metrics
        with torch.no_grad():
            pred = output['probs'].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += (loss.item() * accumulation_steps) * labels.size(0)
        
        if rank == 0 and batch_idx % 10 == 0:
            iterator.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    # Cleanup
    torch.cuda.empty_cache()
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    config: ModelConfig,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """Evaluate model on validation set with AMP."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for flux, delta_t, lengths, labels in tqdm(loader, disable=(rank != 0), desc="Evaluating"):
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=config.use_amp, dtype=torch.bfloat16):
                output = model(flux, delta_t, lengths=lengths)
                loss = F.cross_entropy(output['logits'], labels, weight=class_weights)
            
            pred = output['probs'].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
            if return_predictions:
                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(output['probs'].cpu().numpy())
    
    results = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    if return_predictions:
        results['predictions'] = np.concatenate(all_preds)
        results['labels'] = np.concatenate(all_labels)
        results['probabilities'] = np.concatenate(all_probs)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_acc: float,
    config: ModelConfig,
    stats: Dict,
    output_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'config': config.__dict__,
        'stats': stats
    }
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Roman microlensing classifier')
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='../results', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--window-size', type=int, default=5, help='Causal window size')
    
    # Model variants
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_true', 
                        help='Use hierarchical classification')
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    parser.set_defaults(hierarchical=True)
    
    parser.add_argument('--feature-extraction', type=str, choices=['conv', 'mlp'], default='conv',
                        help='Feature extraction method')
    
    parser.add_argument('--attention-pooling', dest='attention_pooling', action='store_true',
                        help='Use attention pooling')
    parser.add_argument('--no-attention-pooling', dest='attention_pooling', action='store_false')
    parser.set_defaults(attention_pooling=True)
    
    parser.add_argument('--use-residual', dest='use_residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--no-residual', dest='use_residual', action='store_false')
    parser.set_defaults(use_residual=True)
    
    parser.add_argument('--use-layer-norm', dest='use_layer_norm', action='store_true',
                        help='Use layer normalization')
    parser.add_argument('--no-layer-norm', dest='use_layer_norm', action='store_false')
    parser.set_defaults(use_layer_norm=True)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--clip-norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Optimization
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', 
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)
    
    parser.add_argument('--use-gradient-checkpointing', dest='use_gradient_checkpointing', 
                        action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', dest='use_gradient_checkpointing', 
                        action='store_false')
    parser.set_defaults(use_gradient_checkpointing=False)  # Disabled by default for speed
    
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weighting')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile')
    
    # DataLoader optimization
    parser.add_argument('--num-workers', type=int, default=8, 
                        help='DataLoader workers per GPU (8-16 recommended)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Batches to prefetch per worker')
    
    # Evaluation
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=0, 
                        help='Early stopping patience (0=disabled)')
    
    args = parser.parse_args()
    
    # Setup distributed environment
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"d{args.d_model}_l{args.n_layers}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        dist.barrier()
    
    logger = setup_logging(rank, output_dir)
    
    if rank == 0:
        logger.info(f"Random seed: {SEED}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Device: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"World size: {world_size} GPUs")
    
    set_seed_everywhere(SEED, rank)
    
    # Load data and compute splits
    if rank == 0:
        logger.info(f"Loading data indices and computing normalization stats: {args.data}")
    
    train_idx, val_idx, stats = load_labels_and_split(Path(args.data), logger if rank == 0 else None)
    
    if rank == 0:
        logger.info(f"Dataset size: {len(train_idx) + len(val_idx)} samples")
        logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}")
        logger.info(f"Normalization: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
    
    # Create datasets
    train_dataset = MicrolensingLazyDataset(Path(args.data), train_idx, stats)
    val_dataset = MicrolensingLazyDataset(Path(args.data), val_idx, stats)
    
    # Load labels for class weights
    with h5py.File(args.data, 'r') as f:
        all_labels = f['labels'][:]
    labels_train_for_weights = all_labels[train_idx]
    
    # Create samplers
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create DataLoaders with optimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn
    )
    
    if rank == 0:
        logger.info(f"DataLoader workers per GPU: {args.num_workers}")
        logger.info(f"Total DataLoader workers: {args.num_workers * world_size}")
    
    # Create model with optimal dtype for A100
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        hierarchical=args.hierarchical,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        feature_extraction=args.feature_extraction,
        use_attention_pooling=args.attention_pooling,
        use_amp=args.use_amp,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    
    # Use bfloat16 for AMP on A100 (better than float16)
    amp_dtype = torch.bfloat16 if args.use_amp and torch.cuda.is_available() else torch.float32
    model = RomanMicrolensingGRU(config, dtype=amp_dtype).to(device)
    
    if rank == 0:
        n_params = count_parameters(model)
        logger.info(f"Model parameters: {n_params:,}")
        logger.info(f"Architecture: {config}")
        
        effective_batch = args.batch_size * args.accumulation_steps * world_size
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Gradient checkpointing: {args.use_gradient_checkpointing}")
        logger.info(f"Mixed precision (AMP): {args.use_amp}")
    
    # Wrap model in DDP
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
            broadcast_buffers=True,
            static_graph=True  # Optimization for static computation graphs
        )
    
    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode='max-autotune')
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        fused=True  # Fused AdamW for A100
    )
    
    # Create gradient scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Create learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Compute class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(labels_train_for_weights, device)
        if rank == 0:
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(3, device=device)
    
    if is_ddp:
        dist.barrier()
    
    # Training loop
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Training started")
        logger.info("=" * 80)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        with timer(f"Epoch {epoch} training", logger, rank):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scaler,
                class_weights, device, rank, epoch, logger, config,
                accumulation_steps=args.accumulation_steps,
                clip_norm=args.clip_norm
            )
        
        scheduler.step()
        
        should_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        
        if should_eval:
            with timer(f"Epoch {epoch} evaluation", logger, rank):
                val_results = evaluate(
                    model, val_loader, class_weights, device, rank, config,
                    return_predictions=(epoch == args.epochs and rank == 0)
                )
            
            val_loss, val_acc = val_results['loss'], val_results['accuracy']
            
            if rank == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=True
                    )
                    logger.info(f"New best: {best_acc*100:.2f}%")
                else:
                    patience_counter += 1
                
                if epoch % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scaler, epoch, best_acc,
                        config, stats, output_dir, is_best=False
                    )
        else:
            if rank == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
        
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Training complete | Best accuracy: {best_acc*100:.2f}%")
        logger.info("=" * 80)
        
        checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
        if is_ddp:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        final_results = evaluate(
            model, val_loader, class_weights, device, rank, config,
            return_predictions=True
        )
        
        np.savez(
            output_dir / 'final_predictions.npz',
            predictions=final_results['predictions'],
            labels=final_results['labels'],
            probabilities=final_results['probabilities']
        )
        
        cm = confusion_matrix(final_results['labels'], final_results['predictions'])
        logger.info(f"Confusion matrix:\n{cm}")
        np.save(output_dir / 'confusion_matrix.npy', cm)
        
        report = classification_report(
            final_results['labels'], 
            final_results['predictions'],
            target_names=['Flat', 'PSPL', 'Binary']
        )
        logger.info(f"Classification report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # Save training configuration
        config_dict = {
            'model_config': config.__dict__,
            'training_args': vars(args),
            'stats': stats,
            'best_accuracy': float(best_acc),
            'world_size': world_size
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved: {output_dir}")
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
