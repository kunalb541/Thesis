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
from typing import Dict, Any, Tuple, Optional

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
def setup_distributed() -> Tuple[int, int, int, bool]:
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


def set_seed_everywhere(seed: int, rank: int = 0) -> None:
    """
    GOD MODE: Set all random seeds for perfect reproducibility.
    
    Args:
        seed: Base random seed
        rank: Process rank (for rank-specific seeds if needed)
    """
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
        stats: Optional[Dict] = None,
        handle_nans: str = 'zero'
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
        
        # Handle NaN values
        if handle_nans == 'zero':
            self.flux = torch.nan_to_num(self.flux, nan=0.0)
        elif handle_nans == 'median':
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
                iqr = 1.0
            
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - median) / iqr,
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.flux[idx],
            self.delta_t[idx],
            self.lengths[idx],
            self.labels[idx]
        )


def compute_normalization_stats(flux: np.ndarray, sample_size: int = 10000) -> Dict[str, float]:
    """
    Compute median and IQR for flux normalization.
    
    Args:
        flux: (N, T) flux array
        sample_size: Number of events to sample for statistics
        
    Returns:
        Dict with 'median' and 'iqr'
    """
    sample_flux = flux[:sample_size]
    valid_flux = sample_flux[~np.isnan(sample_flux) & (sample_flux != 0.0)]
    
    if len(valid_flux) == 0:
        return {'median': 0.0, 'iqr': 1.0}
    
    median = float(np.median(valid_flux))
    q75, q25 = np.percentile(valid_flux, [75, 25])
    iqr = float(q75 - q25)
    
    if iqr < 1e-6:
        iqr = 1.0
    
    return {'median': median, 'iqr': iqr}


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute class weights for balanced training."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    weight_tensor = torch.zeros(3, device=device)
    weight_tensor[unique] = torch.from_numpy(weights).float()
    return weight_tensor


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    epoch: int,
    logger: logging.Logger,
    config: ModelConfig
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch_idx, (flux, delta_t, lengths, labels) in enumerate(iterator):
        # Move to device
        flux = flux.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=config.use_amp, dtype=torch.bfloat16):
            output = model(flux, delta_t, lengths=lengths)
            
            if config.hierarchical:
                loss = F.cross_entropy(
                    output['logits'], 
                    labels, 
                    weight=class_weights
                )
            else:
                loss = F.cross_entropy(
                    output['logits'],
                    labels,
                    weight=class_weights
                )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # GOD MODE: Memory optimization
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
        
        # Metrics
        with torch.no_grad():
            pred = output['probs'].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        
        if rank == 0:
            iterator.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    rank: int,
    config: ModelConfig,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """Evaluate model."""
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
            
            with torch.cuda.amp.autocast(enabled=config.use_amp, dtype=torch.bfloat16):
                output = model(flux, delta_t, lengths=lengths)
                
                if config.hierarchical:
                    loss = F.cross_entropy(
                        output['logits'],
                        labels,
                        weight=class_weights
                    )
                else:
                    loss = F.cross_entropy(
                        output['logits'],
                        labels,
                        weight=class_weights
                    )
            
            pred = output['probs'].argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
            if return_predictions:
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(output['probs'].cpu().numpy())
    
    results = {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }
    
    if return_predictions:
        results['predictions'] = np.array(all_preds)
        results['labels'] = np.array(all_labels)
        results['probabilities'] = np.array(all_probs)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_acc: float,
    config: ModelConfig,
    stats: Dict,
    output_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    # GOD MODE: Strip DDP wrapper for clean save
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_accuracy': best_acc,
        'config': config.__dict__,
        'normalization_stats': stats,
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Roman microlensing classifier')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--hierarchical', action='store_true', help='Use hierarchical classification')
    parser.add_argument('--feature-extraction', type=str, default='mlp', choices=['mlp', 'conv'])
    parser.add_argument('--use-attention-pooling', action='store_true', help='Use attention pooling')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=WARMUP_EPOCHS, help='Warmup epochs')
    parser.add_argument('--clip-norm', type=float, default=CLIP_NORM, help='Gradient clipping')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    
    # Optimization arguments
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weighting')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    
    # Evaluation arguments
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=0, help='Early stopping patience (0=disabled)')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = f"gru_d{args.d_model}_l{args.n_layers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(rank, output_dir)
    
    # Set random seed
    set_seed_everywhere(SEED, rank)
    
    # Load data
    if rank == 0:
        logger.info(f"Loading data from {args.data}")
    
    data = load_compat(args.data)
    flux = data['flux']
    delta_t = data['delta_t']
    labels = data['labels']
    
    # Split data
    flux_train, flux_val, delta_t_train, delta_t_val, labels_train, labels_val = train_test_split(
        flux, delta_t, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    # Compute normalization stats
    stats = compute_normalization_stats(flux_train)
    
    if rank == 0:
        logger.info(f"Normalization stats: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
    
    # Create datasets
    train_dataset = MicrolensingDataset(flux_train, delta_t_train, labels_train, stats)
    val_dataset = MicrolensingDataset(flux_val, delta_t_val, labels_val, stats)
    
    # Create dataloaders
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        hierarchical=args.hierarchical,
        feature_extraction=args.feature_extraction,
        use_attention_pooling=args.use_attention_pooling,
        use_amp=args.use_amp,
        use_gradient_checkpointing=True
    )
    
    model = RomanMicrolensingGRU(config, dtype=torch.float32).to(device)
    
    if rank == 0:
        logger.info(f"Model parameters: {count_parameters(model):,}")
        logger.info(f"Configuration: {config}")
    
    # Wrap in DDP
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=False
        )
    
    # Compile model
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # GOD MODE FIX: Proper FP16 detection
    use_fp16 = False  # Using BF16, not FP16
    scaler =  None
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Class weights
    if args.use_class_weights:
        class_weights = compute_class_weights(labels_train, device)
        if rank == 0:
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(3, device=device)
    
    # Training loop
    if rank == 0:
        logger.info("=" * 80)
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
                logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation
    if rank == 0:
        logger.info("=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)
        
        # Load best model
        if is_ddp:
            dist.barrier()
        
        best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(best_checkpoint['model_state_dict'])
        
        val_results = evaluate(
            model, val_loader, class_weights, device, rank, config,
            return_predictions=True
        )
        
        logger.info(f"Best Validation Accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"Best Validation Loss: {val_results['loss']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(val_results['labels'], val_results['predictions'])
        logger.info("Confusion Matrix:")
        logger.info("                Predicted")
        logger.info("             Flat  PSPL  Binary")
        for i, row in enumerate(cm):
            class_name = ['Flat', 'PSPL', 'Binary'][i]
            logger.info(f"  {class_name:6s}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")
        
        # Classification report
        class_names = ['Flat', 'PSPL', 'Binary']
        report = classification_report(
            val_results['labels'],
            val_results['predictions'],
            target_names=class_names,
            digits=4
        )
        logger.info("Classification Report:")
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
        
        logger.info(f"Training complete. Results saved to {output_dir}")
        logger.info("=" * 80)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()



