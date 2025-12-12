#!/usr/bin/env python3
"""
Roman Microlensing Classifier Training Engine 
==============================================

High-performance distributed training pipeline for classifying Roman Space
Telescope microlensing light curves into Flat, PSPL, and Binary classes.
Built for extreme throughput, GPU saturation, and scalable experimentation.

CORE CAPABILITIES:
    • End-to-end PyTorch DDP training with NCCL tuning for maximum bandwidth  
    • Zero-copy HDF5 streaming with worker-safe dataset handles  
    • On-the-fly normalization using cached median/IQR statistics  
    • Flash-accelerated attention paths and TF32 matmul where available  
    • Mixed-precision support with dynamic GradScaler and autocast  
    • CUDA streams for asynchronous data prefetching into device memory  
    • Flexible cosine-warmup learning rate schedule for stable convergence  
    • Deterministic—or intentionally non-deterministic—paths depending on 
      performance requirements  

DATA PIPELINE:
    Flux and Δt arrays are pulled from HDF5 storage with sorted index access,
    repaired for missing values, normalized robustly, and masked for variable
    sequence lengths. Batched tensors are delivered via a custom collate
    function tuned for contiguous memory access and transfer speed.

DISTRIBUTED EXECUTION:
    This module dynamically configures a multi-process, GPU-aware environment
    using NCCL backends, safe spawning, and synchronized caching of dataset
    splits. Gradient accumulation enables large effective batch sizes even on
    constrained hardware.

PURPOSE:
    This training engine powers supervised classification experiments for
    Roman microlensing survey data. It is designed to integrate cleanly with
    the accompanying RomanMicrolensingClassifier model definition and with
    large-scale simulation datasets generated for thesis-level research.

Author: Kunal Bhatia  
Institution: University of Heidelberg  
Version: 1.0
"""

from __future__ import annotations
import argparse, gc, json, logging, math, os, random, shutil, sys, time, warnings
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py, numpy as np, torch
import torch.distributed as dist
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def _configure_environment() -> None:
    """Ultra-optimized environment configuration."""
    os.environ.setdefault('NCCL_IB_DISABLE', '0')
    os.environ.setdefault('NCCL_NET_GDR_LEVEL', '3')  # Maximum GPU Direct
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    os.environ.setdefault('NCCL_P2P_LEVEL', '5')  # Maximum P2P
    os.environ.setdefault('NCCL_MIN_NCHANNELS', '16')  # More channels for bandwidth
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                         'expandable_segments:True,garbage_collection_threshold:0.9')
    os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'OFF')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    # Disable profiler overhead
    os.environ.setdefault('KINETO_LOG_LEVEL', '5')

_configure_environment()

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from model import ModelConfig, RomanMicrolensingClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED: int = 42
DEFAULT_CLIP_NORM: float = 1.0
MIN_SEQUENCE_LENGTH: int = 1
DEFAULT_VAL_FRACTION: float = 0.2
PROGRESS_UPDATE_FREQ: int = 50  # Less frequent updates
CLASS_NAMES: Tuple[str, ...] = ('Flat', 'PSPL', 'Binary')
EPS: float = 1e-8

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (Path, os.PathLike)):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)

def format_time(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"

def format_number(n: int) -> str:
    if n < 0:
        return f"-{format_number(-n)}"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_distributed() -> Tuple[int, int, int, bool]:
    if 'RANK' not in os.environ:
        return 0, 0, 1, False
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if rank == 0:
        print(f"Initializing distributed: {world_size} processes", flush=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for distributed training")
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://',
                               timeout=timedelta(seconds=3600),
                               world_size=world_size, rank=rank)
    
    # ULTRA-OPTIMIZED CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    
    # Enable CUDA graphs if available (PyTorch 2.0+)
    if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, local_rank)
        except Exception:
            pass
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return rank, local_rank, world_size, True

def cleanup_distributed() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        finally:
            dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

def synchronize() -> None:
    if dist.is_initialized():
        dist.barrier()

def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ROMAN_TRAIN")
    logger.handlers.clear()
    logger.propagate = False
    
    if is_main_process(rank):
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        log_file = output_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
    
    return logger

def set_global_seeds(seed: int, rank: int = 0) -> torch.Generator:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    
    generator = torch.Generator()
    generator.manual_seed(effective_seed)
    
    # PERFORMANCE: Non-deterministic for speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    return generator

def copy_to_local_storage(data_path: Path, logger: logging.Logger,
                          rank: int, local_dir: Optional[str] = None) -> Path:
    if local_dir is None:
        for candidate in [os.environ.get('TMPDIR'), os.environ.get('LOCAL_SCRATCH'),
                          '/tmp', '/local', '/scratch']:
            if candidate and os.path.isdir(candidate):
                local_dir = candidate
                break
    
    if local_dir is None:
        return data_path
    
    local_path = Path(local_dir) / f"rank{rank}_{data_path.name}"
    
    if not local_path.exists():
        if rank == 0 or not dist.is_initialized():
            logger.info(f"Copying data to local storage: {local_path}")
        start = time.time()
        shutil.copy2(data_path, local_path)
        if rank == 0 or not dist.is_initialized():
            elapsed = time.time() - start
            size_gb = local_path.stat().st_size / 1e9
            logger.info(f"  Copied {size_gb:.1f} GB in {elapsed:.1f}s ({size_gb/elapsed:.1f} GB/s)")
    
    return local_path

def load_or_create_split(data_path: Path, logger: logging.Logger, rank: int = 0,
                         is_ddp: bool = False, val_fraction: float = DEFAULT_VAL_FRACTION,
                         original_data_path: Optional[Path] = None
                         ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    cache_source = original_data_path if original_data_path is not None else data_path
    cache_path = cache_source.parent / f"{cache_source.stem}_split_cache.npz"
    
    if cache_path.exists():
        if is_main_process(rank):
            logger.info(f"Loading cached split from: {cache_path}")
        try:
            cache = np.load(cache_path)
            train_idx = cache['train_idx']
            val_idx = cache['val_idx']
            stats = {'median': float(cache['median']), 'iqr': float(cache['iqr'])}
            if is_main_process(rank):
                logger.info(f"Loaded split: train={format_number(len(train_idx))}, "
                          f"val={format_number(len(val_idx))}")
            if is_ddp:
                synchronize()
            return train_idx, val_idx, stats
        except Exception as e:
            if is_main_process(rank):
                logger.warning(f"Failed to load cache: {e}. Creating new split.")
    
    if is_main_process(rank):
        logger.info("Creating new train/val split...")
        with h5py.File(data_path, 'r', rdcc_nbytes=1024*1024*1024) as f:
            labels = f['labels'][:]
            n_samples = len(labels)
            logger.info(f"Total samples: {format_number(n_samples)}")
            
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(indices, test_size=val_fraction,
                                                  shuffle=True, random_state=SEED,
                                                  stratify=labels)
            
            flux_data = f['flux']
            sample_size = min(50000, len(train_idx))
            sample_idx = np.sort(np.random.choice(train_idx, sample_size, replace=False))
            
            valid_flux_chunks: List[np.ndarray] = []
            chunk_size = 10000
            for i in range(0, len(sample_idx), chunk_size):
                chunk_indices = sample_idx[i:i + chunk_size]
                flux_chunk = flux_data[chunk_indices.tolist()]
                valid = flux_chunk[~np.isnan(flux_chunk) & (flux_chunk != 0.0)]
                if len(valid) > 0:
                    valid_flux_chunks.append(valid)
            
            if valid_flux_chunks:
                all_valid = np.concatenate(valid_flux_chunks)
                median = float(np.median(all_valid))
                q75, q25 = np.percentile(all_valid, [75, 25])
                iqr = float(max(q75 - q25, EPS))
            else:
                logger.warning("No valid flux values found, using defaults")
                median, iqr = 0.0, 1.0
            
            stats = {'median': median, 'iqr': iqr}
            
            unique, counts = np.unique(labels[train_idx], return_counts=True)
            logger.info("Training class distribution:")
            for cls_idx, count in zip(unique, counts):
                pct = 100.0 * count / len(train_idx)
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
                logger.info(f"  {cls_name}: {format_number(count)} ({pct:.2f}%)")
        
        np.savez(cache_path, train_idx=train_idx, val_idx=val_idx,
                median=stats['median'], iqr=stats['iqr'])
        logger.info(f"Cached split to: {cache_path}")
    else:
        train_idx, val_idx, stats = None, None, None
    
    if is_ddp:
        synchronize()
        if not is_main_process(rank):
            cache = np.load(cache_path)
            train_idx = cache['train_idx']
            val_idx = cache['val_idx']
            stats = {'median': float(cache['median']), 'iqr': float(cache['iqr'])}
    
    return train_idx, val_idx, stats

class MicrolensingDatasetOptimized(Dataset):
    def __init__(self, data_path: Path, indices: np.ndarray,
                 stats: Dict[str, float]) -> None:
        self.data_path: str = str(data_path)
        self.indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        self.stats: Dict[str, float] = stats
        self._file: Optional[h5py.File] = None
        self._flux: Optional[h5py.Dataset] = None
        self._delta_t: Optional[h5py.Dataset] = None
        self._labels: Optional[h5py.Dataset] = None
        self._worker_id: Optional[int] = None
    
    def _ensure_open(self) -> None:
        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info else -1
        
        if self._file is None or self._worker_id != current_worker:
            self._close()
            self._file = h5py.File(self.data_path, 'r',
                                  rdcc_nbytes=256*1024*1024,
                                  rdcc_nslots=10007, rdcc_w0=0.75,
                                  libver='latest')
            self._flux = self._file['flux']
            self._delta_t = self._file['delta_t']
            self._labels = self._file['labels']
            self._worker_id = current_worker
    
    def _close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._flux = None
            self._delta_t = None
            self._labels = None
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        self._ensure_open()
        global_idx = int(self.indices[idx])
        label = int(self._labels[global_idx])
        return global_idx, label
    
    def get_batch_data(self, global_indices: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_open()
        sort_order = np.argsort(global_indices)
        sorted_idx = global_indices[sort_order]
        flux = self._flux[sorted_idx.tolist()]
        delta_t = self._delta_t[sorted_idx.tolist()]
        unsort_order = np.argsort(sort_order)
        flux = flux[unsort_order]
        delta_t = delta_t[unsort_order]
        return flux, delta_t
    
    def __del__(self) -> None:
        self._close()

def collate_fn_optimized(batch: List[Tuple[int, int]],
                         dataset: MicrolensingDatasetOptimized,
                         median: float, iqr: float
                         ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    global_indices = np.array([x[0] for x in batch], dtype=np.int64)
    labels = np.array([x[1] for x in batch], dtype=np.int64)
    flux, delta_t = dataset.get_batch_data(global_indices)
    
    flux = np.ascontiguousarray(flux, dtype=np.float32)
    delta_t = np.ascontiguousarray(delta_t, dtype=np.float32)
    
    flux = torch.from_numpy(flux)
    delta_t = torch.from_numpy(delta_t)
    labels = torch.from_numpy(labels).long()
    
    valid_mask = (~torch.isnan(flux)) & (flux != 0.0)
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
    
    flux = (flux - median) / iqr
    flux = flux * valid_mask.float()
    lengths = valid_mask.sum(dim=1).long().clamp(min=MIN_SEQUENCE_LENGTH)
    
    return flux, delta_t, lengths, labels

def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

def create_dataloaders(data_path: Path, train_idx: np.ndarray, val_idx: np.ndarray,
                      stats: Dict[str, float], batch_size: int, num_workers: int,
                      prefetch_factor: int, is_ddp: bool, rank: int,
                      generator: torch.Generator
                      ) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    train_dataset = MicrolensingDatasetOptimized(data_path, train_idx, stats)
    val_dataset = MicrolensingDatasetOptimized(data_path, val_idx, stats)
    
    with h5py.File(data_path, 'r') as f:
        all_labels = f['labels'][:]
    train_labels = all_labels[train_idx]
    
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True,
                                          seed=SEED, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    
    median, iqr = stats['median'], stats['iqr']
    train_collate = partial(collate_fn_optimized, dataset=train_dataset,
                           median=median, iqr=iqr)
    val_collate = partial(collate_fn_optimized, dataset=val_dataset,
                         median=median, iqr=iqr)
    
    use_persistent_workers = num_workers > 0
    use_prefetch = prefetch_factor if num_workers > 0 else None
    
    # PERFORMANCE: pin_memory_device for faster transfers (PyTorch 2.0+)
    pin_memory_device = f'cuda:{rank}' if torch.cuda.is_available() else ''
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
        pin_memory_device=pin_memory_device,
        drop_last=True, persistent_workers=use_persistent_workers,
        prefetch_factor=use_prefetch, worker_init_fn=worker_init_fn,
        collate_fn=train_collate, generator=generator)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2,
        shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        pin_memory_device=pin_memory_device,
        drop_last=False, persistent_workers=use_persistent_workers,
        prefetch_factor=use_prefetch, worker_init_fn=worker_init_fn,
        collate_fn=val_collate, generator=generator)
    
    return train_loader, val_loader, train_labels

def compute_class_weights(labels: np.ndarray, n_classes: int,
                          device: torch.device) -> Tensor:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = torch.ones(n_classes, device=device, dtype=torch.float32)
    for cls_idx, count in zip(unique, counts):
        if count > 0:
            weights[cls_idx] = total / (n_classes * count + EPS)
    weights = weights / (weights.mean() + EPS)
    return weights

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-6, last_epoch: int = -1) -> None:
        self.warmup_epochs: int = max(0, warmup_epochs)
        self.total_epochs: int = max(1, total_epochs)
        self.min_lr: float = max(0.0, min_lr)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        epoch = max(0, self.last_epoch)
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / decay_epochs
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor
                   for base_lr in self.base_lrs]

def should_use_grad_scaler(device: torch.device, use_amp: bool) -> bool:
    if not use_amp or device.type != 'cuda':
        return False
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return False
    return True

class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_data: Optional[Tuple[Tensor, ...]] = None
        self._iter: Optional[Any] = None
    
    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self
    
    def _preload(self) -> None:
        try:
            data = next(self._iter)
        except StopIteration:
            self.next_data = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_data = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, Tensor) else t
                for t in data)
    
    def __next__(self) -> Tuple[Tensor, ...]:
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        data = self.next_data
        if data is None:
            raise StopIteration
        for t in data:
            if isinstance(t, Tensor) and t.is_cuda:
                t.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return data
    
    def __len__(self) -> int:
        return len(self.loader)

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
               scaler: Optional[torch.amp.GradScaler], class_weights: Tensor,
               device: torch.device, rank: int, world_size: int, epoch: int,
               config: ModelConfig, accumulation_steps: int = 1,
               clip_norm: float = DEFAULT_CLIP_NORM, use_prefetcher: bool = True
               ) -> Tuple[float, float]:
    model.train()
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    correct = torch.tensor(0, device=device, dtype=torch.int64)
    total = torch.tensor(0, device=device, dtype=torch.int64)
    
    if use_prefetcher and device.type == 'cuda':
        iterator_base = CUDAPrefetcher(loader, device)
    else:
        iterator_base = loader
    
    if is_main_process(rank):
        iterator = tqdm(iterator_base, desc=f"Epoch {epoch}",
                       leave=False, dynamic_ncols=True, mininterval=1.0)
    else:
        iterator = iterator_base
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config.use_amp
    use_scaler = scaler is not None
    amp_dtype = (torch.bfloat16 if (use_amp and hasattr(torch.cuda, 'is_bf16_supported')
                                   and torch.cuda.is_bf16_supported())
                else torch.float16)
    
    num_batches = len(loader)
    
    for batch_idx, batch_data in enumerate(iterator):
        if use_prefetcher and device.type == 'cuda':
            flux, delta_t, lengths, labels = batch_data
        else:
            flux, delta_t, lengths, labels = batch_data
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = model(flux, delta_t, lengths)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss_scaled = loss / accumulation_steps
        
        if use_scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        
        is_last_batch = (batch_idx + 1) == num_batches
        is_accum_step = ((batch_idx + 1) % accumulation_steps == 0) or is_last_batch
        
        if is_accum_step:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            batch_size = labels.size(0)
            total_loss += loss.detach().double() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum()
            total += batch_size
        
        if is_main_process(rank) and batch_idx % PROGRESS_UPDATE_FREQ == 0:
            current_loss = (total_loss / max(1, total)).item()
            current_acc = (correct / max(1, total)).float().item() * 100
            if hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'loss': f'{current_loss:.4f}',
                                     'acc': f'{current_acc:.1f}%'})
    
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    total_samples = max(1, total.item())
    avg_loss = (total_loss / total_samples).item()
    accuracy = (correct / total_samples).float().item()
    
    return avg_loss, accuracy

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, class_weights: Tensor,
            device: torch.device, rank: int, world_size: int, config: ModelConfig,
            return_predictions: bool = False, use_prefetcher: bool = True
            ) -> Dict[str, Any]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    correct = torch.tensor(0, device=device, dtype=torch.int64)
    total = torch.tensor(0, device=device, dtype=torch.int64)
    
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    
    use_amp = config.use_amp
    amp_dtype = (torch.bfloat16 if (use_amp and hasattr(torch.cuda, 'is_bf16_supported')
                                   and torch.cuda.is_bf16_supported())
                else torch.float16)
    
    if use_prefetcher and device.type == 'cuda':
        iterator_base = CUDAPrefetcher(loader, device)
    else:
        iterator_base = loader
    
    if is_main_process(rank):
        iterator = tqdm(iterator_base, desc="Eval",
                       leave=False, dynamic_ncols=True, mininterval=1.0)
    else:
        iterator = iterator_base
    
    for batch_data in iterator:
        if use_prefetcher and device.type == 'cuda':
            flux, delta_t, lengths, labels = batch_data
        else:
            flux, delta_t, lengths, labels = batch_data
            flux = flux.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = model(flux, delta_t, lengths)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        
        batch_size = labels.size(0)
        preds = logits.argmax(dim=-1)
        
        total_loss += loss.double() * batch_size
        correct += (preds == labels).sum()
        total += batch_size
        
        if return_predictions:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    total_samples = max(1, total.item())
    results: Dict[str, Any] = {
        'loss': (total_loss / total_samples).item(),
        'accuracy': (correct / total_samples).float().item()}
    
    if return_predictions:
        results['predictions'] = np.concatenate(all_preds)
        results['labels'] = np.concatenate(all_labels)
        results['probabilities'] = np.concatenate(all_probs)
    
    return results

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   scaler: Optional[torch.amp.GradScaler], epoch: int,
                   best_acc: float, config: ModelConfig, stats: Dict[str, float],
                   output_dir: Path, is_best: bool = False,
                   extra_info: Optional[Dict[str, Any]] = None) -> Path:
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    elif hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'config': config.to_dict() if hasattr(config, 'to_dict') else asdict(config),
        'stats': stats, 'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'timestamp': datetime.now().isoformat()}
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if extra_info:
        checkpoint['extra_info'] = extra_info
    
    save_path = output_dir / ('best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt')
    temp_path = save_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(save_path)
    
    return save_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train Roman Microlensing Classifier (ULTRA-OPTIMIZED)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, required=True)
    data_group.add_argument('--output-dir', type=str, default='../results')
    data_group.add_argument('--experiment-name', type=str, default=None)
    data_group.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION)
    
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--d-model', type=int, default=64)
    model_group.add_argument('--n-layers', type=int, default=2)
    model_group.add_argument('--dropout', type=float, default=0.3)
    model_group.add_argument('--window-size', type=int, default=5)
    model_group.add_argument('--hierarchical', action='store_true', default=True)
    model_group.add_argument('--no-hierarchical', dest='hierarchical', action='store_false')
    model_group.add_argument('--attention-pooling', action='store_true', default=True)
    model_group.add_argument('--no-attention-pooling', dest='attention_pooling',
                            action='store_false')
    model_group.add_argument('--num-attention-heads', type=int, default=1)
    model_group.add_argument('--use-residual', action='store_true', default=True)
    model_group.add_argument('--no-residual', dest='use_residual', action='store_false')
    model_group.add_argument('--use-layer-norm', action='store_true', default=True)
    model_group.add_argument('--no-layer-norm', dest='use_layer_norm', action='store_false')
    
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch-size', type=int, default=768,
                            help='Batch size per GPU (increased from 512)')
    train_group.add_argument('--accumulation-steps', type=int, default=1)
    train_group.add_argument('--epochs', type=int, default=50)
    train_group.add_argument('--lr', type=float, default=1e-2)
    train_group.add_argument('--weight-decay', type=float, default=1e-3)
    train_group.add_argument('--warmup-epochs', type=int, default=5)
    train_group.add_argument('--clip-norm', type=float, default=DEFAULT_CLIP_NORM)
    train_group.add_argument('--min-lr', type=float, default=1e-6)
    
    amp_group = parser.add_argument_group('Mixed Precision')
    amp_group.add_argument('--use-amp', action='store_true', default=True)
    amp_group.add_argument('--no-amp', dest='use_amp', action='store_false')
    
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--compile', action='store_true', default=True)
    perf_group.add_argument('--no-compile', dest='compile', action='store_false')
    perf_group.add_argument('--compile-mode', type=str, default='max-autotune',
                           choices=['default', 'reduce-overhead', 'max-autotune'],
                           help='Compilation mode (max-autotune for best performance)')
    perf_group.add_argument('--use-prefetcher', action='store_true', default=True)
    perf_group.add_argument('--no-prefetcher', dest='use_prefetcher', action='store_false')
    
    other_group = parser.add_argument_group('Other')
    other_group.add_argument('--use-class-weights', action='store_true', default=True)
    other_group.add_argument('--no-class-weights', dest='use_class_weights', action='store_false')
    other_group.add_argument('--use-gradient-checkpointing', action='store_true', default=False)
    other_group.add_argument('--num-workers', type=int, default=4)
    other_group.add_argument('--prefetch-factor', type=int, default=12,
                            help='Prefetch factor (increased from 8)')
    other_group.add_argument('--eval-every', type=int, default=10)
    other_group.add_argument('--save-every', type=int, default=10)
    other_group.add_argument('--early-stopping-patience', type=int, default=20)
    other_group.add_argument('--broadcast-buffers', action='store_true', default=True)
    other_group.add_argument('--no-broadcast-buffers', dest='broadcast_buffers',
                            action='store_false')
    other_group.add_argument('--seed', type=int, default=SEED)
    other_group.add_argument('--local-copy', action='store_true', default=False)
    other_group.add_argument('--local-dir', type=str, default=None)
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if args.experiment_name is None:
        timestamp = get_timestamp()
        args.experiment_name = f"roman_d{args.d_model}_l{args.n_layers}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.experiment_name
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_ddp:
        synchronize()
    
    logger = setup_logging(rank, output_dir)
    generator = set_global_seeds(args.seed, rank)
    
    if is_main_process(rank):
        logger.info("=" * 80)
        logger.info("ROMAN MICROLENSING CLASSIFIER - ULTRA-OPTIMIZED")
        logger.info("=" * 80)
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA: {torch.version.cuda}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
            logger.info(f"GPU memory: {gpu_mem:.1f} GB")
            logger.info("Performance settings:")
            logger.info(f"  torch.compile: {args.compile} (mode={args.compile_mode})")
            logger.info(f"  CUDA prefetcher: {args.use_prefetcher}")
            logger.info(f"  TF32: {torch.backends.cuda.matmul.allow_tf32}")
            logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            logger.info(f"  Batch size per GPU: {args.batch_size} (optimized)")
            logger.info(f"  Prefetch factor: {args.prefetch_factor} (optimized)")
            if hasattr(torch.cuda, 'is_bf16_supported'):
                logger.info(f"  BF16: {torch.cuda.is_bf16_supported()}")
        logger.info(f"World size: {world_size} GPU(s)")
        logger.info(f"Output: {output_dir}")
    
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Loading data...")
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    original_data_path = data_path
    if args.local_copy:
        if is_main_process(rank):
            logger.info("Copying to local storage...")
        data_path = copy_to_local_storage(data_path, logger, rank, args.local_dir)
        if is_ddp:
            synchronize()
    
    train_idx, val_idx, stats = load_or_create_split(
        data_path, logger, rank, is_ddp, args.val_fraction,
        original_data_path=original_data_path if args.local_copy else None)
    
    if is_main_process(rank):
        logger.info(f"Dataset: {format_number(len(train_idx) + len(val_idx))} samples")
        logger.info(f"Train: {format_number(len(train_idx))}, Val: {format_number(len(val_idx))}")
    
    train_loader, val_loader, train_labels = create_dataloaders(
        data_path, train_idx, val_idx, stats, args.batch_size, args.num_workers,
        args.prefetch_factor, is_ddp, rank, generator)
    
    effective_batch_size = args.batch_size * args.accumulation_steps * world_size
    
    if is_main_process(rank):
        logger.info(f"Batch per GPU: {args.batch_size}")
        logger.info(f"Effective batch: {effective_batch_size}")
        logger.info(f"Workers per GPU: {args.num_workers}")
        logger.info(f"Prefetch: {args.prefetch_factor}")
    
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Building model...")
    
    config = ModelConfig(
        d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout,
        window_size=args.window_size, hierarchical=args.hierarchical,
        use_residual=args.use_residual, use_layer_norm=args.use_layer_norm,
        feature_extraction='conv', use_attention_pooling=args.attention_pooling,
        num_attention_heads=args.num_attention_heads, use_amp=args.use_amp,
        use_gradient_checkpointing=args.use_gradient_checkpointing)
    
    model = RomanMicrolensingClassifier(config).to(device)
    
    if is_main_process(rank):
        n_params = model.count_parameters()
        logger.info(f"Parameters: {format_number(n_params)}")
        logger.info(f"Receptive field: {model.receptive_field} timesteps")
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   gradient_as_bucket_view=True, find_unused_parameters=False,
                   broadcast_buffers=args.broadcast_buffers, static_graph=False)
    
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank):
            logger.info(f"Compiling (mode={args.compile_mode})...")
            logger.info("  First epoch will be slower due to compilation")
        model = torch.compile(model, mode=args.compile_mode,
                             fullgraph=False, dynamic=True)
    
    fused_available = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8, fused=fused_available)
    
    if is_main_process(rank) and fused_available:
        logger.info("Using fused AdamW")
    
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, min_lr=args.min_lr)
    
    use_scaler = should_use_grad_scaler(device, args.use_amp)
    if use_scaler:
        scaler = torch.amp.GradScaler('cuda', enabled=True,
                                     init_scale=65536.0, growth_factor=2.0,
                                     backoff_factor=0.5, growth_interval=2000)
    else:
        scaler = None
    
    if is_main_process(rank):
        if args.use_amp:
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                logger.info("Using bfloat16")
            else:
                logger.info("Using float16 with GradScaler")
        else:
            logger.info("Mixed precision disabled")
    
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, config.n_classes, device)
        if is_main_process(rank):
            logger.info(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        class_weights = torch.ones(config.n_classes, device=device)
    
    if is_ddp:
        synchronize()
        torch.cuda.synchronize()
    
    if is_main_process(rank):
        logger.info("-" * 80)
        logger.info("Starting training...")
        logger.info("-" * 80)
    
    if is_ddp:
        synchronize()
    
    best_acc: float = 0.0
    patience_counter: int = 0
    training_start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            
            if is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scaler, class_weights,
                device, rank, world_size, epoch, config,
                accumulation_steps=args.accumulation_steps,
                clip_norm=args.clip_norm, use_prefetcher=args.use_prefetcher)
            
            scheduler.step()
            
            should_eval = (epoch % args.eval_every == 0 or
                          epoch == 1 or epoch == args.epochs)
            
            if should_eval:
                val_results = evaluate(
                    model, val_loader, class_weights, device, rank, world_size, config,
                    return_predictions=(epoch == args.epochs and is_main_process(rank)),
                    use_prefetcher=args.use_prefetcher)
                
                val_loss = val_results['loss']
                val_acc = val_results['accuracy']
                
                if is_main_process(rank):
                    epoch_time = time.time() - epoch_start_time
                    samples_per_sec = len(train_idx) / epoch_time
                    
                    logger.info(
                        f"Epoch {epoch:3d}/{args.epochs} | "
                        f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                        f"Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
                        f"LR={scheduler.get_last_lr()[0]:.2e} | "
                        f"Time={format_time(epoch_time)} ({samples_per_sec:.0f} samp/s)")
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        patience_counter = 0
                        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                      best_acc, config, stats, output_dir, is_best=True)
                        logger.info(f"  -> New best: {best_acc*100:.2f}%")
                    else:
                        patience_counter += 1
                    
                    if epoch % args.save_every == 0:
                        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                      best_acc, config, stats, output_dir, is_best=False)
            else:
                if is_main_process(rank):
                    epoch_time = time.time() - epoch_start_time
                    samples_per_sec = len(train_idx) / epoch_time
                    logger.info(
                        f"Epoch {epoch:3d}/{args.epochs} | "
                        f"Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                        f"LR={scheduler.get_last_lr()[0]:.2e} | "
                        f"Time={format_time(epoch_time)} ({samples_per_sec:.0f} samp/s)")
            
            if args.early_stopping_patience > 0:
                if patience_counter >= args.early_stopping_patience:
                    if is_main_process(rank):
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    except KeyboardInterrupt:
        if is_main_process(rank):
            logger.info("Training interrupted")
    
    finally:
        total_time = time.time() - training_start_time
        
        if is_main_process(rank):
            logger.info("=" * 80)
            logger.info("Training complete")
            logger.info(f"Total time: {format_time(total_time)}")
            logger.info(f"Best val accuracy: {best_acc*100:.2f}%")
            logger.info(f"Average throughput: {len(train_idx) * args.epochs / total_time:.0f} samp/s")
            logger.info("=" * 80)
            
            best_checkpoint_path = output_dir / 'best_model.pt'
            if best_checkpoint_path.exists():
                checkpoint = torch.load(best_checkpoint_path, map_location=device,
                                       weights_only=False)
                
                base_model = model
                while hasattr(base_model, '_orig_mod'):
                    base_model = base_model._orig_mod
                if isinstance(base_model, DDP):
                    base_model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    base_model.load_state_dict(checkpoint['model_state_dict'])
                
                logger.info("Running final evaluation...")
                final_results = evaluate(model, val_loader, class_weights, device,
                                       rank, world_size, config,
                                       return_predictions=True,
                                       use_prefetcher=args.use_prefetcher)
                
                np.savez(output_dir / 'final_predictions.npz',
                        predictions=final_results['predictions'],
                        labels=final_results['labels'],
                        probabilities=final_results['probabilities'])
                
                cm = confusion_matrix(final_results['labels'], final_results['predictions'])
                logger.info(f"\nConfusion matrix:\n{cm}")
                np.save(output_dir / 'confusion_matrix.npy', cm)
                
                report = classification_report(final_results['labels'],
                                              final_results['predictions'],
                                              target_names=list(CLASS_NAMES), digits=4)
                logger.info(f"\nClassification report:\n{report}")
                
                with open(output_dir / 'classification_report.txt', 'w') as f:
                    f.write(report)
            
            config_dict = {
                'model_config': config.to_dict(),
                'training_args': vars(args),
                'stats': stats,
                'best_accuracy': float(best_acc),
                'world_size': world_size,
                'total_training_time_seconds': total_time,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'timestamp': datetime.now().isoformat(),
                'optimizations': {
                    'torch_compile': args.compile,
                    'compile_mode': args.compile_mode,
                    'cuda_prefetcher': args.use_prefetcher,
                    'fused_optimizer': fused_available,
                    'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'batch_size_optimized': args.batch_size,
                    'prefetch_factor_optimized': args.prefetch_factor}}
            
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"\nResults saved to: {output_dir}")
        
        cleanup_distributed()

if __name__ == '__main__':
    main()
