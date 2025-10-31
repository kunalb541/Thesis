#!/usr/bin/env python3
# train.py
# Multi-node / multi-GPU friendly training script with clean logging,
# single-rank artifact saving, consistent two-stage normalization,
# and graceful DDP shutdown to prevent TCPStore "Broken pipe" warnings.
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import json
import os
import pickle
import random
import socket
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -----------------------------
# Utilities
# -----------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def setup_ddp(backend: Optional[str] = None):
    """Initialize torch.distributed from torchrun environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        
        # FIX: Explicitly set device_ids for the barrier to suppress NCCL warnings
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()
            
        return rank, world_size, local_rank
    else:
        # Single-process (no DDP)
        return 0, 1, 0


def set_seeds(seed: int, rank: int):
    seed = seed + rank  # make rank-unique but deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Sum-reduce tensor across processes. No-op in single-process."""
    if is_dist():
        dist.all_reduce(t, op=op)
    return t


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_experiment_dir(base: str, exp_name: str) -> str:
    ts = now_stamp()
    exp_dir = os.path.join("..", "results", f"{exp_name}_{ts}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    return exp_dir


def rank0_print(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


# -----------------------------
# Data
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X expected shape: (N, 1, T); y: (N,)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_npz_files(pattern: str) -> Dict[str, np.ndarray]:
    """
    Load one or many .npz files (supports glob patterns).
    Expected keys (flexible):
      - Prefer 'X' and 'y'
      - Alternatively 'magnifications' (X) and 'labels' (y)
    Concatenates across files if multiple given.
    """
    files = sorted(glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched data path: {pattern}")

    X_list, y_list = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            if "X" in data and "y" in data:
                X_list.append(data["X"])
                y_list.append(data["y"])
            elif "magnifications" in data and "labels" in data:
                X_list.append(data["magnifications"])
                y_list.append(data["labels"])
            else:
                raise KeyError(
                    f"{f} does not contain expected keys. "
                    f"Expected ('X', 'y') or ('magnifications', 'labels'). "
                    f"Found keys: {list(data.keys())}"
                )

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Ensure X has shape (N, 1, T)
    if X.ndim == 2:
        X = X[:, None, :]  # add channel dim
    elif X.ndim == 3 and X.shape[1] != 1:
        # If shape is (N, T, C), move to (N, C, T) or force single-channel if ambiguous
        if X.shape[-1] == 1:
            X = np.transpose(X, (0, 2, 1))
        else:
            # Fall back: treat last dimension as time and add a singleton channel
            X = X[..., 0]  # take first channel
            X = X[:, None, :]

    y = y.reshape(-1)
    return {"X": X, "y": y}


def fit_two_stage_scalers_rank0(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fit StandardScaler then MinMaxScaler on train data ONLY (on rank 0).
    Returns a dict of parameters to broadcast & reconstruct scalers in all ranks.
    """
    # Fit per-timepoint across the time axis jointly (flatten N and C dims)
    N, C, T = X_train.shape
    flat = X_train.reshape(N, C * T)

    std = StandardScaler(with_mean=True, with_std=True)
    flat_std = std.fit_transform(flat)

    mm = MinMaxScaler(feature_range=(0.0, 1.0))
    mm.fit(flat_std)

    params = {
        "std_mean": std.mean_.astype(np.float64),
        "std_scale": std.scale_.astype(np.float64),
        "mm_min": mm.min_.astype(np.float64),
        "mm_scale": mm.scale_.astype(np.float64),
        "shape": (C, T),
    }
    return params


def apply_two_stage_scalers(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """Apply StandardScaler then MinMaxScaler using provided params."""
    N, C, T = X.shape
    assert (C, T) == tuple(params["shape"]), "Scaler params shape mismatch."
    flat = X.reshape(N, C * T)

    flat = (flat - params["std_mean"]) / (params["std_scale"] + 1e-12)
    flat = flat * params["mm_scale"] + params["mm_min"]
    return flat.reshape(N, C, T)


def save_scalers_rank0(params: Dict[str, np.ndarray], out_dir: str):
    """Reconstruct sklearn scalers and persist to disk on rank 0 only."""
    if get_rank() != 0:
        return
    # Recreate sklearn objects from params (for reproducible inference)
    std = StandardScaler(with_mean=True, with_std=True)
    mm = MinMaxScaler(feature_range=(0.0, 1.0))
    # Minimal objects: just attach learned stats
    std.mean_ = params["std_mean"].copy()
    std.scale_ = params["std_scale"].copy()

    # For MinMaxScaler, scikit stores min_ and scale_. Data min/max not needed for transform
    mm.min_ = params["mm_min"].copy()
    mm.scale_ = params["mm_scale"].copy()
    mm.data_min_ = None
    mm.data_max_ = None
    mm.data_range_ = None
    mm.n_samples_seen_ = None
    mm.feature_range = (0.0, 1.0)

    with open(os.path.join(out_dir, "scaler_standard.pkl"), "wb") as f:
        pickle.dump(std, f)
    with open(os.path.join(out_dir, "scaler_minmax.pkl"), "wb") as f:
        pickle.dump(mm, f)


# FIX: Replaced the entire function with the modern, robust dist.broadcast_object_list
def broadcast_object_from_rank0(obj):
    """
    Broadcast a Python object from rank 0 to all other ranks.
    'obj' is the object to broadcast on rank 0, and should be 'None'
    on all other ranks.
    """
    world_size = get_world_size()
    if world_size == 1:
        return obj

    # We wrap the object in a list because broadcast_object_list
    # expects a list of objects.
    if get_rank() == 0:
        obj_list = [obj]
    else:
        obj_list = [None] # Placeholder

    # This function handles serialization, size, and broadcast internally
    # for both CPU and CUDA tensors, using the correct backend.
    dist.broadcast_object_list(obj_list, src=0)
    
    # Return the broadcasted object
    return obj_list[0]


# -----------------------------
# Model
# -----------------------------
class TDConvClassifier(nn.Module):
    """
    Compact 1D CNN for binary classification.
    Matches the README spirit: stacked Conv1D, BN, ReLU, Dropout,
    then aggregate (mean + max) over time and classify.
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 64),  # concat(mean, max)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B, C=1, T)
        h = self.net(x)
        mean_pool = torch.mean(h, dim=-1)
        max_pool, _ = torch.max(h, dim=-1)
        z = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(z)


# -----------------------------
# Train / Eval
# -----------------------------
def step(model, batch, device, criterion, scaler, optimizer=None):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    is_train = optimizer is not None
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(x)
        loss = criterion(logits, y)
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum()

    if is_train:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss.detach(), correct.detach(), y.numel()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    count_sum = torch.tensor(0.0, device=device)

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        correct = (pred == y).sum()

        loss_sum += loss * y.numel()
        correct_sum += correct
        count_sum += y.numel()

    # Reduce across ranks
    reduce_tensor(loss_sum)
    reduce_tensor(correct_sum)
    reduce_tensor(count_sum)

    loss_avg = (loss_sum / count_sum).item() if count_sum.item() > 0 else 0.0
    acc = (correct_sum / count_sum).item() if count_sum.item() > 0 else 0.0
    return loss_avg, acc


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path or glob to .npz dataset(s)")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seeds(args.seed, rank)

    # Load data (each rank loads the same; deterministic split ensures consistency)
    if rank == 0:
        rank0_print("=" * 76)
        rank0_print("Host:", socket.gethostname())
        rank0_print(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        rank0_print(f"Data pattern: {args.data}")
        rank0_print("=" * 76)
    data = load_npz_files(args.data)
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y) + 1)

    # Deterministic split
    test_size = args.test_split
    val_size = args.val_split / (1.0 - test_size)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=args.seed, stratify=y_trainval
    )

    # -----------------------------
    # Two-stage normalization (fit on rank 0) and broadcast params
    # -----------------------------
    if rank == 0:
        scaler_params = fit_two_stage_scalers_rank0(X_train)
    else:
        scaler_params = None
    scaler_params = broadcast_object_from_rank0(scaler_params)

    X_train = apply_two_stage_scalers(scaler_params, X_train)
    X_val = apply_two_stage_scalers(scaler_params, X_val)
    X_test = apply_two_stage_scalers(scaler_params, X_test)

    # -----------------------------
    # Datasets & Loaders with DistributedSampler
    # -----------------------------
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # -----------------------------
    # Model / Optimizer
    # -----------------------------
    model = TDConvClassifier(in_ch=X.shape[1], n_classes=n_classes, dropout=0.3).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -----------------------------
    # Experiment directory (created by rank 0 and broadcast to others)
    # -----------------------------
    if rank == 0:
        exp_dir = make_experiment_dir("..", args.experiment_name)
    else:
        exp_dir = None
    exp_dir = broadcast_object_from_rank0(exp_dir)

    if rank == 0:
        (Path(exp_dir) / "evaluation").mkdir(parents=True, exist_ok=True)
        # Save config immediately (single-rank)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        # Persist scalers (rank 0 only)
        save_scalers_rank0(scaler_params, exp_dir)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        model.train()

        loss_sum = torch.tensor(0.0, device=device)
        correct_sum = torch.tensor(0.0, device=device)
        count_sum = torch.tensor(0.0, device=device)

        for batch in train_loader:
            loss, correct, count = step(model, batch, device, criterion, scaler, optimizer)
            loss_sum += loss * count
            correct_sum += correct
            count_sum += torch.tensor(count, device=device)

        # Reduce across ranks
        reduce_tensor(loss_sum)
        reduce_tensor(correct_sum)
        reduce_tensor(count_sum)

        train_loss = (loss_sum / count_sum).item() if count_sum.item() > 0 else 0.0
        train_acc = (correct_sum / count_sum).item() if count_sum.item() > 0 else 0.0

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        if get_rank() == 0:
            epoch_time = time.time() - start_time
            rank0_print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | time {epoch_time:.1f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    (model.module if isinstance(model, DDP) else model).state_dict(),
                    os.path.join(exp_dir, "best_model.pt"),
                )

        if is_dist():
            dist.barrier()

    # -----------------------------
    # Final evaluation on test set
    # -----------------------------
    # Load best weights on all ranks for consistent evaluation
    if rank == 0:
        best_path = os.path.join(exp_dir, "best_model.pt")
        state_dict = torch.load(best_path, map_location="cpu")
    else:
        state_dict = None
    # Broadcast state_dict to all ranks
    state_dict = broadcast_object_from_rank0(state_dict)
    (model.module if isinstance(model, DDP) else model).load_state_dict(state_dict)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)

    # Log summary (rank 0 only)
    if rank == 0:
        summary = {
            "final_test_loss": float(test_loss),
            "final_test_acc": float(test_acc),
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "results_dir": exp_dir,
        }
        with open(os.path.join(exp_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        rank0_print("\n" + "=" * 80)
        rank0_print("FINAL EVALUATION")
        rank0_print("=" * 80)
        rank0_print(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
        rank0_print(f"Best  | val acc {best_val_acc:.4f} (epoch {best_epoch})\n")
        rank0_print(f"✓ Training complete! Results saved to: {exp_dir}")

    # -----------------------------
    # Graceful DDP shutdown
    # -----------------------------
    # Fix for TCPStore "Broken pipe"/rendezvous shutdown warnings:
    # 1) Ensure all ranks reach the barrier after logging/saving
    # 2) Destroy the process group cleanly
    if is_dist():
        # Give rank 0 a moment to finish file I/O
        time.sleep(1.0)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
