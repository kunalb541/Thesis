#!/usr/bin/env python3
"""
Transformer-Based Microlensing Classifier with DDP (Fully Debugged)

This script trains a Transformer model to classify microlensing light curves
(PSPL vs Binary) using PyTorch DistributedDataParallel for multi-GPU/multi-node setups.

Author: Kunal Bhatia
Date: November 2025
Version: 5.2 (Fully Debugged, No Placeholders)
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------------------------------------------------------
# MODEL AND UTILITIES IMPORT
# -------------------------------------------------------------------------
from model import TransformerClassifier, count_parameters
from utils import load_npz_dataset


# -------------------------------------------------------------------------
# DDP HELPER FUNCTIONS
# -------------------------------------------------------------------------

def setup_ddp():
    """Initialize torch.distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    is_ddp = world_size > 1

    if is_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank, is_ddp


def cleanup_ddp():
    """Safely terminate DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Return True if this is the main process."""
    return rank == 0


def log_main(message, rank):
    """Print only on main process."""
    if is_main_process(rank):
        print(message, flush=True)


# -------------------------------------------------------------------------
# DATASET CLASS
# -------------------------------------------------------------------------

class MicrolensingDataset(Dataset):
    """Dataset wrapper for microlensing light curves."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------------------
# NORMALIZATION
# -------------------------------------------------------------------------

def normalize_data(X_train, X_val, X_test, pad_value=-1.0):
    """Standard + MinMax normalization (ignores padded values)."""
    N_train, C, T = X_train.shape
    N_val, N_test = X_val.shape[0], X_test.shape[0]
    F = C * T

    X_train_flat = X_train.reshape(N_train, F)
    X_val_flat = X_val.reshape(N_val, F)
    X_test_flat = X_test.reshape(N_test, F)

    pad_train = (X_train_flat == pad_value)
    pad_val = (X_val_flat == pad_value)
    pad_test = (X_test_flat == pad_value)

    means = np.zeros(F, dtype=np.float32)
    for j in range(F):
        valid = X_train_flat[:, j] != pad_value
        means[j] = X_train_flat[valid, j].mean() if valid.any() else 0.0

    X_train_filled = np.where(pad_train, means, X_train_flat)
    X_val_filled = np.where(pad_val, means, X_val_flat)
    X_test_filled = np.where(pad_test, means, X_test_flat)

    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train_filled)
    X_val_std = scaler_std.transform(X_val_filled)
    X_test_std = scaler_std.transform(X_test_filled)

    scaler_mm = MinMaxScaler((0, 1))
    X_train_norm = scaler_mm.fit_transform(X_train_std)
    X_val_norm = scaler_mm.transform(X_val_std)
    X_test_norm = scaler_mm.transform(X_test_std)

    X_train_norm[pad_train] = pad_value
    X_val_norm[pad_val] = pad_value
    X_test_norm[pad_test] = pad_value

    X_train_norm = X_train_norm.reshape(N_train, C, T)
    X_val_norm = X_val_norm.reshape(N_val, C, T)
    X_test_norm = X_test_norm.reshape(N_test, C, T)

    return X_train_norm, X_val_norm, X_test_norm, scaler_std, scaler_mm


# -------------------------------------------------------------------------
# TRAINING AND EVALUATION FUNCTIONS
# -------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, is_ddp, epoch):
    model.train()
    if is_ddp and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    total_loss, correct, total = 0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)

    return total_loss, correct, total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X, return_sequence=False)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)

    return total_loss, correct, total


def aggregate_metrics(local_loss, local_correct, local_total, world_size, device):
    """Aggregate metrics across all DDP processes."""
    if world_size > 1:
        metrics = torch.tensor([local_loss, local_correct, local_total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        global_loss, global_correct, global_total = metrics.tolist()
    else:
        global_loss, global_correct, global_total = local_loss, local_correct, local_total

    if global_total == 0:
        return 0.0, 0.0

    return global_loss / global_total, global_correct / global_total


# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Transformer classifier with DDP")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--experiment_name", default="transformer", help="Experiment name")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--downsample_factor", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # ------------------------ DDP Setup ------------------------
    rank, world_size, local_rank, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process(rank)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    log_main(f"[Rank {rank}] Initialized DDP (world_size={world_size}, device={device})", rank)

    # ------------------------ Load Data ------------------------
    X, y, _, meta = load_npz_dataset(args.data, apply_perm=True)
    pad_value = meta.get("PAD_VALUE", -1.0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed)

    X_train, X_val, X_test, scaler_std, scaler_mm = normalize_data(X_train, X_val, X_test, pad_value)

    train_ds = MicrolensingDataset(X_train, y_train)
    val_ds = MicrolensingDataset(X_val, y_val)
    test_ds = MicrolensingDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, sampler=val_sampler,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, sampler=test_sampler,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ------------------------ Model Setup ------------------------
    model = TransformerClassifier(
        in_channels=X_train.shape[1],
        n_classes=2,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        downsample_factor=args.downsample_factor,
        dropout=args.dropout,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        model_base = model.module
    else:
        model_base = model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------ Experiment Directory ------------------------
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        config = vars(args)
        config.update({
            "n_parameters": count_parameters(model_base),
            "world_size": world_size,
            "data_file": args.data
        })
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(exp_dir / "scaler_standard.pkl", "wb") as f:
            pickle.dump(scaler_std, f)
        with open(exp_dir / "scaler_minmax.pkl", "wb") as f:
            pickle.dump(scaler_mm, f)
    else:
        dist.barrier()
        exp_dir = sorted(Path("../results").glob(f"{args.experiment_name}_*"))[-1]

    # ------------------------ Training Loop ------------------------
    best_val_acc, patience = 0.0, 0

    for epoch in range(args.epochs):
        train_loss_local, train_correct_local, train_total_local = train_epoch(model, train_loader, criterion, optimizer, device, is_ddp, epoch)
        val_loss_local, val_correct_local, val_total_local = evaluate(model, val_loader, criterion, device)

        train_loss, train_acc = aggregate_metrics(train_loss_local, train_correct_local, train_total_local, world_size, device)
        val_loss, val_acc = aggregate_metrics(val_loss_local, val_correct_local, val_total_local, world_size, device)

        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model_base.state_dict(), exp_dir / "best_model.pt")
            else:
                patience += 1

        stop_signal = torch.tensor(int(patience >= args.patience), device=device)
        if is_ddp:
            dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    # ------------------------ Evaluation ------------------------
    if is_main:
        model_base.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
        print(f"\nLoaded best model from {exp_dir}")
    if is_ddp:
        dist.barrier()

    test_loss_local, test_correct_local, test_total_local = evaluate(model, test_loader, criterion, device)
    test_loss, test_acc = aggregate_metrics(test_loss_local, test_correct_local, test_total_local, world_size, device)

    if is_main:
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        with open(exp_dir / "results.json", "w") as f:
            json.dump({
                "best_val_acc": float(best_val_acc),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss),
                "epochs_trained": epoch + 1
            }, f, indent=2)

    cleanup_ddp()


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()