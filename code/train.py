#!/usr/bin/env python3
# train_timedistributed.py - Training with REAL TimeDistributed architecture
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import pickle
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import TimeDistributed models
from model_timedistributed import TimeDistributedCNNSimple, TimeDistributedCNN


# Copy all utility functions from train_fixed.py
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def setup_ddp(backend=None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()
        return rank, world_size, local_rank
    return 0, 1, 0

def set_seeds(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reduce_tensor(t, op=dist.ReduceOp.SUM):
    if is_dist():
        dist.all_reduce(t, op=op)
    return t

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_experiment_dir(base, exp_name):
    ts = now_stamp()
    exp_dir = os.path.join("..", "results", f"{exp_name}_{ts}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    return exp_dir

def rank0_print(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
        sys.stdout.flush()

def broadcast_object_from_rank0(obj):
    world_size = get_world_size()
    if world_size == 1:
        return obj
    if get_rank() == 0:
        obj_list = [obj]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data_with_shuffle(path):
    with np.load(path, allow_pickle=True) as data:
        X = data['X']
        y = data['y']
        if 'perm' in data.files:
            perm = data['perm']
            X = X[perm]
            y = y[perm]
            rank0_print("✓ Applied saved permutation (data shuffled)")
        else:
            rank0_print("⚠️  No permutation found - shuffling now")
            np.random.seed(42)
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
    if y.dtype.kind in ('U', 'S', 'O'):
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.int64)
    else:
        y = y.astype(np.int64)
    return X, y

def fit_two_stage_scalers_rank0(X_train):
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

def apply_two_stage_scalers(params, X):
    N, C, T = X.shape
    assert (C, T) == tuple(params["shape"])
    flat = X.reshape(N, C * T)
    flat = (flat - params["std_mean"]) / (params["std_scale"] + 1e-12)
    flat = flat * params["mm_scale"] + params["mm_min"]
    return flat.reshape(N, C, T)

def save_scalers_rank0(params, out_dir):
    if get_rank() != 0:
        return
    std = StandardScaler(with_mean=True, with_std=True)
    mm = MinMaxScaler(feature_range=(0.0, 1.0))
    std.mean_ = params["std_mean"].copy()
    std.scale_ = params["std_scale"].copy()
    std.n_features_in_ = len(std.mean_)
    mm.min_ = params["mm_min"].copy()
    mm.scale_ = params["mm_scale"].copy()
    mm.n_features_in_ = len(mm.min_)
    mm.feature_range = (0.0, 1.0)
    with open(os.path.join(out_dir, "scaler_standard.pkl"), "wb") as f:
        pickle.dump(std, f)
    with open(os.path.join(out_dir, "scaler_minmax.pkl"), "wb") as f:
        pickle.dump(mm, f)


def step_timedistributed(model, batch, device, criterion, scaler, optimizer=None):
    """Training step for TimeDistributed model"""
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    is_train = optimizer is not None
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        # Get final prediction (not sequence)
        logits = model(x, return_sequence=False)
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
        
        # Final prediction only
        logits = model(x, return_sequence=False)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        correct = (pred == y).sum()

        loss_sum += loss * y.numel()
        correct_sum += correct
        count_sum += y.numel()

    reduce_tensor(loss_sum)
    reduce_tensor(correct_sum)
    reduce_tensor(count_sum)

    loss_avg = (loss_sum / count_sum).item() if count_sum.item() > 0 else 0.0
    acc = (correct_sum / count_sum).item() if count_sum.item() > 0 else 0.0
    return loss_avg, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    
    # TimeDistributed options
    parser.add_argument("--use_lstm", action='store_true', 
                       help='Use LSTM version (slower but better for sequences)')
    parser.add_argument("--window_size", type=int, default=50,
                       help='Window size for LSTM version')
    
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seeds(args.seed, rank)

    if rank == 0:
        rank0_print("=" * 76)
        rank0_print("TIMEDISTRIBUTED CNN TRAINING")
        rank0_print("=" * 76)
        rank0_print("Host:", socket.gethostname())
        rank0_print(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        rank0_print(f"Data: {args.data}")
        rank0_print(f"Model: {'LSTM-based' if args.use_lstm else 'Simple'} TimeDistributed")
        rank0_print("=" * 76)
    
    # Load data with shuffling
    X, y = load_data_with_shuffle(args.data)
    n_classes = int(np.max(y) + 1)
    
    if X.ndim == 2:
        X = X[:, None, :]
    
    # Verify balance
    if rank == 0:
        n_pspl = (y == 0).sum()
        n_binary = (y == 1).sum()
        pspl_pct = n_pspl / len(y) * 100
        rank0_print(f"\n{'='*76}")
        rank0_print("DATA BALANCE CHECK")
        rank0_print(f"{'='*76}")
        rank0_print(f"Total: {len(y):,}")
        rank0_print(f"  PSPL:   {n_pspl:,} ({pspl_pct:.1f}%)")
        rank0_print(f"  Binary: {n_binary:,} ({100-pspl_pct:.1f}%)")
        if pspl_pct < 40 or pspl_pct > 60:
            rank0_print("⚠️  WARNING: Imbalanced data!")
        else:
            rank0_print("✓ Balance OK")
        rank0_print(f"{'='*76}\n")

    # Split
    test_size = args.test_split
    val_size = args.val_split / (1.0 - test_size)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=args.seed, stratify=y_trainval
    )
    
    if rank == 0:
        rank0_print("SPLIT BALANCE")
        rank0_print(f"{'='*76}")
        for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            n_pspl = (labels == 0).sum()
            pspl_pct = n_pspl / len(labels) * 100
            rank0_print(f"{name:<10} {len(labels):<10} PSPL {pspl_pct:.1f}%")
        rank0_print(f"{'='*76}\n")

    # Normalize
    if rank == 0:
        scaler_params = fit_two_stage_scalers_rank0(X_train)
    else:
        scaler_params = None
    scaler_params = broadcast_object_from_rank0(scaler_params)

    X_train = apply_two_stage_scalers(scaler_params, X_train)
    X_val = apply_two_stage_scalers(scaler_params, X_val)
    X_test = apply_two_stage_scalers(scaler_params, X_test)

    # Datasets
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.use_lstm:
        rank0_print("Creating TimeDistributedCNN with LSTM...")
        model = TimeDistributedCNN(
            in_channels=X.shape[1],
            n_classes=n_classes,
            window_size=args.window_size,
            use_lstm=True,
            dropout=0.3
        ).to(device)
    else:
        rank0_print("Creating TimeDistributedCNNSimple...")
        model = TimeDistributedCNNSimple(
            in_channels=X.shape[1],
            n_classes=n_classes,
            dropout=0.3
        ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Experiment dir
    if rank == 0:
        exp_dir = make_experiment_dir("..", args.experiment_name)
    else:
        exp_dir = None
    exp_dir = broadcast_object_from_rank0(exp_dir)

    if rank == 0:
        (Path(exp_dir) / "evaluation").mkdir(parents=True, exist_ok=True)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            config_dict = vars(args).copy()
            config_dict['model_type'] = 'TimeDistributed_LSTM' if args.use_lstm else 'TimeDistributed_Simple'
            json.dump(config_dict, f, indent=2)
        save_scalers_rank0(scaler_params, exp_dir)

    # Training
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
            loss, correct, count = step_timedistributed(model, batch, device, criterion, scaler, optimizer)
            loss_sum += loss * count
            correct_sum += correct
            count_sum += torch.tensor(count, device=device)

        reduce_tensor(loss_sum)
        reduce_tensor(correct_sum)
        reduce_tensor(count_sum)

        train_loss = (loss_sum / count_sum).item()
        train_acc = (correct_sum / count_sum).item()

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        if rank == 0:
            epoch_time = time.time() - start_time
            rank0_print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | time {epoch_time:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    (model.module if isinstance(model, DDP) else model).state_dict(),
                    os.path.join(exp_dir, "best_model.pt"),
                )

        if is_dist():
            dist.barrier()

    # Final eval
    if rank == 0:
        best_path = os.path.join(exp_dir, "best_model.pt")
        state_dict = torch.load(best_path, map_location="cpu")
    else:
        state_dict = None
    state_dict = broadcast_object_from_rank0(state_dict)
    (model.module if isinstance(model, DDP) else model).load_state_dict(state_dict)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)

    if rank == 0:
        summary = {
            "final_test_loss": float(test_loss),
            "final_test_acc": float(test_acc),
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "model_type": 'TimeDistributed_LSTM' if args.use_lstm else 'TimeDistributed_Simple',
            "results_dir": exp_dir,
        }
        with open(os.path.join(exp_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        rank0_print("\n" + "=" * 80)
        rank0_print("FINAL EVALUATION")
        rank0_print("=" * 80)
        rank0_print(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")
        rank0_print(f"Best  | val acc {best_val_acc:.4f} (epoch {best_epoch})")
        rank0_print(f"\n✓ Training complete! Results: {exp_dir}")

    if is_dist():
        time.sleep(1.0)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
