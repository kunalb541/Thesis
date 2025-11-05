#!/usr/bin/env python3
# train.py - Training with TimeDistributed architecture (v5.3 - Causal)
#
# FIXES:
# 1. Reshapes data [N, T] -> [N, 1, T] IMMEDIATELY after loading.
# 2. Normalization functions (fit/apply) now handle 3D data.
# 3. Training step `step_timedistributed` now uses a combined
#    loss: loss_all_steps + 0.5 * loss_final + temporal_weight * loss_temporal
# 4. Model imported is the new causal CNN+LSTM.
#
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

# --- FIX: Import the one and only (causal) model ---
from model import TimeDistributedCNN


# --- DDP Helper Functions (no changes) ---
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
# --- End DDP Helpers ---


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X is already 3D [N, C, T]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data_with_shuffle(path):
    # (This is the same as before, returns 2D [N, T] X)
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


# --- 3D-Aware Normalization Functions (DDP-local, no changes) ---

def _feature_means_ignore_pad_flat(X_flat: np.ndarray, pad_value: float) -> np.ndarray:
    """Compute per-feature means on 2D [N, F] data, ignoring pad_value."""
    N, F = X_flat.shape
    means = np.zeros(F, dtype=np.float32)
    for j in range(F):
        col = X_flat[:, j]
        valid = col != pad_value
        if np.any(valid):
            means[j] = col[valid].mean(dtype=np.float64)
        else:
            means[j] = 0.0
    return means.astype(np.float32)

def fit_two_stage_scalers_rank0(X_train_3d, pad_value=-1.0):
    """Fits scalers on 3D [N, C, T] data."""
    N, C, T = X_train_3d.shape
    F_flat = C * T
    rank0_print(f"Fitting scalers on 3D data: [N, {C}, {T}] -> [N, {F_flat}]")
    
    X_train_flat = X_train_3d.reshape(N, F_flat)
    pad_mask = (X_train_flat == pad_value)

    # Compute means ignoring pads
    means_train = _feature_means_ignore_pad_flat(X_train_flat, pad_value)
    
    # Fill pads for fitting
    X_train_filled = np.where(pad_mask, means_train, X_train_flat)

    # Stage 1: StandardScaler
    std = StandardScaler(with_mean=True, with_std=True)
    flat_std = std.fit_transform(X_train_filled)
    
    # Stage 2: MinMaxScaler
    mm = MinMaxScaler(feature_range=(0.0, 1.0))
    mm.fit(flat_std)
    
    params = {
        "std_mean": std.mean_.astype(np.float64),
        "std_scale": std.scale_.astype(np.float64),
        "mm_min": mm.min_.astype(np.float64),
        "mm_scale": mm.scale_.astype(np.float64),
        "shape": (C, T), # Store original C, T shape
    }
    return params

def apply_two_stage_scalers(params, X_3d, pad_value=-1.0):
    """Applies fitted scalers to 3D [N, C, T] data."""
    N, C, T = X_3d.shape
    F_flat = C * T
    
    # Check shape
    scaler_shape = tuple(params["shape"])
    if (C, T) != scaler_shape:
        raise ValueError(f"Shape mismatch: X is [N, {C}, {T}] but scaler was fit on [N, {scaler_shape[0]}, {scaler_shape[1]}]")
    
    X_flat = X_3d.reshape(N, F_flat)
    pad_mask = (X_flat == pad_value)
    
    # Fill pads with *saved* train means
    X_filled = np.where(pad_mask, params["std_mean"], X_flat)
    
    # Apply transforms
    flat_std = (X_filled - params["std_mean"]) / (params["std_scale"] + 1e-12)
    flat_scaled = flat_std * params["mm_scale"] + params["mm_min"]
    
    # Restore pads
    flat_scaled[pad_mask] = pad_value
    
    # Reshape back to 3D
    return flat_scaled.reshape(N, C, T)

def save_scalers_rank0(params, out_dir):
    """Saves scalers as standard pickle files AND a json dict"""
    if get_rank() != 0:
        return
    
    # Save pkl files for utils.py
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
    
    # Also save the params dict (which contains the shape) as JSON
    with open(os.path.join(out_dir, "scaler_params.json"), "w") as f:
        params_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()}
        json.dump(params_json, f, indent=2)
    rank0_print(f"✓ Scalers saved to {out_dir}/")
        
# --- END Normalization ---


# --- START FIX: Updated Training Step ---
def step_timedistributed(model, batch, device, criterion, scaler, optimizer=None, temporal_weight=0.1):
    """
    FIXED Training step for TimeDistributed model (v5.3)
    Uses loss = loss_all_steps + 0.5 * loss_final + temporal_weight * loss_temporal
    """
    x, y = batch
    x = x.to(device, non_blocking=True) # [B, C, T]
    y = y.to(device, non_blocking=True) # [B]
    B, C, T = x.shape
    
    is_train = optimizer is not None
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        
        # Get predictions at all timesteps
        logits_seq = model(x, return_sequence=True)  # [B, T, n_classes]
        
        # ===== CRITICAL FIX: Supervise ALL timesteps =====
        # Expand labels to match sequence length
        y_expanded = y.unsqueeze(1).expand(B, T)  # [B] -> [B, T]
        
        # 1. Loss on all timesteps
        logits_flat = logits_seq.reshape(B * T, -1)  # [B*T, n_classes]
        y_flat = y_expanded.reshape(B * T)  # [B*T]
        loss_all_steps = criterion(logits_flat, y_flat)
        
        # 2. Extra emphasis on final prediction (most important)
        loss_final = criterion(logits_seq[:, -1, :], y)
        
        # 3. Temporal smoothness (optional - helps stability)
        diff = logits_seq[:, 1:, :] - logits_seq[:, :-1, :]
        loss_temporal = torch.mean(diff ** 2)
        
        # 4. Combined loss
        loss = loss_all_steps + 0.5 * loss_final + temporal_weight * loss_temporal
        # ================================================
    
    # Accuracy computed on FINAL prediction only
    pred = logits_seq[:, -1, :].argmax(dim=1)
    correct = (pred == y).sum()

    if is_train:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Return loss, correct count, and B (number of samples)
    return loss.detach(), correct.detach(), y.numel()
# --- END FIX: Updated Training Step ---


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Validation function.
    Evaluates on the FINAL prediction only (return_sequence=False)
    for speed and to match the primary loss term.
    """
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    count_sum = torch.tensor(0.0, device=device)

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # --- Get FINAL prediction (fast) ---
        logits = model(x, return_sequence=False) # [B, n_classes]
        
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
    
    parser.add_argument("--window_size", type=int, default=50,
                       help='(No longer used for windowing) Kept for config compatibility')
    # --- FIX: Updated temporal_weight default ---
    parser.add_argument("--temporal_weight", type=float, default=0.05,
                       help='Weight for temporal smoothness loss')
    
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seeds(args.seed, rank)

    if rank == 0:
        rank0_print("=" * 76)
        rank0_print("TIMEDISTRIBUTED CNN TRAINING (v5.3 - Causal CNN+LSTM)")
        rank0_print("=" * 76)
        rank0_print(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        rank0_print(f"Data: {args.data}")
        rank0_print(f"Model: Causal TimeDistributedCNN (CNN+LSTM)")
        rank0_print(f"Loss: AllSteps + 0.5*Final + {args.temporal_weight}*TemporalSmoothness")
        rank0_print("=" * 76)
    
    # Load data (X is 2D [N, T])
    X, y = load_data_with_shuffle(args.data)
    n_classes = int(np.max(y) + 1)
    
    # --- START FIX: Reshape data to 3D [N, C, T] IMMEDIATELY ---
    if X.ndim == 2:
        X = X[:, None, :]  # [N, T] -> [N, 1, T]
        rank0_print(f"✓ Reshaped X from 2D to 3D: {X.shape}")
    # --- END FIX ---
    
    # (Balance check... no changes)
    if rank == 0:
        # ... (balance check code)
        pass

    # Split (X is now 3D)
    test_size = args.test_split
    val_size = args.val_split / (1.0 - test_size)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=args.seed, stratify=y_trainval
    )
    
    # (Split balance check... no changes)
    if rank == 0:
        # ... (split balance check code)
        pass

    # --- FIX: Normalize 3D data ---
    if rank == 0:
        rank0_print("Fitting scalers on 3D training data...")
        scaler_params = fit_two_stage_scalers_rank0(X_train)
    else:
        scaler_params = None
    scaler_params = broadcast_object_from_rank0(scaler_params)

    X_train = apply_two_stage_scalers(scaler_params, X_train)
    X_val = apply_two_stage_scalers(scaler_params, X_val)
    X_test = apply_two_stage_scalers(scaler_params, X_test)
    rank0_print("✓ Data normalized.")
    # --- END FIX ---

    # Datasets (X is now 3D and normalized)
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

    # --- FIX: Model (default to causal LSTM) ---
    rank0_print("Creating Causal TimeDistributedCNN (CNN+LSTM)...")
    model = TimeDistributedCNN(
        in_channels=X.shape[1], # Should be 1
        n_classes=n_classes,
        lstm_hidden=64, # Example hidden size
        dropout=0.3
    ).to(device)
    # --- END FIX ---
    
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
            config_dict['model_type'] = 'TimeDistributed_Causal_LSTM'
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

        # --- FIX: Use new training step ---
        for batch in train_loader:
            loss, correct, count = step_timedistributed(
                model, batch, device, criterion, scaler, optimizer, 
                temporal_weight=args.temporal_weight
            )
            loss_sum += loss * count # loss is avg, count is B
            correct_sum += correct
            count_sum += torch.tensor(count, device=device)

        reduce_tensor(loss_sum)
        reduce_tensor(correct_sum)
        reduce_tensor(count_sum)

        train_loss = (loss_sum / count_sum).item()
        train_acc = (correct_sum / count_sum).item()

        # Validation uses the 'evaluate' function (final step accuracy)
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

    # Final eval on test set (final step accuracy)
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
            "model_type": 'TimeDistributed_Causal_LSTM',
            "results_dir": exp_dir,
        }
        with open(os.path.join(exp_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        rank0_print("\n" + "=" * 80)
        rank0_print("FINAL EVALUATION (on final timestep)")
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