#!/usr/bin/env python3
"""
Transformer-Based Microlensing Classifier with DDP (Fixed v5.3)

Author: Kunal Bhatia
Date: November 2025
Version: 5.4 (Patched DDP eval loading and exp_dir broadcast)
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

from model import TransformerClassifier, count_parameters
from utils import load_npz_dataset, two_stage_normalize


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
        # Standalone mode
        rank = 0
        world_size = 1
        local_rank = 0

    is_ddp = world_size > 1

    if is_ddp:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        # Update rank and world_size from dist
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, is_ddp, device


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
# TRAINING AND EVALUATION FUNCTIONS
# -------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, is_ddp, epoch, rank):
    model.train()
    if is_ddp and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    total_loss, correct, total = 0, 0, 0
    
    # Add TQDM progress bar only on the main process
    if is_main_process(rank):
        loader = tqdm(loader, desc=f"Epoch {epoch+1} Train", unit="batch")

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
def evaluate(model, loader, criterion, device, rank):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Add TQDM progress bar only on the main process
    if is_main_process(rank):
        loader = tqdm(loader, desc="Evaluate", unit="batch", leave=False)

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
        metrics = torch.tensor([local_loss, local_correct, local_total], dtype=torch.float64, device=device)
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
    rank, world_size, local_rank, is_ddp, device = setup_ddp()
    is_main = is_main_process(rank)
    
    # Ensure num_workers is adjusted for DDP
    args.num_workers = args.num_workers // world_size

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    log_main(f"[Rank {rank}] Initialized DDP (world_size={world_size}, device={device})", rank)

    # ------------------------ Load Data ------------------------
    # Only rank 0 loads and splits, then we broadcast
    if is_main:
        log_main("Loading dataset...", rank)
        X, y, _, meta = load_npz_dataset(args.data, apply_perm=True)
        pad_value = meta.get("PAD_VALUE", -1.0)

        # Ensure 3D
        if X.ndim == 2:
            X = X[:, None, :]

        log_main(f"Data shape: {X.shape}, Labels: {y.shape}", rank)

        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=args.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed
        )

        log_main(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", rank)

        # Normalize
        log_main("Normalizing data...", rank)
        X_train, X_val, X_test, scaler_std, scaler_mm = two_stage_normalize(
            X_train, X_val, X_test, pad_value
        )
    else:
        # Create empty tensors to receive broadcasted data
        X_train, y_train = np.empty([0,0,0], dtype=np.float32), np.empty([0], dtype=np.int64)
        X_val, y_val     = np.empty([0,0,0], dtype=np.float32), np.empty([0], dtype=np.int64)
        X_test, y_test   = np.empty([0,0,0], dtype=np.float32), np.empty([0], dtype=np.int64)
        scaler_std, scaler_mm = None, None

    # Broadcast data from rank 0 to all other processes
    if is_ddp:
        log_main(f"Broadcasting data to all {world_size} ranks...", rank)
        # 1. Broadcast scalers
        scalers = [scaler_std, scaler_mm]
        dist.broadcast_object_list(scalers, src=0)
        if not is_main:
            scaler_std, scaler_mm = scalers
            
        # 2. Broadcast datasets (using torch.distributed.broadcast)
        def broadcast_tensor(tensor, dtype, rank, device):
            if rank == 0:
                data = torch.from_numpy(tensor).to(device)
            else:
                data = torch.empty(dtype=dtype, device=device) # Shape will be set by broadcast
            
            if rank == 0:
                shape_tensor = torch.tensor(data.shape, device=device, dtype=torch.long)
            else:
                shape_tensor = torch.empty((3 if dtype==torch.float32 else 1), device=device, dtype=torch.long)
            
            dist.broadcast(shape_tensor, src=0)
            
            if rank != 0:
                data = torch.empty(tuple(shape_tensor.cpu().numpy()), dtype=dtype, device=device)
            
            dist.broadcast(data, src=0)
            return data.cpu().numpy()

        X_train = broadcast_tensor(X_train, torch.float32, rank, device)
        y_train = broadcast_tensor(y_train, torch.long, rank, device)
        X_val   = broadcast_tensor(X_val, torch.float32, rank, device)
        y_val   = broadcast_tensor(y_val, torch.long, rank, device)
        X_test  = broadcast_tensor(X_test, torch.float32, rank, device)
        y_test  = broadcast_tensor(y_test, torch.long, rank, device)

    train_ds = MicrolensingDataset(X_train, y_train)
    val_ds = MicrolensingDataset(X_val, y_val)
    test_ds = MicrolensingDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, sampler=val_sampler,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, sampler=test_sampler,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ------------------------ Model Setup ------------------------
    log_main("Building model...", rank)
    # Ensure in_channels matches data
    model = TransformerClassifier(
        in_channels=X_train.shape[1],
        n_classes=len(np.unique(y_train)), # Use 2 if binary, or unique for safety
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        downsample_factor=args.downsample_factor,
        dropout=args.dropout,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, 
                    output_device=local_rank if torch.cuda.is_available() else None, 
                    find_unused_parameters=False)
        model_base = model.module
    else:
        model_base = model

    log_main(f"Model parameters: {count_parameters(model_base):,}", rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------ Experiment Directory ------------------------
    config = None
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(f"../results/{args.experiment_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        exp_dir_str = str(exp_dir)

        config = vars(args).copy()
        config.update({
            "n_parameters": count_parameters(model_base),
            "world_size": world_size,
            "data_file": args.data,
            "timestamp": timestamp,
            "X_train_shape": X_train.shape,
            "y_train_classes": len(np.unique(y_train)),
        })
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save scalers
        with open(exp_dir / "scaler_standard.pkl", "wb") as f:
            pickle.dump(scaler_std, f)
        with open(exp_dir / "scaler_minmax.pkl", "wb") as f:
            pickle.dump(scaler_mm, f)

        log_main(f"Experiment directory: {exp_dir}", rank)
    
    # FIX: Broadcast the experiment directory path from rank 0
    if is_ddp:
        dir_obj = [str(exp_dir) if is_main else None]
        dist.broadcast_object_list(dir_obj, src=0)
        exp_dir = Path(dir_obj[0])
        
        # Broadcast config
        config_obj = [config if is_main else None]
        dist.broadcast_object_list(config_obj, src=0)
        config = config_obj[0]
        
    log_main(f"[Rank {rank}] Using experiment directory: {exp_dir}", rank)

    # Sync all processes before starting training
    if is_ddp:
        dist.barrier()

    # ------------------------ Training Loop ------------------------
    log_main("\nStarting training...", rank)
    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(args.epochs):
        train_loss_local, train_correct_local, train_total_local = train_epoch(
            model, train_loader, criterion, optimizer, device, is_ddp, epoch, rank
        )
        val_loss_local, val_correct_local, val_total_local = evaluate(
            model, val_loader, criterion, device, rank
        )

        train_loss, train_acc = aggregate_metrics(
            train_loss_local, train_correct_local, train_total_local, world_size, device
        )
        val_loss, val_acc = aggregate_metrics(
            val_loss_local, val_correct_local, val_total_local, world_size, device
        )

        if is_main:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model_base.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'config': config 
                }
                torch.save(checkpoint, exp_dir / "best_model.pt")
            else:
                patience_counter += 1

        # Broadcast early stopping signal
        stop_signal = torch.tensor(int(patience_counter >= args.patience), device=device)
        if is_ddp:
            dist.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1:
            log_main(f"\nEarly stopping at epoch {epoch+1}", rank)
            break
            
        if is_ddp:
            dist.barrier()

    # ------------------------ Evaluation ------------------------
    log_main("\nEvaluating on test set...", rank)
    
    # FIX: All processes must load the best model for correct DDP evaluation
    # Load checkpoint onto the correct device
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device)
    model_base.load_state_dict(checkpoint['model_state_dict'])
    log_main(f"[Rank {rank}] Loaded best model from epoch {checkpoint['epoch']+1}", rank)

    # Sync all processes after loading
    if is_ddp:
        dist.barrier()

    test_loss_local, test_correct_local, test_total_local = evaluate(
        model, test_loader, criterion, device, rank
    )
    test_loss, test_acc = aggregate_metrics(
        test_loss_local, test_correct_local, test_total_local, world_size, device
    )

    if is_main:
        log_main(f"\n{'='*60}", rank)
        log_main(f"FINAL RESULTS", rank)
        log_main(f"{'='*60}", rank)
        log_main(f"Best Val Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)", rank)
        log_main(f"Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)", rank)
        log_main(f"Test Loss:          {test_loss:.4f}", rank)
        log_main(f"{'='*60}", rank)

        # Save results
        results = {
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_trained": epoch + 1,
            "best_model_epoch": checkpoint['epoch'] + 1
        }
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        log_main(f"\nResults saved to: {exp_dir}", rank)

    cleanup_ddp()


if __name__ == "__main__":
    # Import tqdm only if main
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    main()