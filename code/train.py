#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — Simple 1D-CNN trainer for microlensing classification
- Loads fast NPZ produced by simulate.py
- Applies saved permutation if present
- Uses config.py for hyperparameters and paths
"""

import argparse
import os
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Config
import config as CFG
# Import unified model and utils
from model import TimeDistributedCNN
from utils import load_npz_dataset


# ---- Dataset ----
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # Optionally map PAD_VALUE to 0 to be neutral in conv
        X = X.copy()
        # Use CFG.PAD_VALUE for consistency
        X[X == CFG.PAD_VALUE] = 0.0
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, L]
        self.y = torch.from_numpy(y).long()                # [N]
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---- Training utils ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure all three are set for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, device, grad_clip=None):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(xb)  # [B, L, 2]
        
        # Aggregate across time (mean pooling for sequence-level loss)
        logits = outputs.mean(dim=1)  # [B, 2]
        
        loss = crit(logits, yb)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), yb) * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        outputs = model(xb)  # [B, L, 2]
        logits = outputs.mean(dim=1)  # [B, 2] - Mean aggregation
        
        loss = crit(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    return total_loss / n, total_acc / n

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .npz produced by simulate.py")
    parser.add_argument("--output", type=str, default=os.path.join(CFG.MODEL_DIR, "baseline.pt"))
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=CFG.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading dataset: {args.data}")
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    L = X.shape[1]
    print(f"Loaded X: {X.shape}, y: {y.shape}, time points: {L}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - CFG.TRAIN_SPLIT), random_state=args.seed, stratify=y
    )
    # The split ratio calculation for X_val/X_test must use the remaining fraction (1.0 - CFG.TRAIN_SPLIT) as the denominator
    # The original was incorrect. Correcting relative split for consistency with CFG.VAL_SPLIT/CFG.TEST_SPLIT
    val_frac = CFG.VAL_SPLIT / (CFG.VAL_SPLIT + CFG.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=args.seed, stratify=y_tmp
    )

    train_ds = NumpyDataset(X_train, y_train)
    val_ds   = NumpyDataset(X_val, y_val)
    test_ds  = NumpyDataset(X_test, y_test)

    # Use CFG worker settings if available
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Unified model instantiation
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2).to(device)
    
    if CFG.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if CFG.LR_SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=CFG.LR_PATIENCE, factor=CFG.LR_FACTOR, verbose=True)
    else:
        scheduler = None

    best_val_acc = -1.0
    best_path = args.output
    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip=CFG.GRAD_CLIP)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_acc)

        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # FIX #5: Corrected checkpoint key to "model_state_dict"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
            if CFG.SAVE_BEST_ONLY:
                print(f"  ↳ saved best model to {best_path}")

        if CFG.SAVE_CHECKPOINTS and (not CFG.SAVE_BEST_ONLY) and (epoch % CFG.CHECKPOINT_FREQ == 0):
            ckpt_path = best_path.replace(".pt", f".epoch{epoch}.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt_path)
            print(f"  ↳ checkpoint saved to {ckpt_path}")

    # Test evaluation
    ckpt = torch.load(best_path, map_location=device)
    # FIX #5: Corrected checkpoint key for final evaluation
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} (best val acc {best_val_acc:.4f})")

if __name__ == "__main__":
    main()