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

# Config
import config as CFG
try:
    from config import PAD_VALUE
except Exception:
    PAD_VALUE = -1

# === Fast NPZ loader that applies saved permutation and normalizes labels ===
def load_npz_dataset(npz_path, apply_perm=True):
    d = np.load(npz_path, allow_pickle=False)
    X = d["X"]
    y = d["y"]
    if apply_perm and "perm" in d.files:
        perm = d["perm"]
        X = X[perm]
        y = y[perm]
    # labels: accept uint8 or strings
    if y.dtype.kind in ("U","S","O"):
        y = np.array([0 if (str(v).lower().startswith("pspl")) else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8, copy=False)
    timestamps = d["timestamps"]
    meta_json = d["meta_json"].item() if "meta_json" in d.files else "{}"
    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}
    return X, y, timestamps, meta

# ---- Model ----
class CNN1D(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        c1, c2, c3 = CFG.CONV1_FILTERS, CFG.CONV2_FILTERS, CFG.CONV3_FILTERS
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, CFG.FC1_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(CFG.DROPOUT_RATE),
            nn.Linear(CFG.FC1_UNITS, 2),  # 2 classes
        )

    def forward(self, x):
        # x: [B, 1, L]
        z = self.net(x)
        return self.head(z)

# ---- Dataset ----
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # Optionally map PAD_VALUE to 0 to be neutral in conv
        X = X.copy()
        X[X == PAD_VALUE] = 0.0
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
    torch.cuda.manual_seed_all(seed)

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
        logits = model(xb)
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
        logits = model(xb)
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
    rel = CFG.TEST_SPLIT / (CFG.TEST_SPLIT + CFG.VAL_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel, random_state=args.seed, stratify=y_tmp
    )

    train_ds = NumpyDataset(X_train, y_train)
    val_ds   = NumpyDataset(X_val, y_val)
    test_ds  = NumpyDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(input_len=L).to(device)
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
            if CFG.SAVE_BEST_ONLY:
                print(f"  ↳ saved best model to {best_path}")

        if CFG.SAVE_CHECKPOINTS and (not CFG.SAVE_BEST_ONLY) and (epoch % CFG.CHECKPOINT_FREQ == 0):
            ckpt_path = best_path.replace(".pt", f".epoch{epoch}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt_path)
            print(f"  ↳ checkpoint saved to {ckpt_path}")

    # Test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} (best val acc {best_val_acc:.4f})")

if __name__ == "__main__":
    main()