#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluate a trained model on an NPZ dataset
- Loads NPZ (applies saved permutation if present)
- Uses PAD_VALUE from config for preprocessing
- Computes Accuracy, ROC-AUC, PR-AUC, Confusion Matrix
- Optional early-detection analysis using fractions from config
- Optional plots (ROC, PR, confusion matrix)

Usage:
  python evaluate.py --model models/baseline.pt --data data/raw/events_baseline_1M.npz --plots results/plots

Author: Kunal Bhatia
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Local imports
import config as CFG
from utils import load_npz_dataset, apply_pad_to_zero

# -------------------------
# Model (must match train)
# -------------------------

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
            nn.Linear(CFG.FC1_UNITS, 2),
        )

    def forward(self, x):
        z = self.net(x)
        return self.head(z)

# -------------------------
# Dataset wrapper
# -------------------------

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = apply_pad_to_zero(X, pad_value=CFG.PAD_VALUE)
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, L]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Evaluation helpers
# -------------------------

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, labels_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        logits_list.append(out.cpu().numpy())
        labels_list.append(yb.numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return logits, labels

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    p1 = probs[:, 1]
    preds = probs.argmax(axis=1)

    acc = (preds == labels).mean().item()
    fpr, tpr, _ = roc_curve(labels, p1)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, p1)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(labels, p1)

    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "average_precision": float(ap),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }

def early_detection_subset(X: np.ndarray, frac: float) -> np.ndarray:
    """
    Crop each sequence to the first `frac` fraction, pad the rest to PAD_VALUE -> then 0.0 for convs.
    """
    L = X.shape[1]
    keep = max(1, int(np.ceil(L * frac)))
    X_early = np.full_like(X, fill_value=CFG.PAD_VALUE)
    X_early[:, :keep] = X[:, :keep]
    X_early = apply_pad_to_zero(X_early, pad_value=CFG.PAD_VALUE)
    return X_early

def plot_curves(save_dir: Optional[str], logits: np.ndarray, labels: np.ndarray):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(labels, probs)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc.png"), dpi=getattr(CFG, "DPI", 150))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"AUC={auc(recall,precision):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr.png"), dpi=getattr(CFG, "DPI", 150))
    plt.close()

    # Confusion matrix
    preds = (probs >= 0.5).astype(np.uint8)
    cm = confusion_matrix(labels, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.xticks([0,1], ["PSPL","Binary"])
    plt.yticks([0,1], ["PSPL","Binary"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=getattr(CFG, "DPI", 150))
    plt.close()

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt saved by train.py")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz produced by simulate.py")
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plots", type=str, default=None, help="Directory to save plots (optional)")
    args = parser.parse_args()

    # Load dataset (perm-aware)
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    X = apply_pad_to_zero(X, pad_value=CFG.PAD_VALUE)
    L = X.shape[1]

    # Build model and load weights
    device = torch.device(args.device)
    model = CNN1D(input_len=L).to(device)
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)

    # Inference
    ds = NumpyDataset(X, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logits, labels = run_inference(model, loader, device)

    # Metrics
    metrics = compute_metrics(logits, labels)
    print("\n=== Evaluation ===")
    for k in ["accuracy", "roc_auc", "pr_auc", "average_precision", "tn", "fp", "fn", "tp"]:
        print(f"{k:>18}: {metrics[k]}")

    # Early detection
    early_metrics = {}
    if getattr(CFG, "EARLY_DETECTION_CHECKPOINTS", None):
        print("\n=== Early Detection ===")
        for frac in CFG.EARLY_DETECTION_CHECKPOINTS:
            Xe = early_detection_subset(X, frac)
            dse = NumpyDataset(Xe, y)
            loe = DataLoader(dse, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            loge, labe = run_inference(model, loe, device)
            me = compute_metrics(loge, labe)
            early_metrics[frac] = me["accuracy"]
            print(f"{frac:>5.2f} observed -> acc {me['accuracy']:.4f}")

    # Plots (optional)
    plot_curves(args.plots, logits, labels)

    # Print save path for JSON summary
    summary = {
        "metrics": metrics,
        "early_detection": early_metrics,
        "meta": meta,
        "data": os.path.abspath(args.data),
        "model": os.path.abspath(args.model),
    }
    out_dir = args.plots or os.path.dirname(args.model) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")

if __name__ == "__main__":
    main()