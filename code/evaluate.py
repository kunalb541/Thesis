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
  python evaluate.py --model models/baseline.pt --data data/raw/events_baseline_1M.npz --output_dir results/plots

Author: Kunal Bhatia
"""
def find_latest_results_dir(experiment_name, base_dir='../results'):
    """Find the most recent results directory for an experiment"""
    base_path = Path(base_dir)
    pattern = f"{experiment_name}_*"
    
    matching_dirs = sorted(base_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No results directories found matching '{pattern}' in {base_dir}")
    
    return matching_dirs[0]

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
from tqdm import tqdm

# Local imports
import config as CFG
from model import TimeDistributedCNN  # FIX #3: Import unified model
from utils import load_npz_dataset  # FIX #3: Import dataset loader from utils


# -------------------------
# Dataset wrapper
# -------------------------

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # The apply_pad_to_zero utility function from utils is not defined in the original diff, 
        # but the logic is simple: map PAD_VALUE to 0.0. Doing this inline for reliability.
        X_copy = X.copy()
        X_copy[X_copy == CFG.PAD_VALUE] = 0.0
        self.X = torch.from_numpy(X_copy).float().unsqueeze(1)  # [N, 1, L]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Evaluation helpers
# -------------------------

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list, labels_list, all_logits_t_list = [], [], []
    for xb, yb in tqdm(loader, desc="Running Inference"):
        xb = xb.to(device)
        outputs = model(xb)  # [B, L, 2]
        all_logits_t_list.append(outputs.cpu().numpy())
        # Aggregate over time dimension to get sequence-level logits
        logits = outputs.mean(dim=1)  # [B, 2]
        logits_list.append(logits.cpu().numpy())
        labels_list.append(yb.numpy())
        
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    logits_t = np.concatenate(all_logits_t_list, axis=0)
    return logits, labels, logits_t # Return per-timestep logits as well

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    p1 = probs[:, 1]
    preds = probs.argmax(axis=1)

    acc = (preds == labels).mean().item()
    
    # Handle case where only one class is present in the batch, or all predictions are the same
    try:
        fpr, tpr, _ = roc_curve(labels, p1)
        roc_auc = auc(fpr, tpr)
    except ValueError: # Happens if all labels are the same
        roc_auc = float('nan')

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
    
    X_early_conv = X_early.copy() # Avoid modifying the pad_value-filled array
    X_early_conv[X_early_conv == CFG.PAD_VALUE] = 0.0
    return X_early_conv

def plot_curves(save_dir: Optional[str], logits: np.ndarray, labels: np.ndarray):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]

    import matplotlib.pyplot as plt
    
    # ROC
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc.png"), dpi=getattr(CFG, "DPI", 150))
        plt.close()
    except ValueError:
        print("Skipping ROC plot: Single class data or all predictions identical.")

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
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["PSPL","Binary"], yticklabels=["PSPL","Binary"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=getattr(CFG, "DPI", 150))
    plt.close()

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (auto-detect if not provided)")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-detect if not provided)")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (for auto-detect)")
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_detection", action="store_true", help="Run early detection analysis")
    args = parser.parse_args()
    
    # Auto-detect model and output_dir if not provided
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError("Must provide either --model and --output_dir, OR --experiment_name for auto-detection")
        
        results_dir = find_latest_results_dir(args.experiment_name)
        print(f"✓ Auto-detected results directory: {results_dir}")
        
        if args.model is None:
            args.model = str(results_dir / "best_model.pt")
        if args.output_dir is None:
            args.output_dir = str(results_dir / "evaluation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)


    # Load dataset (perm-aware)
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    # Note: X passed to NumpyDataset will be 0-padded for the model
    L = X.shape[1]

    # Build model and load weights
    device = torch.device(args.device)
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2).to(device)

    ckpt = torch.load(args.model, map_location=device)
    if "model_state_dict" in ckpt: # Preferred key (from train.py fix)
        state = ckpt["model_state_dict"]
    elif "model" in ckpt: # Legacy key
        state = ckpt["model"]
    else:
        state = ckpt # Raw state dict
    model.load_state_dict(state)

    # Inference
    ds = NumpyDataset(X, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logits, labels, logits_t = run_inference(model, loader, device)

    # Metrics
    metrics = compute_metrics(logits, labels)
    print("\n=== Evaluation ===")
    for k in ["accuracy", "roc_auc", "pr_auc", "average_precision", "tn", "fp", "fn", "tp"]:
        print(f"{k:>18}: {metrics[k]}")

    # Early detection
    early_metrics = {}
    if args.early_detection and getattr(CFG, "EARLY_DETECTION_CHECKPOINTS", None):
        print("\n=== Early Detection ===")
        # Keep track of all accuracies for a full early detection plot if needed later
        early_detection_accuracies = {}
        for frac in CFG.EARLY_DETECTION_CHECKPOINTS:
            # 1. Prepare early data (X is full, but passed to utility which pads)
            Xe_padded_conv = early_detection_subset(X, frac)
            dse = NumpyDataset(Xe_padded_conv, y) # Use the already 0-padded version
            loe = DataLoader(dse, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            # 2. Run inference on early data
            loge, labe, _ = run_inference(model, loe, device)
            # 3. Compute metrics
            me = compute_metrics(loge, labe)
            early_detection_accuracies[frac] = me["accuracy"]
            print(f"{frac:>5.2f} observed -> acc {me['accuracy']:.4f}")
        early_metrics = early_detection_accuracies

    # Plots (optional)
    plot_curves(args.output_dir, logits, labels)

    # Print save path for JSON summary
    summary = {
        "metrics": metrics,
        "early_detection": early_metrics,
        "meta": meta,
        "data": os.path.abspath(args.data),
        "model": os.path.abspath(args.model),
    }
    
    # Save a separate file with the per-timestep logits for later analysis (optional)
    np.savez_compressed(os.path.join(args.output_dir, "predictions_full.npz"), 
                        logits_agg=logits, labels=labels, logits_t=logits_t, timestamps=timestamps)
    
    out_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")

if __name__ == "__main__":
    main()