#!/usr/bin/env python3
"""
Complete Evaluation Script with Notebook-Style Visualizations (Fixed v5.6)

Author: Kunal Bhatia
Date: November 2025
Version: 5.6 - Fixed Path import, added input validation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import TransformerClassifier
from utils import load_npz_dataset, load_scalers, apply_scalers_to_data


# ============================================================
# DATASET WRAPPER
# ============================================================

class MicrolensingDataset(Dataset):
    """Dataset wrapper for evaluation"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# UTILS
# ============================================================

def get_latest_experiment(experiment_name, results_dir='../results'):
    """Find latest experiment directory by timestamp suffix"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_path.resolve()}")
    
    matching = sorted(results_path.glob(f"{experiment_name}_*"))
    if not matching:
        raise ValueError(f"No experiments found matching: {experiment_name} in {results_path.resolve()}")
    
    print(f"Found {len(matching)} matching experiments, using latest: {matching[-1].name}")
    return matching[-1]


@torch.no_grad()
def make_predictions(model, loader, device):
    """Get predictions and probabilities"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for X, y in tqdm(loader, desc="Making predictions"):
        X = X.to(device)
        logits, _ = model(X, return_sequence=False)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.append(probs)
        all_labels.extend(y.numpy())

    return np.array(all_preds), np.vstack(all_probs), np.array(all_labels)


@torch.no_grad()
def get_decision_times(model, X, device, confidence_threshold=0.8, batch_size=128):
    """Compute decision times for each sequence"""
    model.eval()
    N = len(X)
    decision_times = []
    predicted_classes = []

    for i in tqdm(range(0, N, batch_size), desc="Analyzing decision times"):
        X_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        logits_seq, _ = model(X_batch, return_sequence=True)
        probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()

        for sample_probs in probs_seq:
            T_down = sample_probs.shape[0]
            for t in range(T_down):
                confidence = np.max(sample_probs[t])
                if confidence >= confidence_threshold:
                    decision_times.append(t + 1)
                    predicted_classes.append(np.argmax(sample_probs[t]))
                    break
            else:
                decision_times.append(T_down)
                predicted_classes.append(np.argmax(sample_probs[-1]))

    return np.array(decision_times), np.array(predicted_classes)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

@torch.no_grad()
def plot_three_panel_sample(
    model,
    X_original,
    X_normalized,
    y_true,
    sample_idx,
    timestamps,
    device,
    output_path,
    confidence_threshold=0.8,
    pad_value=-1.0
):
    """Three-panel visualization for single event"""
    model.eval()

    X_tensor = torch.from_numpy(X_normalized).float().to(device)
    logits_seq, _ = model(X_tensor, return_sequence=True)
    probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()[0]

    T_orig = len(timestamps)
    T_down = probs_seq.shape[0]

    decision_time_idx = T_down
    predicted_class = np.argmax(probs_seq[-1])
    decision_made = False

    for t in range(T_down):
        confidence = np.max(probs_seq[t])
        if confidence >= confidence_threshold:
            decision_time_idx = t + 1
            predicted_class = np.argmax(probs_seq[t])
            decision_made = True
            break

    probs_clamped = probs_seq.copy()
    if decision_made:
        probs_clamped[decision_time_idx-1:] = probs_seq[decision_time_idx-1]

    downsample_factor = model.module.downsample_factor if hasattr(model, 'module') else model.downsample_factor
    decision_orig_idx = min(decision_time_idx * downsample_factor, T_orig - 1)
    decision_timestamp = timestamps[decision_orig_idx]

    true_label_str = "PSPL" if y_true == 0 else "Binary"
    pred_label_str = "PSPL" if predicted_class == 0 else "Binary"

    fig, axes = plt.subplots(3, 1, figsize=(15, 18), dpi=150)

    # Panel 1: Original Data
    ax1 = axes[0]
    orig_data = X_original[0, 0, :]
    non_pad = orig_data != pad_value
    ax1.scatter(timestamps[non_pad], orig_data[non_pad], color='darkcyan', s=30, alpha=0.7)
    ax1.axvline(x=decision_timestamp, color='red', linestyle='--', lw=2)
    ax1.set_title(f"1. Original Data | True: {true_label_str} | Pred: {pred_label_str}", 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Normalized Input
    ax2 = axes[1]
    model_input = X_normalized[0, 0, :]
    non_pad_input = model_input != pad_value
    ax2.scatter(np.arange(T_orig)[non_pad_input], model_input[non_pad_input], 
                color='darkcyan', s=30, alpha=0.7)
    ax2.axvline(x=decision_orig_idx, color='red', linestyle='--', lw=2)
    ax2.set_title("2. Model Input View (Normalized)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Probabilities
    ax3 = axes[2]
    ax3.plot(np.arange(1, T_down + 1), probs_clamped[:, 0], label="P(PSPL)", lw=2)
    ax3.plot(np.arange(1, T_down + 1), probs_clamped[:, 1], label="P(Binary)", lw=2)
    ax3.axvline(x=decision_time_idx, color='red', linestyle='--', lw=2)
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend()
    ax3.set_title("3. Class Probabilities Over Time (Clamped after decision)", 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Draw confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PSPL', 'Binary'],
                yticklabels=['PSPL', 'Binary'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_time_distribution(decision_times, sequence_length, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(decision_times, bins=range(1, sequence_length + 2), color='skyblue')
    plt.title("Distribution of Decision Time Steps")
    plt.xlabel(f"Time Step (1 to {sequence_length})")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_decision_time(thresholds, accuracies, avg_times, output_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(avg_times, accuracies, 'o-', color='tab:green', label='Accuracy')
    ax1.set_xlabel('Average Decision Time Step')
    ax1.set_ylabel('Accuracy', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(avg_times, thresholds, 'x--', color='tab:blue', label='Threshold')
    ax2.set_ylabel('Confidence Threshold', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Accuracy vs. Average Decision Time')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    args = parser.parse_args()

    # Input validation
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if args.confidence_threshold < 0 or args.confidence_threshold > 1:
        raise ValueError(f"Confidence threshold must be in [0, 1], got {args.confidence_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("EVALUATION WITH NOTEBOOK-STYLE VISUALIZATIONS")
    print("="*80)
    print(f"Device: {device}")

    exp_dir = get_latest_experiment(args.experiment_name)
    print(f"Evaluating experiment: {exp_dir.name}")

    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    print("\nLoading model...")
    model = TransformerClassifier(
        in_channels=1,
        n_classes=2,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 256),
        downsample_factor=config.get('downsample_factor', 3),
        dropout=config.get('dropout', 0.3)
    ).to(device)

    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")
    model.eval()

    print("Loading scalers...")
    scaler_std, scaler_mm = load_scalers(exp_dir)

    print(f"Loading data: {args.data}")
    X_raw, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    pad_value = meta.get("PAD_VALUE", -1.0)
    
    if X_raw.ndim == 2:
        X_raw = X_raw[:, None, :]

    print("Splitting into test set (using same split as training)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.4, random_state=config.get("seed", 42), stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.get("seed", 42), stratify=y_temp
    )
    print(f"Test shape: {X_test.shape}")

    X_test_original = X_test.copy()
    
    print("Applying normalization...")
    X_test_normalized = apply_scalers_to_data(X_test, scaler_std, scaler_mm, pad_value=pad_value)

    test_dataset = MicrolensingDataset(X_test_normalized, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("\nGenerating predictions...")
    y_pred, y_probs, y_true = make_predictions(model, test_loader, device)

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=['PSPL', 'Binary'])
    print(report)

    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, eval_dir / "confusion_matrix.png")

    print("Plotting ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_probs, eval_dir / "roc_curve.png")

    print("Analyzing decision times...")
    decision_times, dec_preds = get_decision_times(
        model, X_test_normalized, device, args.confidence_threshold, args.batch_size
    )

    downsample_factor = model.module.downsample_factor if hasattr(model, 'module') else model.downsample_factor
    T_down = X_test_normalized.shape[2] // downsample_factor
    plot_decision_time_distribution(decision_times, T_down, eval_dir / "decision_time_distribution.png")

    print("Plotting accuracy vs decision time...")
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracies, avg_times = [], []
    for thresh in tqdm(thresholds, desc="Threshold sweep"):
        dec_times, dec_preds = get_decision_times(
            model, X_test_normalized, device, thresh, args.batch_size
        )
        accuracies.append(np.mean(dec_preds == y_true))
        avg_times.append(dec_times.mean())

    plot_accuracy_vs_decision_time(
        thresholds, np.array(accuracies), np.array(avg_times),
        eval_dir / "accuracy_vs_decision_time.png"
    )

    print(f"\nGenerating {args.n_samples} sample visualizations...")
    sample_dir = eval_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    n_samples = min(args.n_samples, len(X_test_normalized))
    indices = np.random.choice(len(X_test_normalized), n_samples, replace=False)
    for idx in tqdm(indices, desc="Sample plots"):
        plot_three_panel_sample(
            model, X_test_original[idx:idx+1], X_test_normalized[idx:idx+1],
            y_true[idx], idx, timestamps, device,
            sample_dir / f"sample_{idx:06d}.png", args.confidence_threshold
        )

    summary = {
        "accuracy": float(np.mean(y_pred == y_true)),
        "roc_auc": float(roc_auc),
        "confidence_threshold": args.confidence_threshold,
        "decision_time_mean": float(np.mean(decision_times)),
        "decision_time_median": float(np.median(decision_times)),
        "decision_time_std": float(np.std(decision_times)),
        "classification_report": report
    }

    with open(eval_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Evaluation complete.")
    print(f"All plots and results saved to: {eval_dir}")
    print("\nSummary:")
    print(f"  Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print(f"  ROC AUC: {summary['roc_auc']:.4f}")
    print(f"  Mean decision time: {summary['decision_time_mean']:.1f} steps")


if __name__ == "__main__":
    main()