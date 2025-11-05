#!/usr/bin/env python3
"""
Complete Evaluation Script with Notebook-Style Visualizations

Generates all plots from original notebook:
1. Three-panel sample predictions (Original, Model Input View, Probabilities)
2. Confusion matrix
3. Decision time distribution
4. Accuracy vs decision time
5. Training curves (if available)

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, Dataset

from model import TransformerClassifier
from utils import load_npz_dataset, apply_scalers_to_data, load_scalers


class MicrolensingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_latest_experiment(experiment_name, results_dir='results'):
    """Find latest experiment directory"""
    results_path = Path(results_dir)
    matching = sorted(results_path.glob(f"{experiment_name}_*"))
    if not matching:
        raise ValueError(f"No experiments found matching: {experiment_name}")
    return matching[-1]


def make_predictions(model, loader, device):
    """Get predictions and probabilities"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits, _ = model(X, return_sequence=False)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.append(probs)
            all_labels.extend(y.numpy())
    
    return np.array(all_preds), np.vstack(all_probs), np.array(all_labels)


def get_decision_times(model, X, device, confidence_threshold=0.8, batch_size=128):
    """Get decision time for each sample (matching original notebook logic)"""
    model.eval()
    N = len(X)
    decision_times = []
    predicted_classes = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            
            # Get per-timestep predictions
            logits_seq, _ = model(X_batch, return_sequence=True)  # [B, T_down, 2]
            probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()  # [B, T_down, 2]
            
            for sample_probs in probs_seq:
                T_down = sample_probs.shape[0]
                decision_made = False
                
                for t in range(T_down):
                    confidence = np.max(sample_probs[t])
                    if confidence >= confidence_threshold:
                        decision_times.append(t + 1)
                        predicted_classes.append(np.argmax(sample_probs[t]))
                        decision_made = True
                        break
                
                if not decision_made:
                    decision_times.append(T_down)
                    predicted_classes.append(np.argmax(sample_probs[-1]))
    
    return np.array(decision_times), np.array(predicted_classes)


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
    """
    Three-panel plot matching original notebook:
    1. Original Data (scattered magnitudes with decision line)
    2. Model Input View (normalized data with decision line)  
    3. Class Probabilities Over Time (clamped after decision)
    """
    model.eval()
    
    # Get predictions
    X_tensor = torch.from_numpy(X_normalized).float().to(device)
    
    with torch.no_grad():
        logits_seq, _ = model(X_tensor, return_sequence=True)
        probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()[0]  # [T_down, 2]
    
    T_orig = len(timestamps)
    T_down = probs_seq.shape[0]
    
    # Find decision point
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
    
    # Clamp probabilities after decision (matching notebook logic)
    probs_clamped = probs_seq.copy()
    if decision_made and decision_time_idx < T_down:
        decision_prob = probs_seq[decision_time_idx - 1, predicted_class]
        for t in range(decision_time_idx - 1, T_down):
            probs_clamped[t, predicted_class] = decision_prob
            probs_clamped[t, 1 - predicted_class] = 1.0 - decision_prob
    
    # Map decision time to original timestamps
    downsample_factor = model.downsample_factor
    # Decision line at END of next block (matching notebook)
    target_block = decision_time_idx + 1
    decision_orig_idx = min((target_block * downsample_factor) - 1, T_orig - 1)
    decision_timestamp = timestamps[decision_orig_idx]
    
    # Labels
    true_label_str = "PSPL" if y_true == 0 else "Binary"
    pred_label_str = "PSPL" if predicted_class == 0 else "Binary"
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    title = (f"Sample {sample_idx} | "
             f"True: {true_label_str} | "
             f"Predicted: {pred_label_str} "
             f"(Decision at step {decision_time_idx})")
    
    # Panel 1: Original Data
    ax1 = axes[0]
    original_data = X_original[0]
    non_pad = original_data != pad_value
    
    ax1.scatter(timestamps[non_pad], original_data[non_pad],
                color='darkcyan', alpha=0.7, s=30,
                label='Original magnitude')
    
    ax1.axvline(x=decision_timestamp, color='red', linestyle='--',
                linewidth=2, label=f'Decision time ≈ {decision_timestamp:.1f}')
    
    ax1.set_title(f"1. Original Data\n{title}", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # Note: No ylabel to match notebook
    
    # Panel 2: Model Input View (Normalized)
    ax2 = axes[1]
    model_input = X_normalized[0, 0, :]
    input_timesteps = np.arange(1, T_orig + 1)
    non_pad_input = model_input != pad_value
    
    ax2.scatter(input_timesteps[non_pad_input], model_input[non_pad_input],
                color='darkcyan', alpha=0.7, s=30,
                label='Normalized flux [0,1], pads=-1.0')
    
    ax2.axvline(x=decision_time_idx, color='red', linestyle='--',
                linewidth=2, label=f'Decision (step {decision_time_idx})')
    
    ax2.set_title(f"2. Model Input View (Normalized)\n{title}", fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'Sequential Timestep (1 to {T_orig})', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Class Probabilities
    ax3 = axes[2]
    prob_timesteps = np.arange(1, T_down + 1)
    
    ax3.plot(prob_timesteps, probs_clamped[:, 0],
             label='P(PSPL)', color='#1f77b4', linewidth=2, alpha=0.8)
    ax3.plot(prob_timesteps, probs_clamped[:, 1],
             label='P(Binary)', color='#ff7f0e', linewidth=2, alpha=0.8)
    
    ax3.axvline(x=decision_time_idx, color='red', linestyle='--',
                linewidth=2, label=f'Decision (step {decision_time_idx})')
    
    ax3.set_title(f"3. Class Probabilities Over Time (Clamped after decision)\n{title}",
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel(f'Sequential Timestep (1 to {T_down})', fontsize=11)
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Confusion matrix matching notebook style"""
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
    """Decision time histogram matching notebook"""
    plt.figure(figsize=(8, 5))
    sns.histplot(decision_times, bins=range(1, sequence_length + 2), 
                 kde=False, color='skyblue')
    plt.title('Distribution of Decision Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Samples')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_decision_time(thresholds, accuracies, avg_times, output_path):
    """Accuracy vs decision time matching notebook"""
    plt.figure(figsize=(8, 5))
    plt.plot(avg_times, accuracies, marker='o', linestyle='-', color='green')
    plt.title('Accuracy vs. Average Decision Time')
    plt.xlabel('Average Decision Time Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path):
    """ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model with notebook-style plots")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    parser.add_argument("--data", required=True, help="Path to test data (.npz)")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of sample plots")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Decision threshold")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("EVALUATION WITH NOTEBOOK-STYLE VISUALIZATIONS")
    print("="*80)
    print(f"Device: {device}")
    
    # Find experiment directory
    exp_dir = get_latest_experiment(args.experiment_name)
    print(f"\nExperiment: {exp_dir}")
    
    # Load config
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    
    # Load model
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
    print("✓ Model loaded")
    
    # Load scalers
    print("\nLoading scalers...")
    scaler_std, scaler_mm = load_scalers(exp_dir)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    X_raw, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    
    if X_raw.ndim == 2:
        X_raw = X_raw[:, None, :]
    
    print(f"Data shape: {X_raw.shape}")
    print(f"Class distribution: PSPL={np.sum(y==0)}, Binary={np.sum(y==1)}")
    
    # Keep original data for plotting
    X_original = X_raw.copy()
    
    # Normalize data
    print("\nNormalizing data...")
    X_normalized = apply_scalers_to_data(X_raw, scaler_std, scaler_mm, pad_value=-1.0)
    
    # Create dataset and loader
    dataset = MicrolensingDataset(X_normalized, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_probs, y_true = make_predictions(model, loader, device)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("CLASSIFICATION METRICS")
    print("="*80)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['PSPL', 'Binary']))
    
    # Create output directory
    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, eval_dir / "confusion_matrix.png")
    
    # 2. ROC Curve
    print("Generating ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_probs, eval_dir / "roc_curve.png")
    
    # 3. Decision Time Analysis
    print("Analyzing decision times...")
    decision_times, _ = get_decision_times(model, X_normalized, device, 
                                          args.confidence_threshold, args.batch_size)
    
    T_down = X_normalized.shape[2] // model.downsample_factor
    plot_decision_time_distribution(decision_times, T_down,
                                   eval_dir / "decision_time_distribution.png")
    
    # 4. Accuracy vs Decision Time
    print("Generating accuracy vs decision time plot...")
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracies = []
    avg_times = []
    
    for thresh in thresholds:
        dec_times, dec_preds = get_decision_times(model, X_normalized, device, 
                                                   thresh, args.batch_size)
        acc = np.mean(dec_preds == y_true)
        accuracies.append(acc)
        avg_times.append(dec_times.mean())
    
    plot_accuracy_vs_decision_time(thresholds, np.array(accuracies),
                                  np.array(avg_times),
                                  eval_dir / "accuracy_vs_decision_time.png")
    
    # 5. Sample Plots
    print(f"\nGenerating {args.n_samples} sample plots...")
    sample_dir = eval_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    # Select random samples
    n_samples = min(args.n_samples, len(X_normalized))
    indices = np.random.choice(len(X_normalized), n_samples, replace=False)
    
    for idx in indices:
        plot_three_panel_sample(
            model,
            X_original[idx:idx+1],
            X_normalized[idx:idx+1],
            y_true[idx],
            idx,
            timestamps,
            device,
            sample_dir / f"sample_{idx:04d}.png",
            args.confidence_threshold
        )
    
    # Save evaluation summary
    accuracy = np.mean(y_pred == y_true)
    avg_decision_time = decision_times.mean()
    
    summary = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'avg_decision_time': float(avg_decision_time),
        'median_decision_time': float(np.median(decision_times)),
        'confidence_threshold': args.confidence_threshold,
        'classification_report': classification_report(y_true, y_pred, 
                                                      target_names=['PSPL', 'Binary'],
                                                      output_dict=True)
    }
    
    with open(eval_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Avg Decision Time: {avg_decision_time:.2f} steps")
    print(f"Median Decision Time: {np.median(decision_times):.2f} steps")
    print(f"\n✓ All visualizations saved to: {eval_dir}")
    print("="*80)


if __name__ == "__main__":
    main()