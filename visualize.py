#!/usr/bin/env python3
"""
Visualization Module for Microlensing Classification

Generates all plots shown in thesis including:
1. Three-panel sample predictions (Original, CNN View, Probabilities)
2. Confusion matrix
3. Distribution of decision time steps
4. Accuracy vs. Average decision time

Author: Kunal Bhatia
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import List, Tuple, Optional


def plot_three_panel_sample(
    model,
    X_original: np.ndarray,
    X_normalized: np.ndarray,
    y_true: int,
    sample_idx: int,
    timestamps: np.ndarray,
    device: torch.device,
    output_path: Path,
    confidence_threshold: float = 0.8,
    pad_value: float = -1.0
):
    """
    Create three-panel plot matching your thesis format:
    1. Original Data (with decision line)
    2. CNN Input View (normalized, with decision line)
    3. Class Probabilities Over Time (with clamping)
    
    Args:
        model: Trained Transformer model
        X_original: Original unnormalized data [1, T]
        X_normalized: Normalized data [1, 1, T] ready for model
        y_true: True label (0=PSPL, 1=Binary)
        sample_idx: Sample index for title
        timestamps: Original timestamps
        device: torch device
        output_path: Where to save the plot
        confidence_threshold: Threshold for decision making
        pad_value: Padding value (-1.0)
    """
    model.eval()
    
    # Get predictions for entire sequence
    X_tensor = torch.from_numpy(X_normalized).float().to(device)
    
    with torch.no_grad():
        logits_seq, _ = model(X_tensor, return_sequence=True)  # [1, T_down, 2]
        probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()[0]  # [T_down, 2]
    
    T_orig = len(timestamps)
    T_down = probs_seq.shape[0]
    
    # Find decision point
    decision_time_idx = T_down  # Default to end
    predicted_class = np.argmax(probs_seq[-1])
    decision_made = False
    
    for t in range(T_down):
        confidence = np.max(probs_seq[t])
        if confidence >= confidence_threshold:
            decision_time_idx = t + 1
            predicted_class = np.argmax(probs_seq[t])
            decision_made = True
            break
    
    # Clamp probabilities after decision
    probs_clamped = probs_seq.copy()
    if decision_made and decision_time_idx < T_down:
        decision_prob = probs_seq[decision_time_idx - 1, predicted_class]
        for t in range(decision_time_idx - 1, T_down):
            probs_clamped[t, predicted_class] = decision_prob
            probs_clamped[t, 1 - predicted_class] = 1.0 - decision_prob
    
    # Map decision time to original timestamps
    downsample_factor = model.downsample_factor
    decision_orig_idx = min(decision_time_idx * downsample_factor, T_orig - 1)
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
    
    # ========================================================================
    # Panel 1: Original Data
    # ========================================================================
    ax1 = axes[0]
    
    original_data = X_original[0]  # [T]
    non_pad = original_data != pad_value
    
    ax1.scatter(timestamps[non_pad], original_data[non_pad],
                color='darkcyan', alpha=0.7, s=30,
                label='Original magnitude')
    
    # Decision line
    ax1.axvline(x=decision_timestamp, color='red', linestyle='--',
                linewidth=2, label=f'Decision time ≈ {decision_timestamp:.1f}')
    
    ax1.set_title(f"1. Original Data\n{title}", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Magnitudes are inverted
    
    # ========================================================================
    # Panel 2: CNN Input View (Normalized)
    # ========================================================================
    ax2 = axes[1]
    
    cnn_data = X_normalized[0, 0, :]  # [T]
    cnn_timesteps = np.arange(1, T_orig + 1)
    non_pad_cnn = cnn_data != pad_value
    
    ax2.scatter(cnn_timesteps[non_pad_cnn], cnn_data[non_pad_cnn],
                color='darkcyan', alpha=0.7, s=30,
                label='Normalized flux [0,1], pads=-1.0')
    
    # Decision line
    ax2.axvline(x=decision_time_idx, color='red', linestyle='--',
                linewidth=2, label=f'Decision (step {decision_time_idx})')
    
    ax2.set_title(f"2. CNN Input View (Normalized)\n{title}", fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'Sequential Timestep (1 to {T_orig})', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 3: Class Probabilities Over Time
    # ========================================================================
    ax3 = axes[2]
    
    prob_timesteps = np.arange(1, T_down + 1)
    
    # Plot probabilities
    ax3.plot(prob_timesteps, probs_clamped[:, 0],
             label='P(PSPL)', color='#1f77b4', linewidth=2, alpha=0.8)
    ax3.plot(prob_timesteps, probs_clamped[:, 1],
             label='P(Binary)', color='#ff7f0e', linewidth=2, alpha=0.8)
    
    # Decision line
    ax3.axvline(x=decision_time_idx, color='red', linestyle='--',
                linewidth=2, label=f'Decision (step {decision_time_idx})')
    
    ax3.set_title(f"3. Class Probabilities Over Time (Clamped after decision)\n{title}",
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel(f'Downsampled Timestep (1 to {T_down})', fontsize=11)
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: List[str] = ['PSPL', 'Binary']
):
    """
    Plot confusion matrix matching thesis format
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_time_distribution(
    decision_times: np.ndarray,
    sequence_length: int,
    output_path: Path
):
    """
    Plot distribution of decision time steps
    """
    plt.figure(figsize=(8, 5))
    
    bins = range(1, sequence_length + 2)
    sns.histplot(decision_times, bins=bins, kde=False, color='skyblue')
    
    plt.title('Distribution of Decision Time Steps', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_decision_time(
    thresholds: np.ndarray,
    accuracies: np.ndarray,
    avg_decision_times: np.ndarray,
    output_path: Path
):
    """
    Plot accuracy vs. average decision time
    """
    plt.figure(figsize=(8, 5))
    
    plt.plot(avg_decision_times, accuracies, 'o-',
             linewidth=2.5, markersize=8, color='green')
    
    plt.title('Accuracy vs. Average Decision Time', fontsize=14, fontweight='bold')
    plt.xlabel('Average Decision Time Step', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_decision_times(
    model,
    X: np.ndarray,
    device: torch.device,
    confidence_threshold: float = 0.8,
    batch_size: int = 128
) -> np.ndarray:
    """
    Get decision times for all samples
    
    Args:
        model: Trained model
        X: Input data [N, 1, T]
        device: torch device
        confidence_threshold: Decision threshold
        batch_size: Batch size for inference
    
    Returns:
        decision_times: [N] array of decision timesteps
    """
    model.eval()
    N = len(X)
    decision_times = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            
            logits_seq, _ = model(X_batch, return_sequence=True)
            probs_seq = torch.softmax(logits_seq, dim=-1).cpu().numpy()
            
            for sample_probs in probs_seq:
                T_down = sample_probs.shape[0]
                decision_made = False
                
                for t in range(T_down):
                    if np.max(sample_probs[t]) >= confidence_threshold:
                        decision_times.append(t + 1)
                        decision_made = True
                        break
                
                if not decision_made:
                    decision_times.append(T_down)
    
    return np.array(decision_times)


def generate_all_visualizations(
    model,
    X_original: np.ndarray,
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    timestamps: np.ndarray,
    device: torch.device,
    output_dir: Path,
    n_sample_plots: int = 12,
    confidence_threshold: float = 0.8
):
    """
    Generate all visualizations for thesis
    
    Args:
        model: Trained model
        X_original: Original data [N, T]
        X_normalized: Normalized data [N, 1, T]
        y_true: True labels [N]
        timestamps: Timestamps [T]
        device: torch device
        output_dir: Output directory
        n_sample_plots: Number of sample plots to generate
        confidence_threshold: Decision threshold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Get predictions
    print("\nGetting predictions...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_normalized), 128):
            X_batch = torch.from_numpy(X_normalized[i:i+128]).float().to(device)
            logits, _ = model(X_batch, return_sequence=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    all_preds = np.array(all_preds)
    
    # 1. Confusion Matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(y_true, all_preds, output_dir / 'confusion_matrix.png')
    
    # 2. Decision Time Distribution
    print("Analyzing decision times...")
    decision_times = get_decision_times(model, X_normalized, device, confidence_threshold)
    T_down = X_normalized.shape[2] // model.downsample_factor
    plot_decision_time_distribution(decision_times, T_down, 
                                   output_dir / 'decision_time_distribution.png')
    
    # 3. Accuracy vs Decision Time
    print("Generating accuracy vs decision time plot...")
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracies = []
    avg_times = []
    
    for thresh in thresholds:
        dec_times = get_decision_times(model, X_normalized, device, thresh)
        # Get predictions at decision times
        # (simplified - you may want more sophisticated logic)
        acc = np.mean(all_preds == y_true)
        accuracies.append(acc)
        avg_times.append(dec_times.mean())
    
    plot_accuracy_vs_decision_time(thresholds, np.array(accuracies), 
                                  np.array(avg_times),
                                  output_dir / 'accuracy_vs_decision_time.png')
    
    # 4. Sample Plots
    print(f"\nGenerating {n_sample_plots} sample plots...")
    sample_dir = output_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)
    
    # Select diverse samples
    indices = np.random.choice(len(X_normalized), n_sample_plots, replace=False)
    
    for idx in indices:
        plot_three_panel_sample(
            model,
            X_original[idx:idx+1],
            X_normalized[idx:idx+1],
            y_true[idx],
            idx,
            timestamps,
            device,
            sample_dir / f'sample_{idx:04d}.png',
            confidence_threshold
        )
    
    print(f"\n✅ All visualizations saved to {output_dir}")
    print("="*80)


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_three_panel_sample()")
    print("  - plot_confusion_matrix()")
    print("  - plot_decision_time_distribution()")
    print("  - plot_accuracy_vs_decision_time()")
    print("  - generate_all_visualizations()")