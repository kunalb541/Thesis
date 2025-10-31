#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_samples.py - Visualize sample light curves with model predictions

Reproduces the original notebook's 3-panel plots:
1. Original data (scatter plot with decision line) - UNSCALED
2. CNN input view (processed data) - SCALED
3. Class probabilities over time (with confidence threshold and clamping)

Author: Kunal Bhatia
Version: 3.2 - FIXED: Proper scaler loading (no duplicate data loading)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from model import TimeDistributedCNN
from utils import load_npz_dataset, load_scalers, apply_scalers_to_data
import config as CFG


def find_latest_results_dir(experiment_name, base_dir='../results'):
    """Find the most recent results directory for an experiment"""
    base_path = Path(base_dir)
    pattern = f"{experiment_name}_*"
    
    matching_dirs = sorted(base_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No results directories found matching '{pattern}' in {base_dir}")
    
    return matching_dirs[0]


def plot_sample_predictions(
    model,
    X_original,  # Unscaled data for plotting
    X_normalized,  # Normalized data for model
    y,
    timestamps,
    device,
    sample_indices,
    output_dir,
    confidence_threshold=0.9,
    pad_value=-1
):
    """
    Generate 3-panel plots for selected samples (like original notebook)
    
    Args:
        model: Trained model
        X_original: Original unscaled data (for Panel 1 visualization)
        X_normalized: Normalized data (for model inference)
        y: True labels
        timestamps: Time values
        device: torch device
        sample_indices: List of sample indices to plot
        output_dir: Where to save plots
        confidence_threshold: Threshold for early decision
        pad_value: Value used for padding
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process normalized data for model (replace PAD_VALUE with 0)
    X_processed = X_normalized.copy()
    X_processed[X_processed == pad_value] = 0.0
    
    X_tensor = torch.from_numpy(X_processed).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)  # [N, L, 2]
        probs = torch.softmax(outputs, dim=2).cpu().numpy()  # [N, L, 2]
    
    L = X_original.shape[1]
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Generating plots")):
        true_label = y[sample_idx]
        true_label_str = "PSPL" if true_label == 0 else "Binary"
        
        # Original unscaled data (for Panel 1)
        original_data = X_original[sample_idx]
        
        # CNN normalized view (for Panel 2)
        cnn_data = X_processed[sample_idx]
        
        # Model probabilities
        sample_probs = probs[sample_idx]  # [L, 2]
        
        # Find decision time
        decision_time = L
        predicted_class = np.argmax(sample_probs[-1])
        decision_made = False
        
        for t in range(L):
            max_conf = np.max(sample_probs[t])
            if max_conf >= confidence_threshold:
                decision_time = t + 1
                predicted_class = np.argmax(sample_probs[t])
                decision_made = True
                break
        
        predicted_label_str = "PSPL" if predicted_class == 0 else "Binary"
        
        # Clamp probabilities after decision (like original notebook)
        plot_probs = sample_probs.copy()
        if decision_made and decision_time <= L:
            decision_idx = decision_time - 1
            decision_prob = sample_probs[decision_idx, predicted_class]
            
            for t in range(decision_idx, L):
                plot_probs[t, predicted_class] = decision_prob
                plot_probs[t, 1 - predicted_class] = 1.0 - decision_prob
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        title_text = (
            f"Sample {sample_idx} | True: {true_label_str} | "
            f"Predicted: {predicted_label_str} | "
            f"Decision at step {decision_time}/{L}"
        )
        
        # --- Panel 1: Original UNSCALED Data ---
        ax1 = axes[0]
        
        # Show non-padded points (original unscaled flux)
        non_pad_mask = (original_data != pad_value)
        if np.any(non_pad_mask):
            ax1.scatter(
                timestamps[non_pad_mask],
                original_data[non_pad_mask],
                color='darkcyan',
                alpha=0.7,
                s=30,
                label='Observed flux (original scale)'
            )
        
        # Decision line (at end of decision block)
        if decision_time <= L:
            # Map decision time to original timestamp
            decision_timestamp = timestamps[min(decision_time-1, L-1)]
            ax1.axvline(
                x=decision_timestamp,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Decision (time≈{decision_timestamp:.1f})'
            )
        
        ax1.set_title(f"1. Original Unscaled Data\n{title_text}", fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Flux (original scale)', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # --- Panel 2: CNN Input View (Normalized) ---
        ax2 = axes[1]
        
        cnn_timesteps = np.arange(1, L + 1)
        non_pad_mask_cnn = (cnn_data != 0.0)
        
        if np.any(non_pad_mask_cnn):
            ax2.scatter(
                cnn_timesteps[non_pad_mask_cnn],
                cnn_data[non_pad_mask_cnn],
                color='darkcyan',
                alpha=0.7,
                s=30,
                label='CNN Input (normalized [0,1])'
            )
        
        # Decision line
        ax2.axvline(
            x=decision_time,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Decision (step {decision_time})'
        )
        
        ax2.set_title(f"2. CNN Input View (Normalized)\n{title_text}", fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'Sequential Timestep (1 to {L})', fontsize=11)
        ax2.set_ylabel('Normalized Flux [0,1]', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # --- Panel 3: Class Probabilities ---
        ax3 = axes[2]
        
        colors = ['blue', 'red']
        labels = ['PSPL', 'Binary']
        
        for class_idx in range(2):
            ax3.plot(
                cnn_timesteps,
                plot_probs[:, class_idx],
                color=colors[class_idx],
                linewidth=2,
                alpha=0.8,
                label=f'P({labels[class_idx]})'
            )
        
        # Decision line
        ax3.axvline(
            x=decision_time,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Decision (step {decision_time})'
        )
        
        # Confidence threshold line
        ax3.axhline(
            y=confidence_threshold,
            color='green',
            linestyle=':',
            linewidth=1.5,
            alpha=0.5,
            label=f'Threshold ({confidence_threshold})'
        )
        
        ax3.set_title(
            f"3. Class Probabilities Over Time (clamped after decision)\n{title_text}",
            fontsize=12,
            fontweight='bold'
        )
        ax3.set_xlabel(f'Sequential Timestep (1 to {L})', fontsize=11)
        ax3.set_ylabel('Probability', fontsize=11)
        ax3.set_ylim([-0.05, 1.05])
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Link x-axes for panels 2 and 3
        ax3.sharex(ax2)
        
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f'sample_{sample_idx:04d}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Saved {len(sample_indices)} sample plots to {output_dir}")


def plot_decision_time_distribution(
    model,
    X,
    device,
    output_dir,
    confidence_threshold=0.9
):
    """
    Plot distribution of decision times (like original notebook)
    """
    model.eval()
    output_dir = Path(output_dir)
    
    # Get predictions
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    X_tensor = torch.from_numpy(X_processed).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=2).cpu().numpy()
    
    L = X.shape[1]
    decision_times = []
    
    for i in range(len(X)):
        decided = False
        for t in range(L):
            if np.max(probs[i, t]) >= confidence_threshold:
                decision_times.append(t + 1)
                decided = True
                break
        if not decided:
            decision_times.append(L)
    
    decision_times = np.array(decision_times)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(decision_times, bins=range(1, L+2), color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Decision Time Step', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Distribution of Decision Time Steps (threshold={confidence_threshold})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = output_dir / 'decision_time_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved decision time distribution to {save_path}")
    
    return decision_times


def main():
    parser = argparse.ArgumentParser(description='Plot sample predictions')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (auto-detect if not provided)')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-detect if not provided)')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (for auto-detect)')
    parser.add_argument('--n_samples', type=int, default=12, help='Number of samples to plot')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
            args.output_dir = str(results_dir / "sample_plots")
    else:
        results_dir = Path(args.model).parent
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAMPLE PREDICTIONS VISUALIZATION (FIXED VERSION)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # =========================================================================
    # FIXED: Load RAW data once, apply saved scalers
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING DATA WITH SAVED SCALERS")
    print("="*80)
    
    # 1. Load RAW unscaled data
    print("\n1. Loading RAW data (normalize=False)...")
    X_original, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X_original.shape[1]
    print(f"✓ Raw data loaded: {X_original.shape}")
    print(f"   Raw data range: [{X_original[X_original != CFG.PAD_VALUE].min():.3f}, {X_original[X_original != CFG.PAD_VALUE].max():.3f}]")
    
    # 2. Load saved scalers from training
    print("\n2. Loading scalers from training...")
    scaler_std, scaler_mm = load_scalers(results_dir)
    print(f"✓ Loaded scalers from {results_dir}")
    
    # 3. Apply saved scalers to get normalized data
    print("\n3. Applying saved scalers to data...")
    X_normalized = apply_scalers_to_data(X_original, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    print(f"✓ Applied same normalization as training")
    print(f"   Normalized data range: [{X_normalized[X_normalized != CFG.PAD_VALUE].min():.3f}, {X_normalized[X_normalized != CFG.PAD_VALUE].max():.3f}]")
    print(f"   Expected: approximately [0.0, 1.0]")
    # =========================================================================
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded")
    
    # Select random samples
    sample_indices = random.sample(range(len(X_original)), min(args.n_samples, len(X_original)))
    print(f"\nGenerating plots for {len(sample_indices)} samples...")
    
    # Generate sample plots
    plot_sample_predictions(
        model,
        X_original,  # Unscaled for Panel 1
        X_normalized,  # Normalized for model
        y,
        timestamps,
        device,
        sample_indices,
        output_dir,
        confidence_threshold=args.confidence_threshold,
        pad_value=CFG.PAD_VALUE
    )
    
    # Generate decision time distribution
    print("\nGenerating decision time distribution...")
    decision_times = plot_decision_time_distribution(
        model,
        X_normalized,  # Use normalized data for model
        device,
        output_dir,
        confidence_threshold=args.confidence_threshold
    )
    
    print(f"\nStatistics:")
    print(f"  Mean decision time: {decision_times.mean():.1f}")
    print(f"  Median decision time: {np.median(decision_times):.1f}")
    print(f"  Min: {decision_times.min()}, Max: {decision_times.max()}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {args.n_samples} sample plots: {output_dir}/sample_*.png")
    print(f"  - Decision time distribution: {output_dir}/decision_time_distribution.png")


if __name__ == "__main__":
    main()