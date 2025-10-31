#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_samples.py - Visualize sample light curves with model predictions

FIXED: Now uses TDConvClassifier from train.py (matching your trained models!)

Author: Kunal Bhatia
Version: 4.0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Define TDConvClassifier exactly as in train.py
class TDConvClassifier(nn.Module):
    """Compact 1D CNN for binary classification - from train.py"""
    def __init__(self, in_ch: int = 1, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        h = self.net(x)
        mean_pool = torch.mean(h, dim=-1)
        max_pool, _ = torch.max(h, dim=-1)
        z = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(z)

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
    X_original,
    X_normalized,
    y,
    timestamps,
    device,
    sample_indices,
    output_dir,
    pad_value=-1
):
    """Generate 2-panel plots for selected samples"""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process normalized data for model
    X_processed = X_normalized.copy()
    X_processed[X_processed == pad_value] = 0.0
    
    X_tensor = torch.from_numpy(X_processed).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)  # [N, 2]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # [N, 2]
    
    L = X_original.shape[1]
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Generating plots")):
        true_label = y[sample_idx]
        true_label_str = "PSPL" if true_label == 0 else "Binary"
        
        # Original unscaled data
        original_data = X_original[sample_idx]
        
        # CNN normalized view
        cnn_data = X_processed[sample_idx]
        
        # Model probabilities
        sample_probs = probs[sample_idx]  # [2]
        predicted_class = np.argmax(sample_probs)
        predicted_label_str = "PSPL" if predicted_class == 0 else "Binary"
        confidence = sample_probs[predicted_class]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        title_text = (
            f"Sample {sample_idx} | True: {true_label_str} | "
            f"Predicted: {predicted_label_str} (conf: {confidence:.3f})"
        )
        
        # --- Panel 1: Original UNSCALED Data ---
        ax1 = axes[0]
        
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
        
        # Add prediction info
        colors = ['blue', 'red']
        labels = ['PSPL', 'Binary']
        
        # Show class probabilities as text
        prob_text = f"P(PSPL) = {sample_probs[0]:.3f}\nP(Binary) = {sample_probs[1]:.3f}"
        ax2.text(0.98, 0.98, prob_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        ax2.set_title(f"2. CNN Input View (Normalized)\n{title_text}", fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'Sequential Timestep (1 to {L})', fontsize=11)
        ax2.set_ylabel('Normalized Flux [0,1]', fontsize=11)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f'sample_{sample_idx:04d}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Saved {len(sample_indices)} sample plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot sample predictions (FIXED)')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--n_samples', type=int, default=12, help='Number of samples to plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError("Must provide --model and --output_dir, OR --experiment_name")
        
        results_dir = find_latest_results_dir(args.experiment_name)
        print(f"✓ Auto-detected: {results_dir}")
        
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
    print("SAMPLE PREDICTIONS VISUALIZATION (FIXED - TDConvClassifier)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    
    # Load RAW data
    print("\nLoading data...")
    X_original, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X_original.shape[1]
    
    # Load scalers and apply
    scaler_std, scaler_mm = load_scalers(results_dir)
    X_normalized = apply_scalers_to_data(X_original, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TDConvClassifier(in_ch=1, n_classes=2, dropout=0.3)
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded (TDConvClassifier)")
    
    # Select random samples
    sample_indices = random.sample(range(len(X_original)), min(args.n_samples, len(X_original)))
    print(f"\nGenerating plots for {len(sample_indices)} samples...")
    
    # Generate plots
    plot_sample_predictions(
        model,
        X_original,
        X_normalized,
        y,
        timestamps,
        device,
        sample_indices,
        output_dir,
        pad_value=CFG.PAD_VALUE
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {args.n_samples} sample plots: {output_dir}/sample_*.png")


if __name__ == "__main__":
    main()