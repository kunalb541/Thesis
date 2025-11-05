#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_samples.py - Visualize sample light curves with model predictions

FIXED (v5.0): 
- Removed hard-coded TDConvClassifier
- Imports models from model.py
- Reads config.json to load the correct model (Simple or LSTM)
- Calls model(x, return_sequence=False) for final prediction

Author: Kunal Bhatia
Version: 5.0
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
import json

# Import model architectures (matches train.py)
from model import TimeDistributedCNNSimple, TimeDistributedCNN

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
    
    # Get predictions for all selected samples
    X_subset = X_processed[sample_indices]
    
    # Input shape [N_samples, T] -> [N_samples, 1, T]
    X_tensor = torch.from_numpy(X_subset).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        # --- FIX: Call model with return_sequence=False ---
        outputs = model(X_tensor, return_sequence=False)  # [N_samples, 2]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # [N_samples, 2]
    
    L = X_original.shape[1]
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Generating plots")):
        true_label = y[sample_idx]
        true_label_str = "PSPL" if true_label == 0 else "Binary"
        
        # Original unscaled data
        original_data = X_original[sample_idx]
        
        # CNN normalized view
        cnn_data = X_processed[sample_idx]
        
        # Model probabilities (from pre-computed batch)
        sample_probs = probs[idx]  # [2]
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
    parser = argparse.ArgumentParser(description='Plot sample predictions (v5.0 - TimeDistributed Compatible)')
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
    print("SAMPLE PREDICTIONS VISUALIZATION (TimeDistributed Compatible)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    
    # Load RAW data
    print("\nLoading data...")
    # Load data, X shape is (N, T)
    X_original, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X_original.shape[1]
    
    # Load scalers and apply
    scaler_std, scaler_mm = load_scalers(results_dir)
    # apply_scalers_to_data expects (N, F) where F=T
    X_normalized = apply_scalers_to_data(X_original, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # --- FIX: Load model based on config.json ---
    config_path = results_dir / "config.json"
    model_type = 'TimeDistributed_Simple'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.get('model_type', 'TimeDistributed_Simple')

    if model_type == 'TimeDistributed_LSTM':
        print("Loading TimeDistributedCNN (LSTM)...")
        window_size = config.get('window_size', 50)
        model = TimeDistributedCNN(
            in_channels=1, 
            n_classes=2, 
            window_size=window_size,
            use_lstm=True,
            dropout=0.3 
        )
    else:
        print("Loading TimeDistributedCNNSimple...")
        model = TimeDistributedCNNSimple(
            in_channels=1, 
            n_classes=2, 
            dropout=0.3
        )
    # --- End Fix ---
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"✓ Model loaded ({model_type})")
    
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