#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_samples.py - Visualize sample light curves (v5.3)

FIXED (v5.3): 
- Loads TimeDistributedCNN (LSTM) model by default
- Loads 3D data [N, 1, T] directly via new utils
- Squeezes 3D raw data to 2D for plotting
- v6.0 FIX: Updated model loading to match simplified CausalCNN

Author: Kunal Bhatia
Version: 6.0
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

# --- FIX: Import the one true model ---
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
    X_original_2d, # Original [N, T] data
    X_normalized_3d, # Normalized [N, 1, T] data
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
    
    # --- START: PADDING FIX ---
    # We NO LONGER replace PAD_VALUE with 0.0
    # X_processed = X_normalized_3d.copy()
    # X_processed[X_processed == pad_value] = 0.0 # <-- REMOVED
    
    # Get predictions for all selected samples
    X_subset = X_normalized_3d[sample_indices] # [n_samples, C, T]
    # --- END: PADDING FIX ---
    
    # Input shape [N_samples, C, T]
    X_tensor = torch.from_numpy(X_subset).float().to(device)
    
    with torch.no_grad():
        # --- Call model with return_sequence=False ---
        # --- NOTE: Model now returns (logits, None) ---
        outputs, _ = model(X_tensor, return_sequence=False)  # [N_samples, 2]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # [N_samples, 2]
    
    L = X_original_2d.shape[1]
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Generating plots")):
        true_label = y[sample_idx]
        true_label_str = "PSPL" if true_label == 0 else "Binary"
        
        # Original unscaled 2D data
        original_data_2d = X_original_2d[sample_idx]
        
        # CNN normalized view (from 3D data, squeezed)
        # --- START: PADDING FIX ---
        # cnn_data = X_processed[sample_idx].squeeze() # [T] # <-- OLD
        cnn_data = X_normalized_3d[sample_idx].squeeze() # [T] # <-- NEW
        # --- END: PADDING FIX ---
        
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
        
        non_pad_mask = (original_data_2d != pad_value)
        if np.any(non_pad_mask):
            ax1.scatter(
                timestamps[non_pad_mask],
                original_data_2d[non_pad_mask],
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
        # --- START: PADDING FIX ---
        # non_pad_mask_cnn = (cnn_data != 0.0) # 0.0 is the normalized pad value # <-- OLD
        non_pad_mask_cnn = (cnn_data != pad_value) # <-- NEW
        # --- END: PADDING FIX ---
        
        if np.any(non_pad_mask_cnn):
            ax2.scatter(
                cnn_timesteps[non_pad_mask_cnn],
                cnn_data[non_pad_mask_cnn],
                color='darkcyan',
                alpha=0.7,
                s=30,
                label='CNN Input (normalized [0,1], pads=-1.0)'
            )
        
        # Add prediction info
        prob_text = f"P(PSPL) = {sample_probs[0]:.3f}\nP(Binary) = {sample_probs[1]:.3f}"
        ax2.text(0.98, 0.98, prob_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        ax2.set_title(f"2. CNN Input View (Normalized)\n{title_text}", fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'Sequential Timestep (1 to {L})', fontsize=11)
        ax2.set_ylabel('Normalized Flux [0,1] (Pads=-1.0)', fontsize=11)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f'sample_{sample_idx:04d}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Saved {len(sample_indices)} sample plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot sample predictions (v5.3)')
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
    print("SAMPLE PREDICTIONS VISUALIZATION (v6.0 - Simplified Causal CNN)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    
    # --- START CHANGED: Load 3D data and squeeze for 2D plot (Fix #4) ---
    print("\nLoading data...")
    # Load RAW 3D data [N, 1, T]
    X_3d_raw, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    # Create 2D version [N, T] for plotting
    X_original_2d = np.squeeze(X_3d_raw, axis=1)
    print(f"✓ Loaded 3D data {X_3d_raw.shape}, squeezed to 2D {X_original_2d.shape}")
    # --- END CHANGED ---
    
    # Load scalers and apply
    try:
        scaler_std, scaler_mm = load_scalers(results_dir)
        # Apply scalers to the 3D raw data
        X_normalized_3d = apply_scalers_to_data(X_3d_raw, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    except Exception as e:
        print(f"⚠ Warning: Could not load or apply scalers: {e}. Using raw data.")
        X_normalized_3d = X_3d_raw # Fallback
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # --- Load TimeDistributedCNN (LSTM) model (no change) ---
    config_path = results_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # --- START: MODEL INSTANTIATION FIX ---
    print("Loading TimeDistributedCNN (Simplified Causal CNN)...")
    model = TimeDistributedCNN(
        in_channels=1, 
        n_classes=2, 
        # window_size=window_size, # REMOVED
        # use_lstm=True,           # REMOVED
        dropout=config.get('dropout', 0.3) # Get dropout from config
    )
    model_type = "TimeDistributed_CausalCNN_Simplified"
    # --- END: MODEL INSTANTIATION FIX ---
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"✓ Model loaded ({model_type})")
    
    # Select random samples
    sample_indices = random.sample(range(len(X_original_2d)), min(args.n_samples, len(X_original_2d)))
    print(f"\nGenerating plots for {len(sample_indices)} samples...")
    
    # Generate plots
    plot_sample_predictions(
        model,
        X_original_2d,   # Pass 2D original for plotting
        X_normalized_3d, # Pass 3D normalized for model
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


if __name__ == "__main__":
    main()