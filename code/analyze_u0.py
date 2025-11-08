#!/usr/bin/env python3
"""
Impact Parameter (u0) Dependency Analysis

Demonstrates the physical detection limit at u0 > 0.3.
Critical for thesis research question #4.

Author: Kunal Bhatia
Version: 6.2
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from tqdm import tqdm

from streaming_transformer import StreamingTransformer
from normalization import CausticPreservingNormalizer
import config as CFG


def load_model_and_data(model_path: str, normalizer_path: str, data_path: str, device: str = 'cuda'):
    """Load model, normalizer, and test data with parameters"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = StreamingTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load normalizer
    print(f"Loading normalizer from {normalizer_path}...")
    normalizer = CausticPreservingNormalizer()
    normalizer.load(normalizer_path)
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    # Get parameters if available
    params = None
    if 'params_binary_json' in data:
        params_binary = json.loads(str(data['params_binary_json']))
        if 'params_pspl_json' in data:
            params_pspl = json.loads(str(data['params_pspl_json']))
            params = {'binary': params_binary, 'pspl': params_pspl}
        else:
            params = {'binary': params_binary}
    
    if params is None:
        print("WARNING: No parameter data found. Cannot perform u0 analysis.")
        print("Re-generate dataset with --save_params flag.")
        return None, None, None, None, None
    
    return model, normalizer, X, y, params, device


@torch.no_grad()
def get_predictions(model, X_norm: np.ndarray, device, batch_size: int = 64):
    """Get model predictions"""
    model.eval()
    
    all_preds = []
    all_probs = []
    
    for i in tqdm(range(0, len(X_norm), batch_size), desc="Predicting"):
        batch = X_norm[i:i+batch_size]
        X_tensor = torch.from_numpy(batch).float().to(device)
        
        outputs = model(X_tensor, return_all_timesteps=False)
        probs = F.softmax(outputs['binary'], dim=-1)
        preds = probs.argmax(dim=-1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def analyze_u0_dependency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    params: dict,
    n_bins: int = 10
):
    """Analyze accuracy as function of u0"""
    
    # Extract u0 values for binary events
    binary_params = params['binary']
    binary_mask = y_true == 1
    
    # Get u0 values
    u0_values = np.array([p['u0'] for p in binary_params])
    
    # Bin by u0
    u0_bins = np.linspace(u0_values.min(), u0_values.max(), n_bins + 1)
    u0_centers = (u0_bins[:-1] + u0_bins[1:]) / 2
    
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        u0_low, u0_high = u0_bins[i], u0_bins[i+1]
        
        # Find events in this bin
        in_bin = (u0_values >= u0_low) & (u0_values < u0_high)
        
        if in_bin.sum() > 0:
            # Get accuracy for this bin
            bin_true = y_true[binary_mask][in_bin]
            bin_pred = y_pred[binary_mask][in_bin]
            
            acc = accuracy_score(bin_true, bin_pred)
            accuracies.append(acc)
            counts.append(in_bin.sum())
        else:
            accuracies.append(np.nan)
            counts.append(0)
    
    return {
        'u0_bins': u0_bins.tolist(),
        'u0_centers': u0_centers.tolist(),
        'accuracies': accuracies,
        'counts': counts,
        'all_u0': u0_values.tolist()
    }


def plot_u0_dependency(results: dict, save_path: Path, threshold: float = 0.3):
    """Plot accuracy vs u0"""
    
    u0_centers = results['u0_centers']
    accuracies = [a*100 if not np.isnan(a) else None for a in results['accuracies']]
    counts = results['counts']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Accuracy plot
    valid_indices = [i for i, a in enumerate(accuracies) if a is not None]
    valid_u0 = [u0_centers[i] for i in valid_indices]
    valid_acc = [accuracies[i] for i in valid_indices]
    
    ax1.plot(valid_u0, valid_acc, 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Physical Limit (u₀ = {threshold})')
    ax1.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='Target (70%)')
    
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=13)
    ax1.set_title('Binary Classification Accuracy vs. Impact Parameter', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add annotations
    for u, a, c in zip(valid_u0, valid_acc, [counts[i] for i in valid_indices]):
        ax1.annotate(f'{a:.1f}%\n(n={c})', 
                    xy=(u, a),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Count histogram
    ax2.bar(u0_centers, counts, width=(u0_centers[1]-u0_centers[0])*0.8, 
           color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Impact Parameter u₀', fontsize=13)
    ax2.set_ylabel('Number of Events', fontsize=13)
    ax2.set_title('Distribution of Impact Parameters', fontsize=12)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")


def plot_u0_distribution_2d(results: dict, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot 2D distribution of u0 vs performance"""
    
    u0_values = np.array(results['all_u0'])
    binary_mask = y_true == 1
    
    # Separate correct and incorrect
    correct_mask = y_pred[binary_mask] == y_true[binary_mask]
    
    u0_correct = u0_values[correct_mask]
    u0_incorrect = u0_values[~correct_mask]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histograms
    bins = np.linspace(u0_values.min(), u0_values.max(), 30)
    ax.hist(u0_correct, bins=bins, alpha=0.6, color='green', label='Correct', density=True)
    ax.hist(u0_incorrect, bins=bins, alpha=0.6, color='red', label='Incorrect', density=True)
    
    ax.axvline(x=0.3, color='black', linestyle='--', linewidth=2, label='Physical Limit (u₀=0.3)')
    
    ax.set_xlabel('Impact Parameter u₀', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Distribution of Correct vs. Incorrect Classifications by u₀', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to {save_path}")


def generate_report(results: dict, save_path: Path, threshold: float = 0.3):
    """Generate u0 analysis report"""
    
    u0_centers = results['u0_centers']
    accuracies = results['accuracies']
    counts = results['counts']
    
    # Find accuracy at threshold
    threshold_idx = np.argmin(np.abs(np.array(u0_centers) - threshold))
    acc_at_threshold = accuracies[threshold_idx] if not np.isnan(accuracies[threshold_idx]) else None
    
    # Count events above/below threshold
    u0_all = np.array(results['all_u0'])
    n_below = (u0_all < threshold).sum()
    n_above = (u0_all >= threshold).sum()
    
    # Average accuracies
    valid_acc = [a for a in accuracies if not np.isnan(a)]
    
    report = {
        'threshold': threshold,
        'accuracy_at_threshold': float(acc_at_threshold) if acc_at_threshold else None,
        'events_below_threshold': int(n_below),
        'events_above_threshold': int(n_above),
        'fraction_above_threshold': float(n_above / len(u0_all)),
        'mean_accuracy_all': float(np.mean(valid_acc)) if valid_acc else None,
        'u0_bins': {
            'centers': u0_centers,
            'accuracies': [float(a) if not np.isnan(a) else None for a in accuracies],
            'counts': counts
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("U0 DEPENDENCY ANALYSIS REPORT")
    print("="*60)
    print(f"Physical Detection Threshold: u₀ = {threshold}")
    print(f"Accuracy at threshold: {acc_at_threshold*100:.2f}%" if acc_at_threshold else "N/A")
    print(f"\nEvent Distribution:")
    print(f"  Below threshold (u₀ < {threshold}): {n_below} ({n_below/len(u0_all)*100:.1f}%)")
    print(f"  Above threshold (u₀ ≥ {threshold}): {n_above} ({n_above/len(u0_all)*100:.1f}%)")
    print(f"\nMean Accuracy: {np.mean(valid_acc)*100:.2f}%" if valid_acc else "N/A")
    print(f"\nReport saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze u0 dependency')
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--data', required=True, help='Path to dataset with parameters')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of u0 bins')
    parser.add_argument('--threshold', type=float, default=0.3, help='Physical limit threshold')
    parser.add_argument('--output_dir', type=str, help='Custom output directory')
    
    args = parser.parse_args()
    
    # Find experiment
    results_dir = Path(CFG.RESULTS_DIR)
    exp_dirs = sorted(results_dir.glob(f"{args.experiment_name}_*"))
    
    if not exp_dirs:
        print(f"ERROR: No experiment found matching '{args.experiment_name}'")
        return
    
    exp_dir = exp_dirs[-1]
    print(f"Using experiment: {exp_dir.name}")
    
    # Setup paths
    model_path = exp_dir / 'best_model.pt'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    if not model_path.exists() or not normalizer_path.exists():
        print(f"ERROR: Missing model or normalizer in {exp_dir}")
        return
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = exp_dir / 'u0_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load everything
    result = load_model_and_data(
        str(model_path), str(normalizer_path), args.data
    )
    
    if result[0] is None:
        return
    
    model, normalizer, X, y, params, device = result
    
    # Normalize data
    print("\nNormalizing data...")
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    X_norm = normalizer.transform(X).squeeze(1)
    
    # Get predictions
    print("\nGetting predictions...")
    y_pred, y_probs = get_predictions(model, X_norm, device)
    
    # Analyze u0 dependency
    print("\nAnalyzing u0 dependency...")
    results = analyze_u0_dependency(y, y_pred, y_probs, params, args.n_bins)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_u0_dependency(results, output_dir / 'u0_dependency.png', args.threshold)
    plot_u0_distribution_2d(results, y, y_pred, output_dir / 'u0_distribution.png')
    
    # Generate report
    generate_report(results, output_dir / 'u0_report.json', args.threshold)
    
    print(f"\n✅ U0 analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()