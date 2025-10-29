#!/usr/bin/env python3
"""
Real-Time Binary Detection Analysis

This is the CORE analysis for your thesis:
- At what point during observation can we confidently identify binaries?
- How does confidence evolve over time?
- Can we trigger follow-up earlier than traditional methods?

Even with modest overall accuracy, early detection at high confidence is valuable.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

def load_model_and_data(model_path, data_path):
    """Load trained model and test data"""
    from model import TimeDistributedCNN
    
    # Load data
    data = np.load(data_path)
    X, y, timestamps = data['X'], data['y'], data['timestamps']
    
    # Apply perm
    if 'perm' in data.files:
        perm = data['perm']
        X, y = X[perm], y[perm]
    
    # Convert labels
    if y.dtype.kind in ('U', 'S', 'O'):
        y = np.array([0 if str(v).lower().startswith('pspl') else 1 for v in y], dtype=np.uint8)
    
    # Preprocess
    X_proc = X.copy()
    X_proc[X_proc == -1] = 0.0
    
    # Load model
    L = X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, X_proc, y, timestamps, device


def get_timestep_predictions(model, X, device, batch_size=256):
    """Get predictions at each timestep for all events"""
    N = X.shape[0]
    X_tensor = torch.from_numpy(X).float().unsqueeze(1).to(device)  # [N, 1, L]
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)  # [B, L, 2]
            all_logits.append(outputs.cpu())
    
    logits_t = torch.cat(all_logits, dim=0).numpy()  # [N, L, 2]
    probs_t = torch.softmax(torch.from_numpy(logits_t), dim=-1).numpy()
    
    return probs_t  # [N, L, 2]


def analyze_detection_timing(probs_t, labels, timestamps, thresholds=[0.6, 0.7, 0.8, 0.9]):
    """
    For each binary event, find when confidence first crosses each threshold.
    
    This is THE KEY METRIC for real-time triggering.
    """
    N, L, _ = probs_t.shape
    binary_probs = probs_t[:, :, 1]  # [N, L]
    
    binary_mask = (labels == 1)
    n_binary = binary_mask.sum()
    
    results = {}
    
    for thresh in thresholds:
        detection_fractions = []
        detection_times = []
        
        for i in np.where(binary_mask)[0]:
            # Find first timestep where confidence > threshold
            confident_steps = np.where(binary_probs[i] > thresh)[0]
            if len(confident_steps) > 0:
                det_step = confident_steps[0]
                det_frac = (det_step + 1) / L
                det_time = timestamps[det_step]
                detection_fractions.append(det_frac)
                detection_times.append(det_time)
        
        n_detected = len(detection_fractions)
        detection_rate = n_detected / n_binary if n_binary > 0 else 0
        
        results[thresh] = {
            'n_detected': n_detected,
            'detection_rate': detection_rate,
            'fractions': detection_fractions,
            'times': detection_times,
            'median_fraction': np.median(detection_fractions) if detection_fractions else None,
            'mean_fraction': np.mean(detection_fractions) if detection_fractions else None,
            'p25_fraction': np.percentile(detection_fractions, 25) if detection_fractions else None,
            'p75_fraction': np.percentile(detection_fractions, 75) if detection_fractions else None,
        }
    
    return results


def plot_confidence_evolution(probs_t, labels, timestamps, save_path, n_examples=9):
    """
    Plot how confidence evolves over time for example binary events.
    This shows the "real-time classification" story visually.
    """
    binary_probs = probs_t[:, :, 1]
    binary_indices = np.where(labels == 1)[0]
    
    np.random.seed(42)
    examples = np.random.choice(binary_indices, size=min(n_examples, len(binary_indices)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, event_idx in enumerate(examples):
        ax = axes[idx]
        
        # Plot confidence over time
        ax.plot(timestamps, binary_probs[event_idx], linewidth=2, color='darkblue', label='Binary Confidence')
        
        # Add threshold lines
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Mark detection points
        for thresh, color in [(0.7, 'orange'), (0.9, 'red')]:
            det_idx = np.where(binary_probs[event_idx] > thresh)[0]
            if len(det_idx) > 0:
                det_time = timestamps[det_idx[0]]
                det_frac = (det_idx[0] + 1) / len(timestamps)
                ax.axvline(det_time, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
                ax.text(det_time, 0.95, f'{det_frac:.0%}', 
                       color=color, fontsize=9, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel('Binary Probability', fontsize=10)
        ax.set_title(f'Binary Event {event_idx}', fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence evolution plot saved to {save_path}")


def plot_detection_histograms(results, save_path):
    """Plot when binaries are detected at different thresholds"""
    thresholds = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, thresh in enumerate(thresholds[:4]):
        ax = axes[idx]
        fracs = results[thresh]['fractions']
        
        if fracs:
            ax.hist(fracs, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            median = results[thresh]['median_fraction']
            ax.axvline(median, color='red', linestyle='--', linewidth=2, 
                      label=f'Median: {median:.1%}')
            
            ax.set_xlabel('Fraction of Event Observed at Detection', fontsize=11)
            ax.set_ylabel('Number of Binary Events', fontsize=11)
            ax.set_title(f'Detection at {thresh:.0%} Confidence Threshold', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Add text with statistics
            text = f"Detected: {results[thresh]['n_detected']}\n"
            text += f"Rate: {results[thresh]['detection_rate']:.1%}\n"
            text += f"Mean: {results[thresh]['mean_fraction']:.1%}"
            ax.text(0.97, 0.97, text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Detection histograms saved to {save_path}")


def plot_detection_summary(results, save_path):
    """Summary plot: median detection time vs threshold"""
    thresholds = sorted(results.keys())
    medians = [results[t]['median_fraction'] * 100 for t in thresholds if results[t]['median_fraction']]
    detection_rates = [results[t]['detection_rate'] * 100 for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Detection timing
    ax1.plot(thresholds, medians, 'o-', linewidth=2, markersize=10, color='darkblue')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Median Detection Point (% of Event)', fontsize=12)
    ax1.set_title('When Are Binaries Detected?', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Annotate key points
    for t, m in zip(thresholds, medians):
        ax1.text(t, m + 3, f'{m:.0f}%', ha='center', fontsize=10)
    
    # Plot 2: Detection rate
    ax2.plot(thresholds, detection_rates, 's-', linewidth=2, markersize=10, color='darkred')
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Detection Rate (% of Binaries)', fontsize=12)
    ax2.set_title('What Fraction of Binaries Are Detected?', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 100])
    
    for t, r in zip(thresholds, detection_rates):
        ax2.text(t, r + 3, f'{r:.0f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Detection summary saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze real-time binary detection')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.6, 0.7, 0.8, 0.9],
                       help='Confidence thresholds to analyze')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("REAL-TIME BINARY DETECTION ANALYSIS")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Thresholds: {args.thresholds}")
    
    # Load model and data
    print("\nLoading model and data...")
    model, X, y, timestamps, device = load_model_and_data(args.model, args.data)
    print(f"Loaded {len(y)} events")
    print(f"Binary events: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"Device: {device}")
    
    # Get timestep predictions
    print("\nComputing predictions at each timestep...")
    probs_t = get_timestep_predictions(model, X, device)
    print(f"Prediction shape: {probs_t.shape}")
    
    # Analyze detection timing
    print("\nAnalyzing detection timing...")
    results = analyze_detection_timing(probs_t, y, timestamps, thresholds=args.thresholds)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for thresh in sorted(results.keys()):
        r = results[thresh]
        print(f"\nThreshold: {thresh:.0%} confidence")
        print(f"  Binaries detected: {r['n_detected']} ({r['detection_rate']:.1%})")
        if r['median_fraction']:
            print(f"  Median detection at: {r['median_fraction']:.1%} of event")
            print(f"  Mean detection at: {r['mean_fraction']:.1%} of event")
            print(f"  25th-75th percentile: {r['p25_fraction']:.1%} - {r['p75_fraction']:.1%}")
    
    # Save results
    results_serializable = {}
    for thresh, r in results.items():
        results_serializable[float(thresh)] = {
            'n_detected': int(r['n_detected']),
            'detection_rate': float(r['detection_rate']),
            'median_fraction': float(r['median_fraction']) if r['median_fraction'] else None,
            'mean_fraction': float(r['mean_fraction']) if r['mean_fraction'] else None,
            'p25_fraction': float(r['p25_fraction']) if r['p25_fraction'] else None,
            'p75_fraction': float(r['p75_fraction']) if r['p75_fraction'] else None,
        }
    
    results_file = output_dir / 'realtime_detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_confidence_evolution(probs_t, y, timestamps, 
                             output_dir / 'confidence_evolution.png', n_examples=9)
    plot_detection_histograms(results, output_dir / 'detection_histograms.png')
    plot_detection_summary(results, output_dir / 'detection_summary.png')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey Finding for Thesis:")
    thresh_70 = results.get(0.7, {})
    if thresh_70.get('median_fraction'):
        print(f"  At 70% confidence threshold:")
        print(f"  → Median detection at {thresh_70['median_fraction']:.1%} of event")
        print(f"  → {100 - thresh_70['median_fraction']*100:.0f}% time savings vs complete light curve")
        print(f"  → {thresh_70['detection_rate']:.1%} of binaries detected")

if __name__ == "__main__":
    main()
