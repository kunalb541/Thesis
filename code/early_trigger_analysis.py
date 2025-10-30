#!/usr/bin/env python3
"""
Real-time trigger decision analysis
Determines optimal threshold for follow-up observations
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
import json
from pathlib import Path
from tqdm import tqdm

import config as CFG
from model import TimeDistributedCNN
from utils import load_npz_dataset


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        X = X.copy()
        X[X == CFG.PAD_VALUE] = 0.0
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_predictions_at_fraction(model, X, y, device, fraction, batch_size=128):
    """Get model predictions using only first `fraction` of observations"""
    L = X.shape[1]
    cutoff = int(L * fraction)
    
    # Zero out observations after cutoff
    X_partial = X.copy()
    X_partial[:, cutoff:] = 0.0
    
    dataset = NumpyDataset(X_partial, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)
    
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"Inference ({int(fraction*100)}%)", leave=False):
            xb = xb.to(device)
            outputs = model(xb)  # [B, L, 2]
            
            # Only use predictions up to cutoff point
            logits = outputs[:, :cutoff, :].mean(dim=1)  # [B, 2]
            probs = torch.softmax(logits, dim=1)  # [B, 2]
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return probs, labels


def compute_trigger_metrics(probs, labels, threshold):
    """
    Compute metrics for a given confidence threshold
    
    Trigger if P(Binary) > threshold
    """
    binary_probs = probs[:, 1]  # Probability of Binary class
    
    triggered = binary_probs > threshold
    
    # Ground truth
    true_binary = (labels == 1)
    true_pspl = (labels == 0)
    
    # Outcomes
    true_positives = np.sum(triggered & true_binary)
    false_positives = np.sum(triggered & true_pspl)
    false_negatives = np.sum(~triggered & true_binary)
    true_negatives = np.sum(~triggered & true_pspl)
    
    # Metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    trigger_rate = np.mean(triggered)  # Fraction of events triggered
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'trigger_rate': trigger_rate,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives)
    }


def analyze_early_triggering(model_path, data_path, output_dir, device='cuda'):
    """
    Comprehensive analysis of early detection triggering
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EARLY DETECTION TRIGGER ANALYSIS")
    print("="*80)
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    X, y, timestamps, meta = load_npz_dataset(data_path, apply_perm=True)
    L = X.shape[1]
    
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Data loaded: {X.shape}")
    
    # Test different observation fractions
    fractions = [0.25, 0.33, 0.5, 0.67, 0.83, 1.0]
    thresholds = np.linspace(0.5, 0.95, 20)  # Test various confidence thresholds
    
    results = {}
    
    for frac in fractions:
        print(f"\n{'='*80}")
        print(f"ANALYSIS: {int(frac*100)}% of Event Observed")
        print(f"{'='*80}")
        
        # Get predictions at this fraction
        probs, labels = get_predictions_at_fraction(model, X, y, device, frac)
        
        # Test different thresholds
        frac_results = []
        for threshold in thresholds:
            metrics = compute_trigger_metrics(probs, labels, threshold)
            frac_results.append(metrics)
        
        results[frac] = {
            'probs': probs,
            'labels': labels,
            'threshold_analysis': frac_results
        }
        
        # Print optimal threshold (max F1)
        best = max(frac_results, key=lambda x: x['f1'])
        print(f"\nOptimal Threshold: {best['threshold']:.3f}")
        print(f"  Precision: {best['precision']:.3f}")
        print(f"  Recall:    {best['recall']:.3f}")
        print(f"  F1 Score:  {best['f1']:.3f}")
        print(f"  Trigger Rate: {best['trigger_rate']*100:.1f}% of events")
        print(f"  True Positives:  {best['true_positives']:,}")
        print(f"  False Positives: {best['false_positives']:,}")
    
    # Generate plots
    plot_threshold_analysis(results, output_dir)
    plot_roc_curves(results, output_dir)
    plot_trigger_efficiency(results, output_dir)
    
    # Save results
    save_results(results, output_dir, fractions, thresholds)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete. Results saved to {output_dir}")
    print(f"{'='*80}")


def plot_threshold_analysis(results, output_dir):
    """Plot precision/recall/F1 vs threshold for each fraction"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    fractions = sorted(results.keys())
    
    for idx, frac in enumerate(fractions):
        ax = axes[idx]
        analysis = results[frac]['threshold_analysis']
        
        thresholds = [r['threshold'] for r in analysis]
        precisions = [r['precision'] for r in analysis]
        recalls = [r['recall'] for r in analysis]
        f1s = [r['f1'] for r in analysis]
        
        ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        ax.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
        
        # Mark optimal F1
        best_idx = np.argmax(f1s)
        ax.axvline(thresholds[best_idx], color='gray', linestyle='--', alpha=0.5)
        ax.plot(thresholds[best_idx], f1s[best_idx], 'g*', markersize=15)
        
        ax.set_xlabel('Confidence Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{int(frac*100)}% Observed', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: threshold_analysis.png")


def plot_roc_curves(results, output_dir):
    """Plot ROC curves for each observation fraction"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fractions = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    for frac, color in zip(fractions, colors):
        probs = results[frac]['probs'][:, 1]  # Binary probability
        labels = results[frac]['labels']
        
        fpr, tpr, _ = roc_curve(labels, probs)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2.5, 
               label=f'{int(frac*100)}% obs (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves: Early Detection Performance', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_early.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: roc_curves_early.png")


def plot_trigger_efficiency(results, output_dir):
    """Plot trigger efficiency: what % of true binaries captured vs observation time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fractions = sorted(results.keys())
    
    # For each "desired precision" level, plot recall vs observation fraction
    target_precisions = [0.7, 0.8, 0.9, 0.95]
    colors = ['red', 'orange', 'green', 'blue']
    
    for target_prec, color in zip(target_precisions, colors):
        recalls = []
        trigger_rates = []
        
        for frac in fractions:
            analysis = results[frac]['threshold_analysis']
            
            # Find threshold that gives closest to target precision
            valid = [r for r in analysis if r['precision'] >= target_prec]
            if valid:
                best = max(valid, key=lambda x: x['recall'])
                recalls.append(best['recall'])
                trigger_rates.append(best['trigger_rate'])
            else:
                recalls.append(0)
                trigger_rates.append(0)
        
        ax.plot([f*100 for f in fractions], [r*100 for r in recalls], 
               'o-', color=color, linewidth=2.5, markersize=10,
               label=f'Precision ≥ {target_prec:.0%}')
    
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% recall target')
    ax.set_xlabel('Event Observed (%)', fontsize=13)
    ax.set_ylabel('Recall (% True Binaries Detected)', fontsize=13)
    ax.set_title('Early Detection Efficiency', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trigger_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: trigger_efficiency.png")


def save_results(results, output_dir, fractions, thresholds):
    """Save numerical results to JSON"""
    summary = {}
    
    for frac in fractions:
        analysis = results[frac]['threshold_analysis']
        
        # Find optimal threshold (max F1)
        best = max(analysis, key=lambda x: x['f1'])
        
        summary[f'{int(frac*100)}%_observed'] = {
            'optimal_threshold': best['threshold'],
            'precision': best['precision'],
            'recall': best['recall'],
            'f1_score': best['f1'],
            'trigger_rate': best['trigger_rate'],
            'true_positives': best['true_positives'],
            'false_positives': best['false_positives']
        }
    
    with open(output_dir / 'trigger_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved: trigger_analysis.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Early detection trigger analysis')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()
    
    analyze_early_triggering(args.model, args.data, args.output_dir, args.device)


if __name__ == "__main__":
    main()
