#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Evaluation with saved scalers (FIXED VERSION)

Author: Kunal Bhatia
Version: 3.1 - Fixed to load scalers from training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

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

def early_exit_inference(model, X, device, confidence_threshold=0.9, batch_size=128):
    """
    Reproduce original make_decision logic with early-exit
    
    Returns:
        predictions: Final class predictions
        decision_times: Timestep at which decision was made
        confidences: Confidence at decision time
    """
    model.eval()
    
    predictions = []
    decision_times = []
    confidences = []
    
    # Process in batches
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X[i:batch_end]
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_batch).float().unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)  # [B, L, 2]
            probs = torch.softmax(outputs, dim=2)  # [B, L, 2]
        
        probs_np = probs.cpu().numpy()
        B, L, C = probs_np.shape
        
        for j in range(B):
            decided = False
            
            # Check each timestep for confident prediction
            for t in range(L):
                class_probs = probs_np[j, t]
                max_conf = np.max(class_probs)
                pred_class = np.argmax(class_probs)
                
                if max_conf >= confidence_threshold:
                    predictions.append(pred_class)
                    decision_times.append(t + 1)  # 1-indexed
                    confidences.append(max_conf)
                    decided = True
                    break
            
            # Fallback: use final timestep
            if not decided:
                predictions.append(np.argmax(probs_np[j, -1]))
                decision_times.append(L)
                confidences.append(np.max(probs_np[j, -1]))
    
    return np.array(predictions), np.array(decision_times), np.array(confidences)

def sweep_confidence_threshold(model, X, y, device, thresholds, batch_size=128):
    """
    Reproduce 'Accuracy vs. Average Decision Time' plot
    """
    results = []
    
    for thresh in tqdm(thresholds, desc="Sweeping thresholds"):
        preds, times, confs = early_exit_inference(model, X, device, thresh, batch_size)
        
        acc = (preds == y).mean()
        avg_time = times.mean()
        
        results.append({
            'threshold': float(thresh),
            'accuracy': float(acc),
            'avg_decision_time': float(avg_time)
        })
    
    return results

def early_detection_analysis(model, X, y, device, fractions=[0.1, 0.25, 0.33, 0.5, 0.67, 0.83, 1.0], batch_size=128):
    """
    Test performance when only observing fraction of light curve
    """
    results = {}
    L = X.shape[1]
    
    for frac in tqdm(fractions, desc="Early detection analysis"):
        # Truncate data
        cutoff = max(1, int(L * frac))
        X_truncated = X.copy()
        X_truncated[:, cutoff:] = CFG.PAD_VALUE
        
        # Replace PAD_VALUE with 0.0 for model
        X_processed = X_truncated.copy()
        X_processed[X_processed == CFG.PAD_VALUE] = 0.0
        
        # Get predictions
        preds, _, _ = early_exit_inference(model, X_processed, device, 0.9, batch_size)
        acc = (preds == y).mean()
        
        results[frac] = float(acc)
        print(f"  {frac:>4.0%} observed -> accuracy {acc:.4f}")
    
    return results

def plot_results(output_dir, cm, sweep_results, early_results, y_true, probs):
    """Generate all evaluation plots"""
    output_dir = Path(output_dir)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['PSPL', 'Binary'], 
                yticklabels=['PSPL', 'Binary'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Confidence Threshold = 0.9)')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion_matrix.png")
    
    # 2. ROC Curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved roc_curve.png")
    
    # 3. Accuracy vs Decision Time
    if sweep_results:
        thresholds = [r['threshold'] for r in sweep_results]
        accuracies = [r['accuracy'] for r in sweep_results]
        avg_times = [r['avg_decision_time'] for r in sweep_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_times, accuracies, 'o-', linewidth=2.5, markersize=8, color='green')
        
        # Mark optimal point
        max_idx = np.argmax(accuracies)
        plt.annotate(f"Optimal\nAcc: {accuracies[max_idx]:.3f}\nTime: {avg_times[max_idx]:.1f}",
                    xy=(avg_times[max_idx], accuracies[max_idx]),
                    xytext=(20, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Average Decision Time (timesteps)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Average Decision Time', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_vs_decision_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved accuracy_vs_decision_time.png")
    
    # 4. Early Detection Performance
    if early_results:
        fractions = sorted(early_results.keys())
        accuracies = [early_results[f] for f in fractions]
        
        plt.figure(figsize=(10, 6))
        plt.plot([f*100 for f in fractions], accuracies, 'o-', linewidth=2.5, markersize=10, color='blue')
        plt.xlabel('Percentage of Light Curve Observed (%)', fontsize=12)
        plt.ylabel('Classification Accuracy', fontsize=12)
        plt.title('Early Detection Performance', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(output_dir / 'early_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved early_detection.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with early-exit strategy')
    parser.add_argument("--model", type=str, default=None, help='Path to model checkpoint (auto-detect if not provided)')
    parser.add_argument("--data", type=str, required=True, help='Path to test data')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory (auto-detect if not provided)')
    parser.add_argument("--experiment_name", type=str, default=None, help='Experiment name (for auto-detect)')
    parser.add_argument("--early_detection", action='store_true', help='Run early detection analysis')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for inference')
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
            args.output_dir = str(results_dir / "evaluation")
    else:
        results_dir = Path(args.model).parent
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EVALUATION WITH SAVED SCALERS (FIXED VERSION)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # =========================================================================
    # CRITICAL FIX: Load RAW data and apply saved scalers
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING DATA AND SCALERS")
    print("="*80)
    
    # Load RAW data (no normalization)
    print("\n1. Loading RAW data (normalize=False)...")
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X.shape[1]
    print(f"✓ Raw data loaded: {X.shape}")
    print(f"   Raw data range: [{X[X != CFG.PAD_VALUE].min():.3f}, {X[X != CFG.PAD_VALUE].max():.3f}]")
    
    # Load saved scalers from training
    print("\n2. Loading scalers from training...")
    scaler_std, scaler_mm = load_scalers(results_dir)
    print(f"✓ Loaded scalers from {results_dir}")
    
    # Apply same transformation used during training
    print("\n3. Applying saved scalers to data...")
    X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    print(f"✓ Applied same normalization as training")
    print(f"   Normalized data range: [{X[X != CFG.PAD_VALUE].min():.3f}, {X[X != CFG.PAD_VALUE].max():.3f}]")
    print(f"   Expected: approximately [0.0, 1.0]")
    # =========================================================================
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded")
    
    # Replace PAD_VALUE with 0.0 for inference
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # 1. Single threshold evaluation (0.9)
    print("\n" + "="*80)
    print("INFERENCE WITH CONFIDENCE THRESHOLD = 0.9")
    print("="*80)
    
    preds, times, confs = early_exit_inference(model, X_processed, device, confidence_threshold=0.9, batch_size=args.batch_size)
    acc = (preds == y).mean()
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Avg decision time: {times.mean():.1f} / {L}")
    print(f"  Median decision time: {np.median(times):.1f}")
    print(f"  Min decision time: {times.min()}")
    print(f"  Max decision time: {times.max()}")
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=['PSPL', 'Binary']))
    
    # Get probabilities for ROC
    probs = confs  # Confidence values
    
    # 2. Threshold sweep
    print("\n" + "="*80)
    print("CONFIDENCE THRESHOLD SWEEP")
    print("="*80)
    
    thresholds = np.arange(0.5, 1.0, 0.05)
    sweep_results = sweep_confidence_threshold(model, X_processed, y, device, thresholds, args.batch_size)
    
    # Print summary
    print("\nThreshold Sweep Results:")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Avg Time':<12}")
    print("-" * 40)
    for res in sweep_results:
        print(f"{res['threshold']:<12.2f} {res['accuracy']:<12.4f} {res['avg_decision_time']:<12.1f}")
    
    # 3. Early detection analysis
    early_results = {}
    if args.early_detection:
        print("\n" + "="*80)
        print("EARLY DETECTION ANALYSIS")
        print("="*80)
        
        early_results = early_detection_analysis(model, X, y, device, batch_size=args.batch_size)
    
    # 4. Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_results(output_dir, cm, sweep_results, early_results, y, probs)
    
    # 5. Save results
    results_summary = {
        'single_threshold_0.9': {
            'accuracy': float(acc),
            'avg_decision_time': float(times.mean()),
            'median_decision_time': float(np.median(times)),
            'confusion_matrix': cm.tolist()
        },
        'threshold_sweep': sweep_results,
        'early_detection': early_results,
        'metadata': meta,
        'data_path': str(args.data),
        'model_path': str(args.model),
        'scalers_used': str(results_dir)
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/evaluation_summary.json")
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if acc > 0.65:
        print(f"✅ Good performance! Accuracy = {acc:.4f}")
    elif acc > 0.55:
        print(f"⚠️  Moderate performance. Accuracy = {acc:.4f}")
        print("   Consider training longer or adjusting hyperparameters")
    else:
        print(f"❌ Low performance. Accuracy = {acc:.4f}")
        print("   Check data generation and model architecture")
    
    print(f"\nKey outputs:")
    print(f"  - Confusion matrix: {output_dir}/confusion_matrix.png")
    print(f"  - Accuracy vs Time: {output_dir}/accuracy_vs_decision_time.png")
    if args.early_detection:
        print(f"  - Early detection: {output_dir}/early_detection.png")

if __name__ == "__main__":
    main()