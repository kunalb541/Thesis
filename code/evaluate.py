#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Model Evaluation (v5.0 - TimeDistributed Compatible)

FIXED:
- Removed hard-coded TDConvClassifier
- Imports models from model.py
- Reads config.json to load the correct model (Simple or LSTM)
- Calls model(x, return_sequence=False) for final prediction

Author: Kunal Bhatia
Version: 5.0
Date: November 2025
"""

import warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve, 
                             accuracy_score, precision_score, recall_score, f1_score)
import sys
import os

# Suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import model architectures (matches train.py)
from model import TimeDistributedCNNSimple, TimeDistributedCNN

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_npz_dataset, load_scalers, apply_scalers_to_data
import config as CFG


def find_latest_results_dir(experiment_name, base_dir='../results'):
    """Find the most recent results directory for an experiment"""
    base_path = Path(base_dir)
    pattern = f"{experiment_name}_*"
    
    matching_dirs = sorted(base_path.glob(pattern), 
                          key=lambda x: x.stat().st_mtime, 
                          reverse=True)
    
    if not matching_dirs:
        raise FileNotFoundError(
            f"No results directories found matching '{pattern}' in {base_dir}"
        )
    
    return matching_dirs[0]


def evaluate_model(model, X, device, batch_size=128, quiet=False):
    """
    Evaluate model and return predictions and confidences
    """
    model.eval()
    
    predictions = []
    confidences = []
    all_probs = []
    
    n_samples = len(X)
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    if not quiet:
        print(f"\nEvaluating {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X[i:batch_end]
            
            # Input shape [B, T] -> [B, 1, T]
            X_tensor = torch.from_numpy(X_batch).float().unsqueeze(1).to(device)
            
            # --- FIX: Call model with return_sequence=False ---
            outputs = model(X_tensor, return_sequence=False)
            probs = torch.softmax(outputs, dim=1)
            
            probs_np = probs.cpu().numpy()
            
            for j in range(len(probs_np)):
                pred_class = np.argmax(probs_np[j])
                max_conf = np.max(probs_np[j])
                
                predictions.append(pred_class)
                confidences.append(max_conf)
                all_probs.append(probs_np[j])
            
            if not quiet and (i // batch_size) % 10 == 0:
                progress = min(100, int((i / n_samples) * 100))
                print(f"  Progress: {progress}%", end='\r')
    
    if not quiet:
        print(f"  Progress: 100%")
    
    return (np.array(predictions), 
            np.array(confidences), 
            np.array(all_probs))


def plot_results(output_dir, cm, y_true, probs, quiet=False):
    """Generate evaluation plots"""
    output_dir = Path(output_dir)
    
    if not quiet:
        print("\nGenerating plots...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['PSPL', 'Binary'], 
                yticklabels=['PSPL', 'Binary'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if not quiet:
        print("  ✓ confusion_matrix.png")
    
    # 2. ROC Curve
    if len(np.unique(y_true)) > 1:
        # Use binary class probability
        y_prob_binary = probs[:, 1]  # Probability of class 1 (Binary)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2.5, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})', color='darkblue')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if not quiet:
            print("  ✓ roc_curve.png")
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
        # Note: Using trapz(recall, precision) is more standard for PR-AUC
        pr_auc = auc(recall, precision) 
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2.5, 
                label=f'PR Curve (AUC = {pr_auc:.3f})', color='darkgreen')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if not quiet:
            print("  ✓ pr_curve.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model (v5.0 - TimeDistributed Compatible)')
    parser.add_argument("--model", type=str, default=None, 
                       help='Path to model checkpoint')
    parser.add_argument("--data", type=str, required=True, 
                       help='Path to test data')
    parser.add_argument("--output_dir", type=str, default=None, 
                       help='Output directory')
    parser.add_argument("--experiment_name", type=str, default=None, 
                       help='Experiment name (for auto-detect)')
    parser.add_argument("--batch_size", type=int, default=128, 
                       help='Batch size for inference')
    parser.add_argument("--quiet", action='store_true', 
                       help='Suppress progress messages')
    args = parser.parse_args()
    
    # Auto-detect model and output_dir if not provided
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError(
                "Must provide either --model and --output_dir, "
                "OR --experiment_name for auto-detection"
            )
        
        results_dir = find_latest_results_dir(args.experiment_name)
        
        if not args.quiet:
            print(f"✓ Auto-detected: {results_dir}")
        
        if args.model is None:
            args.model = str(results_dir / "best_model.pt")
        if args.output_dir is None:
            args.output_dir = str(results_dir / "evaluation")
    else:
        results_dir = Path(args.model).parent
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.quiet:
        print("=" * 80)
        print("MODEL EVALUATION (v5.0 - TimeDistributed Compatible)")
        print("=" * 80)
        print(f"\nModel: {args.model}")
        print(f"Data: {args.data}")
        print(f"Output: {output_dir}")
    
    # Load data and scalers
    if not args.quiet:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
    
    # Load data, X shape is (N, T)
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, 
                                              normalize=False)
    
    if not args.quiet:
        print(f"\n✓ Loaded {len(X)} samples")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {np.unique(y, return_counts=True)}")
    
    # Load scalers
    try:
        scaler_std, scaler_mm = load_scalers(results_dir)
        if not args.quiet:
            print(f"✓ Loaded scalers from {results_dir}")
    except Exception as e:
        print(f"⚠ Warning: Could not load scalers: {e}")
        print("  Using data as-is (may affect accuracy)")
        scaler_std = scaler_mm = None
    
    # Apply scalers if available
    if scaler_std is not None and scaler_mm is not None:
        # apply_scalers_to_data expects (N, F) where F=T
        X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
        if not args.quiet:
            print("✓ Applied normalization")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not args.quiet:
        print(f"\nDevice: {device}")
    
    # --- FIX: Load model based on config.json ---
    config_path = results_dir / "config.json"
    model_type = 'TimeDistributed_Simple'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.get('model_type', 'TimeDistributed_Simple')
    
    if model_type == 'TimeDistributed_LSTM':
        if not args.quiet:
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
        if not args.quiet:
            print("Loading TimeDistributedCNNSimple...")
        model = TimeDistributedCNNSimple(
            in_channels=1, 
            n_classes=2, 
            dropout=0.3
        )
    # --- End Fix ---

    try:
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        # Handle checkpoints saved from DDP
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v 
                         for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        if not args.quiet:
            print(f"✓ Model loaded ({model_type})")
    
    except Exception as e:
        print(f"❌ Error loading model state_dict: {e}")
        print("   This often happens if the model architecture in model.py")
        print("   does not match the saved checkpoint.")
        sys.exit(1)
    
    # Replace PAD_VALUE with 0.0 for CNN
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # Evaluate
    if not args.quiet:
        print("\n" + "=" * 80)
        print("INFERENCE")
        print("=" * 80)
    
    preds, confs, probs = evaluate_model(model, X_processed, device, 
                                        batch_size=args.batch_size,
                                        quiet=args.quiet)
    
    # Compute metrics
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, average='weighted', zero_division=0)
    recall = recall_score(y, preds, average='weighted', zero_division=0)
    f1 = f1_score(y, preds, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    
    # ROC AUC
    if len(np.unique(y)) > 1:
        from sklearn.metrics import roc_auc_score
        try:
            roc_auc = roc_auc_score(y, probs[:, 1])
        except:
            roc_auc = None
    else:
        roc_auc = None
    
    # Print results
    if not args.quiet:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
    
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {acc:.4f}")
    print(f"{'Precision':<20} {precision:.4f}")
    print(f"{'Recall':<20} {recall:.4f}")
    print(f"{'F1-Score':<20} {f1:.4f}")
    if roc_auc is not None:
        print(f"{'ROC AUC':<20} {roc_auc:.4f}")
    print(f"{'Avg Confidence':<20} {confs.mean():.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    if not args.quiet:
        print("\nClassification Report:")
        print(classification_report(y, preds, 
                                   target_names=['PSPL', 'Binary'],
                                   digits=4))
    
    # Generate plots
    plot_results(output_dir, cm, y, probs, quiet=args.quiet)
    
    # Save results
    results_summary = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'avg_confidence': float(confs.mean()),
        'min_confidence': float(confs.min()),
        'max_confidence': float(confs.max()),
        'confusion_matrix': cm.tolist(),
        'metadata': meta,
        'data_path': str(args.data),
        'model_path': str(args.model),
        'model_architecture': model_type,
        'scalers_used': str(results_dir),
        'n_samples': int(len(X)),
        'class_distribution': {
            'PSPL': int((y == 0).sum()),
            'Binary': int((y == 1).sum())
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    if not args.quiet:
        print(f"\n✓ Results saved to {output_dir}/evaluation_summary.json")
    
    # Final summary
    if not args.quiet:
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
    
    if acc > 0.70:
        status = "✅ Excellent"
    elif acc > 0.65:
        status = "✓ Good"
    elif acc > 0.55:
        status = "⚠ Moderate"
    else:
        status = "❌ Low"
    
    print(f"\n{status} performance: Accuracy = {acc:.4f}")
    
    if not args.quiet:
        print(f"\nKey outputs:")
        print(f"  - Confusion matrix: {output_dir}/confusion_matrix.png")
        print(f"  - ROC curve: {output_dir}/roc_curve.png")
        print(f"  - PR curve: {output_dir}/pr_curve.png")
        print(f"  - Summary: {output_dir}/evaluation_summary.json")


if __name__ == "__main__":
    main()