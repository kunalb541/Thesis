#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Model Evaluation (v6.1 - MASKED LOSS)

FIXED:
- v6.1: CRITICAL FIX: Updated `evaluate_model_final_step` and
        `run_early_detection_analysis` to use masked loss logic
        and compute accuracy on the LAST VALID timestep.
- v6.0: Updated model loading to match simplified CausalCNN.

Author: Kunal Bhatia
Version: 6.1
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
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
import sys
import os

# Suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- FIX: Import the one true model ---
from model import TimeDistributedCNN

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


# --- START: CRITICAL FIX - MASKED EVALUATION ---
def evaluate_model_final_step(model, X_3d, y_true, device, batch_size=128, quiet=False):
    """
    Evaluate model (v6.1)
    Returns predictions and confidences based on the LAST VALID timestep.
    X_3d is shape [N, C, T] (with -1.0 pads)
    """
    model.eval()
    
    predictions = []
    confidences = []
    all_probs = []
    
    n_samples = len(X_3d)
    
    if not quiet:
        print(f"\nEvaluating {n_samples} samples (last valid step)...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_3d[batch_start:batch_end]
            
            X_tensor = torch.from_numpy(X_batch).float().to(device)
            B, C, T = X_tensor.shape

            # Get predictions for the *entire* sequence
            logits_seq, _ = model(X_tensor, return_sequence=True) # [B, T, n_classes]
            
            # --- Find last valid timestep ---
            pad_mask = (X_tensor[:, 0, :] != -1.0) # [B, T]
            last_valid_idx = pad_mask.long().sum(dim=1) - 1 # [B]
            last_valid_idx = torch.clamp(last_valid_idx, min=0, max=T-1)
            
            # Gather predictions from last valid timestep
            batch_indices = torch.arange(B, device=device)
            outputs = logits_seq[batch_indices, last_valid_idx, :] # [B, n_classes]
            
            # --- Standard logic from here ---
            probs = torch.softmax(outputs, dim=1)
            
            probs_np = probs.cpu().numpy()
            preds_np = np.argmax(probs_np, axis=1)
            confs_np = np.max(probs_np, axis=1)
            
            predictions.extend(preds_np)
            confidences.extend(confs_np)
            all_probs.extend(probs_np)
            
            if not quiet and (i // batch_size) % 10 == 0:
                print(f"  Progress: {int((i / n_samples) * 100)}%", end='\r')
    
    if not quiet:
        print(f"  Progress: 100%     ")
    
    return (np.array(predictions), 
            np.array(confidences), 
            np.array(all_probs))
# --- END: CRITICAL FIX ---


# --- START: CRITICAL FIX - MASKED EARLY DETECTION ---
@torch.no_grad()
def run_early_detection_analysis(model, X_3d, y, device, checkpoints, batch_size=128):
    """
    Evaluates model accuracy at different observation checkpoints (v6.1)
    Uses sequence truncation.
    Computes accuracy on LAST VALID timestep *within the truncated sequence*.
    X_3d is shape [N, C, T] (with -1.0 pads)
    """
    model.eval()
    N, C, T = X_3d.shape
    results = {}
    
    print("\n" + "=" * 80)
    print("EARLY DETECTION ANALYSIS (Causal Truncation, Last Valid Step)")
    print("=" * 80)
    
    for chk in checkpoints:
        if chk <= 0 or chk > 1.0:
            continue
            
        # Get timestep index to truncate *to*
        t_idx = int(T * chk)
        if t_idx < 1: t_idx = 1 # Must have at least one timestep
        
        # Create a truncated view of the data [N, C, t_idx]
        X_truncated = X_3d[:, :, :t_idx]
        
        all_preds = []
        
        # Evaluate on the truncated sequences
        for i in range(0, N, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, N)
            
            X_batch_np = X_truncated[batch_start:batch_end]
            X_tensor = torch.from_numpy(X_batch_np).float().to(device)
            B_b, C_b, T_b = X_tensor.shape # T_b is the truncated length
            
            # Get FULL sequence predictions *for the truncated input*
            logits_seq, _ = model(X_tensor, return_sequence=True)  # [B_b, T_b, n_classes]
            
            # --- Find last valid timestep *within this truncated view* ---
            pad_mask = (X_tensor[:, 0, :] != -1.0) # [B_b, T_b]
            last_valid_idx = pad_mask.long().sum(dim=1) - 1 # [B_b]
            last_valid_idx = torch.clamp(last_valid_idx, min=0, max=T_b-1)
            
            # Gather predictions
            batch_indices = torch.arange(B_b, device=device)
            logits = logits_seq[batch_indices, last_valid_idx, :] # [B_b, n_classes]
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
        
        # Calculate accuracy for this checkpoint
        acc = accuracy_score(y, np.array(all_preds))
        
        chk_str = f"{chk*100:.0f}%"
        results[chk_str] = acc
        
        print(f"  Accuracy at {chk_str:>4} ({t_idx}/{T} steps): {acc:.4f}")

    print("=" * 80)
    return results
# --- END: CRITICAL FIX ---


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
    plt.title('Confusion Matrix (Last Valid Step)', fontsize=14, fontweight='bold')
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
        plt.title('ROC Curve (Last Valid Step)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if not quiet:
            print("  ✓ roc_curve.png")
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
        pr_auc = auc(recall, precision) 
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2.5, 
                label=f'PR Curve (AUC = {pr_auc:.3f})', color='darkgreen')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve (Last Valid Step)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if not quiet:
            print("  ✓ pr_curve.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model (v6.1 - Masked Loss)')
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
    # --- New flag for early detection ---
    parser.add_argument("--early_detection", action='store_true',
                       help='Run early detection analysis')
    
    args = parser.parse_args()
    
    # Auto-detect model and output_dir
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError("Must provide --model/--output_dir OR --experiment_name")
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
        print("MODEL EVALUATION (v6.1 - Causal Simplified CNN, Masked Loss)")
        print("=" * 80)
        print(f"\nModel: {args.model}")
        print(f"Data: {args.data}")
        print(f"Output: {output_dir}")
    
    # Load data and scalers
    if not args.quiet:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
    
    # --- START CHANGED: Load 3D data [N, 1, T] directly (Fix #4) ---
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, 
                                              normalize=False)
    
    if not args.quiet:
        print(f"✓ Loaded 3D data: {X.shape}")
    # --- END CHANGED ---

    # Load scalers
    try:
        scaler_std, scaler_mm = load_scalers(results_dir)
        if not args.quiet:
            print(f"✓ Loaded scalers from {results_dir}")
    except Exception as e:
        print(f"⚠ Warning: Could not load scalers: {e}. Using data as-is.")
        scaler_std = scaler_mm = None
    
    # Apply scalers if available
    if scaler_std is not None and scaler_mm is not None:
        # Apply scalers to 3D data
        X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
        if not args.quiet:
            print("✓ Applied normalization")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.quiet:
        print(f"\nDevice: {device}")
    
    # --- Load TimeDistributedCNN (LSTM) model (no change) ---
    config_path = results_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # --- START: MODEL INSTANTIATION FIX (already fixed) ---
    if not args.quiet:
        print("Loading TimeDistributedCNN (Simplified Causal CNN)...")
    model = TimeDistributedCNN(
        in_channels=1, 
        n_classes=2, 
        dropout=config.get('dropout', 0.3) # Get dropout from config
    )
    model_type = "TimeDistributed_CausalCNN_Simplified"
    # --- END: MODEL INSTANTIATION FIX ---

    try:
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
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
        sys.exit(1)
    
    # --- We pass X (with -1.0 pads) directly to evaluation ---
    
    # Evaluate (Final Step)
    if not args.quiet:
        print("\n" + "=" * 80)
        print("INFERENCE (Last Valid Timestep)")
        print("=" * 80)
    
    # --- CRITICAL FIX: Pass 'y' to the evaluation function ---
    preds, confs, probs = evaluate_model_final_step(model, X, y, device, # Pass X and y
                                                    batch_size=args.batch_size,
                                                    quiet=args.quiet)
    
    # Compute metrics for final step
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, average='weighted', zero_division=0)
    recall = recall_score(y, preds, average='weighted', zero_division=0)
    f1 = f1_score(y, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y, preds)
    roc_auc = roc_auc_score(y, probs[:, 1]) if len(np.unique(y)) > 1 else None
    
    # Print results (final step)
    if not args.quiet:
        print("\n" + "=" * 80)
        print("RESULTS (Last Valid Timestep)")
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
    
    # Generate plots (final step)
    plot_results(output_dir, cm, y, probs, quiet=args.quiet)
    
    # --- START: Run Causal Early Detection Analysis (Fix #3) ---
    early_detection_results = None
    if args.early_detection:
        checkpoints = [0.1, 0.25, 0.33, 0.5, 0.67, 0.83, 1.0]
        early_detection_results = run_early_detection_analysis(
            model, X, y, device, checkpoints, args.batch_size # Pass X
        )
    # --- END: Run Early Detection Analysis ---
    
    # Save results
    results_summary = {
        'final_step_metrics': {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'avg_confidence': float(confs.mean()),
        },
        'early_detection_accuracy': early_detection_results,
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
    
    print(f"\nFinal Step Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
