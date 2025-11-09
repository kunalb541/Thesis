#!/usr/bin/env python3
"""
FIXED Evaluation Script for MicrolensingTransformer
Matches training architecture and includes early detection analysis

Author: Kunal Bhatia
Version: FIXED for thesis experiments
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix
)
import sys

# Import the CORRECT transformer class
sys.path.insert(0, '/pfs/data6/home/hd/hd_hd/hd_vm305/Thesis/code')
from transformer import MicrolensingTransformer


class StableNormalizer:
    """Same normalizer as training - CRITICAL for consistency"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, X):
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            self.mean = np.median(valid_values)
            self.std = np.median(np.abs(valid_values - self.mean))
            
            if self.std < 1e-8:
                self.std = 1.0
            
            self.mean = np.clip(self.mean, -100, 100)
            self.std = np.clip(self.std, 0.01, 100)
        
        return self
    
    def transform(self, X):
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            X_norm[valid_mask] = (X[valid_mask] - self.mean) / self.std
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        return np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)


def load_model(model_path: str, device='cuda'):
    """Load trained model with auto-detected architecture"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Auto-detect architecture
    d_model = state_dict['input_embed.0.weight'].shape[0] // 2 * 2  # Approximate
    
    # Try to find exact d_model from embeddings
    if 'pos_encoding' in state_dict:
        d_model = state_dict['pos_encoding'].shape[2]
    
    num_layers = len([k for k in state_dict.keys() 
                     if 'layers.' in k and '.norm.weight' in k])
    
    print(f"Detected architecture:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  nhead: 8")  # From config
    
    # Create model
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=d_model,
        nhead=8,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=0.1,
        pad_value=-1.0
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device


@torch.no_grad()
def evaluate_model(model, X, y, device, batch_size=64):
    """Get predictions on full dataset"""
    
    all_preds = []
    all_probs = []
    
    for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
        batch = X[i:i+batch_size]
        X_tensor = torch.from_numpy(batch).float().to(device)
        
        outputs = model(X_tensor, return_all=False)
        logits = outputs['binary']
        probs = F.softmax(logits, dim=1)
        
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    
    return y_pred, y_probs


@torch.no_grad()
def evaluate_early_detection(model, X, y, device, 
                             fractions=[0.1, 0.25, 0.5, 0.67, 0.83, 1.0],
                             batch_size=64):
    """
    Evaluate model at different observation completeness levels
    Key for real-time detection analysis
    """
    
    results = {}
    
    for frac in tqdm(fractions, desc="Early detection analysis"):
        n_points = int(1500 * frac)
        
        # Create partially observed light curves
        X_partial = np.full_like(X, -1.0)  # Fill with padding
        X_partial[:, :n_points] = X[:, :n_points]  # Copy observed portion
        
        # Get predictions
        y_pred, y_probs = evaluate_model(model, X_partial, y, device, batch_size)
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        
        # Binary-specific metrics
        binary_mask = y == 1
        if binary_mask.sum() > 0:
            binary_acc = accuracy_score(y[binary_mask], y_pred[binary_mask])
        else:
            binary_acc = 0.0
        
        results[frac] = {
            'accuracy': acc,
            'binary_recall': binary_acc,
            'n_points': n_points
        }
        
        print(f"  {frac*100:.0f}% observed ({n_points} points): "
              f"Acc={acc*100:.2f}%, Binary Recall={binary_acc*100:.2f}%")
    
    return results


def plot_evaluation_results(y_true, y_pred, y_probs, output_dir):
    """Generate comprehensive evaluation plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['PSPL', 'Binary'],
           yticklabels=['PSPL', 'Binary'],
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=14, fontweight='bold')
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrix")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved ROC curve (AUC={roc_auc:.3f})")
    
    # 3. Confidence Distribution
    correct = y_pred == y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_probs[correct].max(axis=1), bins=50, alpha=0.7, 
           label='Correct', color='green', edgecolor='black')
    ax.hist(y_probs[~correct].max(axis=1), bins=50, alpha=0.7,
           label='Incorrect', color='red', edgecolor='black')
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confidence distribution")


def plot_early_detection_results(early_results, output_dir):
    """Plot early detection performance"""
    
    fractions = sorted(early_results.keys())
    accuracies = [early_results[f]['accuracy'] * 100 for f in fractions]
    binary_recalls = [early_results[f]['binary_recall'] * 100 for f in fractions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot([f*100 for f in fractions], accuracies, 
           'o-', linewidth=2.5, markersize=10, label='Overall Accuracy', color='blue')
    ax.plot([f*100 for f in fractions], binary_recalls,
           's-', linewidth=2.5, markersize=10, label='Binary Recall', color='red')
    
    # Add reference lines
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='70% threshold')
    ax.axvline(x=50, color='green', linestyle=':', alpha=0.5, label='50% observed')
    
    ax.set_xlabel('Observation Completeness (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Early Detection Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add annotations
    for frac, acc, recall in zip(fractions, accuracies, binary_recalls):
        if frac in [0.25, 0.5, 1.0]:
            ax.annotate(f'{acc:.1f}%', 
                       xy=(frac*100, acc),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'early_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved early detection plot")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MicrolensingTransformer')
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--early_detection', action='store_true', 
                       help='Run early detection analysis')
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f"{args.experiment_name}_*"))
    
    if not exp_dirs:
        print(f"ERROR: No experiment found matching '{args.experiment_name}'")
        return
    
    exp_dir = exp_dirs[-1]
    print(f"Using experiment: {exp_dir.name}")
    
    model_path = exp_dir / 'best_model.pt'
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    # Create output directory
    output_dir = exp_dir / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: Binary={np.sum(y==1)}, PSPL={np.sum(y==0)}")
    
    # Normalize data (same as training)
    print("\nNormalizing data...")
    normalizer = StableNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit(X).transform(X)
    
    # Load model
    print("\nLoading model...")
    model, device = load_model(str(model_path))
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    y_pred, y_probs = evaluate_model(model, X_norm, y, device, args.batch_size)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    roc_auc = roc_auc_score(y, y_probs[:, 1])
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Save metrics
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_evaluation_results(y, y_pred, y_probs, output_dir)
    
    # Early detection analysis
    if args.early_detection:
        print("\n" + "="*60)
        print("EARLY DETECTION ANALYSIS")
        print("="*60)
        
        early_results = evaluate_early_detection(
            model, X_norm, y, device,
            fractions=[0.1, 0.25, 0.5, 0.67, 0.83, 1.0],
            batch_size=args.batch_size
        )
        
        results['early_detection'] = early_results
        
        print("\nGenerating early detection plot...")
        plot_early_detection_results(early_results, output_dir)
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
