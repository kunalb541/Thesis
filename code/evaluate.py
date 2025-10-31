#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Evaluation with saved scalers (FIXED VERSION)

FIXED: Now uses TDConvClassifier from train.py (matching your trained models!)

Author: Kunal Bhatia
Version: 4.0 - Fixed to match train.py architecture
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

# Import the CORRECT model architecture from train.py
# This matches what you actually trained!
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Define TDConvClassifier exactly as in train.py
class TDConvClassifier(nn.Module):
    """
    Compact 1D CNN for binary classification.
    THIS IS THE MODEL FROM TRAIN.PY - matches your trained models!
    """
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
            nn.Linear(32 * 2, 64),  # concat(mean, max)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B, C=1, T)
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

def evaluate_model(model, X, device, batch_size=128):
    """
    Simple evaluation - get predictions for all samples
    """
    model.eval()
    
    predictions = []
    confidences = []
    
    # Process in batches
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X[i:batch_end]
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_batch).float().unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)  # [B, 2]
            probs = torch.softmax(outputs, dim=1)  # [B, 2]
        
        probs_np = probs.cpu().numpy()
        
        for j in range(len(probs_np)):
            pred_class = np.argmax(probs_np[j])
            max_conf = np.max(probs_np[j])
            
            predictions.append(pred_class)
            confidences.append(max_conf)
    
    return np.array(predictions), np.array(confidences)

def plot_results(output_dir, cm, y_true, probs):
    """Generate evaluation plots"""
    output_dir = Path(output_dir)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['PSPL', 'Binary'], 
                yticklabels=['PSPL', 'Binary'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate model (FIXED for train.py compatibility)')
    parser.add_argument("--model", type=str, default=None, help='Path to model checkpoint (auto-detect if not provided)')
    parser.add_argument("--data", type=str, required=True, help='Path to test data')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory (auto-detect if not provided)')
    parser.add_argument("--experiment_name", type=str, default=None, help='Experiment name (for auto-detect)')
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
    print("EVALUATION (FIXED - USES TDConvClassifier FROM TRAIN.PY)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # Load RAW data and apply saved scalers
    print("\n" + "="*80)
    print("LOADING DATA AND SCALERS")
    print("="*80)
    
    print("\n1. Loading RAW data (normalize=False)...")
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X.shape[1]
    print(f"✓ Raw data loaded: {X.shape}")
    
    print("\n2. Loading scalers from training...")
    scaler_std, scaler_mm = load_scalers(results_dir)
    print(f"✓ Loaded scalers from {results_dir}")
    
    print("\n3. Applying saved scalers to data...")
    X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    print(f"✓ Applied same normalization as training")
    
    # Load model (USING TDConvClassifier!)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Use TDConvClassifier from train.py!
    model = TDConvClassifier(in_ch=1, n_classes=2, dropout=0.3)
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded (TDConvClassifier from train.py)")
    
    # Replace PAD_VALUE with 0.0 for inference
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # Evaluate
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    
    preds, confs = evaluate_model(model, X_processed, device, batch_size=args.batch_size)
    acc = (preds == y).mean()
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Avg confidence: {confs.mean():.3f}")
    print(f"  Min confidence: {confs.min():.3f}")
    print(f"  Max confidence: {confs.max():.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=['PSPL', 'Binary']))
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_results(output_dir, cm, y, confs)
    
    # Save results
    results_summary = {
        'accuracy': float(acc),
        'avg_confidence': float(confs.mean()),
        'confusion_matrix': cm.tolist(),
        'metadata': meta,
        'data_path': str(args.data),
        'model_path': str(args.model),
        'model_architecture': 'TDConvClassifier',  # Document which architecture was used
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
    else:
        print(f"❌ Low performance. Accuracy = {acc:.4f}")
    
    print(f"\nKey outputs:")
    print(f"  - Confusion matrix: {output_dir}/confusion_matrix.png")
    print(f"  - ROC curve: {output_dir}/roc_curve.png")
    print(f"  - Summary: {output_dir}/evaluation_summary.json")

if __name__ == "__main__":
    main()