#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Binary Microlensing Classification

Generates all metrics, plots, and analyses needed for thesis.

Author: Kunal Bhatia
Version: 6.2 - Complete evaluation suite
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from streaming_transformer import StreamingTransformer
from normalization import CausticPreservingNormalizer
import config as CFG


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path: str, normalizer_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = StreamingTransformer().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle DDP checkpoint
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load normalizer
        print(f"Loading normalizer from {normalizer_path}...")
        self.normalizer = CausticPreservingNormalizer()
        self.normalizer.load(normalizer_path)
        
        print(f"Evaluator initialized on {self.device}")
    
    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 64) -> dict:
        """Get predictions for dataset"""
        self.model.eval()
        
        n_samples = len(X)
        all_probs = []
        all_logits = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            batch = X[i:i+batch_size]
            X_tensor = torch.from_numpy(batch).float().to(self.device)
            
            outputs = self.model(X_tensor, return_all_timesteps=False)
            logits = outputs['binary']
            probs = F.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
        
        return {
            'probs': np.vstack(all_probs),
            'logits': np.vstack(all_logits),
            'preds': np.vstack(all_probs).argmax(axis=1)
        }
    
    @torch.no_grad()
    def early_detection_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fractions: list = [0.1, 0.25, 0.5, 0.67, 0.83, 1.0],
        batch_size: int = 64
    ) -> dict:
        """Analyze performance with partial observations"""
        self.model.eval()
        
        results = {}
        
        for frac in fractions:
            print(f"\nEvaluating at {frac*100:.0f}% observation completeness...")
            
            # Truncate sequences
            n_points = int(CFG.N_POINTS * frac)
            X_truncated = X.copy()
            X_truncated[:, n_points:] = CFG.PAD_VALUE
            
            # Get predictions
            preds_dict = self.predict(X_truncated, batch_size)
            
            # Compute metrics
            acc = accuracy_score(y, preds_dict['preds'])
            
            results[frac] = {
                'accuracy': acc,
                'n_points': n_points,
                'preds': preds_dict['preds'],
                'probs': preds_dict['probs']
            }
            
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        return results
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> dict:
        """Compute comprehensive metrics"""
        
        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        except:
            roc_auc = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, target_names=['PSPL', 'Binary'], output_dict=True)
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['PSPL', 'Binary'],
                   yticklabels=['PSPL', 'Binary'])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, save_path: Path):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_early_detection(self, results: dict, save_path: Path):
        """Plot early detection performance"""
        fractions = sorted(results.keys())
        accuracies = [results[f]['accuracy'] for f in fractions]
        
        plt.figure(figsize=(10, 6))
        plt.plot([f*100 for f in fractions], [a*100 for a in accuracies],
                'o-', linewidth=2.5, markersize=10, color='#2E86AB')
        plt.xlabel('Observation Completeness (%)', fontsize=12)
        plt.ylabel('Classification Accuracy (%)', fontsize=12)
        plt.title('Early Detection Performance', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.ylim([0, 100])
        
        # Add annotations
        for f, a in zip(fractions, accuracies):
            plt.annotate(f'{a*100:.1f}%', 
                        xy=(f*100, a*100),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_examples(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        timestamps: np.ndarray,
        save_path: Path,
        n_examples: int = 12
    ):
        """Plot example predictions"""
        
        # Select examples: correct PSPL, correct Binary, incorrect each
        pspl_correct = np.where((y_true == 0) & (y_pred == 0))[0]
        binary_correct = np.where((y_true == 1) & (y_pred == 1))[0]
        pspl_wrong = np.where((y_true == 0) & (y_pred == 1))[0]
        binary_wrong = np.where((y_true == 1) & (y_pred == 0))[0]
        
        examples = []
        labels = []
        
        # Add examples from each category
        for indices, label in [
            (pspl_correct, 'PSPL ✓'),
            (binary_correct, 'Binary ✓'),
            (pspl_wrong, 'PSPL ✗'),
            (binary_wrong, 'Binary ✗')
        ]:
            if len(indices) > 0:
                n = min(3, len(indices))
                examples.extend(indices[:n])
                labels.extend([label] * n)
        
        # Plot
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (ex_idx, label) in enumerate(zip(examples, labels)):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            flux = X[ex_idx]
            valid = flux != CFG.PAD_VALUE
            
            if valid.any():
                ax.scatter(timestamps[valid], flux[valid], s=1, alpha=0.7,
                          color='green' if '✓' in label else 'red')
                
                pred_class = 'Binary' if y_pred[ex_idx] == 1 else 'PSPL'
                conf = y_probs[ex_idx].max()
                
                ax.set_title(f'{label} | Pred: {pred_class} ({conf:.2%})', fontsize=10)
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Normalized Flux')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Example Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--data', required=True, help='Path to test dataset')
    parser.add_argument('--early_detection', action='store_true', help='Run early detection analysis')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, help='Custom output directory')
    
    args = parser.parse_args()
    
    # Find experiment directory
    results_dir = Path(CFG.RESULTS_DIR)
    exp_dirs = sorted(results_dir.glob(f"{args.experiment_name}_*"))
    
    if not exp_dirs:
        print(f"ERROR: No experiment found matching '{args.experiment_name}'")
        return
    
    exp_dir = exp_dirs[-1]  # Use most recent
    print(f"Using experiment: {exp_dir.name}")
    
    # Setup paths
    model_path = exp_dir / 'best_model.pt'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    if not model_path.exists() or not normalizer_path.exists():
        print(f"ERROR: Missing model or normalizer in {exp_dir}")
        return
    
    # Create evaluation output directory
    if args.output_dir:
        eval_dir = Path(args.output_dir)
    else:
        eval_dir = exp_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"\nLoading test data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    timestamps = data['timestamps']
    
    # Apply same normalization
    print("Applying normalization...")
    evaluator = ModelEvaluator(str(model_path), str(normalizer_path))
    
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    
    X_norm = evaluator.normalizer.transform(X).squeeze(1)
    
    print(f"Test data: {X_norm.shape}, Labels: {y.shape}")
    print(f"Class distribution: PSPL={np.sum(y==0)}, Binary={np.sum(y==1)}")
    
    # Main evaluation
    print("\n" + "="*60)
    print("MAIN EVALUATION")
    print("="*60)
    
    preds_dict = evaluator.predict(X_norm, args.batch_size)
    metrics = evaluator.compute_metrics(y, preds_dict['preds'], preds_dict['probs'])
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"              Predicted")
    print(f"              PSPL  Binary")
    print(f"Actual PSPL   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Binary {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(cm, eval_dir / 'confusion_matrix.png')
    evaluator.plot_roc_curve(y, preds_dict['probs'], eval_dir / 'roc_curve.png')
    evaluator.plot_examples(X_norm, y, preds_dict['preds'], preds_dict['probs'],
                           timestamps, eval_dir / 'example_predictions.png')
    
    # Early detection analysis
    if args.early_detection:
        print("\n" + "="*60)
        print("EARLY DETECTION ANALYSIS")
        print("="*60)
        
        early_results = evaluator.early_detection_analysis(X_norm, y, batch_size=args.batch_size)
        evaluator.plot_early_detection(early_results, eval_dir / 'early_detection.png')
        
        # Save early detection results
        early_summary = {
            frac: {
                'accuracy': float(res['accuracy']),
                'n_points': int(res['n_points'])
            }
            for frac, res in early_results.items()
        }
        metrics['early_detection'] = early_summary
    
    # Save summary
    summary_path = eval_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        metrics_json = json.loads(json.dumps(metrics, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        json.dump({
            'experiment': args.experiment_name,
            'test_data': str(args.data),
            'n_test_samples': int(len(y)),
            'final_test_acc': float(metrics['accuracy']),
            'metrics': metrics_json
        }, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to {eval_dir}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Confusion Matrix: {eval_dir / 'confusion_matrix.png'}")
    print(f"   - ROC Curve: {eval_dir / 'roc_curve.png'}")
    print(f"   - Examples: {eval_dir / 'example_predictions.png'}")
    if args.early_detection:
        print(f"   - Early Detection: {eval_dir / 'early_detection.png'}")


if __name__ == "__main__":
    main()