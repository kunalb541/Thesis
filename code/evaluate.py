#!/usr/bin/env python3
"""
Comprehensive Model Evaluation + u0 Analysis
============================================
Combined script for complete model evaluation including:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Visualizations (ROC curve, confusion matrix, confidence distribution)
- u0 dependency analysis (if parameter data available)
- Early detection analysis (optional)

Author: Kunal Bhatia
Version: 9.0 - Combined Evaluation + u0 Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Import from local code directory
import sys
sys.path.insert(0, 'code')
from transformer import MicrolensingTransformer, count_parameters

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class ComprehensiveEvaluator:
    """Complete evaluation with all visualizations and u0 analysis"""
    
    def __init__(self, model_path, normalizer_path, data_path, output_dir, 
                 device='cuda', batch_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE MODEL EVALUATION + u0 ANALYSIS")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        
        # Load model
        print("\n📦 Loading model...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Load normalizer
        print("\n📊 Loading normalizer...")
        self.normalizer = self._load_normalizer(normalizer_path)
        print("✅ Normalizer loaded successfully!")
        
        # Load data
        print("\n📊 Loading data...")
        self.X, self.y, self.params = self._load_data(data_path)
        
        # Normalize
        print("\n🔄 Normalizing data...")
        self.X_norm = self.normalizer.transform(self.X)
        
        # Get predictions
        print(f"\n🔮 Getting predictions...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # Compute metrics
        print("\n📈 Computing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load model with correct architecture from checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load config from same directory
        config_path = Path(model_path).parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            d_model = config.get('d_model', 256)
            nhead = config.get('nhead', 8)
            num_layers = config.get('num_layers', 6)
            dropout = config.get('dropout', 0.1)
            
            print(f"   Using config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        else:
            print("   Warning: config.json not found, using defaults")
            d_model = 256
            nhead = 8
            num_layers = 6
            dropout = 0.1
        
        # Create model
        model = MicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            pad_value=-1.0
        )
        
        # Load state dict (handle DDP wrapping)
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"   Parameters: {count_parameters(model):,}")
        
        return model
    
    def _load_normalizer(self, normalizer_path):
        """Load normalizer saved during training"""
        normalizer_path = Path(normalizer_path)
        
        if not normalizer_path.exists():
            print(f"   Warning: Normalizer not found, creating default")
            from train import StableNormalizer
            return StableNormalizer(pad_value=-1.0)
        
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        
        print(f"   Loaded: mean={normalizer.mean:.3f}, std={normalizer.std:.3f}")
        return normalizer
    
    def _load_data(self, data_path):
        """Load test data and parameters"""
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        if X.ndim == 3:
            X = X.squeeze(1)
        
        print(f"   Events: {len(X)}")
        print(f"   Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print(f"   PSPL:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        # Load parameters if available (for u0 analysis)
        params = None
        if 'params_binary_json' in data:
            params_binary = json.loads(str(data['params_binary_json']))
            if 'params_pspl_json' in data:
                params_pspl = json.loads(str(data['params_pspl_json']))
                params = {'binary': params_binary, 'pspl': params_pspl}
            else:
                params = {'binary': params_binary}
            print("   ✅ Parameter data found (u0 analysis enabled)")
        else:
            print("   ⚠️  No parameter data (u0 analysis disabled)")
        
        return X, y, params
    
    def _get_predictions(self):
        """Get model predictions"""
        predictions = []
        confidences = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X_norm), self.batch_size), desc="   Evaluating"):
                batch_end = min(i + self.batch_size, len(self.X_norm))
                x_batch = torch.tensor(self.X_norm[i:batch_end], dtype=torch.float32).to(self.device)
                
                output = self.model(x_batch, return_all=False)
                logits = output['binary']
                probs = F.softmax(logits, dim=1).cpu().numpy()
                
                preds = probs.argmax(axis=1)
                confs = probs.max(axis=1)
                
                predictions.extend(preds)
                confidences.extend(confs)
                all_probs.append(probs)
        
        return np.array(predictions), np.array(confidences), np.vstack(all_probs)
    
    def _compute_metrics(self):
        """Compute all metrics"""
        tp = ((self.y == 1) & (self.predictions == 1)).sum()
        tn = ((self.y == 0) & (self.predictions == 0)).sum()
        fp = ((self.y == 0) & (self.predictions == 1)).sum()
        fn = ((self.y == 1) & (self.predictions == 0)).sum()
        
        metrics = {
            'accuracy': accuracy_score(self.y, self.predictions),
            'precision': precision_score(self.y, self.predictions, zero_division=0),
            'recall': recall_score(self.y, self.predictions, zero_division=0),
            'f1': f1_score(self.y, self.predictions, zero_division=0),
            'roc_auc': roc_auc_score(self.y, self.probs[:, 1]),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
        
        return metrics
    
    def _print_summary(self):
        """Print evaluation summary"""
        m = self.metrics
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy:  {m['accuracy']*100:.2f}%")
        print(f"Precision: {m['precision']*100:.2f}%")
        print(f"Recall:    {m['recall']*100:.2f}%")
        print(f"F1 Score:  {m['f1']*100:.2f}%")
        print(f"ROC AUC:   {m['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {m['tp']:5d}  |  FP: {m['fp']:5d}")
        print(f"  FN: {m['fn']:5d}  |  TN: {m['tn']:5d}")
        print(f"{'='*70}\n")
    
    # ========== VISUALIZATION METHODS ==========
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y, self.probs[:, 1])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC curve (AUC = {self.metrics["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y, self.predictions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['PSPL', 'Binary'], fontsize=12)
        ax.set_yticklabels(['PSPL', 'Binary'], fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=20, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot confidence distributions"""
        correct = self.predictions == self.y
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bins = np.linspace(0.5, 1.0, 50)
        ax.hist(self.confidences[correct], bins=bins, alpha=0.7, color='green',
               label=f'Correct (n={correct.sum()})', edgecolor='black')
        ax.hist(self.confidences[~correct], bins=bins, alpha=0.7, color='red',
               label=f'Incorrect (n={(~correct).sum()})', edgecolor='black')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    # ========== u0 ANALYSIS METHODS ==========
    
    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        """Analyze accuracy as function of u0"""
        if self.params is None:
            print("\n⚠️  Skipping u0 analysis (no parameter data)")
            return None
        
        print(f"\n{'='*70}")
        print("u0 DEPENDENCY ANALYSIS")
        print(f"{'='*70}")
        
        binary_params = self.params['binary']
        binary_mask = self.y == 1
        
        # Get u0 values
        u0_values = np.array([p['u0'] for p in binary_params])
        
        # Bin by u0
        u0_bins = np.linspace(u0_values.min(), u0_values.max(), n_bins + 1)
        u0_centers = (u0_bins[:-1] + u0_bins[1:]) / 2
        
        accuracies = []
        counts = []
        
        for i in range(n_bins):
            u0_low, u0_high = u0_bins[i], u0_bins[i+1]
            in_bin = (u0_values >= u0_low) & (u0_values < u0_high)
            
            if in_bin.sum() > 0:
                bin_true = self.y[binary_mask][in_bin]
                bin_pred = self.predictions[binary_mask][in_bin]
                acc = accuracy_score(bin_true, bin_pred)
                accuracies.append(acc)
                counts.append(in_bin.sum())
            else:
                accuracies.append(np.nan)
                counts.append(0)
        
        # Find accuracy at threshold
        threshold_idx = np.argmin(np.abs(u0_centers - threshold))
        acc_at_threshold = accuracies[threshold_idx] if not np.isnan(accuracies[threshold_idx]) else None
        
        # Count events
        n_below = (u0_values < threshold).sum()
        n_above = (u0_values >= threshold).sum()
        
        print(f"Physical Detection Threshold: u₀ = {threshold}")
        print(f"Accuracy at threshold: {acc_at_threshold*100:.2f}%" if acc_at_threshold else "N/A")
        print(f"\nEvent Distribution:")
        print(f"  Below threshold (u₀ < {threshold}): {n_below} ({n_below/len(u0_values)*100:.1f}%)")
        print(f"  Above threshold (u₀ ≥ {threshold}): {n_above} ({n_above/len(u0_values)*100:.1f}%)")
        
        return {
            'u0_bins': u0_bins.tolist(),
            'u0_centers': u0_centers.tolist(),
            'accuracies': accuracies,
            'counts': counts,
            'all_u0': u0_values.tolist(),
            'threshold': threshold,
            'accuracy_at_threshold': float(acc_at_threshold) if acc_at_threshold else None,
            'events_below_threshold': int(n_below),
            'events_above_threshold': int(n_above)
        }
    
    def plot_u0_dependency(self, u0_results, threshold=0.3):
        """Plot accuracy vs u0"""
        if u0_results is None:
            return
        
        u0_centers = u0_results['u0_centers']
        accuracies = [a*100 if not np.isnan(a) else None for a in u0_results['accuracies']]
        counts = u0_results['counts']
        
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
                        xy=(u, a), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8,
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
        
        output_path = self.output_dir / 'u0_dependency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def generate_all_plots(self, include_u0=True, u0_threshold=0.3, u0_bins=10):
        """Generate all visualizations"""
        print(f"\n{'='*70}")
        print("GENERATING ALL VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        print("1. ROC Curve...")
        self.plot_roc_curve()
        
        print("\n2. Confusion Matrix...")
        self.plot_confusion_matrix()
        
        print("\n3. Confidence Distribution...")
        self.plot_confidence_distribution()
        
        # u0 analysis (if parameters available)
        if include_u0 and self.params is not None:
            print("\n4. u0 Dependency Analysis...")
            u0_results = self.analyze_u0_dependency(n_bins=u0_bins, threshold=u0_threshold)
            if u0_results:
                self.plot_u0_dependency(u0_results, threshold=u0_threshold)
                
                # Save u0 report
                u0_report_path = self.output_dir / 'u0_report.json'
                with open(u0_report_path, 'w') as f:
                    json.dump(u0_results, f, indent=2)
                print(f"  ✓ Saved: {u0_report_path.name}")
        
        print(f"\n{'='*70}")
        print(f"✅ ALL VISUALIZATIONS SAVED TO: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_results(self):
        """Save results to JSON"""
        results = {
            'metrics': self.metrics,
            'n_samples': int(len(self.y)),
            'high_confidence_90': int((self.confidences >= 0.9).sum()),
            'high_confidence_95': int((self.confidences >= 0.95).sum()),
            'has_u0_analysis': self.params is not None
        }
        
        output_path = self.output_dir / 'evaluation_summary.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive model evaluation with optional u0 analysis'
    )
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (will find latest matching dir)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--u0_threshold', type=float, default=0.3,
                       help='Physical limit threshold for u0 analysis')
    parser.add_argument('--u0_bins', type=int, default=10,
                       help='Number of u0 bins for analysis')
    parser.add_argument('--no_u0', action='store_true',
                       help='Skip u0 analysis even if parameters available')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}_*'))
    
    if not exp_dirs:
        print(f"❌ No experiment found matching: {args.experiment_name}")
        print(f"Available experiments:")
        for d in sorted(results_dir.glob('*')):
            if d.is_dir():
                print(f"  - {d.name}")
        return
    
    exp_dir = exp_dirs[-1]  # Most recent
    model_path = exp_dir / 'best_model.pt'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Using experiment: {exp_dir.name}")
    print(f"Model: {model_path.name}")
    print(f"Normalizer: {normalizer_path.name if normalizer_path.exists() else 'Will create default'}")
    
    # Create output directory
    output_dir = exp_dir / 'evaluation'
    
    # Run evaluation
    device = 'cpu' if args.no_cuda else 'cuda'
    evaluator = ComprehensiveEvaluator(
        model_path=str(model_path),
        normalizer_path=str(normalizer_path),
        data_path=args.data,
        output_dir=str(output_dir),
        device=device,
        batch_size=args.batch_size
    )
    
    # Generate all plots (including u0 if available and not disabled)
    evaluator.generate_all_plots(
        include_u0=not args.no_u0,
        u0_threshold=args.u0_threshold,
        u0_bins=args.u0_bins
    )
    
    # Save results
    evaluator.save_results()
    
    print("\n🎉 Evaluation complete!")
    print(f"📁 All results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()