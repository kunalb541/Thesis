#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with All Visualizations
======================================================
- Evaluates model performance
- Generates ALL thesis figures in one run
- Uses scatter plots (realistic observations)
- Hides masked/padded data
- Creates 12×12 example grids
- DDP-compatible

Author: Kunal Bhatia
Date: November 2025
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
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from scipy.signal import find_peaks
import sys

# Add code directory to path
sys.path.insert(0, '/pfs/data6/home/hd/hd_hd/hd_vm305/Thesis/code')
from transformer import MicrolensingTransformer

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class StableNormalizer:
    """Robust normalizer matching training"""
    
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


class ComprehensiveEvaluator:
    """Complete evaluation with all visualizations"""
    
    def __init__(self, model_path, data_path, output_dir, device='cuda', batch_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        
        # Load model
        print("\n📦 Loading model...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Load data
        print("\n📊 Loading data...")
        self.X, self.y = self._load_data(data_path)
        
        # Normalize
        print("\n🔄 Normalizing data...")
        self.normalizer = StableNormalizer(pad_value=-1.0)
        self.X_norm = self.normalizer.fit(self.X).transform(self.X)
        
        # Get predictions
        print(f"\n🔮 Getting predictions...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # Compute metrics
        print("\n📈 Computing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load model with auto-detected architecture"""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Detect architecture
        if 'pos_encoding' in state_dict:
            d_model = state_dict['pos_encoding'].shape[2]
        elif 'input_embed.4.weight' in state_dict:
            d_model = state_dict['input_embed.4.weight'].shape[0]
        else:
            d_model = 256
        
        # Count layers
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith('layers.'):
                layer_idx = key.split('.')[1]
                if layer_idx.isdigit():
                    layer_indices.add(int(layer_idx))
        
        num_layers = len(layer_indices) if layer_indices else 6
        
        # Detect nhead
        if d_model <= 64:
            nhead = 4
        elif d_model <= 128:
            nhead = 8
        else:
            nhead = 8
        
        print(f"   Architecture: d_model={d_model}, num_layers={num_layers}, nhead={nhead}")
        
        model = MicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            pad_value=-1.0
        )
        
        model.load_state_dict(state_dict)
        return model
    
    def _load_data(self, data_path):
        """Load test data"""
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        if X.ndim == 3:
            X = X.squeeze(1)
        
        print(f"   Loaded {len(X)} events")
        print(f"   Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print(f"   PSPL:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        return X, y
    
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
        correct = self.predictions == self.y
        
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
        
        # Add text annotations
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
        
        # Plot correct vs incorrect
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
    
    def plot_calibration_curve(self):
        """Plot calibration curve"""
        correct = self.predictions == self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax = axes[0]
        conf_bins = np.linspace(0.5, 1.0, 11)
        accuracies, bin_centers, counts = [], [], []
        
        for i in range(len(conf_bins)-1):
            mask = (self.confidences >= conf_bins[i]) & (self.confidences < conf_bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct[mask].mean())
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                counts.append(mask.sum())
        
        if len(bin_centers) > 0:
            bars = ax.bar(bin_centers, accuracies, width=0.04, alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                bar.set_facecolor(plt.cm.Blues(0.3 + 0.7 * cnt / max(counts)))
            
            for bc, acc, cnt in zip(bin_centers, accuracies, counts):
                ax.text(bc, acc + 0.02, f'n={cnt}', ha='center', fontsize=7)
        
        ax.plot([0.5, 1.0], [0.5, 1.0], 'r--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Model Calibration', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.05])
        
        # Scatter plot
        ax = axes[1]
        n_plot = min(5000, len(correct))
        idx = np.random.choice(len(correct), n_plot, replace=False)
        jitter = np.random.normal(0, 0.01, size=n_plot)
        
        ax.scatter(self.confidences[idx][correct[idx]] + jitter[correct[idx]],
                  correct[idx][correct[idx]],
                  alpha=0.3, s=5, color='green', label='Correct')
        ax.scatter(self.confidences[idx][~correct[idx]] + jitter[~correct[idx]],
                  correct[idx][~correct[idx]],
                  alpha=0.3, s=5, color='red', label='Incorrect')
        
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correctness')
        ax.set_title('Confidence vs Correctness', fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'calibration.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_example_grid(self, n_examples=144):
        """Plot 12×12 grid of examples using scatter plots"""
        print(f"\n  Generating {n_examples} example plots...")
        
        # Get diverse examples
        correct = self.predictions == self.y
        binary_pred = self.predictions == 1
        
        # 4 categories
        tp = np.where((self.y == 1) & binary_pred & correct)[0]
        tn = np.where((self.y == 0) & ~binary_pred & correct)[0]
        fp = np.where((self.y == 0) & binary_pred & ~correct)[0]
        fn = np.where((self.y == 1) & ~binary_pred & ~correct)[0]
        
        # Select examples
        n_per_cat = n_examples // 4
        examples = []
        
        for idx_arr, label, color in [
            (tp, 'TP', 'green'),
            (tn, 'TN', 'blue'),
            (fp, 'FP', 'orange'),
            (fn, 'FN', 'red')
        ]:
            if len(idx_arr) > 0:
                selected = np.random.choice(idx_arr, min(n_per_cat, len(idx_arr)), replace=False)
                for idx in selected:
                    examples.append((idx, label, color))
        
        # Create 12×12 grid
        fig, axes = plt.subplots(12, 12, figsize=(30, 30))
        axes = axes.flatten()
        
        for i, (idx, label, color) in enumerate(examples[:144]):
            ax = axes[i]
            
            # Filter masked values
            light_curve = self.X[idx]
            valid_mask = light_curve != -1.0
            times = np.arange(len(light_curve))[valid_mask]
            fluxes = light_curve[valid_mask]
            
            # Scatter plot
            ax.scatter(times, fluxes, c=color, s=0.5, alpha=0.6)
            ax.set_title(f'{label} ({self.confidences[idx]:.0%})', fontsize=6, color=color)
            ax.axis('off')
        
        # Hide unused
        for i in range(len(examples), 144):
            axes[i].axis('off')
        
        plt.suptitle(f'Example Predictions (12×12 Grid)', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'example_grid_12x12.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_real_time_evolution(self, event_idx=None):
        """Plot real-time classification evolution (two panels)"""
        if event_idx is None:
            # Find a good binary event
            binary_correct = np.where((self.y == 1) & (self.predictions == 1) & (self.confidences > 0.9))[0]
            if len(binary_correct) == 0:
                binary_correct = np.where(self.y == 1)[0]
            event_idx = binary_correct[0]
        
        light_curve = self.X[event_idx]
        light_curve_norm = self.X_norm[event_idx]
        true_label = self.y[event_idx]
        
        # Sample different completeness levels
        fractions = np.linspace(0.1, 1.0, 10)
        binary_probs = []
        confidences = []
        
        with torch.no_grad():
            for frac in fractions:
                n_points = int(1500 * frac)
                partial_curve = np.full(1500, -1.0)
                partial_curve[:n_points] = light_curve_norm[:n_points]
                
                x = torch.tensor([partial_curve], dtype=torch.float32).to(self.device)
                output = self.model(x, return_all=False)
                logits = output['binary']
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                binary_probs.append(probs[1])
                confidences.append(probs.max())
        
        # Create two-panel figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        days = fractions * 1500
        
        # Panel 1: Binary probability
        ax = axes[0]
        ax.plot(days, binary_probs, 'o-', linewidth=3, markersize=8, color='darkblue', label='Binary Probability')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.axhline(y=0.8, color='orange', linestyle=':', linewidth=2, label='High Confidence')
        
        ax.fill_between(days, 0.8, 1.0, alpha=0.2, color='green')
        ax.fill_between(days, 0.5, 0.8, alpha=0.2, color='yellow')
        ax.fill_between(days, 0.0, 0.5, alpha=0.2, color='lightblue')
        
        true_str = 'Binary' if true_label == 1 else 'Single'
        ax.set_ylabel('Binary Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Real-Time Classification Evolution (True: {true_str})', fontsize=13, fontweight='bold')
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # Annotate key point
        high_conf = np.where(np.array(binary_probs) > 0.8)[0]
        if len(high_conf) > 0:
            first_conf = high_conf[0]
            ax.annotate(f'First high-confidence\nbinary prediction\n({fractions[first_conf]*100:.0f}% data)',
                       xy=(days[first_conf], binary_probs[first_conf]),
                       xytext=(days[first_conf] + 200, 0.9),
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        # Panel 2: Overall confidence
        ax = axes[1]
        ax.plot(days, confidences, 's-', linewidth=3, markersize=8, color='purple', label='Overall Confidence')
        ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% Threshold')
        
        ax.set_xlabel('Time Points Observed', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Evolution Over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.05])
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'real_time_evolution_event_{event_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_early_detection(self):
        """Plot early detection performance"""
        print("\n  Computing early detection performance...")
        
        fractions = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
        overall_accs = []
        binary_recalls = []
        
        for frac in tqdm(fractions, desc="    Testing fractions"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(self.X_norm), self.batch_size):
                    batch_end = min(i + self.batch_size, len(self.X_norm))
                    
                    # Create partial curves
                    partial_curves = []
                    for j in range(i, batch_end):
                        n_points = int(1500 * frac)
                        partial_curve = np.full(1500, -1.0)
                        partial_curve[:n_points] = self.X_norm[j][:n_points]
                        partial_curves.append(partial_curve)
                    
                    x_batch = torch.tensor(partial_curves, dtype=torch.float32).to(self.device)
                    output = self.model(x_batch, return_all=False)
                    logits = output['binary']
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    
                    predictions.extend(probs.argmax(axis=1))
            
            predictions = np.array(predictions)
            overall_accs.append(accuracy_score(self.y, predictions))
            binary_recalls.append(recall_score(self.y, predictions))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        completeness = [f*100 for f in fractions]
        
        ax.plot(completeness, [a*100 for a in overall_accs], 'o-', linewidth=3, markersize=10,
               color='blue', label='Overall Accuracy')
        ax.plot(completeness, [r*100 for r in binary_recalls], 's-', linewidth=3, markersize=10,
               color='red', label='Binary Recall')
        
        ax.axhline(y=70, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='70% threshold')
        ax.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='50% observed')
        
        # Annotate 50% point
        idx_50 = fractions.index(0.5)
        ax.annotate(f'{overall_accs[idx_50]*100:.1f}%',
                   xy=(50, overall_accs[idx_50]*100),
                   xytext=(55, overall_accs[idx_50]*100 - 5),
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Observation Completeness (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
        ax.set_title('Early Detection Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'early_detection.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def generate_all_plots(self, include_early_detection=False, n_evolution_examples=3):
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
        
        print("\n4. Calibration Curve...")
        self.plot_calibration_curve()
        
        print("\n5. 12×12 Example Grid (scatter plots)...")
        self.plot_example_grid(n_examples=144)
        
        print(f"\n6. Real-Time Evolution ({n_evolution_examples} examples)...")
        for i in range(n_evolution_examples):
            self.plot_real_time_evolution()
        
        if include_early_detection:
            print("\n7. Early Detection Performance...")
            self.plot_early_detection()
        
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
        }
        
        output_path = self.output_dir / 'results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (will find latest matching dir)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--early_detection', action='store_true',
                       help='Include early detection analysis (slower)')
    parser.add_argument('--n_evolution_examples', type=int, default=3,
                       help='Number of real-time evolution examples to generate')
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
            print(f"  - {d.name}")
        return
    
    exp_dir = exp_dirs[-1]  # Most recent
    model_path = exp_dir / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Using experiment: {exp_dir.name}")
    print(f"Model path: {model_path}")
    
    # Create output directory
    output_dir = exp_dir / 'evaluation'
    
    # Run evaluation
    device = 'cpu' if args.no_cuda else 'cuda'
    evaluator = ComprehensiveEvaluator(
        model_path=str(model_path),
        data_path=args.data,
        output_dir=str(output_dir),
        device=device,
        batch_size=args.batch_size
    )
    
    # Generate all plots
    evaluator.generate_all_plots(
        include_early_detection=args.early_detection,
        n_evolution_examples=args.n_evolution_examples
    )
    
    # Save results
    evaluator.save_results()
    
    print("\n🎉 Evaluation complete!")
    print(f"📁 All results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()
