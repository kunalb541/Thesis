#!/usr/bin/env python3
"""
PRODUCTION Comprehensive Model Evaluation + u0 Analysis
========================================================

ACTUAL FIXES IN THIS VERSION:
1. Vectorized tensor creation in plot_early_detection (was still using loop)
2. Memory-efficient batch processing
3. Removed redundant array indexing

Author: Kunal Bhatia  
Version: 10.0 - Production Ready (FIXED)
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

import sys
from transformer import MicrolensingTransformer, count_parameters

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class StableNormalizer:
    """Robust normalizer matching training (for pickle compatibility)"""
    
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
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ComprehensiveEvaluator:
    """Complete evaluation with ALL visualizations and u0 analysis"""
    
    def __init__(self, model_path, normalizer_path, data_path, output_dir, 
                 device='cuda', batch_size=128, n_samples=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"PRODUCTION EVALUATION + u0 ANALYSIS (v10.1)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        if n_samples:
            print(f"Sample limit: {n_samples} events (for faster evaluation)")
        
        print("\n📦 Loading model...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        print("\n📊 Loading normalizer...")
        self.normalizer = self._load_normalizer(normalizer_path)
        print("✅ Normalizer loaded successfully!")
        
        print("\n📊 Loading data...")
        self.X, self.y, self.params, self.timestamps = self._load_data(data_path)
        
        print("\n🔄 Normalizing data...")
        self.X_norm = self.normalizer.transform(self.X)
        
        print(f"\n🔮 Getting predictions...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        print("\n📈 Computing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
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
        
        model = MicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            pad_value=-1.0
        )
        
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"   Parameters: {count_parameters(model):,}")
        
        return model
    
    def _load_normalizer(self, normalizer_path):
        normalizer_path = Path(normalizer_path)
        
        if not normalizer_path.exists():
            print(f"   Warning: Normalizer not found, creating default")
            return StableNormalizer(pad_value=-1.0)
        
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        
        print(f"   Loaded: mean={normalizer.mean:.3f}, std={normalizer.std:.3f}")
        return normalizer
    
    def _load_data(self, data_path):
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        if X.ndim == 3:
            X = X.squeeze(1)
        
        if 'timestamps' in data:
            timestamps = data['timestamps']
        else:
            timestamps = np.linspace(-100, 100, X.shape[1])
        
        params = None
        if 'params_binary_json' in data:
            params_binary = json.loads(str(data['params_binary_json']))
            if 'params_pspl_json' in data:
                params_pspl = json.loads(str(data['params_pspl_json']))
                params = {'binary': params_binary, 'pspl': params_pspl}
            else:
                params = {'binary': params_binary}
        
        if self.n_samples is not None and self.n_samples < len(X):
            print(f"   ⚡ Sampling {self.n_samples} events for faster evaluation...")
            
            binary_mask = y == 1
            pspl_mask = y == 0
            
            n_binary = min(self.n_samples // 2, binary_mask.sum())
            n_pspl = min(self.n_samples // 2, pspl_mask.sum())
            
            binary_indices = np.random.choice(np.where(binary_mask)[0], n_binary, replace=False)
            pspl_indices = np.random.choice(np.where(pspl_mask)[0], n_pspl, replace=False)
            
            all_indices = np.concatenate([binary_indices, pspl_indices])
            np.random.shuffle(all_indices)
            
            X = X[all_indices]
            y = y[all_indices]
            
            if params is not None and 'binary' in params:
                all_binary_indices = np.where(binary_mask)[0]
                
                binary_param_indices = []
                for idx in binary_indices:
                    param_idx = np.where(all_binary_indices == idx)[0][0]
                    binary_param_indices.append(param_idx)
                
                params['binary'] = [params['binary'][i] for i in binary_param_indices]
                
                if 'pspl' in params:
                    all_pspl_indices = np.where(pspl_mask)[0]
                    pspl_param_indices = []
                    for idx in pspl_indices:
                        param_idx = np.where(all_pspl_indices == idx)[0][0]
                        pspl_param_indices.append(param_idx)
                    params['pspl'] = [params['pspl'][i] for i in pspl_param_indices]
        
        print(f"   Events: {len(X)}")
        print(f"   Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print(f"   PSPL:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        if params is not None:
            print(f"   ✅ Parameter data found (u0 analysis enabled)")
            if 'binary' in params:
                print(f"      Binary params: {len(params['binary'])}")
            if 'pspl' in params:
                print(f"      PSPL params: {len(params['pspl'])}")
        else:
            print("   ⚠️  No parameter data (u0 analysis disabled)")
        
        return X, y, params, timestamps
    
    def _get_predictions(self):
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
    
    def plot_roc_curve(self):
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
    
    def plot_calibration_curve(self):
        correct = self.predictions == self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
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
    
    def plot_example_grid(self, n_per_class=3):
        print(f"  Generating {n_per_class * 4} example plots (astronomical style)...")
        
        correct = self.predictions == self.y
        binary_pred = self.predictions == 1
        
        tp = np.where((self.y == 1) & binary_pred & correct)[0]
        tn = np.where((self.y == 0) & ~binary_pred & correct)[0]
        fp = np.where((self.y == 0) & binary_pred & ~correct)[0]
        fn = np.where((self.y == 1) & ~binary_pred & ~correct)[0]
        
        examples = []
        for idx_arr, label, color in [
            (tp, 'True Positive\n(Binary → Binary)', 'green'),
            (tn, 'True Negative\n(PSPL → PSPL)', 'blue'),
            (fp, 'False Positive\n(PSPL → Binary)', 'orange'),
            (fn, 'False Negative\n(Binary → PSPL)', 'red')
        ]:
            if len(idx_arr) >= n_per_class:
                conf_sorted = idx_arr[np.argsort(-self.confidences[idx_arr])]
                selected = conf_sorted[:n_per_class]
            elif len(idx_arr) > 0:
                selected = idx_arr[:n_per_class]
            else:
                selected = []
            
            for idx in selected:
                examples.append((idx, label, color))
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i, (idx, label, color) in enumerate(examples):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            flux = self.X[idx]
            
            valid_mask = flux != -1.0
            times = self.timestamps[valid_mask]
            fluxes = flux[valid_mask]
            
            baseline = 20.0
            magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
            
            ax.scatter(times, magnitudes, c=color, s=8, alpha=0.7, edgecolors='black', linewidth=0.3)
            ax.invert_yaxis()
            
            true_label = 'Binary' if self.y[idx] == 1 else 'PSPL'
            pred_label = 'Binary' if self.predictions[idx] == 1 else 'PSPL'
            
            ax.set_title(f'{label}\nTrue: {true_label}, Pred: {pred_label}\nConf: {self.confidences[idx]:.2f}',
                        fontsize=9, color=color, fontweight='bold')
            ax.set_xlabel('Time (days)', fontsize=8)
            ax.set_ylabel('Magnitude', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=7)
        
        plt.suptitle('Example Light Curves (3 per class, astronomical convention)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'example_grid_3x4_astronomical.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_real_time_evolution(self, event_idx=None, event_type='binary'):
        """Plot real-time evolution showing BOTH PSPL and Binary probabilities"""
        if event_idx is None:
            if event_type == 'binary':
                binary_correct = np.where((self.y == 1) & (self.predictions == 1) & (self.confidences > 0.8))[0]
                if len(binary_correct) == 0:
                    binary_correct = np.where(self.y == 1)[0]
                if len(binary_correct) == 0:
                    return
                event_idx = np.random.choice(binary_correct)
            else:
                pspl_correct = np.where((self.y == 0) & (self.predictions == 0) & (self.confidences > 0.8))[0]
                if len(pspl_correct) == 0:
                    pspl_correct = np.where(self.y == 0)[0]
                if len(pspl_correct) == 0:
                    return
                event_idx = np.random.choice(pspl_correct)
        
        light_curve = self.X[event_idx]
        light_curve_norm = self.X_norm[event_idx]
        true_label = self.y[event_idx]
        
        fractions = np.linspace(0.1, 1.0, 10)
        binary_probs = []
        pspl_probs = []
        confidences = []
        
        with torch.no_grad():
            for frac in fractions:
                n_points = int(1500 * frac)
                partial_curve = np.full(1500, -1.0, dtype=np.float32)
                partial_curve[:n_points] = light_curve_norm[:n_points]
                
                x = torch.from_numpy(partial_curve).unsqueeze(0).to(self.device)
                output = self.model(x, return_all=False)
                logits = output['binary']
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                pspl_probs.append(probs[0])
                binary_probs.append(probs[1])
                confidences.append(probs.max())
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.3)
        
        # Top: Light curve
        ax1 = fig.add_subplot(gs[0])
        valid_mask = light_curve != -1.0
        times = self.timestamps[valid_mask]
        fluxes = light_curve[valid_mask]
        baseline = 20.0
        magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
        
        color = 'darkblue' if true_label == 1 else 'darkred'
        ax1.scatter(times, magnitudes, c=color, s=15, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.invert_yaxis()
        
        true_str = 'Binary' if true_label == 1 else 'PSPL'
        pred_str = 'Binary' if self.predictions[event_idx] == 1 else 'PSPL'
        ax1.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        ax1.set_title(f'Light Curve - True: {true_str}, Predicted: {pred_str} (Conf: {self.confidences[event_idx]:.2f})',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Middle: BOTH PROBABILITIES
        ax2 = fig.add_subplot(gs[1])
        days = fractions * 1500
        ax2.plot(days, binary_probs, 'o-', linewidth=3, markersize=8, 
                color='darkblue', label='Binary Probability')
        ax2.plot(days, pspl_probs, 's-', linewidth=3, markersize=8, 
                color='darkred', label='PSPL Probability', alpha=0.8)
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='High Confidence')
        
        ax2.fill_between(days, 0.8, 1.0, alpha=0.15, color='green')
        ax2.fill_between(days, 0.5, 0.8, alpha=0.15, color='yellow')
        ax2.fill_between(days, 0.0, 0.5, alpha=0.15, color='lightblue')
        
        ax2.set_ylabel('Class Probability', fontsize=12, fontweight='bold')
        ax2.legend(loc='center left', fontsize=9, bbox_to_anchor=(1, 0.5))
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        # Bottom: Overall confidence
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(days, confidences, 'd-', linewidth=3, markersize=8, color='purple', label='Overall Confidence')
        ax3.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
        ax3.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% Threshold')
        ax3.set_xlabel('Time Points Observed', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.4, 1.05])
        
        plt.suptitle(f'Real-Time Classification Evolution - {true_str} Event', 
                    fontsize=14, fontweight='bold')
        
        event_type_str = 'binary' if true_label == 1 else 'pspl'
        output_path = self.output_dir / f'real_time_evolution_{event_type_str}_event_{event_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_early_detection(self):
        """
        FIXED v10.1: Properly vectorized tensor creation
        """
        print("  Computing early detection performance...")
        
        fractions = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
        overall_accs = []
        binary_recalls = []
        
        for frac in tqdm(fractions, desc="    Testing fractions"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(self.X_norm), self.batch_size):
                    batch_end = min(i + self.batch_size, len(self.X_norm))
                    batch_size_actual = batch_end - i
                    n_points = int(1500 * frac)
                    
                    # PROPERLY VECTORIZED: No loop needed!
                    partial_curves_np = np.full((batch_size_actual, 1500), -1.0, dtype=np.float32)
                    partial_curves_np[:, :n_points] = self.X_norm[i:batch_end, :n_points]
                    
                    # Convert to tensor
                    x_batch = torch.from_numpy(partial_curves_np).to(self.device)
                    
                    output = self.model(x_batch, return_all=False)
                    logits = output['binary']
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    
                    predictions.extend(probs.argmax(axis=1))
            
            predictions = np.array(predictions)
            overall_accs.append(accuracy_score(self.y, predictions))
            binary_recalls.append(recall_score(self.y, predictions))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        completeness = [f*100 for f in fractions]
        
        ax.plot(completeness, [a*100 for a in overall_accs], 'o-', linewidth=3, markersize=10,
               color='blue', label='Overall Accuracy')
        ax.plot(completeness, [r*100 for r in binary_recalls], 's-', linewidth=3, markersize=10,
               color='red', label='Binary Recall')
        
        ax.axhline(y=70, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='70% threshold')
        ax.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='50% observed')
        
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
    
    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        if self.params is None:
            print("\n⚠️  Skipping u0 analysis (no parameter data)")
            return None
        
        print(f"\n{'='*70}")
        print("u0 DEPENDENCY ANALYSIS")
        print(f"{'='*70}")
        
        binary_params = self.params['binary']
        binary_mask = self.y == 1
        
        u0_values = np.array([p['u_0'] for p in binary_params])
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
                counts.append(int(in_bin.sum()))
            else:
                accuracies.append(np.nan)
                counts.append(0)
        
        threshold_idx = np.argmin(np.abs(u0_centers - threshold))
        acc_at_threshold = accuracies[threshold_idx] if not np.isnan(accuracies[threshold_idx]) else None
        
        n_below = int((u0_values < threshold).sum())
        n_above = int((u0_values >= threshold).sum())
        
        print(f"Physical Detection Threshold: u₀ = {threshold}")
        print(f"Accuracy at threshold: {acc_at_threshold*100:.2f}%" if acc_at_threshold else "N/A")
        print(f"\nEvent Distribution:")
        print(f"  Below threshold (u₀ < {threshold}): {n_below} ({n_below/len(u0_values)*100:.1f}%)")
        print(f"  Above threshold (u₀ ≥ {threshold}): {n_above} ({n_above/len(u0_values)*100:.1f}%)")
        
        return {
            'u0_bins': [float(x) for x in u0_bins],
            'u0_centers': [float(x) for x in u0_centers],
            'accuracies': [float(a) if not np.isnan(a) else None for a in accuracies],
            'counts': counts,
            'all_u0': [float(x) for x in u0_values],
            'threshold': float(threshold),
            'accuracy_at_threshold': float(acc_at_threshold) if acc_at_threshold else None,
            'events_below_threshold': n_below,
            'events_above_threshold': n_above
        }
    
    def plot_u0_dependency(self, u0_results, threshold=0.3):
        if u0_results is None:
            return
        
        u0_centers = u0_results['u0_centers']
        accuracies = [a*100 if a is not None else None for a in u0_results['accuracies']]
        counts = u0_results['counts']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
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
        
        for u, a, c in zip(valid_u0, valid_acc, [counts[i] for i in valid_indices]):
            ax1.annotate(f'{a:.1f}%\n(n={c})', 
                        xy=(u, a), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
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
    
    def generate_all_plots(self, include_u0=True, include_early=False, n_evolution_per_type=3, 
                          u0_threshold=0.3, u0_bins=10):
        """Generate all visualizations with BOTH probability plots"""
        print(f"\n{'='*70}")
        print("GENERATING ALL VISUALIZATIONS (v10.1 - PROPERLY FIXED)")
        print(f"{'='*70}\n")
        
        print("1. ROC Curve...")
        self.plot_roc_curve()
        
        print("\n2. Confusion Matrix...")
        self.plot_confusion_matrix()
        
        print("\n3. Confidence Distribution...")
        self.plot_confidence_distribution()
        
        print("\n4. Calibration Curve...")
        self.plot_calibration_curve()
        
        print("\n5. 3×4 Example Grid (astronomical convention)...")
        self.plot_example_grid(n_per_class=3)
        
        print(f"\n6. Real-Time Evolution ({n_evolution_per_type} Binary + {n_evolution_per_type} PSPL examples)...")
        print("   Generating Binary event evolutions...")
        for i in range(n_evolution_per_type):
            self.plot_real_time_evolution(event_type='binary')
        
        print("   Generating PSPL event evolutions...")
        for i in range(n_evolution_per_type):
            self.plot_real_time_evolution(event_type='pspl')
        
        if include_early:
            print("\n7. Early Detection Performance (PROPERLY VECTORIZED)...")
            self.plot_early_detection()
        
        if include_u0 and self.params is not None:
            print("\n8. u0 Dependency Analysis...")
            u0_results = self.analyze_u0_dependency(n_bins=u0_bins, threshold=u0_threshold)
            if u0_results:
                self.plot_u0_dependency(u0_results, threshold=u0_threshold)
                
                u0_report_path = self.output_dir / 'u0_report.json'
                with open(u0_report_path, 'w') as f:
                    json.dump(u0_results, f, indent=2)
                print(f"  ✓ Saved: {u0_report_path.name}")
        
        print(f"\n{'='*70}")
        print(f"✅ ALL VISUALIZATIONS SAVED TO: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_results(self):
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
        description='Production evaluation (v10.1) with proper vectorization'
    )
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--u0_threshold', type=float, default=0.3)
    parser.add_argument('--u0_bins', type=int, default=10)
    parser.add_argument('--no_u0', action='store_true')
    parser.add_argument('--early_detection', action='store_true')
    parser.add_argument('--n_evolution_per_type', type=int, default=3,
                       help='Number of evolution plots per event type (binary and PSPL)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}_*'))
    
    if not exp_dirs:
        print(f"❌ No experiment found matching: {args.experiment_name}")
        return
    
    exp_dir = exp_dirs[-1]
    model_path = exp_dir / 'best_model.pt'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Using experiment: {exp_dir.name}")
    output_dir = exp_dir / 'evaluation'
    
    device = 'cpu' if args.no_cuda else 'cuda'
    evaluator = ComprehensiveEvaluator(
        model_path=str(model_path),
        normalizer_path=str(normalizer_path),
        data_path=args.data,
        output_dir=str(output_dir),
        device=device,
        batch_size=args.batch_size,
        n_samples=args.n_samples
    )
    
    evaluator.generate_all_plots(
        include_u0=not args.no_u0,
        include_early=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type,
        u0_threshold=args.u0_threshold,
        u0_bins=args.u0_bins
    )
    
    evaluator.save_results()
    print("\n🎉 Production evaluation complete (v10.1)!\n")


if __name__ == '__main__':
    main()
