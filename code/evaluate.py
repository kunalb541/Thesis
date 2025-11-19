#!/usr/bin/env python3
"""
Model Evaluation

Compatible with MicrolensingTransformer v1.0
- Uses flux + delta_t inputs (not normalized flux)
- Works with ModelConfig
- Handles variable sequence lengths
- Full evaluation suite with all plots

Author: Kunal Bhatia
Version: 1.0.0
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
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from scipy.stats import ks_2samp

import sys

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def create_delta_t_from_timestamps(timestamps):
    """Create delta_t array from timestamps."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    
    return delta_t


def compute_lengths_from_flux(flux, pad_value=-1.0):
    """Compute valid sequence lengths from padded flux."""
    return np.sum(flux != pad_value, axis=1).astype(np.int64)


class ComprehensiveEvaluator:
    """Complete evaluation with v1.0 Transformer support"""
    
    def __init__(self, model_path, data_path, output_dir, 
                 device='cuda', batch_size=128, n_samples=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("MICROLENSING TRANSFORMER EVALUATION v1.0")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        if n_samples:
            print(f"Sample limit: {n_samples}")
        
        print("\nLoading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")
        
        print("\nLoading data...")
        self.flux, self.delta_t, self.y, self.params, self.timestamps, \
            self.n_classes, self.n_points = self._load_data(data_path)
        
        print("\nComputing lengths...")
        self.lengths = compute_lengths_from_flux(self.flux)
        print(f"   Valid lengths: min={self.lengths.min()}, max={self.lengths.max()}, "
              f"mean={self.lengths.mean():.1f}")
        
        print("\nGetting predictions...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        print("\nComputing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load model with v1.0 architecture"""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"   Config from checkpoint")
        else:
            config_path = Path(model_path).parent / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                
                # Create ModelConfig from dict
                sys.path.insert(0, str(Path(__file__).parent))
                from transformer import ModelConfig
                
                config = ModelConfig(
                    d_model=config_dict.get('d_model', 128),
                    n_heads=config_dict.get('n_heads', 8),
                    n_layers=config_dict.get('n_layers', 4),
                    dropout=config_dict.get('dropout', 0.1),
                    attention_window=config_dict.get('attention_window', 64),
                    train_final_only=True,
                    use_adaptive_normalization=True
                )
                print(f"   Config from config.json")
            else:
                raise FileNotFoundError(f"Cannot find config in checkpoint or config.json")
        
        print(f"   d_model={config.d_model}, n_heads={config.n_heads}, "
              f"n_layers={config.n_layers}")
        
        # Import model
        sys.path.insert(0, str(Path(__file__).parent))
        from transformer import MicrolensingTransformer, count_parameters
        
        # Create model
        model = MicrolensingTransformer(config)
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        
        # Load temperature if available
        if 'calibration' in checkpoint and 'temperature' in checkpoint['calibration']:
            temp = checkpoint['calibration']['temperature']
            model.set_temperature(temp)
            print(f"   Temperature: {temp:.4f}")
        
        print(f"   Parameters: {count_parameters(model):,}")
        
        return model, config
    
    def _load_data(self, data_path):
        """Load data with flux and delta_t"""
        data = np.load(data_path)
        
        # Load flux
        if 'flux' in data:
            flux = data['flux']
        elif 'X' in data:
            # Legacy format
            flux = data['X']
        else:
            raise KeyError("Data must contain 'flux' or 'X'")
        
        # Remove channel dimension if present
        if flux.ndim == 3:
            flux = flux.squeeze(1)
        
        # Load or create delta_t
        if 'delta_t' in data:
            delta_t = data['delta_t']
            if delta_t.ndim == 3:
                delta_t = delta_t.squeeze(1)
            print("   Using delta_t from file")
        elif 'timestamps' in data:
            timestamps = data['timestamps']
            if timestamps.ndim == 1:
                # Broadcast to all samples
                timestamps_array = np.tile(timestamps, (len(flux), 1))
            else:
                timestamps_array = timestamps
            delta_t = create_delta_t_from_timestamps(timestamps_array)
            print("   Created delta_t from timestamps")
        else:
            # Create uniform delta_t
            print("   Warning: No timestamps, creating uniform delta_t")
            n_points = flux.shape[1]
            time_range = 200.0  # -100 to 100
            dt = time_range / n_points
            delta_t = np.full_like(flux, dt)
            delta_t[:, 0] = 0.0  # First point has no delta
        
        # Load labels
        if 'labels' in data:
            y = data['labels']
        elif 'y' in data:
            y = data['y']
        else:
            raise KeyError("Data must contain 'labels' or 'y'")
        
        # Get actual data dimensions
        n_points = flux.shape[1]
        n_classes = len(np.unique(y))
        
        # Load timestamps
        if 'timestamps' in data:
            timestamps = data['timestamps']
            if timestamps.ndim == 1:
                timestamps = np.tile(timestamps, (len(flux), 1))
        else:
            # Create uniform timestamps
            timestamps = np.linspace(-100, 100, n_points)
            timestamps = np.tile(timestamps, (len(flux), 1))
        
        print(f"   Dataset: {n_classes} classes")
        print(f"   Data shape: {flux.shape} (n_points={n_points})")
        
        # Load parameters if available
        params = None
        if 'params_binary_json' in data:
            params_binary = json.loads(str(data['params_binary_json']))
            params_dict = {'binary': params_binary}
            
            if 'params_pspl_json' in data:
                params_pspl = json.loads(str(data['params_pspl_json']))
                params_dict['pspl'] = params_pspl
            
            if 'params_flat_json' in data and n_classes == 3:
                params_flat = json.loads(str(data['params_flat_json']))
                params_dict['flat'] = params_flat
            
            params = params_dict
        
        # Sample if requested
        if self.n_samples is not None and self.n_samples < len(flux):
            print(f"   Sampling {self.n_samples} events...")
            
            indices_per_class = []
            for c in range(n_classes):
                class_mask = y == c
                n_class = min(self.n_samples // n_classes, class_mask.sum())
                class_indices = np.random.choice(np.where(class_mask)[0], n_class, replace=False)
                indices_per_class.append(class_indices)
            
            all_indices = np.concatenate(indices_per_class)
            np.random.shuffle(all_indices)
            
            flux = flux[all_indices]
            delta_t = delta_t[all_indices]
            timestamps = timestamps[all_indices]
            y = y[all_indices]
            
            if params is not None:
                for key in params.keys():
                    if key in ['binary', 'pspl', 'flat']:
                        class_id = {'flat': 0, 'pspl': 1, 'binary': 2}.get(key, -1)
                        if class_id >= 0:
                            class_mask = y == class_id
                            params[key] = [params[key][i] for i in range(min(len(params[key]), class_mask.sum()))]
        
        print(f"   Events: {len(flux)}")
        if n_classes == 3:
            print(f"   Flat:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
            print(f"   PSPL:   {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
            print(f"   Binary: {(y == 2).sum()} ({(y == 2).mean()*100:.1f}%)")
        
        if params is not None:
            print(f"   Parameter data available (u0 analysis enabled)")
        else:
            print("   WARNING: No parameter data (u0 analysis disabled)")
        
        return flux, delta_t, y, params, timestamps, n_classes, n_points
    
    def _get_predictions(self):
        """Get predictions using new model API"""
        predictions = []
        confidences = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.flux), self.batch_size), desc="   Evaluating"):
                batch_end = min(i + self.batch_size, len(self.flux))
                
                # Get batch data
                flux_batch = torch.tensor(self.flux[i:batch_end], dtype=torch.float32).to(self.device)
                delta_t_batch = torch.tensor(self.delta_t[i:batch_end], dtype=torch.float32).to(self.device)
                lengths_batch = torch.tensor(self.lengths[i:batch_end], dtype=torch.long).to(self.device)
                
                # Forward pass
                output = self.model(
                    flux_batch, 
                    delta_t_batch, 
                    lengths_batch,
                    return_all_timesteps=False
                )
                
                # Extract predictions
                probs = output['probs'].cpu().numpy()
                preds = output['predictions'].cpu().numpy()
                confs = output['confidence'].cpu().numpy()
                
                predictions.extend(preds)
                confidences.extend(confs)
                all_probs.append(probs)
        
        return np.array(predictions), np.array(confidences), np.vstack(all_probs)
    
    def _compute_metrics(self):
        """Compute classification metrics"""
        accuracy = accuracy_score(self.y, self.predictions)
        
        if self.n_classes == 3:
            target_names = ['Flat', 'PSPL', 'Binary']
        else:
            target_names = ['PSPL', 'Binary']
        
        report = classification_report(
            self.y, self.predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(self.y, self.predictions)
        
        metrics = {
            'accuracy': accuracy,
            'n_classes': self.n_classes,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        for i, name in enumerate(target_names):
            metrics[f'{name.lower()}_precision'] = report[name]['precision']
            metrics[f'{name.lower()}_recall'] = report[name]['recall']
            metrics[f'{name.lower()}_f1'] = report[name]['f1-score']
        
        return metrics
    
    def _print_summary(self):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS ({self.n_classes} classes)")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {self.metrics['accuracy']*100:.2f}%")
        print(f"\nPer-Class Performance:")
        
        if self.n_classes == 3:
            classes = ['flat', 'pspl', 'binary']
            names = ['Flat', 'PSPL', 'Binary']
        else:
            classes = ['pspl', 'binary']
            names = ['PSPL', 'Binary']
        
        for cls, name in zip(classes, names):
            prec = self.metrics[f'{cls}_precision']
            rec = self.metrics[f'{cls}_recall']
            f1 = self.metrics[f'{cls}_f1']
            print(f"  {name:8s}: Prec={prec*100:5.1f}%, Rec={rec*100:5.1f}%, F1={f1*100:5.1f}%")
        
        print(f"{'='*70}\n")
    
    def plot_roc_curve(self):
        """Plot ROC curves (one-vs-rest)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (self.y == i).astype(int)
            y_score = self.probs[:, i]
            
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                auc = roc_auc_score(y_true_binary, y_score)
                
                ax.plot(fpr, tpr, linewidth=3, color=color,
                       label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curves ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = np.array(self.metrics['confusion_matrix'])
        
        if self.n_classes == 3:
            labels = ['Flat', 'PSPL', 'Binary']
        else:
            labels = ['PSPL', 'Binary']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=16, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot confidence distribution"""
        correct = self.predictions == self.y
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bins = np.linspace(0.3 if self.n_classes == 3 else 0.5, 1.0, 50)
        ax.hist(self.confidences[correct], bins=bins, alpha=0.7, color='green',
               label=f'Correct (n={correct.sum()})', edgecolor='black')
        ax.hist(self.confidences[~correct], bins=bins, alpha=0.7, color='red',
               label=f'Incorrect (n={(~correct).sum()})', edgecolor='black')
        
        ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Confidence Distribution ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_calibration_curve(self):
        """Plot calibration curve"""
        correct = self.predictions == self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax = axes[0]
        conf_min = 0.3 if self.n_classes == 3 else 0.5
        conf_bins = np.linspace(conf_min, 1.0, 11)
        accuracies, bin_centers, counts = [], [], []
        
        for i in range(len(conf_bins)-1):
            mask = (self.confidences >= conf_bins[i]) & (self.confidences < conf_bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct[mask].mean())
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                counts.append(mask.sum())
        
        if len(bin_centers) > 0:
            bars = ax.bar(bin_centers, accuracies, width=0.06, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                bar.set_facecolor(plt.cm.Blues(0.3 + 0.7 * cnt / max(counts)))
            
            for bc, acc, cnt in zip(bin_centers, accuracies, counts):
                ax.text(bc, acc + 0.02, f'n={cnt}', ha='center', fontsize=7)
        
        ax.plot([conf_min, 1.0], [conf_min, 1.0], 'r--', linewidth=2, 
               alpha=0.5, label='Perfect')
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Calibration', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3 if self.n_classes == 3 else 0.4, 1.05])
        
        # Confidence vs Correctness scatter
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
        
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correctness')
        ax.set_title('Confidence vs Correctness', fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'calibration.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_example_grid(self, n_per_class=4):
        """Plot example grid"""
        print(f"  Generating example plots...")
        
        correct = self.predictions == self.y
        examples = []
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        for true_class, class_name, color in zip(range(self.n_classes), class_names, colors):
            true_mask = self.y == true_class
            correct_mask = true_mask & correct
            incorrect_mask = true_mask & ~correct
            
            # Correct examples
            if correct_mask.sum() > 0:
                indices = np.where(correct_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:n_per_class]
                for idx in selected:
                    examples.append((idx, f'{class_name} (Correct)', 'green'))
            
            # Incorrect example
            if incorrect_mask.sum() > 0:
                indices = np.where(incorrect_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:1]
                for idx in selected:
                    pred_name = class_names[self.predictions[idx]]
                    examples.append((idx, f'{class_name}→{pred_name}', 'red'))
        
        n_examples = len(examples)
        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, label, color) in enumerate(examples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            flux = self.flux[idx]
            valid_mask = (flux != -1.0) & np.isfinite(flux)
            
            times = self.timestamps[idx][valid_mask]
            fluxes = flux[valid_mask]
            
            # Convert to magnitudes
            baseline = 20.0
            magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
            
            true_name = class_names[self.y[idx]]
            pred_name = class_names[self.predictions[idx]]
            
            ax.scatter(times, magnitudes, c=color, s=8, alpha=0.7, 
                      edgecolors='black', linewidth=0.3)
            ax.invert_yaxis()
            
            ax.set_title(f'{label}\nTrue: {true_name}, Pred: {pred_name}\n'
                        f'Conf: {self.confidences[idx]:.2f}',
                        fontsize=9, color=color, fontweight='bold')
            ax.set_xlabel('Time (days)', fontsize=8)
            ax.set_ylabel('Magnitude', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for i in range(len(examples), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Example Light Curves ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f'example_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_high_res_evolution(self, event_idx=None, event_type='binary'):
        """Ultra-high-resolution classification evolution (100 points)"""
        if event_idx is None:
            if self.n_classes == 3:
                target_class = {'flat': 0, 'pspl': 1, 'binary': 2}.get(event_type, 2)
            else:
                target_class = {'pspl': 0, 'binary': 1}.get(event_type, 1)
            
            good_examples = np.where(
                (self.y == target_class) & 
                (self.predictions == target_class) & 
                (self.confidences > 0.7)
            )[0]
            
            if len(good_examples) == 0:
                good_examples = np.where(self.y == target_class)[0]
            
            if len(good_examples) == 0:
                return
            
            event_idx = np.random.choice(good_examples)
        
        flux = self.flux[event_idx]
        delta_t = self.delta_t[event_idx]
        true_label = self.y[event_idx]
        full_length = self.lengths[event_idx]
        
        # Ultra-high resolution: 100 fractions
        fractions = np.linspace(0.05, 1.0, 100)
        
        if self.n_classes == 3:
            flat_probs, pspl_probs, binary_probs = [], [], []
        else:
            pspl_probs, binary_probs = [], []
        
        confidences = []
        
        with torch.no_grad():
            for frac in fractions:
                n_points = max(1, int(full_length * frac))
                
                # Create partial observation
                partial_flux = torch.tensor(flux[:n_points], dtype=torch.float32).unsqueeze(0).to(self.device)
                partial_delta = torch.tensor(delta_t[:n_points], dtype=torch.float32).unsqueeze(0).to(self.device)
                partial_length = torch.tensor([n_points], dtype=torch.long).to(self.device)
                
                output = self.model(partial_flux, partial_delta, partial_length, 
                                   return_all_timesteps=False)
                probs = output['probs'].cpu().numpy()[0]
                
                if self.n_classes == 3:
                    flat_probs.append(probs[0])
                    pspl_probs.append(probs[1])
                    binary_probs.append(probs[2])
                else:
                    pspl_probs.append(probs[0])
                    binary_probs.append(probs[1])
                
                confidences.append(probs.max())
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 1], hspace=0.3)
        
        # Top: Light curve
        ax1 = fig.add_subplot(gs[0])
        valid_mask = (flux != -1.0) & np.isfinite(flux)
        times = self.timestamps[event_idx][valid_mask]
        fluxes = flux[valid_mask]
        baseline = 20.0
        magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        color = colors[true_label]
        ax1.scatter(times, magnitudes, c=color, s=15, alpha=0.7, 
                   edgecolors='black', linewidth=0.5)
        ax1.invert_yaxis()
        
        true_str = class_names[true_label]
        pred_str = class_names[self.predictions[event_idx]]
        ax1.set_ylabel('Magnitude', fontsize=13, fontweight='bold')
        ax1.set_title(f'ULTRA-HIGH-RES Evolution (100 Points) - True: {true_str}, '
                     f'Pred: {pred_str} (Conf: {self.confidences[event_idx]:.2f})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Middle: Class probabilities
        ax2 = fig.add_subplot(gs[1])
        completeness = [f*100 for f in fractions]
        
        if self.n_classes == 3:
            ax2.plot(completeness, flat_probs, '-', linewidth=1.5, 
                    color='gray', label='Flat', alpha=0.8)
            ax2.plot(completeness, pspl_probs, '-', linewidth=1.5, 
                    color='darkred', label='PSPL', alpha=0.8)
            ax2.plot(completeness, binary_probs, '-', linewidth=1.5, 
                    color='darkblue', label='Binary', alpha=0.8)
        else:
            ax2.plot(completeness, pspl_probs, '-', linewidth=1.5, 
                    color='darkred', label='PSPL', alpha=0.8)
            ax2.plot(completeness, binary_probs, '-', linewidth=1.5, 
                    color='darkblue', label='Binary', alpha=0.8)
        
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, 
                   label='50% Threshold')
        ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, 
                   alpha=0.7, label='High Conf')
        
        ax2.set_ylabel('Class Probability', fontsize=13, fontweight='bold')
        ax2.legend(loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        # Bottom: Confidence
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(completeness, confidences, '-', linewidth=1.5, 
                color='purple', label='Confidence')
        ax3.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80%')
        ax3.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90%')
        ax3.set_xlabel('Observation Completeness (%)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Prediction Confidence', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.3 if self.n_classes == 3 else 0.4, 1.05])
        
        plt.suptitle(f'ULTRA-HIGH-RES Classification Evolution ({self.n_classes}-Class, 100 Points)', 
                    fontsize=15, fontweight='bold')
        
        event_type_str = event_type.lower()
        output_path = self.output_dir / f'ultrahighres_evolution_{event_type_str}_{event_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def plot_fine_early_detection(self):
        """Fine-grained early detection analysis (50 fractions)"""
        print("  Computing fine-grained early detection...")
        
        # 50 fractions for smooth curves
        fractions = np.linspace(0.05, 1.0, 50)
        overall_accs = []
        per_class_recalls = [[] for _ in range(self.n_classes)]
        
        for frac in tqdm(fractions, desc="    Testing fractions"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(self.flux), self.batch_size):
                    batch_end = min(i + self.batch_size, len(self.flux))
                    batch_size_actual = batch_end - i
                    
                    # Create partial observations
                    partial_flux_list = []
                    partial_delta_list = []
                    partial_lengths = []
                    
                    for j in range(i, batch_end):
                        n_points = max(1, int(self.lengths[j] * frac))
                        partial_flux_list.append(
                            torch.tensor(self.flux[j, :n_points], dtype=torch.float32)
                        )
                        partial_delta_list.append(
                            torch.tensor(self.delta_t[j, :n_points], dtype=torch.float32)
                        )
                        partial_lengths.append(n_points)
                    
                    # Pad to max length in batch
                    max_len = max(partial_lengths)
                    padded_flux = torch.full((batch_size_actual, max_len), -1.0)
                    padded_delta = torch.zeros((batch_size_actual, max_len))
                    
                    for k, (f, d) in enumerate(zip(partial_flux_list, partial_delta_list)):
                        padded_flux[k, :len(f)] = f
                        padded_delta[k, :len(d)] = d
                    
                    padded_flux = padded_flux.to(self.device)
                    padded_delta = padded_delta.to(self.device)
                    lengths_tensor = torch.tensor(partial_lengths, dtype=torch.long).to(self.device)
                    
                    output = self.model(padded_flux, padded_delta, lengths_tensor,
                                       return_all_timesteps=False)
                    preds = output['predictions'].cpu().numpy()
                    
                    predictions.extend(preds)
            
            predictions = np.array(predictions)
            overall_accs.append(accuracy_score(self.y, predictions))
            
            # Per-class recall
            for c in range(self.n_classes):
                class_mask = self.y == c
                if class_mask.sum() > 0:
                    class_recall = (predictions[class_mask] == c).mean()
                    per_class_recalls[c].append(class_recall)
                else:
                    per_class_recalls[c].append(0.0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        completeness = [f*100 for f in fractions]
        
        ax.plot(completeness, [a*100 for a in overall_accs], '-', linewidth=3.5, 
               color='purple', label='Overall Accuracy', zorder=3)
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        for c, (name, color) in enumerate(zip(class_names, colors)):
            ax.plot(completeness, [r*100 for r in per_class_recalls[c]], '-', 
                   linewidth=3, color=color, label=f'{name} Recall', alpha=0.8)
        
        ax.axhline(y=33.3 if self.n_classes == 3 else 50, color='red', 
                  linestyle='--', linewidth=1.5, alpha=0.5, label='Random')
        ax.axhline(y=70, color='gray', linestyle=':', linewidth=1.5, 
                  alpha=0.5, label='Target (70%)')
        ax.axhline(y=80, color='green', linestyle=':', linewidth=1.5, 
                  alpha=0.5, label='Excellent (80%)')
        
        milestones = [0.10, 0.25, 0.50, 0.75]
        for milestone in milestones:
            if milestone in fractions:
                idx = list(fractions).index(milestone)
                ax.axvline(x=milestone*100, color='gray', linestyle=':', 
                          linewidth=1, alpha=0.3)
                ax.text(milestone*100, 5, f'{int(milestone*100)}%', 
                       ha='center', fontsize=9, color='gray')
        
        ax.set_xlabel('Observation Completeness (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Fine-Grained Early Detection ({self.n_classes}-Class, 50 Points)', 
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.set_xlim([0, 105])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'fine_early_detection.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def diagnose_temporal_bias(self):
        """Temporal bias diagnostics"""
        if self.params is None or 'pspl' not in self.params or 'binary' not in self.params:
            print("\nWARNING: Skipping temporal bias diagnosis (no parameter data)")
            return
        
        print("\n" + "="*70)
        print("TEMPORAL BIAS DIAGNOSIS")
        print("="*70)
        
        pspl_t0 = np.array([p['t_0'] for p in self.params['pspl']])
        binary_t0 = np.array([p['t_0'] for p in self.params['binary']])
        
        # Test 1: KS test
        stat, pval = ks_2samp(pspl_t0, binary_t0)
        
        print(f"\nTest 1: t0 Distribution Comparison")
        print(f"  PSPL t0:   mean={np.mean(pspl_t0):.1f}, std={np.std(pspl_t0):.1f}")
        print(f"  Binary t0: mean={np.mean(binary_t0):.1f}, std={np.std(binary_t0):.1f}")
        print(f"  KS statistic: {stat:.4f}")
        print(f"  P-value: {pval:.4f}")
        
        if pval < 0.05:
            print(f"  WARNING: SIGNIFICANT DIFFERENCE - Potential data leakage!")
        else:
            print(f"  No significant difference (good!)")
        
        # Test 2: Correlation
        print(f"\nTest 2: t0 vs Predicted Class Correlation")
        
        pspl_mask = self.y == 1
        binary_mask = self.y == 2
        
        pspl_correct_t0 = pspl_t0[(self.predictions[pspl_mask] == 1)]
        pspl_wrong_t0 = pspl_t0[(self.predictions[pspl_mask] != 1)]
        
        binary_correct_t0 = binary_t0[(self.predictions[binary_mask] == 2)]
        binary_wrong_t0 = binary_t0[(self.predictions[binary_mask] != 2)]
        
        print(f"\n  PSPL events:")
        print(f"    Correct (n={len(pspl_correct_t0)}): mean t0={np.mean(pspl_correct_t0):.1f}")
        if len(pspl_wrong_t0) > 0:
            print(f"    Wrong (n={len(pspl_wrong_t0)}): mean t0={np.mean(pspl_wrong_t0):.1f}")
        
        if len(pspl_correct_t0) > 0 and len(pspl_wrong_t0) > 0:
            stat, pval = ks_2samp(pspl_correct_t0, pspl_wrong_t0)
            print(f"    KS test: stat={stat:.4f}, p={pval:.4f}")
            if pval < 0.05:
                print(f"    WARNING: Timing affects PSPL classification!")
        
        print(f"\n  Binary events:")
        print(f"    Correct (n={len(binary_correct_t0)}): mean t0={np.mean(binary_correct_t0):.1f}")
        if len(binary_wrong_t0) > 0:
            print(f"    Wrong (n={len(binary_wrong_t0)}): mean t0={np.mean(binary_wrong_t0):.1f}")
        
        if len(binary_correct_t0) > 0 and len(binary_wrong_t0) > 0:
            stat, pval = ks_2samp(binary_correct_t0, binary_wrong_t0)
            print(f"    KS test: stat={stat:.4f}, p={pval:.4f}")
            if pval < 0.05:
                print(f"    WARNING: Timing affects Binary classification!")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax = axes[0]
        ax.hist(pspl_t0, bins=30, alpha=0.6, color='darkred', 
               label='PSPL (True)', edgecolor='black')
        ax.hist(binary_t0, bins=30, alpha=0.6, color='darkblue', 
               label='Binary (True)', edgecolor='black')
        ax.axvline(np.mean(pspl_t0), color='darkred', linestyle='--', linewidth=2)
        ax.axvline(np.mean(binary_t0), color='darkblue', linestyle='--', linewidth=2)
        ax.set_xlabel('Peak Time t₀ (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('t₀ Distribution by True Class', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        ax = axes[1]
        if len(pspl_correct_t0) > 0:
            ax.hist(pspl_correct_t0, bins=20, alpha=0.5, color='green', 
                   label='PSPL Correct', edgecolor='black')
        if len(pspl_wrong_t0) > 0:
            ax.hist(pspl_wrong_t0, bins=20, alpha=0.5, color='red', 
                   label='PSPL Wrong', edgecolor='black')
        if len(binary_correct_t0) > 0:
            ax.hist(binary_correct_t0, bins=20, alpha=0.5, color='lightgreen', 
                   label='Binary Correct', edgecolor='black', hatch='//')
        if len(binary_wrong_t0) > 0:
            ax.hist(binary_wrong_t0, bins=20, alpha=0.5, color='lightcoral', 
                   label='Binary Wrong', edgecolor='black', hatch='//')
        ax.set_xlabel('Peak Time t₀ (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('t₀: Correct vs Wrong', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'temporal_bias_diagnosis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path.name}")
        plt.close()
        
        print("="*70)
    
    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        """Analyze u0 dependency (Binary class only)"""
        if self.params is None or 'binary' not in self.params:
            print("\nWARNING: Skipping u0 analysis (no binary parameter data)")
            return None
        
        print(f"\n{'='*70}")
        print("u0 DEPENDENCY ANALYSIS (Binary Class)")
        print(f"{'='*70}")
        
        binary_params = self.params['binary']
        
        if self.n_classes == 3:
            binary_mask = self.y == 2
        else:
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
        if acc_at_threshold:
            print(f"Accuracy at threshold: {acc_at_threshold*100:.1f}%")
        print(f"\nBinary Event Distribution:")
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
        """Plot u0 dependency"""
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
        
        ax1.set_ylabel('Binary Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Binary Accuracy vs. Impact Parameter', 
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
        ax2.set_xlabel('Impact Parameter u₀', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax2.set_title('Distribution of Impact Parameters', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'u0_dependency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    def generate_all_plots(self, include_u0=True, include_early=True, 
                          n_evolution_per_type=3, temporal_bias_check=True,
                          u0_threshold=0.3, u0_bins=10):
        """Generate all visualizations"""
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        print("1. ROC Curves...")
        self.plot_roc_curve()
        
        print("\n2. Confusion Matrix...")
        self.plot_confusion_matrix()
        
        print("\n3. Confidence Distribution...")
        self.plot_confidence_distribution()
        
        print("\n4. Calibration Curve...")
        self.plot_calibration_curve()
        
        print("\n5. Example Grid...")
        self.plot_example_grid(n_per_class=3)
        
        print(f"\n6. ULTRA-HIGH-RES Evolution ({n_evolution_per_type} per class)...")
        
        if self.n_classes == 3:
            event_types = ['flat', 'pspl', 'binary']
        else:
            event_types = ['pspl', 'binary']
        
        for event_type in event_types:
            print(f"   {event_type.capitalize()}...")
            for i in range(n_evolution_per_type):
                self.plot_high_res_evolution(event_type=event_type)
        
        if include_early:
            print("\n7. Fine-Grained Early Detection...")
            self.plot_fine_early_detection()
        
        if temporal_bias_check:
            print("\n8. Temporal Bias Diagnosis...")
            self.diagnose_temporal_bias()
        
        if include_u0 and self.params is not None and 'binary' in self.params:
            print("\n9. u0 Dependency Analysis...")
            u0_results = self.analyze_u0_dependency(n_bins=u0_bins, threshold=u0_threshold)
            if u0_results:
                self.plot_u0_dependency(u0_results, threshold=u0_threshold)
                
                u0_report_path = self.output_dir / 'u0_report.json'
                with open(u0_report_path, 'w') as f:
                    json.dump(u0_results, f, indent=2)
                print(f"  Saved: {u0_report_path.name}")
        
        print(f"\n{'='*70}")
        print(f"All visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_results(self):
        """Save evaluation results"""
        results = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in self.metrics.items() 
                       if k not in ['classification_report', 'confusion_matrix']},
            'classification_report': self.metrics['classification_report'],
            'confusion_matrix': self.metrics['confusion_matrix'],
            'n_classes': self.n_classes,
            'n_points': int(self.n_points),
            'n_samples': int(len(self.y)),
            'high_confidence_80': int((self.confidences >= 0.8).sum()),
            'high_confidence_90': int((self.confidences >= 0.9).sum()),
            'has_u0_analysis': self.params is not None and 'binary' in self.params,
            'version': '1.0.0'
        }
        
        output_path = self.output_dir / 'evaluation_summary.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation v1.0 - Updated for new Transformer architecture'
    )
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (will find latest matching)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Limit number of samples')
    parser.add_argument('--u0_threshold', type=float, default=0.3,
                       help='u0 threshold for analysis')
    parser.add_argument('--u0_bins', type=int, default=10,
                       help='Number of bins for u0 analysis')
    parser.add_argument('--no_u0', action='store_true',
                       help='Skip u0 analysis')
    parser.add_argument('--no_temporal_bias_check', action='store_true',
                       help='Skip temporal bias diagnosis')
    parser.add_argument('--early_detection', action='store_true',
                       help='Run early detection analysis (slower)')
    parser.add_argument('--n_evolution_per_type', type=int, default=3,
                       help='Number of evolution plots per class')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Force CPU')
    
    args = parser.parse_args()
    
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}_*'))
    
    if not exp_dirs:
        print(f"ERROR: No experiment found matching: {args.experiment_name}")
        return
    
    exp_dir = exp_dirs[-1]
    
    # Try both model names
    model_path = exp_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = exp_dir / 'final_model.pt'
    
    if not model_path.exists():
        print(f"ERROR: Model not found in: {exp_dir}")
        return
    
    print(f"Using experiment: {exp_dir.name}")
    print(f"Using model: {model_path.name}")
    
    output_dir = exp_dir / 'evaluation'
    
    device = 'cpu' if args.no_cuda else 'cuda'
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        model_path=str(model_path),
        data_path=args.data,
        output_dir=str(output_dir),
        device=device,
        batch_size=args.batch_size,
        n_samples=args.n_samples
    )
    
    # Generate all plots
    evaluator.generate_all_plots(
        include_u0=not args.no_u0,
        include_early=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type,
        temporal_bias_check=not args.no_temporal_bias_check,
        u0_threshold=args.u0_threshold,
        u0_bins=args.u0_bins
    )
    
    # Save results
    evaluator.save_results()
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()