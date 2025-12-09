#!/usr/bin/env python3
"""
Causal Hybrid Model Evaluation and Diagnostic Suite

Performs inference using a trained CausalHybridModel on test data.
Computes classification metrics and generates diagnostic visualizations regarding
temporal evolution and physical parameter dependencies.

Functions:
- Loads model checkpoints and .npz test data.
- Executes batched inference with causal masking.
- Computes Accuracy, AUROC, Precision, Recall, and F1-score.
- Generates visualizations: ROC curves, Confusion Matrices, Calibration curves.
- Performs physics-based diagnostics:
  - Temporal Bias Check (t0 distribution analysis).
  - Impact Parameter (u0) dependency analysis.
  - Time-step evolution of class probabilities.

Usage:
    python evaluate.py --experiment_name "exp_id" --data "../data/test.npz"

Author: Kunal Bhatia
Version: 3.5.0
Date: December 2025
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
import sys
import warnings
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from scipy.stats import ks_2samp
from datetime import datetime

# Filter warnings
warnings.filterwarnings("ignore")

# --- Dynamic Import for Core Components ---
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import CausalHybridModel, CausalConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"CRITICAL ERROR: 'transformer.py' not found.")
    print(f"Details: {e}")
    sys.exit(1)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')

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
    lens = np.sum(flux != pad_value, axis=1).astype(np.int64)
    return np.maximum(lens, 1)


# =============================================================================
# CORE EVALUATOR CLASS
# =============================================================================
class ComprehensiveEvaluator:
    """
    Complete evaluation suite with CausalHybridModel support.
    """
    
    def __init__(self, model_path, data_path, output_dir, 
                 device='cuda', batch_size=128, n_samples=None):
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        
        # Output setup (Matches your version 2.0 logic)
        self.output_dir = Path(output_dir) / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("CAUSAL HYBRID MODEL EVALUATION")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        if n_samples:
            print(f"Sample limit: {n_samples}")
        
        # 1. Load Model
        print("\nLoading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Load Data (With JSON Fix)
        print("\nLoading data...")
        self.flux, self.delta_t, self.y, self.params, self.timestamps, \
            self.n_classes, self.n_points = self._load_data(data_path)
        
        # 3. Lengths
        print("\nComputing lengths...")
        self.lengths = compute_lengths_from_flux(self.flux)
        
        # 4. Inference
        print("\nGetting predictions...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # 5. Metrics
        print("\nComputing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load CausalHybridModel with robust config handling."""
        print(f"   Reading {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
        
        # Extract Config
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_dict = checkpoint['config']
            valid_keys = CausalConfig().__dict__.keys()
            filtered_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            config = CausalConfig(**filtered_conf)
        else:
            print("   Warning: Config not found. Using default 128/8/2.")
            config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        # Initialize
        model = CausalHybridModel(config)
        
        # Load Weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            return checkpoint, config

        # Handle DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"   Params: {count_parameters(model):,}")
        return model, config
    
    def _load_data(self, data_path):
        """Robust Data Loading with JSON Unwrapping Fix."""
        print(f"   Reading {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        flux = data.get('flux', data.get('X'))
        if flux is None: raise KeyError("Data missing 'flux'")
        if flux.ndim == 3: flux = flux.squeeze(1)
        
        y = data.get('labels', data.get('y'))
        if y is None: raise KeyError("Data missing 'labels'")
        
        if 'delta_t' in data:
            delta_t = data['delta_t']
            if delta_t.ndim == 3: delta_t = delta_t.squeeze(1)
        elif 'timestamps' in data:
            ts = data['timestamps']
            if ts.ndim == 1: ts = np.tile(ts, (len(flux), 1))
            delta_t = create_delta_t_from_timestamps(ts)
        else:
            delta_t = np.zeros_like(flux)
            
        timestamps = data.get('timestamps')
        if timestamps is None or timestamps.ndim == 1:
             timestamps = np.tile(np.linspace(-100, 100, flux.shape[1]), (len(flux), 1))
        
        n_points = flux.shape[1]
        n_classes = len(np.unique(y))
        
        # --- FIXED PARAMETER LOADING ---
        params_dict = {}
        target_keys = ['params_binary_json', 'params_pspl_json', 'params_flat_json']
        
        for key in target_keys:
            if key in data:
                try:
                    raw = data[key]
                    if isinstance(raw, np.ndarray):
                        raw = raw.item() if raw.size == 1 else raw[0]
                    if isinstance(raw, bytes): raw = raw.decode('utf-8')
                    
                    cat = key.split('_')[1]
                    params_dict[cat] = json.loads(str(raw))
                    print(f"   Loaded {len(params_dict[cat])} params for '{cat}'")
                except Exception as e:
                    print(f"   Warning: Failed to load {key}: {e}")

        params = params_dict if params_dict else None
        
        # Sampling
        if self.n_samples is not None and self.n_samples < len(flux):
            print(f"   Subsampling to {self.n_samples}...")
            idx = np.random.choice(len(flux), self.n_samples, replace=False)
            flux = flux[idx]
            delta_t = delta_t[idx]
            y = y[idx]
            timestamps = timestamps[idx]

        return flux, delta_t, y, params, timestamps, n_classes, n_points
    
    def _get_predictions(self):
        """Batched Inference for Causal Model."""
        predictions, confidences, all_probs = [], [], []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.flux), self.batch_size), desc="   Inferring"):
                end = min(i + self.batch_size, len(self.flux))
                
                f_b = torch.tensor(self.flux[i:end]).float().to(self.device)
                d_b = torch.tensor(self.delta_t[i:end]).float().to(self.device)
                l_b = torch.tensor(self.lengths[i:end]).long().to(self.device)
                
                # Model Call
                out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=True)
                
                # Extract Last Timestep
                if 'probs' in out:
                    batch_idx = torch.arange(f_b.size(0), device=self.device)
                    last_idx = (l_b - 1).clamp(min=0)
                    final_probs = out['probs'][batch_idx, last_idx]
                    
                    probs_np = final_probs.cpu().numpy()
                    predictions.extend(probs_np.argmax(axis=1))
                    confidences.extend(probs_np.max(axis=1))
                    all_probs.append(probs_np)
                else:
                    print("Error: Model output missing 'probs'.")
                    sys.exit(1)
        
        return np.array(predictions), np.array(confidences), np.vstack(all_probs)
    
    def _compute_metrics(self):
        acc = accuracy_score(self.y, self.predictions)
        names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        
        report = classification_report(self.y, self.predictions, target_names=names, output_dict=True, zero_division=0)
        cm = confusion_matrix(self.y, self.predictions).tolist()
        
        metrics = {'accuracy': acc, 'report': report, 'confusion_matrix': cm}
        
        try:
            if len(np.unique(self.y)) > 1:
                metrics['auroc_macro'] = roc_auc_score(self.y, self.probs, multi_class='ovr', average='macro')
            else:
                metrics['auroc_macro'] = 0.0
        except:
            metrics['auroc_macro'] = 0.0
            
        for i, name in enumerate(names):
            metrics[f'{name.lower()}_precision'] = report[name]['precision']
            metrics[f'{name.lower()}_recall'] = report[name]['recall']
            metrics[f'{name.lower()}_f1'] = report[name]['f1-score']
            
        return metrics
    
    def _print_summary(self):
        print(f"\n{'='*70}")
        print(f"RESULTS ({self.n_classes} classes)")
        print(f"{'='*70}")
        print(f"Accuracy: {self.metrics['accuracy']*100:.2f}%")
        print(f"AUROC:    {self.metrics['auroc_macro']:.4f}")
        
        names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        print(f"\nPer-Class Performance:")
        for name in names:
            prec = self.metrics[f'{name.lower()}_precision']
            rec = self.metrics[f'{name.lower()}_recall']
            print(f"   {name:8s}: Prec={prec*100:5.1f}%, Rec={rec*100:5.1f}%")
        print(f"{'='*70}\n")

    # =========================================================================
    # PLOTTING & DIAGNOSTICS
    # =========================================================================

    def plot_roc_curve(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'darkred', 'darkblue'] if self.n_classes == 3 else ['darkred', 'darkblue']
        
        for i, (name, color) in enumerate(zip(names, colors)):
            y_true_binary = (self.y == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, self.probs[:, i])
                auc = roc_auc_score(y_true_binary, self.probs[:, i])
                ax.plot(fpr, tpr, linewidth=3, color=color, label=f'{name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.legend()
        ax.set_title('ROC Curves')
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

    def plot_confusion_matrix(self):
        cm = np.array(self.metrics['confusion_matrix'])
        names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
        plt.title('Confusion Matrix')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_calibration_curve(self):
        correct = self.predictions == self.y
        bins = np.linspace(0, 1, 11)
        accs, centers = [], []
        
        for i in range(len(bins)-1):
            mask = (self.confidences >= bins[i]) & (self.confidences < bins[i+1])
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                centers.append((bins[i]+bins[i+1])/2)
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(centers, accs, 'o-', label='Model')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration')
        plt.legend()
        plt.savefig(self.output_dir / 'calibration.png')
        plt.close()

    def plot_example_grid(self, n_per_class=4):
        """Plot grid of correct/incorrect examples."""
        print(f"   Generating example plots...")
        correct = self.predictions == self.y
        examples = []
        
        class_names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'darkred', 'darkblue'] if self.n_classes == 3 else ['darkred', 'darkblue']
        
        for true_class, class_name, color in zip(range(self.n_classes), class_names, colors):
            true_mask = self.y == true_class
            correct_mask = true_mask & correct
            incorrect_mask = true_mask & ~correct
            
            # Select High Confidence Correct
            if correct_mask.sum() > 0:
                indices = np.where(correct_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:n_per_class]
                for idx in selected:
                    examples.append((idx, f'{class_name} (Correct)', 'green'))
            
            # Select High Confidence Incorrect
            if incorrect_mask.sum() > 0:
                indices = np.where(incorrect_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:1]
                for idx in selected:
                    pred_name = class_names[self.predictions[idx]]
                    examples.append((idx, f'{class_name}->{pred_name}', 'red'))

        n_examples = len(examples)
        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1: axes = axes.reshape(1, -1)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        for i, (idx, label, color) in enumerate(examples):
            ax = axes_flat[i]
            flux = self.flux[idx]
            valid_mask = (flux != -1.0)
            times = self.timestamps[idx][valid_mask]
            fluxes = flux[valid_mask]
            
            # Plot
            ax.scatter(times, fluxes, c=color, s=8, alpha=0.7)
            ax.set_title(f'{label}\nConf: {self.confidences[idx]:.2f}', 
                         fontsize=9, color=color, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
        # Hide empty
        for i in range(len(examples), len(axes_flat)):
            axes_flat[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'example_grid.png')
        plt.close()

    def plot_high_res_evolution(self, event_idx=None, event_type='binary'):
        """Generates evolution plot for a single event."""
        if event_idx is None:
            target = {'flat': 0, 'pspl': 1, 'binary': 2}.get(event_type, 2)
            cands = np.where((self.y == target) & (self.predictions == target) & (self.confidences > 0.8))[0]
            if len(cands) == 0: cands = np.where(self.y == target)[0]
            if len(cands) == 0: return
            event_idx = np.random.choice(cands)
            
        flux = self.flux[event_idx]
        delta = self.delta_t[event_idx]
        length = self.lengths[event_idx]
        
        f_in = torch.tensor(flux).float().unsqueeze(0).to(self.device)
        d_in = torch.tensor(delta).float().unsqueeze(0).to(self.device)
        l_in = torch.tensor([length]).long().to(self.device)
        
        with torch.no_grad():
            out = self.model(f_in, d_in, lengths=l_in, return_all_timesteps=True)
            probs = out['probs'][0, :length].cpu().numpy()
            
        times = self.timestamps[event_idx][:length]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.scatter(times, flux[:length], c='k', s=10, alpha=0.5)
        ax1.set_title(f'Evolution: Event {event_idx} ({event_type})')
        ax1.set_ylabel('Flux')
        
        if self.n_classes == 3:
            ax2.plot(times, probs[:, 0], color='gray', label='Flat', alpha=0.6)
            ax2.plot(times, probs[:, 1], color='red', label='PSPL', alpha=0.6)
            ax2.plot(times, probs[:, 2], color='blue', label='Binary', alpha=0.6)
        
        ax2.plot(times, probs.max(axis=1), color='purple', linestyle='--', label='Confidence')
        ax2.legend()
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('Time')
        ax2.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'evolution_{event_type}_{event_idx}.png')
        plt.close()

    def plot_fine_early_detection(self):
        print("   Running fine-grained early detection...")
        n_test = min(len(self.flux), 500)
        idx = np.random.choice(len(self.flux), n_test, replace=False)
        
        f_sub = self.flux[idx]
        d_sub = self.delta_t[idx]
        l_sub = self.lengths[idx]
        y_sub = self.y[idx]
        
        fractions = np.linspace(0.05, 1.0, 20)
        accuracies = []
        
        with torch.no_grad():
            for frac in tqdm(fractions):
                preds = []
                for i in range(0, n_test, self.batch_size):
                    end = min(i+self.batch_size, n_test)
                    
                    cur_len = (l_sub[i:end] * frac).astype(int)
                    cur_len = np.maximum(cur_len, 1)
                    max_l = cur_len.max()
                    
                    f_b = torch.full((len(cur_len), max_l), -1.0, device=self.device)
                    d_b = torch.zeros((len(cur_len), max_l), device=self.device)
                    
                    for k in range(len(cur_len)):
                        f_b[k, :cur_len[k]] = torch.from_numpy(f_sub[i+k, :cur_len[k]])
                        d_b[k, :cur_len[k]] = torch.from_numpy(d_sub[i+k, :cur_len[k]])
                        
                    l_b = torch.from_numpy(cur_len).long().to(self.device)
                    out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=True)
                    
                    last_idx = (l_b - 1).clamp(min=0)
                    final_p = out['probs'][torch.arange(len(l_b)), last_idx]
                    preds.extend(final_p.argmax(dim=1).cpu().numpy())
                
                accuracies.append(accuracy_score(y_sub, preds))
                
        plt.figure(figsize=(10, 6))
        plt.plot(fractions*100, accuracies, 'o-', color='purple')
        plt.xlabel('Completeness (%)')
        plt.ylabel('Accuracy')
        plt.title('Early Detection')
        plt.grid(True)
        plt.savefig(self.output_dir / 'fine_early_detection.png')
        plt.close()

    def diagnose_temporal_bias(self):
        if not self.params: return
        if self.n_samples: return 
        print("   Running Temporal Bias Check...")
        
        try:
            pspl_t0 = [p.get('t_0', 0) for p in self.params['pspl']]
            binary_t0 = [p.get('t_0', 0) for p in self.params['binary']]
            
            plt.figure(figsize=(10, 6))
            plt.hist(pspl_t0, bins=30, alpha=0.5, label='PSPL t0', density=True)
            plt.hist(binary_t0, bins=30, alpha=0.5, label='Binary t0', density=True)
            plt.legend()
            plt.title('Temporal Bias Check (t0)')
            plt.savefig(self.output_dir / 'temporal_bias_check.png')
            plt.close()
            
            stat, pval = ks_2samp(pspl_t0, binary_t0)
            if pval < 0.05: print("   WARNING: Significant t0 bias detected!")
        except Exception as e:
            print(f"   Skip temporal check: {e}")

    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        if not self.params or 'binary' not in self.params: return
        if self.n_samples: return
        print("   Running u0 Dependency Check...")
        
        bin_mask = (self.y == 2) if self.n_classes == 3 else (self.y == 1)
        bin_preds = self.predictions[bin_mask]
        bin_y = self.y[bin_mask]
        
        u0s = np.array([p.get('u_0', -1) for p in self.params['binary']])
        if len(u0s) != len(bin_y): return 
        
        bins = np.logspace(np.log10(max(1e-4, u0s.min())), np.log10(u0s.max()), n_bins)
        accs, centers = [], []
        
        for i in range(len(bins)-1):
            m = (u0s >= bins[i]) & (u0s < bins[i+1])
            if m.sum() > 0:
                accs.append((bin_preds[m] == bin_y[m]).mean())
                centers.append(np.sqrt(bins[i]*bins[i+1]))
        
        plt.figure(figsize=(8, 6))
        plt.semilogx(centers, accs, 'o-')
        plt.axvline(threshold, color='r', linestyle='--')
        plt.xlabel('u0')
        plt.ylabel('Accuracy')
        plt.title('Binary Accuracy vs u0')
        plt.grid(True)
        plt.savefig(self.output_dir / 'u0_dependency.png')
        plt.close()

    def generate_all_plots(self):
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_calibration_curve()
        self.plot_fine_early_detection()
        
        if self.n_classes == 3: self.plot_high_res_evolution(event_type='flat')
        self.plot_high_res_evolution(event_type='pspl')
        self.plot_high_res_evolution(event_type='binary')
        
        self.diagnose_temporal_bias()
        self.analyze_u0_dependency()
        
        # Save summary
        res = {k: float(v) for k,v in self.metrics.items() if isinstance(v, (float, np.float32))}
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(res, f, indent=4)
            
        print(f"\nEvaluation Complete. Saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full Causal Evaluation with Experiment Search")
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name to search in ../results')
    parser.add_argument('--data', type=str, required=True, help='Path to .npz data')
    parser.add_argument('--n_samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Find Experiment Directory
    results_dir = Path('../results')
    if not results_dir.exists():
        # Fallback if running from a different relative path
        results_dir = Path('results')
        
    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}*'))
    
    if not exp_dirs:
        print(f"ERROR: No experiment directory found matching '{args.experiment_name}*' in {results_dir.resolve()}")
        return
        
    # Pick latest
    exp_dir = exp_dirs[-1]
    
    # 2. Find Model File
    # Priority: best_model.pt -> final_model.pt -> *.pt
    if (exp_dir / 'best_model.pt').exists():
        model_path = exp_dir / 'best_model.pt'
    elif (exp_dir / 'final_model.pt').exists():
        model_path = exp_dir / 'final_model.pt'
    else:
        # Try finding any .pt file
        pts = list(exp_dir.glob('*.pt'))
        if pts:
            model_path = pts[0]
        else:
            print(f"ERROR: No .pt model found inside {exp_dir}")
            return

    print(f"Selected Experiment: {exp_dir.name}")
    print(f"Selected Model:      {model_path.name}")

    # 3. Run Evaluation
    ComprehensiveEvaluator(
        model_path=str(model_path),
        data_path=args.data,
        output_dir=str(exp_dir), # Save results INSIDE experiment folder
        device='cpu' if args.no_cuda else 'cuda',
        batch_size=args.batch_size,
        n_samples=args.n_samples
    ).generate_all_plots()

if __name__ == '__main__':
    main()
