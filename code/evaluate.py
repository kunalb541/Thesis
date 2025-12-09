#!/usr/bin/env python3
"""
Causal Hybrid Model Visualization and Diagnostics

Executes inference using a trained CausalHybridModel to generate performance metrics
and physics-based diagnostic visualizations.

Functional Overview:
- Inference: Runs batched forward passes with causal masking on test data.
- Metrics: Calculates accuracy, precision, recall, F1-score, and AUROC.
- Visualization: Generates ROC curves, confusion matrices, and reliability diagrams.
- Diagnostics:
  - Event Evolution: Plots flux, class probability trajectories, and confidence over time.
  - Temporal Bias: Compares peak time (t0) distributions across classes using KS-tests.
  - Parameter Dependency: Analyzes classification accuracy as a function of impact parameter (u0).

Usage:
    python evaluate.py --experiment_name "exp_id" --data "../data/test.npz"

Author: Kunal Bhatia
Version: 1.0
Date: December 2025
"""

import os
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Dynamic Import for Core Components ---
# This block ensures we can find the model file even if running from a subdir
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from transformer import CausalHybridModel, CausalConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'causal_hybrid_model.py'.")
    print(f"Ensure the model file is in: {current_dir}")
    print(f"Python Error: {e}\n")
    sys.exit(1)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('ggplot')

sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_delta_t_from_timestamps(timestamps):
    """Create delta_t array from timestamps safely."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    
    # Handle negative delta_t (should not happen in sorted time, but safety first)
    delta_t = np.maximum(delta_t, 0.0)
    return delta_t


def compute_lengths_from_flux(flux, pad_value=-1.0):
    """Compute valid sequence lengths from padded flux."""
    # Boolean mask of valid data
    valid_mask = (flux != pad_value)
    # Sum valid points
    lengths = np.sum(valid_mask, axis=1).astype(np.int64)
    # Clamp to minimum 1 to avoid indexing errors
    return np.maximum(lengths, 1)


# =============================================================================
# CORE EVALUATOR CLASS
# =============================================================================
class ComprehensiveEvaluator:
    """
    Complete evaluation suite with CausalHybridModel support.
    Includes all diagnostic and visualization tools.
    """
    
    def __init__(self, model_path, data_path, output_dir, 
                 device='cuda', batch_size=128, n_samples=None):
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        
        # Setup Output Directory with Timestamp to avoid overwrites
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f'eval_{self.timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("CAUSAL HYBRID MODEL EVALUATION (FULL SUITE)")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Device:    {self.device}")
        print(f"Output:    {self.output_dir}")
        if n_samples:
            print(f"Sampling:  {n_samples} events")
        
        # 1. Load Model
        print("\n[1/5] Loading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Load Data
        print("\n[2/5] Loading data...")
        self.flux, self.delta_t, self.y, self.params, self.timestamps, \
            self.n_classes, self.n_points = self._load_data(data_path)
        
        # 3. Compute Lengths
        print("\n[3/5] Computing sequence lengths...")
        self.lengths = compute_lengths_from_flux(self.flux)
        print(f"   Seq Lengths: min={self.lengths.min()}, max={self.lengths.max()}, mean={self.lengths.mean():.1f}")
        
        # 4. Run Inference
        print("\n[4/5] Running inference...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # 5. Compute Metrics
        print("\n[5/5] Computing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load CausalHybridModel with robust config handling."""
        print(f"   Reading checkpoint: {Path(model_path).name}")
        try:
            # Map location fixes device mismatches (e.g. loading gpu model on cpu)
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"   Error loading checkpoint: {e}")
            sys.exit(1)
        
        # Extract Config
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_dict = checkpoint['config']
            # Reconstruct CausalConfig safely (filtering unknown keys for forward compatibility)
            valid_keys = CausalConfig().__dict__.keys()
            filtered_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            config = CausalConfig(**filtered_conf)
        else:
            print("   Warning: Config not found in checkpoint. Using default (128/8/2).")
            config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        print(f"   Architecture: d_model={config.d_model}, heads={config.n_heads}, layers={config.n_transformer_layers}")

        # Initialize Model
        model = CausalHybridModel(config)
        
        # Load State Dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            # Assume dict itself is state_dict
            state_dict = checkpoint
        else:
            # Checkpoint might be the full model object
            return checkpoint, config

        # Handle DDP prefixes (remove 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
            
        # Load weights
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"   Warning: Strict loading failed ({str(e)[:50]}...). Trying strict=False.")
            model.load_state_dict(new_state_dict, strict=False)
            
        print(f"   Model Loaded. Parameters: {count_parameters(model):,}")
        return model, config
    
    def _load_data(self, data_path):
        """
        Robust Data Loading with JSON FIX applied.
        Fixes the numpy-wrapped-string issue common in NPZ files.
        """
        print(f"   Reading {data_path}...")
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            print(f"   Error reading data: {e}")
            sys.exit(1)
        
        # Load Flux
        flux = data.get('flux', data.get('X'))
        if flux is None: raise KeyError("Data missing 'flux' or 'X'")
        if flux.ndim == 3: flux = flux.squeeze(1)
        
        # Load Labels
        y = data.get('labels', data.get('y'))
        if y is None: raise KeyError("Data missing 'labels' or 'y'")
        
        # Load/Create Delta T
        if 'delta_t' in data:
            delta_t = data['delta_t']
            if delta_t.ndim == 3: delta_t = delta_t.squeeze(1)
            print("   Using pre-computed delta_t.")
        elif 'timestamps' in data:
            ts = data['timestamps']
            if ts.ndim == 1: ts = np.tile(ts, (len(flux), 1))
            delta_t = create_delta_t_from_timestamps(ts)
            print("   Computed delta_t from timestamps.")
        else:
            print("   Warning: No temporal data. Creating dummy delta_t (dt=1.0).")
            delta_t = np.ones_like(flux)
            delta_t[:, 0] = 0
            
        # Timestamps for plotting
        timestamps = data.get('timestamps')
        if timestamps is None or timestamps.ndim == 1:
             n_points = flux.shape[1]
             timestamps = np.linspace(0, 100, n_points)
             timestamps = np.tile(timestamps, (len(flux), 1))
        
        n_points = flux.shape[1]
        n_classes = len(np.unique(y))
        
        # --- FIXED PARAMETER LOADING (JSON UNWRAPPING) ---
        params_dict = {}
        target_keys = ['params_binary_json', 'params_pspl_json', 'params_flat_json']
        
        for key in target_keys:
            if key in data:
                try:
                    raw = data[key]
                    
                    # 1. Unwrap numpy array (0-d or 1-d)
                    if isinstance(raw, np.ndarray):
                        if raw.size == 1: 
                            raw = raw.item()
                        else: 
                            raw = raw[0] # Fallback
                    
                    # 2. Decode bytes to string
                    if isinstance(raw, bytes): 
                        raw = raw.decode('utf-8')
                    
                    # 3. Parse JSON
                    cat = key.split('_')[1] # extracts 'binary' from 'params_binary_json'
                    params_dict[cat] = json.loads(str(raw))
                    print(f"   Loaded {len(params_dict[cat])} params for '{cat}'")
                except Exception as e:
                    print(f"   Warning: Failed to load {key}: {e}")

        params = params_dict if params_dict else None
        
        # Sampling
        if self.n_samples is not None and self.n_samples < len(flux):
            print(f"   Subsampling to {self.n_samples} events...")
            # We shuffle indices to get a random subset
            idx = np.random.choice(len(flux), self.n_samples, replace=False)
            flux = flux[idx]
            delta_t = delta_t[idx]
            y = y[idx]
            timestamps = timestamps[idx]
            # Note: We do NOT filter 'params' here. 
            # Filtering params requires exact index matching which is complex without IDs.
            # We will disable u0 checks if sampling is active to prevent mismatch errors.

        return flux, delta_t, y, params, timestamps, n_classes, n_points
    
    def _get_predictions(self):
        """Batched Inference compatible with CausalHybridModel output."""
        predictions = []
        confidences = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.flux), self.batch_size), desc="   Inferring"):
                end = min(i + self.batch_size, len(self.flux))
                
                # Convert to Tensor
                f_b = torch.tensor(self.flux[i:end]).float().to(self.device)
                d_b = torch.tensor(self.delta_t[i:end]).float().to(self.device)
                l_b = torch.tensor(self.lengths[i:end]).long().to(self.device)
                
                # Model call - Causal model returns sequence
                out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=True)
                
                # Handling Output Dictionary
                if 'probs' in out:
                    # out['probs'] shape: [B, SeqLen, Classes]
                    # We need the prediction at the END of the valid sequence.
                    batch_idx = torch.arange(f_b.size(0), device=self.device)
                    # l_b is 1-based length, so index is l_b - 1. Clamp to 0 just in case.
                    last_idx = (l_b - 1).clamp(min=0)
                    
                    # Gather last step
                    final_probs = out['probs'][batch_idx, last_idx] # [B, Classes]
                    
                    probs_np = final_probs.cpu().numpy()
                    preds_np = probs_np.argmax(axis=1)
                    confs_np = probs_np.max(axis=1)
                    
                else:
                    # Fallback if dictionary keys missing
                    print("Error: Model output missing 'probs'. Check model implementation.")
                    sys.exit(1)
                
                predictions.extend(preds_np)
                confidences.extend(confs_np)
                all_probs.append(probs_np)
        
        return np.array(predictions), np.array(confidences), np.vstack(all_probs)
    
    def _compute_metrics(self):
        """Compute standard classification metrics."""
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
        
        # AUROC (Macro and Weighted)
        try:
            if len(np.unique(self.y)) > 1:
                metrics['auroc_macro'] = roc_auc_score(self.y, self.probs, multi_class='ovr', average='macro')
                metrics['auroc_weighted'] = roc_auc_score(self.y, self.probs, multi_class='ovr', average='weighted')
            else:
                metrics['auroc_macro'] = 0.0
                metrics['auroc_weighted'] = 0.0
        except:
            metrics['auroc_macro'] = 0.0
            metrics['auroc_weighted'] = 0.0
            
        # Unpack per-class stats for easier access
        for i, name in enumerate(target_names):
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
            f1 = self.metrics[f'{name.lower()}_f1']
            print(f"   {name:8s}: Prec={prec*100:5.1f}%, Rec={rec*100:5.1f}%, F1={f1*100:5.1f}%")
        print(f"{'='*70}\n")

    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================

    def plot_roc_curve(self):
        """Plot One-vs-Rest ROC curves"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        class_names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'darkred', 'darkblue'] if self.n_classes == 3 else ['darkred', 'darkblue']
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (self.y == i).astype(int)
            # Only plot if class exists in ground truth
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, self.probs[:, i])
                auc = roc_auc_score(y_true_binary, self.probs[:, i])
                ax.plot(fpr, tpr, linewidth=3, color=color, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves', fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

    def plot_confusion_matrix(self):
        """Plot Confusion Matrix Heatmap"""
        cm = np.array(self.metrics['confusion_matrix'])
        labels = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title(f'Confusion Matrix ({self.n_classes}-Class)', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_calibration_curve(self):
        """Plot Reliability Diagram and Confidence Histograms"""
        correct = self.predictions == self.y
        bins = np.linspace(0, 1, 11)
        accs, centers, counts = [], [], []
        
        for i in range(len(bins)-1):
            mask = (self.confidences >= bins[i]) & (self.confidences < bins[i+1])
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                centers.append((bins[i]+bins[i+1])/2)
                counts.append(mask.sum())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reliability Diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
        ax1.plot(centers, accs, 'o-', label='Model', color='blue', linewidth=2)
        ax1.bar(centers, accs, width=0.08, alpha=0.2, color='blue')
        ax1.set_xlabel('Confidence', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Calibration Curve', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence Histogram
        ax2.hist(self.confidences[correct], bins=20, alpha=0.6, color='green', label='Correct')
        ax2.hist(self.confidences[~correct], bins=20, alpha=0.6, color='red', label='Incorrect')
        ax2.set_xlabel('Confidence', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Confidence Distribution', fontweight='bold')
        ax2.legend()
        
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

        if not examples: return

        n_examples = len(examples)
        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1: axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        for i, (idx, label, color) in enumerate(examples):
            ax = axes_flat[i]
            flux = self.flux[idx]
            valid_mask = (flux != -1.0)
            times = self.timestamps[idx][valid_mask]
            fluxes = flux[valid_mask]
            
            ax.scatter(times, fluxes, c=color, s=8, alpha=0.7)
            ax.set_title(f'{label}\nConf: {self.confidences[idx]:.2f}', 
                         fontsize=9, color=color, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
        for i in range(len(examples), len(axes_flat)):
            axes_flat[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'example_grid.png')
        plt.close()

    def plot_high_res_evolution(self, event_idx=None, event_type='binary'):
        """
        Generates evolution plot for a single event using Causal Model capabilities.
        Visualizes Flux, Class Probability Trajectory, and Confidence over time.
        """
        if event_idx is None:
            # Find a good candidate (Correctly predicted, High confidence)
            target_class = {'flat': 0, 'pspl': 1, 'binary': 2}.get(event_type, 2)
            candidates = np.where((self.y == target_class) & (self.predictions == target_class) & (self.confidences > 0.8))[0]
            if len(candidates) == 0: 
                candidates = np.where((self.y == target_class) & (self.predictions == target_class))[0]
            if len(candidates) == 0: return
            event_idx = np.random.choice(candidates)
            
        flux = self.flux[event_idx]
        delta_t = self.delta_t[event_idx]
        full_len = self.lengths[event_idx]
        
        # Prepare single batch
        f_in = torch.tensor(flux).float().unsqueeze(0).to(self.device)
        d_in = torch.tensor(delta_t).float().unsqueeze(0).to(self.device)
        l_in = torch.tensor([full_len]).long().to(self.device)
        
        # Run model to get full sequence output
        with torch.no_grad():
            out = self.model(f_in, d_in, lengths=l_in, return_all_timesteps=True)
            # out['probs'] shape: [1, SeqLen, Classes]
            probs_seq = out['probs'][0, :full_len].cpu().numpy()
            
        # Create 3-panel plot
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 1], hspace=0.3)
        
        # Panel 1: Light Curve
        ax1 = fig.add_subplot(gs[0])
        valid_mask = (flux != -1.0) & (np.arange(len(flux)) < full_len)
        times = self.timestamps[event_idx][valid_mask]
        fluxes = flux[valid_mask]
        
        ax1.scatter(times, fluxes, c='black', s=15, alpha=0.7, label='Flux')
        ax1.set_title(f'Event {event_idx} ({event_type.upper()}) - Evolution Analysis', fontweight='bold')
        ax1.set_ylabel('Flux', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Probabilities (Time-synced)
        # Ensure times matches probs_seq length
        plot_len = min(len(times), len(probs_seq))
        
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if self.n_classes == 3:
            ax2.plot(times[:plot_len], probs_seq[:plot_len, 0], label='Flat', color='gray', alpha=0.7)
            ax2.plot(times[:plot_len], probs_seq[:plot_len, 1], label='PSPL', color='red', alpha=0.7)
            ax2.plot(times[:plot_len], probs_seq[:plot_len, 2], label='Binary', color='blue', alpha=0.7)
        else:
            ax2.plot(times[:plot_len], probs_seq[:plot_len, 0], label='PSPL', color='red')
            ax2.plot(times[:plot_len], probs_seq[:plot_len, 1], label='Binary', color='blue')
            
        ax2.axhline(0.5, color='k', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Class Probability', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Confidence
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        confidence_seq = probs_seq[:plot_len].max(axis=1)
        ax3.plot(times[:plot_len], confidence_seq, color='purple', linewidth=2, label='Confidence')
        ax3.axhline(0.9, color='green', linestyle='--', label='90% Threshold')
        ax3.set_ylabel('Confidence', fontweight='bold')
        ax3.set_xlabel('Time', fontweight='bold')
        ax3.set_ylim(0.2, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / f'evolution_{event_type}_{event_idx}.png')
        plt.close()

    def plot_fine_early_detection(self):
        """
        Fine-grained accuracy over time (aggregated).
        Leverages Causal Transformer to do this efficiently without re-inference.
        """
        print("   Running fine-grained early detection...")
        
        # Subsample for memory efficiency if dataset is huge
        n_test = min(len(self.flux), 1000) 
        indices = np.random.choice(len(self.flux), n_test, replace=False)
        
        f_sub = self.flux[indices]
        d_sub = self.delta_t[indices]
        l_sub = self.lengths[indices]
        y_sub = self.y[indices]
        
        # 1. Run inference once on full sequences to get all timesteps
        all_probs_seq = []
        
        with torch.no_grad():
            for i in range(0, n_test, self.batch_size):
                end = min(i+self.batch_size, n_test)
                f_b = torch.tensor(f_sub[i:end]).float().to(self.device)
                d_b = torch.tensor(d_sub[i:end]).float().to(self.device)
                l_b = torch.tensor(l_sub[i:end]).long().to(self.device)
                
                out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=True)
                # Store CPU numpy
                all_probs_seq.append(out['probs'].cpu().numpy())
        
        # 2. Iterate fractions and gather results
        fractions = np.linspace(0.05, 1.0, 50)
        accuracies = []
        
        for frac in fractions:
            correct_count = 0
            total_count = 0
            
            current_batch_start = 0
            for batch_probs in all_probs_seq:
                batch_size = batch_probs.shape[0]
                batch_lengths = l_sub[current_batch_start : current_batch_start + batch_size]
                batch_y = y_sub[current_batch_start : current_batch_start + batch_size]
                
                # Calculate index corresponding to this fraction of the lightcurve
                # Index = floor(Length * Frac) - 1
                target_indices = (batch_lengths * frac).astype(int)
                target_indices = np.maximum(target_indices - 1, 0)
                
                preds = []
                for b in range(batch_size):
                    # Clamp index to valid range
                    valid_idx = min(target_indices[b], batch_probs.shape[1]-1)
                    p = batch_probs[b, valid_idx]
                    preds.append(np.argmax(p))
                
                correct_count += np.sum(np.array(preds) == batch_y)
                total_count += batch_size
                current_batch_start += batch_size
                
            accuracies.append(correct_count / total_count if total_count > 0 else 0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(fractions*100, accuracies, 'o-', linewidth=2, color='purple')
        plt.xlabel('Percentage of Light Curve Observed', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('Early Detection Performance (Aggregated)', fontweight='bold')
        plt.axhline(0.8, color='green', linestyle=':', label='80% Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.savefig(self.output_dir / 'fine_early_detection.png')
        plt.close()

    def diagnose_temporal_bias(self):
        """Diagnose if model is cheating using t0 (Peak Time)."""
        if not self.params: return
        if self.n_samples: return 
        print("   Running Temporal Bias Check...")
        
        try:
            pspl_t0 = [p.get('t_0', 0) for p in self.params['pspl']]
            binary_t0 = [p.get('t_0', 0) for p in self.params['binary']]
            
            plt.figure(figsize=(10, 6))
            plt.hist(pspl_t0, bins=30, alpha=0.5, label='PSPL t0', density=True, color='red')
            plt.hist(binary_t0, bins=30, alpha=0.5, label='Binary t0', density=True, color='blue')
            plt.legend()
            plt.title('Temporal Bias Check (t0)')
            plt.xlabel('Peak Time (t0)')
            plt.savefig(self.output_dir / 'temporal_bias_check.png')
            plt.close()
            
            stat, pval = ks_2samp(pspl_t0, binary_t0)
            if pval < 0.05: 
                print("   WARNING: Significant t0 bias detected! (p < 0.05)")
                print("   The model might be using 'time of peak' as a cheat feature.")
            else:
                print("   PASSED: t0 distributions are statistically similar.")
        except Exception as e:
            print(f"   Skip temporal check: {e}")

    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        """Analyze accuracy vs impact parameter (u0)."""
        if not self.params or 'binary' not in self.params: return
        if self.n_samples: return
        print("   Running u0 Dependency Check...")
        
        # Identify binary events
        bin_mask = (self.y == 2) if self.n_classes == 3 else (self.y == 1)
        bin_preds = self.predictions[bin_mask]
        bin_y = self.y[bin_mask]
        
        # Extract u0s (Assuming ordered match)
        u0s = np.array([p.get('u_0', -1) for p in self.params['binary']])
        
        if len(u0s) != len(bin_y): 
            print("   Mismatch in binary params vs labels length. Skipping.")
            return 
        
        # Log-space binning
        bins = np.logspace(np.log10(max(1e-4, u0s.min())), np.log10(u0s.max()), n_bins)
        accs, centers, counts = [], [], []
        
        for i in range(len(bins)-1):
            m = (u0s >= bins[i]) & (u0s < bins[i+1])
            if m.sum() > 0:
                accs.append((bin_preds[m] == bin_y[m]).mean())
                centers.append(np.sqrt(bins[i]*bins[i+1]))
                counts.append(m.sum())
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Accuracy line
        ax1.semilogx(centers, accs, 'o-', color='tab:blue', linewidth=2, label='Accuracy')
        ax1.axvline(threshold, color='r', linestyle='--', label=f'u0={threshold}')
        ax1.set_xlabel('Impact Parameter (u0)', fontweight='bold')
        ax1.set_ylabel('Accuracy', color='tab:blue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, which="both", alpha=0.3)
        
        # Histogram on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(centers, counts, width=np.diff(bins), alpha=0.1, color='black', label='Count')
        ax2.set_ylabel('Count', color='black', fontweight='bold')
        ax2.set_yscale('log')
        
        plt.title('Binary Accuracy vs Impact Parameter (u0)', fontweight='bold')
        plt.savefig(self.output_dir / 'u0_dependency.png')
        plt.close()

    def generate_all_plots(self):
        """Orchestrate all visualizations."""
        print("\n[Visualizations] Generating plots...")
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_calibration_curve()
        self.plot_fine_early_detection()
        
        print("   Generating High-Res Evolution examples...")
        if self.n_classes == 3:
            self.plot_high_res_evolution(event_type='flat')
        self.plot_high_res_evolution(event_type='pspl')
        self.plot_high_res_evolution(event_type='binary')
        
        self.diagnose_temporal_bias()
        self.analyze_u0_dependency()
        
        # Save JSON summary
        summary = {
            'metrics': {k: float(v) if isinstance(v, (float, np.float32)) else v 
                        for k,v in self.metrics.items() if 'matrix' not in k and 'report' not in k},
            'timestamp': self.timestamp,
            'model_config': str(self.config.__dict__)
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"\nEvaluation Complete. Results in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full Causal Evaluation with Experiment Search")
    parser.add_argument('--experiment_name', type=str, required=True, 
                        help='Experiment name to search in ../results (e.g., "exp1")')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to .npz data file')
    parser.add_argument('--n_samples', type=int, default=None, 
                        help='Limit samples for faster debugging')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Find Experiment Directory
    # We look in both ../results and ./results to be helpful
    possible_roots = [Path('../results'), Path('results'), Path('.')]
    results_dir = None
    
    for r in possible_roots:
        if r.exists() and list(r.glob(f'{args.experiment_name}*')):
            results_dir = r
            break
            
    if results_dir is None:
        print(f"ERROR: Could not find any directory matching '{args.experiment_name}*' in {possible_roots}")
        return

    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}*'))
    
    # Pick latest experiment
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
