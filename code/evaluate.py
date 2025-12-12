import os
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import h5py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

def load_compat(path):
    """Hybrid loader for NPZ and HDF5."""
    import h5py
    import numpy as np
    path = str(path)
    if path.endswith('.h5') or path.endswith('.hdf5'):
        with h5py.File(path, 'r') as f:
            # Load all datasets into memory to mimic np.load dictionary behavior
            return {k: f[k][:] for k in f.keys()}
    return np.load(path, allow_pickle=True)


# =============================================================================
# IMPORT MODEL
# =============================================================================
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    
    from model import RomanMicrolensingGRU, ModelConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'model.py'")
    print(f"Error: {e}\n")
    print("Make sure model.py is in the same directory.")
    sys.exit(1)
    

    
# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
COLORS = ['#95a5a6', '#e74c3c', '#3498db']  # Flat (Grey), PSPL (Red), Binary (Blue)
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

# Physical constants from simulate.py
AB_ZEROPOINT_JY = 3631.0
MISSION_DURATION_DAYS = 1826.25


# =============================================================================
# COMPREHENSIVE EVALUATOR
# =============================================================================
class RomanEvaluator:
    """
    Comprehensive evaluation suite for Roman microlensing classifier.
    
    Capabilities:
        - Standard metrics (accuracy, precision, recall, F1)
        - ROC curves and AUC scores
        - Confusion matrices
        - Calibration plots
        - Physical parameter analysis
        - Temporal bias diagnostics
        - Early detection analysis
        - Probability evolution visualization
    """
    
    def __init__(
        self, 
        experiment-name: str, 
        data_path: str, 
        output-dir: Optional[str] = None, 
        device: str = 'cuda', 
        batch-size: int = 128, 
        n-samples: Optional[int] = None,
        early-detection: bool = False, 
        n-evolution-per-type: int = 0
    ):
        """
        Args:
            experiment-name: Name of experiment to evaluate
            data_path: Path to test dataset (.npz)
            output-dir: Optional custom output directory
            device: 'cuda' or 'cpu'
            batch-size: Batch size for inference
            n-samples: Subsample test set (for speed)
            early-detection: Run early detection analysis
            n-evolution-per-type: Number of evolution plots per class
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch-size = batch-size
        self.n-samples = n-samples
        self.run_early-detection = early-detection
        self.n-evolution-per-type = n-evolution-per-type
        
        # Locate and load model
        self.model_path, self.exp_dir = self._find_model(experiment-name)
        
        # Setup output directory
        if output-dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name = Path(data_path).stem
            self.output-dir = self.exp_dir / f'eval_{data_name}_{timestamp}'
        else:
            self.output-dir = Path(output-dir)
            
        self.output-dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ROMAN SPACE TELESCOPE - COMPREHENSIVE EVALUATION")
        print("=" * 80)
        print(f"Experiment:  {experiment-name}")
        print(f"Model Path:  {self.model_path}")
        print(f"Data Path:   {data_path}")
        print(f"Output Dir:  {self.output-dir}")
        print(f"Device:      {self.device}")
        
        # Load model and data
        self.model, self.config, self.checkpoint = self._load_model()
        self.data_dict = self._load_data(data_path)
        
        # Extract for convenience
        self.flux = self.data_dict['norm_flux']
        self.raw_flux = self.data_dict['raw_flux']
        self.delta_t = self.data_dict['delta_t']
        self.y = self.data_dict['y']
        self.lengths = self.data_dict['lengths']
        self.timestamps = self.data_dict['timestamps']
        self.params = self.data_dict['params']
        
        # Run inference
        print("\nRunning inference on test set...")
        self.probs, self.preds, self.confs = self._run_inference()
        
        # Compute metrics
        self.metrics = self._compute_metrics()

    def _find_model(self, exp_name: str) -> Tuple[Path, Path]:
        """Find best_model.pt for experiment."""
        search_roots = [Path('../results'), Path('results'), Path('.')]
        candidates = []
        
        for root in search_roots:
            if root.exists():
                candidates.extend(list(root.glob(f"*{exp_name}*")))
        
        if not candidates:
            if Path(exp_name).exists():
                candidates = [Path(exp_name)]
            else:
                raise FileNotFoundError(
                    f"No experiment found matching '{exp_name}'\n"
                    f"Searched in: {search_roots}"
                )
        
        # Get most recent
        exp_dir = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        model_file = exp_dir / "best_model.pt"
        
        if not model_file.exists():
            pt_files = list(exp_dir.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt file found in {exp_dir}")
        
        return model_file, exp_dir

    def _load_model(self):
        """Load model from checkpoint."""
        print(f"\nLoading model from {self.model_path.name}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract config
        config_dict = checkpoint.get('config', {})
        valid_keys = set(ModelConfig.__annotations__.keys())
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ModelConfig(**clean_conf)
        
        # Create model
        model = RomanMicrolensingGRU(config, dtype=torch.float32).to(self.device)
        
        # Load weights
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        
        n_params = count_parameters(model)
        print(f"  Model: RomanMicrolensingGRU")
        print(f"  Parameters: {n_params:,}")
        print(f"  Config: d_model={config.d_model}, n_layers={config.n_layers}")
        
        return model, config, checkpoint

    def _load_data(self, path):
        """Load and preprocess test data."""
        print(f"\nLoading data from {path}...")
        data = load_compat(path)
        
        # Extract arrays
        raw_flux = data['flux'].astype(np.float32)
        if raw_flux.ndim == 3:
            raw_flux = raw_flux.squeeze(-1)
        
        y = data['labels'].astype(np.int64)
        
        # Delta_t
        if 'delta_t' in data:
            delta_t = data['delta_t'].astype(np.float32)
            if delta_t.ndim == 3:
                delta_t = delta_t.squeeze(-1)
        else:
            print("  Warning: delta_t missing, using zeros")
            delta_t = np.zeros_like(raw_flux)
        
        # Timestamps
        if 'timestamps' in data:
            timestamps = data['timestamps'].astype(np.float32)
            if timestamps.ndim == 3:
                timestamps = timestamps.squeeze(-1)
        else:
            print("  Warning: timestamps missing, generating linear grid")
            timestamps = np.linspace(0, MISSION_DURATION_DAYS, raw_flux.shape[1])
            timestamps = np.tile(timestamps, (len(raw_flux), 1))
        
        # Lengths
        if 'lengths' in data:
            lengths = data['lengths'].astype(np.int64)
        else:
            lengths = np.maximum((raw_flux != 0).sum(axis=1), 1)
        
        # Subsample if requested
        if self.n-samples is not None:
            n = min(self.n-samples, len(raw_flux))
            indices = np.random.choice(len(raw_flux), n, replace=False)
            raw_flux = raw_flux[indices]
            y = y[indices]
            delta_t = delta_t[indices]
            timestamps = timestamps[indices]
            lengths = lengths[indices]
        
        print(f"  Loaded {len(raw_flux):,} samples")
        print(f"  Classes: {np.bincount(y)}")
        
        # Check for physical realism flags
        if 'physical_realism' in data:
            print(f"  Physical realism: {data['physical_realism']}")
        if 'ab_zeropoint_jy' in data:
            print(f"  AB zero point: {data['ab_zeropoint_jy']} Jy")
        
        # Normalize flux (matching train.py)
        print("\nNormalizing flux...")
        
        # Use checkpoint stats if available
        if 'normalization_stats' in self.checkpoint:
            stats = self.checkpoint['normalization_stats']
            print(f"  Using checkpoint stats: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
        else:
            # Compute from data (matching train.py logic)
            stats = self._compute_normalization_stats(raw_flux)
            print(f"  Computed from data: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
        
        # Apply normalization
        mask = (raw_flux != 0)
        norm_flux = np.where(mask, (raw_flux - stats['median']) / stats['iqr'], 0.0)
        
        # Load parameters if available
        params = self._load_parameters(data, y)
        
        return {
            'raw_flux': raw_flux,
            'norm_flux': norm_flux,
            'delta_t': delta_t,
            'y': y,
            'lengths': lengths,
            'timestamps': timestamps,
            'params': params,
            'stats': stats
        }
    
    def _compute_normalization_stats(self, flux: np.ndarray) -> dict:
        """Compute normalization statistics matching train.py."""
        sample_size = min(10000, len(flux))
        sample_flux = flux[:sample_size]
        
        valid_flux = sample_flux[sample_flux != 0]
        valid_flux = valid_flux[~np.isnan(valid_flux)]
        
        if len(valid_flux) == 0:
            return {'median': 0.0, 'iqr': 1.0}
        
        median = float(np.median(valid_flux))
        q75, q25 = np.percentile(valid_flux, [75, 25])
        iqr = float(q75 - q25)
        
        if iqr < 1e-6:
            iqr = 1.0
        
        return {'median': median, 'iqr': iqr}
    
    def _load_parameters(self, data, labels):
        """Load event parameters if available."""
        params = {'flat': [], 'pspl': [], 'binary': []}
        
        for key in ['params_flat', 'params_pspl', 'params_binary']:
            short_key = key.replace('params_', '')
            if key in data and f"{key}_keys" in data:
                keys = data[f"{key}_keys"]
                values = data[key]
                
                for i, row in enumerate(values):
                    param_dict = {k: float(row[j]) for j, k in enumerate(keys)}
                    params[short_key].append(param_dict)
        
        return params

    def _run_inference(self):
        """Run inference on test set."""
        n = len(self.flux)
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, n, self.batch-size), desc="Inference"):
                batch_end = min(i + self.batch-size, n)
                
                flux_batch = torch.tensor(
                    self.flux[i:batch_end], 
                    dtype=torch.float32
                , device=self.device)
                
                dt_batch = torch.tensor(
                    self.delta_t[i:batch_end], 
                    dtype=torch.float32
                , device=self.device)
                
                len_batch = torch.tensor(
                    self.lengths[i:batch_end], 
                    dtype=torch.long
                , device=self.device)
                
                output = self.model(flux_batch, dt_batch, lengths=len_batch)
                probs = output['probs'].cpu().numpy()
                all_probs.append(probs)
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs

    def _compute_metrics(self):
        """Compute classification metrics."""
        print("\nComputing metrics...")
        
        metrics = {
            'accuracy': accuracy_score(self.y, self.preds),
            'precision_macro': precision_score(self.y, self.preds, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y, self.preds, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y, self.preds, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for i, name in enumerate(CLASS_NAMES):
            metrics[f'precision_{name}'] = precision_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            metrics[f'recall_{name}'] = recall_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            metrics[f'f1_{name}'] = f1_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
        
        print("\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
        
        return metrics

    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(self.y, self.preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax1, cbar_kws={'label': 'Count'}
        )
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(
            cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax2, cbar_kws={'label': 'Fraction'}
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output-dir / 'confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self):
        """Plot ROC curves."""
        print("\nGenerating ROC curves...")
        
        # One-hot encode labels
        y_onehot = np.eye(3)[self.y]
        
        plt.figure(figsize=(10, 8))
        
        for i, name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_onehot[:, i], self.probs[:, i])
            auc = roc_auc_score(y_onehot[:, i], self.probs[:, i])
            plt.plot(fpr, tpr, color=COLORS[i], lw=2, 
                    label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output-dir / 'roc_curves.png')
        plt.close()

    def plot_calibration(self):
        """Plot calibration curve."""
        print("\nGenerating calibration plot...")
        
        plt.figure(figsize=(10, 8))
        
        bins = np.linspace(0, 1, 11)
        
        for i, name in enumerate(CLASS_NAMES):
            class_mask = self.y == i
            if class_mask.sum() == 0:
                continue
            
            probs_class = self.probs[class_mask, i]
            preds_class = self.preds[class_mask]
            correct = (preds_class == i).astype(float)
            
            bin_means_pred = []
            bin_means_true = []
            
            for j in range(len(bins) - 1):
                bin_mask = (probs_class >= bins[j]) & (probs_class < bins[j+1])
                if bin_mask.sum() > 0:
                    bin_means_pred.append(probs_class[bin_mask].mean())
                    bin_means_true.append(correct[bin_mask].mean())
            
            if bin_means_pred:
                plt.plot(bin_means_pred, bin_means_true, 'o-', 
                        color=COLORS[i], lw=2, markersize=8, label=name)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output-dir / 'calibration.png')
        plt.close()

    def analyze_physical_limits(self):
        """Analyze accuracy vs physical parameters."""
        print("\nAnalyzing physical parameter dependencies...")
        
        if 'binary' not in self.params or len(self.params['binary']) == 0:
            print("  No binary parameters available")
            return
        
        try:
            # Extract u0 for binary events
            u0_vals = np.array([p.get('u0', p.get('u_0', np.nan)) 
                               for p in self.params['binary']])
            
            # Find binary events in predictions
            bin_mask = self.y == 2
            bin_preds = self.preds[bin_mask]
            bin_correct = (bin_preds == 2).astype(int)
            
            # Align arrays (handle subsampling)
            min_len = min(len(u0_vals), len(bin_correct))
            u0_vals = u0_vals[:min_len]
            bin_correct = bin_correct[:min_len]
            
            # Remove NaNs
            valid = ~np.isnan(u0_vals)
            u0_vals = u0_vals[valid]
            bin_correct = bin_correct[valid]
            
            if len(u0_vals) == 0:
                print("  No valid u0 values found")
                return
            
            # Bin by u0
            bins = np.logspace(np.log10(max(1e-4, u0_vals.min())), 
                             np.log10(min(1.0, u0_vals.max())), 15)
            digitized = np.digitize(u0_vals, bins)
            
            accs, centers = [], []
            for i in range(1, len(bins)):
                mask = digitized == i
                if mask.sum() > 5:
                    accs.append(bin_correct[mask].mean())
                    centers.append(np.sqrt(bins[i-1] * bins[i]))
            
            if not accs:
                print("  Insufficient data for u0 analysis")
                return
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(centers, accs, 'o-', color='blue', lw=2, markersize=8)
            plt.xlabel('Impact Parameter (u₀)')
            plt.ylabel('Binary Classification Accuracy')
            plt.title('Physical Detection Limits')
            plt.axvspan(1e-4, 0.15, color='green', alpha=0.1, label='Strong Caustics')
            plt.axvspan(0.3, 1.0, color='red', alpha=0.1, label='Weak/PSPL-like')
            plt.ylim([0, 1.05])
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output-dir / 'u0_dependency.png')
            plt.close()
            
        except Exception as e:
            print(f"  Error in physical limits analysis: {e}")

    def diagnose_temporal_bias(self):
        """Check for temporal bias in classifications."""
        print("\nDiagnosing temporal bias...")
        
        if 'pspl' not in self.params or 'binary' not in self.params:
            print("  Parameters not available")
            return
        
        if len(self.params['pspl']) == 0 or len(self.params['binary']) == 0:
            print("  Insufficient parameter data")
            return
        
        try:
            # Extract t0 values
            t0_pspl = np.array([p.get('t0', p.get('t_0', np.nan)) 
                               for p in self.params['pspl']])
            t0_bin = np.array([p.get('t0', p.get('t_0', np.nan)) 
                              for p in self.params['binary']])
            
            # Remove NaNs
            t0_pspl = t0_pspl[~np.isnan(t0_pspl)]
            t0_bin = t0_bin[~np.isnan(t0_bin)]
            
            if len(t0_pspl) == 0 or len(t0_bin) == 0:
                print("  No valid t0 values found")
                return
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(t0_pspl, bins=30, alpha=0.6, density=True, 
                    color=COLORS[1], label='PSPL', edgecolor='black')
            plt.hist(t0_bin, bins=30, alpha=0.6, density=True, 
                    color=COLORS[2], label='Binary', edgecolor='black')
            plt.xlabel('Peak Time (t₀) [days]')
            plt.ylabel('Density')
            plt.title('t₀ Distribution (Should be Identical)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # KS test
            stat, pval = ks_2samp(t0_pspl, t0_bin)
            
            plt.subplot(1, 2, 2)
            plt.text(0.1, 0.5, 
                    f"Kolmogorov-Smirnov Test:\n\n"
                    f"Statistic: {stat:.4f}\n"
                    f"p-value: {pval:.4f}\n\n"
                    f"Interpretation:\n"
                    f"p > 0.05: Distributions are similar (good)\n"
                    f"p < 0.05: Distributions differ (bias!)",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output-dir / 'temporal_bias_check.png')
            plt.close()
            
            print(f"  KS test: stat={stat:.4f}, p-value={pval:.4f}")
            if pval < 0.05:
                print("  WARNING: Significant temporal bias detected!")
            else:
                print("  No significant temporal bias (good)")
                
        except Exception as e:
            print(f"  Error in temporal bias analysis: {e}")

    def plot_evolution_examples(self):
        """Plot probability evolution for sample events."""
        print(f"\nGenerating {self.n-evolution-per-type} evolution plots per class...")
        
        for i, cls_name in enumerate(CLASS_NAMES):
            # Find correctly classified events
            candidates = np.where((self.y == i) & (self.preds == i))[0]
            
            if len(candidates) == 0:
                print(f"  No correct predictions for {cls_name}")
                continue
            
            # Random selection
            n_plot = min(len(candidates), self.n-evolution-per-type)
            selection = np.random.choice(candidates, n_plot, replace=False)
            
            for idx in tqdm(selection, desc=f"  {cls_name}"):
                self._plot_single_evolution(idx, cls_name, i)

    def _plot_single_evolution(self, idx: int, cls_name: str, cls_idx: int):
        """Plot single event evolution."""
        f = torch.tensor(self.flux[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        d = torch.tensor(self.delta_t[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            output = self.model(f, d, lengths=l, return_all_timesteps=True)
            if 'probs_seq' not in output:
                return
            probs = output['probs_seq'][0].cpu().numpy()
        
        T = self.lengths[idx]
        times = self.timestamps[idx][:T]
        raw_flux = self.raw_flux[idx][:T]
        probs = probs[:T]
        
        # Convert flux to magnitude for visualization
        valid_mask = raw_flux > 0
        mags = np.full_like(raw_flux, 25.0)
        if valid_mask.any():
            # Use AB system: m = -2.5 * log10(flux / 3631) + offset
            # Adjust offset for typical baseline
            mags[valid_mask] = 22.0 - 2.5 * np.log10(raw_flux[valid_mask] / AB_ZEROPOINT_JY * 1e10)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1], hspace=0.1)
        
        # 1. Light curve
        ax0 = fig.add_subplot(gs[0])
        plot_mask = (raw_flux != 0)
        ax0.scatter(times[plot_mask], mags[plot_mask], c='k', s=10, alpha=0.6)
        ax0.invert_yaxis()
        ax0.set_ylabel('Magnitude (AB)')
        ax0.set_title(f'Event {idx} - {cls_name}')
        ax0.grid(True, alpha=0.3)
        
        # 2. Probabilities
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        for i, name in enumerate(CLASS_NAMES):
            ax1.plot(times, probs[:, i], color=COLORS[i], lw=2, label=name)
        ax1.set_ylabel('Probability')
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 3. Confidence
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        conf = probs.max(axis=1)
        ax2.plot(times, conf, color='purple', lw=2)
        ax2.axhline(0.9, color='green', ls='--', alpha=0.5, label='High Confidence')
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output-dir / f'evolution_{cls_name}_{idx}.png')
        plt.close()

    def verify_causal_inference(self):
        """
        ✅ GOD MODE: Verify that model maintains causality in streaming mode.
        Tests that predictions at timestep T only depend on observations <= T.
        """
        print("\nVerifying causal inference...")
        
        # Take a random sample
        idx = np.random.randint(len(self.flux))
        
        # Prepare tensors correctly
        f_full = torch.tensor(self.flux[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        d_full = torch.tensor(self.delta_t[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        length = self.lengths[idx]
        
        # Full sequence prediction
        with torch.no_grad():
            l_tensor = torch.tensor([length], dtype=torch.long, device=self.device)
            out_full = self.model(f_full, d_full, lengths=l_tensor, return_all_timesteps=True)
            
            # Handle dictionary output safely
            if 'probs_seq' in out_full:
                probs_full = out_full['probs_seq'][0, :length].cpu().numpy()
            else:
                probs_full = torch.softmax(out_full['logits_seq'][0, :length], dim=-1).cpu().numpy()
        
        # Verify causality by truncating (stride for speed)
        step_size = max(1, length // 10)
        
        for t in range(10, length, step_size):
            f_trunc = f_full[:, :t]
            d_trunc = d_full[:, :t]
            l_trunc = torch.tensor([t], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                out_trunc = self.model(f_trunc, d_trunc, lengths=l_trunc, return_all_timesteps=True)
                
                if 'probs_seq' in out_trunc:
                    probs_trunc = out_trunc['probs_seq'][0, -1].cpu().numpy()
                else:
                    probs_trunc = torch.softmax(out_trunc['logits_seq'][0, -1], dim=-1).cpu().numpy()
            
            # The prediction at the last step of the truncated sequence (t-1)
            # must match the prediction at step t-1 of the full sequence.
            diff = np.abs(probs_trunc - probs_full[t-1]).max()
            
            if diff > 1e-5:
                print(f"  ⚠️  CAUSALITY VIOLATION at t={t}: max diff={diff:.6f}")
                return False
        
        print("  ✅ Causality verified: Predictions are properly causal")
        return True

    def run(self):
        """Run full evaluation suite."""
        print("\n" + "=" * 80)
        print("RUNNING EVALUATION SUITE")
        print("=" * 80)
        
        # 1. Verify causality (Critical for Roman Science)
        if self.run_early-detection: 
             if not self.verify_causal_inference():
                 print("CRITICAL: Causality check failed. Aborting evaluation.")
                 return

        # 2. Standard Plots
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_calibration()
        self.analyze_physical_limits()
        self.diagnose_temporal_bias()
        
        # 3. Evolution Plots
        if self.n-evolution-per-type > 0:
            self.plot_evolution_examples()
        
        # 4. Save Summary
        summary = {
            'metrics': {k: float(v) for k, v in self.metrics.items()},
            'model_config': {k: v for k, v in self.config.__dict__.items()},
            'test_samples': int(len(self.y)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output-dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 5. Print Classification Report
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(self.y, self.preds, target_names=CLASS_NAMES, digits=4))
        
        print("\n" + "=" * 80)
        print(f"Evaluation complete. Results saved to: {self.output-dir}")
        print("=" * 80)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite for Roman microlensing classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--experiment-name', required=True, 
                       help="Name of experiment to evaluate")
    parser.add_argument('--data', required=True, 
                       help="Path to test dataset (.npz)")
    parser.add_argument('--output-dir', default=None, 
                       help="Optional custom output directory")
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--n-samples', type=int, default=None,
                       help="Subsample test set for speed")
                       
    parser.add_argument('--device', default='cuda',
                       help="Device: cuda or cpu")
                       
    parser.add_argument('--early-detection', action='store_true', 
                       help='Run early detection analysis')
    
    parser.add_argument('--n-evolution-per-type', type=int, default=5,
                       help="Number of evolution plots per class")
    
    args = parser.parse_args()
    
    evaluator = RomanEvaluator(
        experiment-name=args.experiment_name,
        data_path=args.data,
        output-dir=args.output-dir,
        device=args.device,
        batch-size=args.batch-size,
        n-samples=args.n_samples, early-detection=args.early-detection,
        n-evolution-per-type=args.n_evolution_per_type
    )
    
    evaluator.run()
