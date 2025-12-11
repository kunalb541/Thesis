import os
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import matplotlib
try:
    matplotlib.use('Agg') # Essential for HPC/Server use
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORT MODEL
# =============================================================================
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    
    # Try importing the God Mode model first
    try:
        from model import GodModeCausalGRU, GRUConfig
        ModelClass = GodModeCausalGRU
        ConfigClass = GRUConfig
        print(">> Imported GodModeCausalGRU successfully.")
    except ImportError:
        # Fallback to older class names if file hasn't been updated
        from model import CausalHybridModel, CausalConfig
        ModelClass = CausalHybridModel
        ConfigClass = CausalConfig
        print(">> Imported CausalHybridModel (Fallback).")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'model.py'")
    print(f"Error: {e}\n")
    sys.exit(1)

# =============================================================================
# PLOTTING SETUP
# =============================================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
COLORS = ['#95a5a6', '#e74c3c', '#3498db'] # Flat (Grey), PSPL (Red), Binary (Blue)

# =============================================================================
# CORE EVALUATOR CLASS
# =============================================================================
class ComprehensiveEvaluator:
    def __init__(self, experiment_name: str, data_path: str, output_dir: str = None, 
                 device: str = 'cuda', batch_size: int = 128, n_samples: Optional[int] = None,
                 early_detection: bool = False, n_evolution_per_type: int = 0):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        
        # 1. Locate and Load Model
        self.model_path, self.exp_dir = self._find_model(experiment_name)
        
        # 2. Setup Output Directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name = Path(data_path).stem
            self.output_dir = self.exp_dir / f'eval_{data_name}_{timestamp}'
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ROMAN SPACE TELESCOPE - EVALUATION SUITE (Normalized)")
        print("=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Model Path: {self.model_path}")
        print(f"Data Path:  {data_path}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Device:     {self.device}")
        
        # 3. Initialize Model and Data
        self.model, self.config = self._load_model()
        self.flux, self.delta_t, self.y, self.params, self.timestamps = self._load_data(data_path)
        self.lengths = self._compute_lengths()
        
        # 4. Run Core Inference
        print("\n[Inference] Generating predictions...")
        self.probs, self.preds, self.confs, self.seq_probs_batches = self._run_inference()
        
        # 5. Compute Metrics
        self.metrics = self._compute_base_metrics()

    def _find_model(self, exp_name: str) -> Tuple[Path, Path]:
        """Find the best_model.pt for the given experiment name."""
        search_roots = [Path('../results'), Path('results'), Path('.')]
        candidates = []
        clean_name = exp_name.replace('*', '')
        
        for root in search_roots:
            if root.exists():
                candidates.extend(list(root.glob(f"*{clean_name}*")))
        
        if not candidates:
            if Path(exp_name).exists():
                candidates = [Path(exp_name)]
            else:
                raise FileNotFoundError(f"No experiment found matching '{exp_name}'")
            
        best_exp = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        model_file = best_exp / "best_model.pt"
        
        if not model_file.exists():
            pt_files = list(best_exp.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0]
            else:
                raise FileNotFoundError(f"No model file (.pt) found in {best_exp}")
            
        return model_file, best_exp

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        config_dict = checkpoint.get('config', {})
        valid_keys = ConfigClass().__init__.__code__.co_varnames
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ConfigClass(**clean_conf)
        
        model = ModelClass(config).to(self.device)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        
        print(f"Model Loaded: {count_parameters(model):,} parameters")
        return model, config

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        
        flux = data['flux']
        y = data['labels']
        
        if 'delta_t' not in data:
            print("WARNING: 'delta_t' missing. Assuming constant cadence.")
            delta_t = np.zeros_like(flux)
        else:
            delta_t = data['delta_t']
            
        # Handle Timestamps
        if 'timestamps' in data:
            raw_ts = data['timestamps']
        else:
            raw_ts = np.linspace(0, 100, flux.shape[1])
            
        # Load params if available
        params = {}
        for key in ['params_flat', 'params_pspl', 'params_binary']:
            short_key = key.replace('params_', '')
            if key in data:
                if f"{key}_keys" in data:
                    keys = data[f"{key}_keys"]
                    values = data[key]
                    params[short_key] = [dict(zip(keys, v)) for v in values]
                else:
                    params[short_key] = data[key]

        # Normalization (CRITICAL STEP)
        print("Normalizing flux data...")
        flux = self._normalize_flux(flux)

        # Subsampling
        if self.n_samples and self.n_samples < len(flux):
            print(f"Subsampling {len(flux)} -> {self.n_samples} events...")
            idx = np.random.choice(len(flux), self.n_samples, replace=False)
            flux, delta_t, y = flux[idx], delta_t[idx], y[idx]
            
            if raw_ts.ndim == 1:
                timestamps = np.tile(raw_ts, (len(flux), 1))
            else:
                timestamps = raw_ts[idx]
                
            print("Note: Physical limit checks will use population stats due to subsampling.")
        else:
            if raw_ts.ndim == 1:
                timestamps = np.tile(raw_ts, (len(flux), 1))
            else:
                timestamps = raw_ts
            
        return flux, delta_t, y, params, timestamps
    
    def _normalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """Applies Median/IQR normalization to match training data."""
        # Calculate stats on a subset for speed, similar to train.py
        subset_size = min(len(flux), 10000)
        # Flatten to calculate global stats, ignoring zeros (padding)
        sample_vals = flux[:subset_size].flatten()
        sample_vals = sample_vals[sample_vals != 0]
        
        if len(sample_vals) == 0:
            return flux # Should not happen

        median = np.median(sample_vals)
        q75, q25 = np.percentile(sample_vals, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6:
            iqr = 1.0
            
        print(f"  > Norm Stats | Median: {median:.4f} | IQR: {iqr:.4f}")
        
        # Apply to full dataset
        # We process in chunks to save memory if dataset is huge
        norm_flux = np.zeros_like(flux, dtype=np.float32)
        chunk_size = 50000
        
        for i in range(0, len(flux), chunk_size):
            end = min(i + chunk_size, len(flux))
            chunk = flux[i:end]
            # Zero is padding, preserve it
            mask = (chunk != 0)
            chunk_norm = np.where(mask, (chunk - median) / iqr, 0.0)
            norm_flux[i:end] = chunk_norm
            
        return norm_flux

    def _compute_lengths(self):
        # Compute lengths based on non-zero values (assumes padding is 0.0)
        # Using a small epsilon to be safe with floats
        return np.maximum((np.abs(self.flux) > 1e-6).sum(axis=1), 1)

    def _run_inference(self):
        all_probs = []
        seq_probs_batches = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.flux), self.batch_size), desc="Inferring"):
                end = min(i + self.batch_size, len(self.flux))
                
                f = torch.tensor(self.flux[i:end], dtype=torch.float32).to(self.device)
                d = torch.tensor(self.delta_t[i:end], dtype=torch.float32).to(self.device)
                l = torch.tensor(self.lengths[i:end], dtype=torch.long).to(self.device)
                
                # CRITICAL: Only return sequences if we actually need them
                return_seq = self.run_early_detection or (self.n_evolution_per_type > 0)
                
                out = self.model(f, d, lengths=l, return_all_timesteps=return_seq)
                all_probs.append(out['probs'].cpu().numpy())
                
                if return_seq:
                    if 'probs_seq' in out:
                        # Store as list of batch arrays to prevent massive allocation
                        seq_probs_batches.append(out['probs_seq'].cpu().numpy())
                    else:
                        pass 
                
        probs = np.concatenate(all_probs)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs, seq_probs_batches

    def _compute_base_metrics(self):
        acc = accuracy_score(self.y, self.preds)
        try:
            auc = roc_auc_score(self.y, self.probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
            
        report = classification_report(self.y, self.preds, output_dict=True)
        cm = confusion_matrix(self.y, self.preds)
        
        print(f"\nResults:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC:      {auc:.4f}")
        
        return {'acc': acc, 'auc': auc, 'report': report, 'cm': cm}

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================

    def plot_roc_curve(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        classes = ['Flat', 'PSPL', 'Binary']
        
        for i, cls in enumerate(classes):
            if i >= self.probs.shape[1]: continue
            y_bin = (self.y == i).astype(int)
            if len(np.unique(y_bin)) < 2: continue
            
            fpr, tpr, _ = roc_curve(y_bin, self.probs[:, i])
            auc = roc_auc_score(y_bin, self.probs[:, i])
            ax.plot(fpr, tpr, lw=3, label=f'{cls} (AUC={auc:.3f})', color=COLORS[i])
            
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

    def plot_confusion_matrix(self):
        cm = self.metrics['cm']
        # Normalize with safety
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Flat', 'PSPL', 'Binary'],
                   yticklabels=['Flat', 'PSPL', 'Binary'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_calibration(self):
        """Reliability Diagram + Confidence Histogram"""
        correct = (self.preds == self.y)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reliability
        bins = np.linspace(0, 1, 11)
        accs, centers = [], []
        for i in range(len(bins)-1):
            mask = (self.confs >= bins[i]) & (self.confs < bins[i+1])
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                centers.append((bins[i] + bins[i+1])/2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.plot(centers, accs, 'o-', lw=3, color='purple')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        
        # Histogram
        ax2.hist(self.confs[correct], bins=20, alpha=0.5, color='green', label='Correct', density=True)
        ax2.hist(self.confs[~correct], bins=20, alpha=0.5, color='red', label='Incorrect', density=True)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.png')
        plt.close()

    def plot_fine_early_detection(self):
        """
        Calculates accuracy at 50 different observation percentages.
        Refactored to handle batch-wise sequence probabilities correctly.
        """
        if not self.seq_probs_batches:
            print("[Warning] No sequence probabilities found. Skipping Early Detection.")
            return

        print("\n[Analysis] Running Fine-Grained Early Detection...")
        
        fractions = np.linspace(0.05, 1.0, 50)
        
        # We accumulate correct/total counts per fraction
        total_correct = {f: 0 for f in fractions}
        total_count = {f: 0 for f in fractions}
        
        current_idx = 0
        
        for batch_prob in tqdm(self.seq_probs_batches, desc="Early Detect Batches"):
            B, T, C = batch_prob.shape
            
            # Get corresponding metadata for this batch
            batch_lens = self.lengths[current_idx : current_idx + B]
            batch_y = self.y[current_idx : current_idx + B]
            
            for frac in fractions:
                # Vectorized index calculation
                t_idx = (batch_lens * frac).astype(int)
                t_idx = np.clip(t_idx - 1, 0, T-1)
                
                # Gather predictions: batch_prob[row, t_idx]
                preds_at_frac = batch_prob[np.arange(B), t_idx].argmax(axis=1)
                
                correct = (preds_at_frac == batch_y).sum()
                total_correct[frac] += correct
                total_count[frac] += B
            
            current_idx += B
            
        # Compile results
        acc_list = [total_correct[f] / max(total_count[f], 1) for f in fractions]
            
        plt.figure(figsize=(10, 6))
        plt.plot(fractions*100, acc_list, 'o-', color='purple', lw=2)
        plt.axhline(0.9, color='green', ls='--', label='90% Accuracy')
        plt.xlabel('% of Light Curve Observed')
        plt.ylabel('Accuracy')
        plt.title('Early Detection Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fine_early_detection.png')
        plt.close()

    def analyze_physical_limits(self):
        """Analyzes Accuracy vs Impact Parameter (u0)."""
        print("\n[Analysis] Verifying Physical Limits (u0 dependency)...")
        if 'binary' not in self.params:
            print("  Skipping: No binary parameters found.")
            return

        bin_mask = (self.y == 2)
        if bin_mask.sum() == 0: return
        
        try:
            u0_vals = np.array([p['u_0'] for p in self.params['binary']])
            
            bin_preds = self.preds[bin_mask]
            bin_correct = (bin_preds == 2).astype(int)
            
            min_len = min(len(u0_vals), len(bin_correct))
            u0_vals = u0_vals[:min_len]
            bin_correct = bin_correct[:min_len]
            
            bins = np.logspace(np.log10(1e-4), np.log10(1.0), 15)
            digitized = np.digitize(u0_vals, bins)
            
            accs, centers = [], []
            for i in range(1, len(bins)):
                mask = digitized == i
                if mask.sum() > 5:
                    accs.append(bin_correct[mask].mean())
                    centers.append(np.sqrt(bins[i-1] * bins[i]))
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(centers, accs, 'o-', color='blue', lw=2)
            plt.xlabel('Impact Parameter ($u_0$)')
            plt.ylabel('Binary Classification Accuracy')
            plt.title('Physical Detection Limits')
            plt.axvspan(1e-4, 0.15, color='green', alpha=0.1, label='Strong Caustics')
            plt.axvspan(0.3, 1.0, color='red', alpha=0.1, label='Weak/PSPL-like')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'u0_dependency.png')
            plt.close()
            
        except Exception as e:
            print(f"  Error in physical limits: {e}")

    def diagnose_temporal_bias(self):
        """Checks if model is cheating using t0."""
        print("\n[Analysis] Checking Temporal Bias (t0 distribution)...")
        if 'pspl' not in self.params or 'binary' not in self.params: return
        
        try:
            t0_pspl = [p['t_0'] for p in self.params['pspl']]
            t0_bin = [p['t_0'] for p in self.params['binary']]
            
            plt.figure(figsize=(10, 6))
            plt.hist(t0_pspl, bins=30, alpha=0.5, density=True, label='PSPL t0')
            plt.hist(t0_bin, bins=30, alpha=0.5, density=True, label='Binary t0')
            plt.xlabel('Peak Time ($t_0$)')
            plt.title('Temporal Bias Diagnostic')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_bias_check.png')
            plt.close()
            
            stat, pval = ks_2samp(t0_pspl, t0_bin)
            print(f"  t0 Distribution Check: KS-stat={stat:.3f}, p-value={pval:.3f}")
        except Exception as e:
            print(f"  Error in temporal bias: {e}")

    def plot_evolution_examples(self):
        """Plots trajectory of Flux, Probability, and Confidence."""
        print(f"\n[Visualizations] Generating {self.n_evolution_per_type} evolution plots per class...")
        
        classes = ['Flat', 'PSPL', 'Binary']
        
        for i, cls in enumerate(classes):
            candidates = np.where((self.y == i) & (self.preds == i))[0]
            if len(candidates) == 0: continue
            
            selection = np.random.choice(candidates, min(len(candidates), self.n_evolution_per_type), replace=False)
            
            for idx in selection:
                self._plot_single_evolution(idx, cls)

    def _plot_single_evolution(self, idx, true_cls):
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.delta_t[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            out = self.model(f, d, lengths=l, return_all_timesteps=True)
            if 'probs_seq' not in out: return
            probs = out['probs_seq'][0].cpu().numpy()
            
        T = self.lengths[idx]
        times = self.timestamps[idx][:T]
        flux = self.flux[idx][:T]
        probs = probs[:T]
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1], hspace=0.1)
        
        ax0 = fig.add_subplot(gs[0])
        ax0.scatter(times, flux, c='k', s=10, alpha=0.6, label='Flux')
        ax0.set_ylabel('Flux')
        ax0.set_title(f'Event {idx} Evolution (True: {true_cls})')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        for i, cls in enumerate(['Flat', 'PSPL', 'Binary']):
            ax1.plot(times, probs[:, i], color=COLORS[i], lw=2, label=cls)
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        conf = probs.max(axis=1)
        ax2.plot(times, conf, color='purple', lw=2, label='Confidence')
        ax2.axhline(0.9, color='green', ls='--', label='Trigger')
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / f'evolution_{true_cls}_{idx}.png')
        plt.close()

    def run(self):
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_calibration()
        
        if self.run_early_detection:
            self.plot_fine_early_detection()
            
        self.analyze_physical_limits()
        self.diagnose_temporal_bias()
        
        if self.n_evolution_per_type > 0:
            self.plot_evolution_examples()
            
        summary = {
            'metrics': {k: float(v) for k, v in self.metrics.items() if isinstance(v, (int, float))},
            'config': str(self.config),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n[Done] Evaluation complete. Results at: {self.output_dir}")

# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="God Mode Evaluation Suite")
    
    parser.add_argument('--experiment_name', required=True, type=str, help="Name of experiment to find model")
    parser.add_argument('--data', required=True, type=str, help="Path to .npz data")
    parser.add_argument('--output_dir', type=str, default=None, help="Optional custom output path")
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_samples', type=int, default=None, help="Subsample test set")
    
    parser.add_argument('--early_detection', action='store_true', help="Run fine-grained early detection")
    parser.add_argument('--n_evolution_per_type', type=int, default=0, help="Number of evolution plots per class")
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(
        experiment_name=args.experiment_name,
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        early_detection=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type
    )
    
    evaluator.run()
