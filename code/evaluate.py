import os
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import matplotlib
try:
    # Use Agg backend for non-interactive plotting environments
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import h5py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve # Ensure this is available globally for the function

warnings.filterwarnings("ignore")

# =============================================================================
# UTILITIES AND CONFIGURATION
# =============================================================================
def load_compat(path: str) -> Dict[str, Any]:
    """Hybrid loader for NPZ and HDF5, loading all datasets."""
    path = str(path)
    if path.endswith('.h5') or path.endswith('.hdf5'):
        # --- FIX: Check for structured array params ---
        data = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                data[k] = f[k][:]
        return data
        
    return np.load(path, allow_pickle=True)

# Try importing model for type hinting and parameter counting
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    
    from model import RomanMicrolensingClassifier as RomanMicrolensingGRU, ModelConfig 
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
except ImportError as e:
    # Placeholder classes if model.py cannot be imported
    class ModelConfig:
        def __init__(self, d_model=64, n_layers=2, **kwargs):
            self.d_model = d_model
            self.n_layers = n_layers
    class RomanMicrolensingGRU(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            print(f"\n[WARNING] Model class not found. Using dummy model.")
            
    def count_parameters(model):
        return 0
    

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
COLORS = ['#95a5a6', '#e74c3c', '#3498db']  # Flat (Grey), PSPL (Red), Binary (Blue)
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']
MISSION_DURATION_DAYS = 1826.25 # Roman 5-year mission


# =============================================================================
# COMPREHENSIVE EVALUATOR
# =============================================================================
class RomanEvaluator:
    """
    Comprehensive evaluation suite for Roman microlensing classifier.
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        data_path: str, 
        output_dir: Optional[str] = None, 
        device: str = 'cuda', 
        batch_size: int = 128, 
        n_samples: Optional[int] = None,
        early_detection: bool = False, 
        n_evolution_per_type: int = 0,
        n_example_grid_per_type: int = 4
    ):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        self.n_example_grid_per_type = n_example_grid_per_type
        
        self.model_path, self.exp_dir = self._find_model(experiment_name)
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name = Path(data_path).stem
            self.output_dir = self.exp_dir / f'eval_{data_name}_{timestamp}'
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ROMAN SPACE TELESCOPE - COMPREHENSIVE EVALUATION")
        print("=" * 80)
        print(f"Experiment:  {experiment_name}")
        print(f"Model Path:  {self.model_path}")
        print(f"Data Path:   {data_path}")
        print(f"Output Dir:  {self.output_dir}")
        print(f"Device:      {self.device}")
        
        self.model, self.config, self.checkpoint = self._load_model()
        self.data_dict = self._load_data(data_path)
        
        self.flux = self.data_dict['norm_flux']
        self.raw_flux = self.data_dict['raw_flux']
        self.delta_t = self.data_dict['delta_t']
        self.y = self.data_dict['y']
        self.lengths = self.data_dict['lengths']
        self.timestamps = self.data_dict['timestamps']
        self.params = self.data_dict['params']
        self.norm_median = self.data_dict['norm_median']
        self.norm_iqr = self.data_dict['norm_iqr']
        
        print("\nRunning inference on test set...")
        self.probs, self.preds, self.confs = self._run_inference()
        
        self.metrics = self._compute_metrics()

    def _find_model(self, exp_name: str) -> Tuple[Path, Path]:
        """Find best_model.pt for experiment."""
        search_roots = [Path('../results'), Path('results'), Path('.')]
        candidates = []
        
        for root in search_roots:
            if root.exists():
                candidates.extend(list(root.glob(f"*{exp_name}*")))
        
        if not candidates:
            if Path(exp_name).exists() and Path(exp_name).is_dir():
                exp_dir = Path(exp_name)
            else:
                raise FileNotFoundError(
                    f"No experiment found matching '{exp_name}'\n"
                    f"Searched in: {search_roots}"
                )
        else:
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
        
        try:
            valid_keys = set(ModelConfig.__annotations__.keys())
            config_dict = checkpoint.get('config', {})
            clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            config = ModelConfig(**clean_conf)
        except NameError:
            config = ModelConfig() 
        
        model = RomanMicrolensingGRU(config).to(self.device)
        
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        clean_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        
        n_params = count_parameters(model)
        print(f"  Model: RomanMicrolensingGRU")
        print(f"  Parameters: {n_params:,}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        print(f"  Config: d_model={config.d_model}, n_layers={config.n_layers}")
        
        return model, config, checkpoint

    def _load_data(self, path):
        """Load and preprocess test data, including structured params."""
        print(f"\nLoading data from {path}...")
        data = load_compat(path)
        
        # --- Core Data Extraction ---
        raw_flux = data['flux'].astype(np.float32)
        if raw_flux.ndim == 3:
            raw_flux = raw_flux.squeeze(-1)
        
        y = data['labels'].astype(np.int64)
        
        delta_t = data.get('delta_t', np.ones_like(raw_flux, dtype=np.float32))
        
        timestamps = data.get('timestamps', np.tile(np.linspace(0, MISSION_DURATION_DAYS, raw_flux.shape[1]), (len(raw_flux), 1)).astype(np.float32))
            
        lengths = data.get('lengths', np.maximum((raw_flux != 0).sum(axis=1), 1).astype(np.int64))
        
        # --- Parameter Extraction Fix (Looking for structured arrays) ---
        params: Dict[str, Any] = {}
        for ptype in ['flat', 'pspl', 'binary']:
            key = f'params_{ptype}'
            param_array = data.get(key, data.get(ptype, None))
            # Need to filter the array to only include the events of the correct type
            if param_array is not None and param_array.dtype.fields is not None:
                # Find indices in the full array 'y' that correspond to this ptype
                ptype_idx = CLASS_NAMES.index(ptype.capitalize())
                original_indices = np.where(y == ptype_idx)[0]
                
                # Check if the param_array is the same length as the number of events of that type
                if len(param_array) == len(original_indices):
                    # We can use the param_array directly as it corresponds to the correct subset
                    params[ptype] = [dict(zip(param_array.dtype.names, row)) for row in param_array]
                else:
                    # Fallback/Debug: The param array might be the full set of parameters
                    # We will rely on _plot_diagnostic to handle the length check
                    # For safety, let's skip if the lengths are drastically mismatched
                    if len(param_array) > 0:
                        params[ptype] = [dict(zip(param_array.dtype.names, row)) for row in param_array]
        
        if not params:
            print("  Physical parameters (params_pspl, params_binary, etc.) not found.")

            
        # --- Subsample if requested ---
        if self.n_samples is not None and len(raw_flux) > self.n_samples:
            n = min(self.n_samples, len(raw_flux))
            indices = np.random.choice(len(raw_flux), n, replace=False)
            raw_flux = raw_flux[indices]
            y = y[indices]
            delta_t = delta_t[indices]
            timestamps = timestamps[indices]
            lengths = lengths[indices]
            # NOTE: Param dict is NOT subsampled here. It is handled in _plot_diagnostic based on the filtered 'y'

        
        print(f"  Loaded {len(raw_flux):,} samples")
        print(f"  Classes: {np.unique(y)}")
        
        # --- Flux Normalization ---
        print("\nNormalizing flux...")
        if 'normalization_stats' in self.checkpoint:
            stats = self.checkpoint['normalization_stats']
            median = stats.get('median', 0.0)
            iqr = stats.get('iqr', 1.0)
            print(f"  Computed from checkpoint: median={median:.4f}, iqr={iqr:.4f}")
        else:
            valid_mask = raw_flux != 0
            if valid_mask.sum() == 0:
                warnings.warn("All flux values are zero. Skipping normalization.")
                median = 0.0
                iqr = 1.0
            else:
                median = np.median(raw_flux[valid_mask])
                q75, q25 = np.percentile(raw_flux[valid_mask], [75, 25])
                iqr = q75 - q25
                iqr = max(iqr, 1e-6)
            print(f"  Computed from data: median={median:.4f}, iqr={iqr:.4f}")
            
        norm_flux = (raw_flux - median) / iqr
        norm_flux[raw_flux == 0] = 0.0 # Re-apply mask

        return {
            'norm_flux': norm_flux,
            'raw_flux': raw_flux,
            'delta_t': delta_t,
            'y': y,
            'lengths': lengths,
            'timestamps': timestamps,
            'params': params,
            'norm_median': median,
            'norm_iqr': iqr
        }

    # =========================================================================
    # CORE LOGIC
    # =========================================================================
    
    def _run_inference(self):
        """Run inference on test set."""
        n = len(self.flux)
        all_probs = []
        
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, n, self.batch_size), desc="Inference"):
                batch_end = min(i + self.batch_size, n)
                
                flux_batch = torch.tensor(self.flux[i:batch_end], dtype=torch.float32, device=self.device)
                dt_batch = torch.tensor(self.delta_t[i:batch_end], dtype=torch.float32, device=self.device)
                len_batch = torch.tensor(self.lengths[i:batch_end], dtype=torch.long, device=self.device)
                
                logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                
                probs_tensor = torch.softmax(logits, dim=-1)

                if probs_tensor.ndim == 3:
                    probs = probs_tensor[:, -1, :]
                elif probs_tensor.ndim == 2:
                    probs = probs_tensor
                else:
                    raise ValueError(f"Unexpected 'probs' tensor dimension: {probs_tensor.ndim}")
                
                probs = probs.cpu().numpy()
                all_probs.append(probs)
                
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs

    def _compute_metrics(self):
        """Compute classification metrics."""
        print("\nComputing metrics...")
        
        self.preds = self.preds[:len(self.y)]
        self.probs = self.probs[:len(self.y)]

        metrics = {
            'accuracy': accuracy_score(self.y, self.preds),
            'precision_macro': precision_score(self.y, self.preds, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y, self.preds, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y, self.preds, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for i, name in enumerate(CLASS_NAMES):
            metrics[f'f1_{name}'] = f1_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            
        try:
            auc_scores = roc_auc_score(self.y, self.probs, average=None, multi_class='ovr')
            metrics['auc_macro'] = float(np.mean(auc_scores))
            for i, name in enumerate(CLASS_NAMES):
                metrics[f'auc_{name}'] = float(auc_scores[i])
        except ValueError:
             print("\n[WARNING] Cannot compute AUC: Only one class present in test labels.")

        print("\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1:        {metrics['f1_macro']:.4f}")
        if 'auc_macro' in metrics:
            print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
        
        with open(self.output_dir / 'metrics.json', 'w') as f:
            # Convert all NumPy types to native Python types
            serializable_metrics = {k: float(v) if isinstance(v, np.number) else v for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=2)
            
        return metrics

    def run_all_analysis(self):
        """Run all evaluation analyses and plots."""
        
        print("\nRunning full analysis and generating plots...")
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration_curve() 
        
        if self.n_example_grid_per_type > 0:
            self.plot_example_grid()
            
        self.plot_parameter_diagnostics()
        
        if self.run_early_detection:
            self.run_early_detection_analysis()
            
        if self.n_evolution_per_type > 0:
            self.plot_evolution_examples()
            
        self.plot_temporal_bias()
        
        print(f"\nEvaluation complete. Results saved to: {self.output_dir}")

    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================
    
    def _raw_flux_to_relative_mag(self, raw_flux: np.ndarray) -> np.ndarray:
        """Converts raw flux to relative magnitude (brighter is up)."""
        # Relative Mag = 2.5 * log10(F_median / F)
        # We plot the NEGATIVE of this value to keep brighter light curves at the top.
        
        # Avoid log(0) and potential division by zero by replacing 0 with a small epsilon
        epsilon = 1e-10 
        flux = np.where(raw_flux == 0, epsilon, raw_flux)

        # Relative magnitude: M = -2.5 * log10(F / F_median)
        # This is equivalent to plotting: 2.5 * log10(F_median / F)
        relative_mag = 2.5 * np.log10(self.norm_median / flux)

        # Plot NEGATIVE relative mag to keep brighter/peak flux UP
        return -relative_mag
    
    def plot_example_grid(self):
        """Plot a 3x4 grid of relative magnitude light curves (4 per class, scatter plot, no padding, inverted y-axis)."""
        print(f"\nGenerating {self.n_example_grid_per_type} example grid plots per class...")
        
        n_rows = 3
        n_cols = self.n_example_grid_per_type
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), 
                                 sharex=False, sharey=False)
        # Handle case where only 1 row or 1 column exists (e.g., if n_example_grid_per_type=1)
        if n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes)

        fig.suptitle('Example Simulated Light Curves (Relative Magnitude)', fontsize=16)
        
        for i, cls_name in enumerate(CLASS_NAMES):
            candidates = np.where(self.y == i)[0]
            if len(candidates) < n_cols:
                selection = candidates
            else:
                selection = np.random.choice(candidates, n_cols, replace=False)
            
            for j, idx in enumerate(selection):
                ax = axes[i, j]
                length = self.lengths[idx]
                
                # --- Magnitude Conversion and Slicing ---
                time_axis = self.timestamps[idx, :length]
                raw_flux = self.raw_flux[idx, :length]
                # Convert only the non-padded, actual data points
                mag_data = self._raw_flux_to_relative_mag(raw_flux)
                
                # Scatter plot for discrete data points, excluding pad values
                ax.scatter(time_axis, mag_data, color=COLORS[i], s=5, linewidths=0)
                
                # Invert Y-axis for standard astronomical convention (fainter/higher magnitude number is up)
                # Since we plotted -Relative_Mag, the brighter events (lower mag number) are UP, so we DON'T invert.
                # If we plot Relative_Mag (brighter is positive), we DO invert.
                # Let's stick to the convention where smaller mag number is UP (brighter).
                # Since the peak is now at the max positive value, we do NOT invert.
                
                # ax.invert_yaxis() # DO NOT INVERT Y-AXIS - MAGNITUDE IS ALREADY INVERTED
                
                ax.set_title(f'True: {cls_name}', fontsize=10)
                
                if i == n_rows - 1:
                    ax.set_xlabel('Time (Days)', fontsize=8)
                if j == 0:
                    ax.set_ylabel('-Relative Magnitude', fontsize=8) # Label reflects the inverted axis
                
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Add a line at 0 for median flux/mag
                ax.axhline(0, color='k', linestyle=':', alpha=0.5, linewidth=1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(self.output_dir / 'example_data_grid.png')
        plt.close()
        print("  Generated Example Data Grid.")


    def plot_confusion_matrix(self):
        """Plot and save the normalized confusion matrix."""
        cm = confusion_matrix(self.y, self.preds, normalize='true')
        
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
        print("  Generated Confusion Matrix.")

    def plot_roc_curves(self):
        """Plot and save the ROC curve for each class."""
        plt.figure(figsize=(8, 6))
        
        for i, class_name in enumerate(CLASS_NAMES):
            y_true_binary = (self.y == i).astype(int)
            y_score = self.probs[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = roc_auc_score(y_true_binary, y_score)
                
                plt.plot(
                    fpr, tpr, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})', 
                    color=COLORS[i], 
                    linewidth=2
                )
            except ValueError:
                pass

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png')
        plt.close()
        print("  Generated ROC Curves.")

    def plot_calibration_curve(self):
        """Plot and save the reliability diagram (calibration curve)."""
        
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Aggregate all probabilities and true labels (one-hot)
        y_prob = self.probs.ravel()
        y_true_one_hot = np.eye(len(CLASS_NAMES))[self.y].ravel()
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_one_hot, y_prob, n_bins=10
        )

        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Model Calibration Curve (Reliability Diagram)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curve.png')
        plt.close()
        print("  Generated Calibration Curve.")

    def plot_parameter_diagnostics(self):
        """Plot diagnostic metrics against physical parameters (if available)."""
        if not self.params:
            print("  Physical parameters unavailable for parameter diagnostics.")
            return

        print("\nRunning parameter diagnostics...")
        
        try:
            # These calls now handle their own success/failure prints
            self._plot_binary_u0_accuracy() 
            self._plot_binary_q_accuracy()
            self._plot_pspl_tE_accuracy()
        except Exception as e:
            print(f"  Error in parameter diagnostics: {e}")

    def _plot_diagnostic(self, data_type, param_key, param_label, class_idx, class_name):
        """Generalized plotting function for parameter diagnostics."""
        
        if data_type not in self.params:
            return

        try:
            param_list = self.params[data_type]
            
            class_mask = self.y == class_idx
            if class_mask.sum() == 0: return

            class_preds = self.preds[class_mask]
            
            if len(param_list) != class_mask.sum():
                 warnings.warn(f"Mismatch in counts for {data_type}: params={len(param_list)}, y_true={class_mask.sum()}. Skipping plot.")
                 print(f"  [DEBUG] Skipping {class_name} {param_key} plot due to data mismatch: {len(param_list)} params vs {class_mask.sum()} samples in the true class.")
                 return

            param_vals = np.array([p.get(param_key, np.nan) for p in param_list])
            class_correct = (class_preds == class_idx).astype(int)

            # Filter out NaN or zero values, as is required for log-plotting and meaningful analysis
            valid = ~np.isnan(param_vals) & (param_vals > 0)
            
            # --- CRITICAL DEBUGGING PRINT ---
            if valid.sum() == 0:
                print(f"  [DEBUG] Skipping {class_name} {param_key} plot: {class_mask.sum()} samples found, but 0 valid parameter values (>0 and not NaN) remain after filtering.")
            # --------------------------------
            
            param_vals = param_vals[valid]
            class_correct = class_correct[valid]

            if len(param_vals) == 0:
                return # Final exit if no valid data remains for plotting

            # Plotting code starts here (only if data is valid)
            log_param = np.log10(param_vals)
            bins = np.linspace(log_param.min(), log_param.max(), 10)
            
            accs, centers, counts = [], [], []
            for i in range(len(bins) - 1):
                mask = (log_param >= bins[i]) & (log_param < bins[i+1])
                if mask.sum() > 5: # Require minimum 5 samples per bin
                    accs.append(class_correct[mask].mean())
                    centers.append((bins[i] + bins[i+1]) / 2)
                    counts.append(mask.sum())
            
            if not accs: return

            plt.figure(figsize=(7, 5))
            
            plt.plot(10**np.array(centers), accs, 'o-', color=COLORS[class_idx], label=f'{class_name} Accuracy')
            plt.ylabel(f"Accuracy ({class_name} Class)")
            plt.ylim([0, 1.05])
            
            ax2 = plt.gca().twinx()
            bar_width = (10**bins[1]-10**bins[0])*0.8
            ax2.bar(10**np.array(centers), counts, width=bar_width, color='gray', alpha=0.3, label='Sample Count')
            ax2.set_ylabel("Count per Bin (Log Scale)", color='gray')
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', labelcolor='gray')
            
            plt.title(f"{class_name} Classification Accuracy vs. {param_label}")
            plt.xlabel(param_label)
            plt.xscale('log')
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            
            # Successful Save Print
            file_name = f'diag_{data_type}_{param_key}.png'
            plt.savefig(self.output_dir / file_name)
            plt.close()
            print(f"  Generated Parameter Diagnostic: {file_name}") 
            
        except Exception as e:
            print(f"  Error plotting {data_type} {param_key} accuracy: {e}")

    def _plot_binary_u0_accuracy(self):
        """Plot accuracy vs. source-lens separation (u0) for binary events. (Generates diag_binary_u0.png)"""
        self._plot_diagnostic(
            data_type='binary', param_key='u0', param_label='Minimum Source-Lens Separation ($u_0$)', 
            class_idx=2, class_name='Binary'
        )

    def _plot_binary_q_accuracy(self):
        """Plot accuracy vs. mass ratio (q) for binary events."""
        self._plot_diagnostic(
            data_type='binary', param_key='q', param_label='Mass Ratio ($q = m_2/m_1$)', 
            class_idx=2, class_name='Binary'
        )

    def _plot_pspl_tE_accuracy(self):
        """Plot accuracy vs. Einstein radius crossing time (tE) for PSPL events."""
        self._plot_diagnostic(
            data_type='pspl', param_key='tE', param_label='Einstein Radius Crossing Time ($t_E$) [Days]', 
            class_idx=1, class_name='PSPL'
        )

    def plot_temporal_bias(self):
        """Analyze temporal bias using Kolmogorov-Smirnov test."""
        
        y_true = self.y
        correct_mask = self.preds == y_true
        
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            print("  Cannot run temporal bias analysis: Zero correct or incorrect predictions.")
            return

        time_indices_correct = self.lengths[correct_mask] - 1
        time_indices_incorrect = self.lengths[~correct_mask] - 1
        
        t_correct = np.array([self.timestamps[i, idx] for i, idx in zip(np.where(correct_mask)[0], time_indices_correct)])
        t_incorrect = np.array([self.timestamps[i, idx] for i, idx in zip(np.where(~correct_mask)[0], time_indices_incorrect)])
        
        ks_stat, p_value = ks_2samp(t_correct, t_incorrect)
        
        print(f"\nTemporal Bias Diagnostics (KS Test on Last Timestep):")
        print(f"  KS Statistic: {ks_stat:.4f}")
        print(f"  P-Value: {p_value:.4e}")
        
        if p_value < 0.05:
            print("  Temporal bias detected (p < 0.05): Last data time for correct vs. incorrect events are significantly different.")
        else:
            print("  No significant temporal bias detected (p >= 0.05).")
        
        plt.figure(figsize=(7, 5))
        sns.histplot(t_correct, bins=30, kde=True, stat="density", label='Correct Predictions', color='green', alpha=0.5)
        sns.histplot(t_incorrect, bins=30, kde=True, stat="density", label='Incorrect Predictions', color='red', alpha=0.5)
        plt.title("Distribution of Last Data Timestep for Correct vs. Incorrect Predictions")
        plt.xlabel("Time (Days)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_bias_last_timestep.png')
        plt.close()

    def plot_evolution_examples(self):
        """Plot the 3-panel probability evolution for sample events."""
        print(f"\nGenerating {self.n_evolution_per_type} evolution plots per class...")
        
        self.model.eval()
        
        for i, cls_name in enumerate(CLASS_NAMES):
            # Find correctly classified events
            candidates = np.where((self.y == i) & (self.preds == i))[0]
            if len(candidates) == 0:
                print(f"  No correct predictions for {cls_name}")
                continue
            
            n_plot = min(len(candidates), self.n_evolution_per_type)
            selection = np.random.choice(candidates, n_plot, replace=False)
            
            for idx in tqdm(selection, desc=f"  {cls_name}"):
                self._plot_single_evolution(idx, cls_name, i)

    def _plot_single_evolution(self, idx: int, cls_name: str, cls_idx: int):
        """Plot single event evolution (3-panel: Mag (scatter), Probs, Confidence)."""
        
        # --- 1. Run time-series inference ---
        f = torch.tensor(self.flux[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        d = torch.tensor(self.delta_t[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        length = self.lengths[idx]
        
        probs_seq = []
        with torch.no_grad():
            for t in range(1, length + 1):
                f_trunc = f[:, :t]
                d_trunc = d[:, :t]
                l_trunc = torch.tensor([t], dtype=torch.long, device=self.device)
                
                logits_final = self.model(f_trunc, d_trunc, lengths=l_trunc)
                
                probs_tensor = torch.softmax(logits_final, dim=-1)
                
                if probs_tensor.ndim == 3:
                    probs = probs_tensor[:, -1, :]
                elif probs_tensor.ndim == 2:
                    probs = probs_tensor
                else:
                    raise ValueError(f"Unexpected tensor dimension in evolution plot: {probs_tensor.ndim}")
                    
                probs = probs.squeeze(0).cpu().numpy()
                probs_seq.append(probs)
        
        if not probs_seq: return
        probs_full = np.array(probs_seq)
        time_axis = self.timestamps[idx, :length]

        # --- 2. Plotting (3 Panels) ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Probability Evolution (True: {cls_name}, Predicted: {CLASS_NAMES[self.preds[idx]]})', fontsize=14)

        # Panel 1: Light Curve (Magnitude)
        ax1 = axes[0]
        raw_flux = self.raw_flux[idx, :length]
        
        # --- Magnitude Conversion and Zero Plotting Fix ---
        mag_data = self._raw_flux_to_relative_mag(raw_flux)
        
        # FIX: Use ONLY scatter plot, which implicitly handles no plotting of the padded zeros (since they are filtered in _raw_flux_to_relative_mag)
        ax1.scatter(time_axis, mag_data, color='gray', s=5, linewidths=0, label='Light Curve')
        
        ax1.set_ylabel('-Relative Magnitude')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.axhline(0, color='k', linestyle=':', alpha=0.5, linewidth=1) # Line for median flux/mag

        # Panel 2: Class Probability Evolution
        ax2 = axes[1]
        for i, name in enumerate(CLASS_NAMES):
            ax2.plot(time_axis, probs_full[:, i], label=f'P({name})', color=COLORS[i], linewidth=2)
            
        ax2.axhline(0.5, color='k', linestyle=':', alpha=0.7, linewidth=1)
        ax2.set_ylabel('Class Probability')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', ncol=3, fontsize=9)

        # Panel 3: Classification Confidence
        ax3 = axes[2]
        confidence = probs_full.max(axis=1)
        
        ax3.plot(time_axis, confidence, label='Max Confidence', color='black', linewidth=2)
        ax3.fill_between(time_axis, 0, confidence, color=COLORS[self.preds[idx]], alpha=0.3)
        
        ax3.axhline(0.5, color='k', linestyle=':', alpha=0.7, linewidth=1)
        ax3.set_ylabel('Confidence (Max P)')
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlabel('Time (Days)')
        ax3.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(self.output_dir / f'evolution_{cls_name.lower()}_{idx}.png')
        plt.close()

    def run_early_detection_analysis(self):
        """Run early detection analysis by computing metrics at sequence length quantiles."""
        print("\nRunning early detection analysis (re-running inference)...")
        
        time_fractions = np.linspace(0.2, 1.0, 5)
        results = []
        
        max_len = self.flux.shape[1]
        
        for frac in time_fractions:
            print(f"  Inference at {frac*100:.0f}% of full length...")
            
            new_lengths = np.clip((self.lengths * frac).astype(np.int64), a_min=1, a_max=max_len)
            current_max_len = new_lengths.max()
            
            n = len(self.flux)
            all_probs = []
            
            with torch.no_grad():
                for i in tqdm(range(0, n, self.batch_size), desc="    Inference"):
                    batch_end = min(i + self.batch_size, n)
                    
                    flux_batch = torch.tensor(self.flux[i:batch_end, :current_max_len], dtype=torch.float32, device=self.device)
                    dt_batch = torch.tensor(self.delta_t[i:batch_end, :current_max_len], dtype=torch.float32, device=self.device)
                    
                    len_batch = torch.tensor(new_lengths[i:batch_end], dtype=torch.long, device=self.device)
                    
                    logits = self.model(flux_batch, dt_batch, lengths=len_batch)

                    probs_tensor = torch.softmax(logits, dim=-1)

                    if probs_tensor.ndim == 3:
                        probs = probs_tensor[:, -1, :]
                    elif probs_tensor.ndim == 2:
                        probs = probs_tensor
                    else:
                        raise ValueError(f"Unexpected 'probs' tensor dimension: {probs_tensor.ndim}")
                    
                    probs = probs.cpu().numpy()
                    all_probs.append(probs)

            probs_frac = np.concatenate(all_probs, axis=0)
            preds_frac = probs_frac.argmax(axis=1)
            
            acc = accuracy_score(self.y, preds_frac)
            f1 = f1_score(self.y, preds_frac, average='macro', zero_division=0)
            
            results.append({
                'fraction': float(frac),
                'max_sequence_length': int(current_max_len), 
                'accuracy': float(acc),
                'f1_macro': float(f1)
            })

        # --- Plotting Early Detection Curve ---
        fractions = [r['fraction'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_macro'] for r in results]
        
        plt.figure(figsize=(7, 5))
        plt.plot(fractions, accuracies, 'o-', label='Accuracy', color='blue')
        plt.plot(fractions, f1_scores, 's--', label='F1 (macro)', color='red')
        
        plt.title('Early Detection Performance vs. Sequence Length')
        plt.xlabel('Fraction of Full Sequence Length')
        plt.ylabel('Metric Score')
        plt.ylim([0.0, 1.05])
        plt.xticks(fractions, [f'{f:.0%}' for f in fractions])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'early_detection_curve.png')
        plt.close()
        print("  Generated Early Detection Curve.")
        
        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Comprehensive Evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--experiment_name', required=True, 
                       help="Name of experiment to evaluate")
    parser.add_argument('--data', required=True, 
                       help="Path to test dataset (.npz or .h5)")
    parser.add_argument('--output_dir', default=None, 
                       help="Optional custom output directory")
    
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--n_samples', type=int, default=None,
                       help="Subsample test set for speed")
                       
    parser.add_argument('--device', default='cuda',
                       help="Device: cuda or cpu")
                       
    parser.add_argument('--early_detection', action='store_true', 
                       help='Run early detection analysis')
    
    parser.add_argument('--n_evolution_per_type', type=int, default=10,
                       help="Number of 3-panel evolution plots to generate per class.")
                       
    parser.add_argument('--n_example_grid_per_type', type=int, default=4,
                       help="Number of light curves per class for the example data grid (e.g., 4 makes a 3x4 grid).")
    
    args = parser.parse_args()
    
    evaluator = RomanEvaluator(
        experiment_name=args.experiment_name,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples, 
        early_detection=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type,
        n_example_grid_per_type=args.n_example_grid_per_type
    )
    
    evaluator.run_all_analysis()
