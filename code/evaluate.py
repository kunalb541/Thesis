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
    
    # Import the model and config classes
    from model import RomanMicrolensingClassifier as RomanMicrolensingGRU, ModelConfig 
    
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
        experiment_name: str, 
        data_path: str, 
        output_dir: Optional[str] = None, 
        device: str = 'cuda', 
        batch_size: int = 128, 
        n_samples: Optional[int] = None,
        early_detection: bool = False, 
        n_evolution_per_type: int = 0
    ):
        """
        Args:
            experiment_name: Name of experiment to evaluate
            data_path: Path to test dataset (.npz)
            output_dir: Optional custom output directory
            device: 'cuda' or 'cpu'
            batch_size: Batch size for inference
            n_samples: Subsample test set (for speed)
            early_detection: Run early detection analysis
            n_evolution_per_type: Number of evolution plots per class
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        
        # Locate and load model
        self.model_path, self.exp_dir = self._find_model(experiment_name)
        
        # Setup output directory
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
        model = RomanMicrolensingGRU(config).to(self.device)
        
        # Load weights
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
        else:
            print("  delta_t missing, generating linear grid")
            # Create a dummy delta_t if missing (essential for the model input)
            delta_t = np.ones_like(raw_flux, dtype=np.float32)
        
        # Timestamps
        if 'timestamps' in data:
            timestamps = data['timestamps'].astype(np.float32)
        else:
            print("  timestamps missing, generating linear grid")
            timestamps = np.linspace(0, MISSION_DURATION_DAYS, raw_flux.shape[1])
            timestamps = np.tile(timestamps, (len(raw_flux), 1))
            
        # Lengths
        if 'lengths' in data:
            lengths = data['lengths'].astype(np.int64)
        else:
            # Dynamically determine lengths from non-zero flux values
            lengths = np.maximum((raw_flux != 0).sum(axis=1), 1)
        
        # Params (optional, for analysis)
        if 'params' in data:
            params = data['params']
        else:
            params = None
            
        # Subsample if requested
        if self.n_samples is not None:
            n = min(self.n_samples, len(raw_flux))
            indices = np.random.choice(len(raw_flux), n, replace=False)
            raw_flux = raw_flux[indices]
            y = y[indices]
            delta_t = delta_t[indices]
            timestamps = timestamps[indices]
            lengths = lengths[indices]
            if params is not None:
                params = params[indices]
        
        print(f"  Loaded {len(raw_flux):,} samples")
        print(f"  Classes: {np.unique(y)}")
        
        # Normalize flux (matching train.py)
        print("\nNormalizing flux...")
        # Use checkpoint stats if available
        if 'normalization_stats' in self.checkpoint:
            stats = self.checkpoint['normalization_stats']
            median = stats['median']
            iqr = stats['iqr']
            print(f"  Computed from checkpoint: median={median:.4f}, iqr={iqr:.4f}")
        else:
            # Fallback to computing robust stats if not in checkpoint
            valid_mask = raw_flux != 0
            if valid_mask.sum() == 0:
                warnings.warn("All flux values are zero. Skipping normalization.")
                median = 0.0
                iqr = 1.0
            else:
                median = np.median(raw_flux[valid_mask])
                q75, q25 = np.percentile(raw_flux[valid_mask], [75, 25])
                iqr = q75 - q25
                iqr = max(iqr, 1e-6) # Ensure non-zero IQR
            print(f"  Computed from data: median={median:.4f}, iqr={iqr:.4f}")
            
        norm_flux = (raw_flux - median) / iqr
        
        # Apply mask
        norm_flux[raw_flux == 0] = 0.0
        
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

    def _run_inference(self):
        """
        Run inference on test set.
        FIX: Handles model output as a Tensor (logits), converts to probabilities,
        and applies the robust dimension check.
        """
        n = len(self.flux)
        all_probs = []
        
        # Ensure model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, n, self.batch_size), desc="Inference"):
                batch_end = min(i + self.batch_size, n)
                
                # Prepare batches
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
                
                # Forward pass: model returns logits (a Tensor)
                logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                
                # --- START OF FIX: Robust Dimension Check ---
                
                # 1. Convert logits to probabilities
                probs_tensor = torch.softmax(logits, dim=-1)

                # 2. Check and handle tensor dimensions
                if probs_tensor.ndim == 3:
                    # TENSOR IS [Batch, Sequence, Classes] -> Select last timestep
                    probs = probs_tensor[:, -1, :]
                elif probs_tensor.ndim == 2:
                    # TENSOR IS [Batch, Classes] -> Use directly
                    probs = probs_tensor
                else:
                    raise ValueError(
                        f"Unexpected 'probs' tensor dimension: {probs_tensor.ndim} "
                        f"(Expected 2 or 3)"
                    )
                
                # Convert to numpy and append
                probs = probs.cpu().numpy()
                all_probs.append(probs)
                # --- END OF FIX ---
                
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs

    def _compute_metrics(self):
        """Compute classification metrics."""
        print("\nComputing metrics...")
        
        # Ensure predictions are aligned with ground truth
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
            metrics[f'precision_{name}'] = precision_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            metrics[f'recall_{name}'] = recall_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            metrics[f'f1_{name}'] = f1_score(
                self.y, self.preds, labels=[i], average='macro', zero_division=0
            )
            
        # AUC Score (requires one-hot encoding or probability selection)
        try:
            auc_scores = roc_auc_score(self.y, self.probs, average=None, multi_class='ovr')
            metrics['auc_macro'] = np.mean(auc_scores)
            for i, name in enumerate(CLASS_NAMES):
                metrics[f'auc_{name}'] = auc_scores[i]
        except ValueError as e:
            # Handle case where only one class is present in y_true
            if "Only one class present" in str(e):
                 print("\n[WARNING] Cannot compute AUC: Only one class present in test labels.")
            else:
                 raise e

        print("\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1:        {metrics['f1_macro']:.4f}")
        if 'auc_macro' in metrics:
            print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
        
        # Save metrics to JSON
        with open(self.output_dir / 'metrics.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = {k: float(v) if isinstance(v, np.number) else v for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=2)
            
        return metrics

    def run_all_analysis(self):
        """Run all evaluation analyses and plots."""
        
        print("\nRunning full analysis and generating plots...")
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration_curve()
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
                # Skip if only one class is present in the true labels for this binary separation
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
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        y_prob = self.probs.ravel()
        y_true_one_hot = np.eye(len(CLASS_NAMES))[self.y].ravel()
        
        # Use 10 bins for the calibration curve
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
        if self.params is None:
            print("  Physical parameters unavailable for parameter diagnostics.")
            return

        print("\nRunning parameter diagnostics...")
        
        try:
            # --- Diagnostic 1: Binary Events vs. u0 ---
            self._plot_binary_u0_accuracy()
            
            # --- Diagnostic 2: Binary Events vs. q ---
            self._plot_binary_q_accuracy()
            
            # --- Diagnostic 3: PSPL Events vs. tE ---
            self._plot_pspl_tE_accuracy()
            
            print("  Generated Parameter Diagnostics.")
        except Exception as e:
            print(f"  Error in parameter diagnostics: {e}")

    def _plot_binary_u0_accuracy(self):
        """Plot accuracy vs. source-lens separation (u0) for binary events."""
        if 'binary' not in self.params:
            return

        try:
            u0_vals = np.array([p.get('u0', p.get('u_0', np.nan)) for p in self.params['binary']])
            
            bin_mask = self.y == 2
            if bin_mask.sum() == 0: return

            bin_preds = self.preds[bin_mask]
            bin_correct = (bin_preds == 2).astype(int)

            min_len = min(len(u0_vals), len(bin_correct))
            u0_vals = u0_vals[:min_len]
            bin_correct = bin_correct[:min_len]

            valid = ~np.isnan(u0_vals) & (u0_vals > 0)
            u0_vals = u0_vals[valid]
            bin_correct = bin_correct[valid]

            if len(u0_vals) == 0: return

            # Bin by u0 on a log scale
            log_u0 = np.log10(u0_vals)
            bins = np.linspace(log_u0.min(), log_u0.max(), 10)
            
            accs, centers, counts = [], [], []
            for i in range(len(bins) - 1):
                mask = (log_u0 >= bins[i]) & (log_u0 < bins[i+1])
                if mask.sum() > 0:
                    accs.append(bin_correct[mask].mean())
                    centers.append((bins[i] + bins[i+1]) / 2)
                    counts.append(mask.sum())
            
            if not accs: return

            plt.figure(figsize=(7, 5))
            plt.plot(10**np.array(centers), accs, 'o-', color=COLORS[2])
            
            # Add bar plot for counts
            ax2 = plt.gca().twinx()
            ax2.bar(10**np.array(centers), counts, width=(10**bins[1]-10**bins[0])*0.8, color='gray', alpha=0.3)
            ax2.set_ylabel("Count per Bin (Log Scale)", color='gray')
            ax2.set_yscale('log')
            
            plt.title("Binary Classification Accuracy vs. u₀")
            plt.xlabel("Minimum Source-Lens Separation (u₀)")
            plt.ylabel(f"Accuracy (Binary Class)")
            plt.xscale('log')
            plt.ylim([0, 1.05])
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(self.output_dir / 'diag_binary_u0.png')
            plt.close()
        except Exception as e:
            print(f"  Error plotting binary u0 accuracy: {e}")

    def _plot_binary_q_accuracy(self):
        """Plot accuracy vs. mass ratio (q) for binary events."""
        if 'binary' not in self.params:
            return

        try:
            q_vals = np.array([p.get('q', np.nan) for p in self.params['binary']])
            
            bin_mask = self.y == 2
            if bin_mask.sum() == 0: return

            bin_preds = self.preds[bin_mask]
            bin_correct = (bin_preds == 2).astype(int)

            min_len = min(len(q_vals), len(bin_correct))
            q_vals = q_vals[:min_len]
            bin_correct = bin_correct[:min_len]

            valid = ~np.isnan(q_vals) & (q_vals > 0)
            q_vals = q_vals[valid]
            bin_correct = bin_correct[valid]

            if len(q_vals) == 0: return

            # Bin by q on a log scale
            log_q = np.log10(q_vals)
            bins = np.linspace(log_q.min(), log_q.max(), 10)
            
            accs, centers, counts = [], [], []
            for i in range(len(bins) - 1):
                mask = (log_q >= bins[i]) & (log_q < bins[i+1])
                if mask.sum() > 0:
                    accs.append(bin_correct[mask].mean())
                    centers.append((bins[i] + bins[i+1]) / 2)
                    counts.append(mask.sum())
            
            if not accs: return

            plt.figure(figsize=(7, 5))
            plt.plot(10**np.array(centers), accs, 'o-', color=COLORS[2])
            
            # Add bar plot for counts
            ax2 = plt.gca().twinx()
            ax2.bar(10**np.array(centers), counts, width=(10**bins[1]-10**bins[0])*0.8, color='gray', alpha=0.3)
            ax2.set_ylabel("Count per Bin (Log Scale)", color='gray')
            ax2.set_yscale('log')
            
            plt.title("Binary Classification Accuracy vs. Mass Ratio (q)")
            plt.xlabel("Mass Ratio (q = m₂/m₁)")
            plt.ylabel(f"Accuracy (Binary Class)")
            plt.xscale('log')
            plt.ylim([0, 1.05])
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(self.output_dir / 'diag_binary_q.png')
            plt.close()
        except Exception as e:
            print(f"  Error plotting binary q accuracy: {e}")

    def _plot_pspl_tE_accuracy(self):
        """Plot accuracy vs. Einstein radius crossing time (tE) for PSPL events."""
        if 'pspl' not in self.params:
            return

        try:
            tE_vals = np.array([p.get('tE', np.nan) for p in self.params['pspl']])
            
            pspl_mask = self.y == 1
            if pspl_mask.sum() == 0: return

            pspl_preds = self.preds[pspl_mask]
            pspl_correct = (pspl_preds == 1).astype(int)

            min_len = min(len(tE_vals), len(pspl_correct))
            tE_vals = tE_vals[:min_len]
            pspl_correct = pspl_correct[:min_len]

            valid = ~np.isnan(tE_vals) & (tE_vals > 0)
            tE_vals = tE_vals[valid]
            pspl_correct = pspl_correct[valid]

            if len(tE_vals) == 0: return

            # Bin by tE on a log scale
            log_tE = np.log10(tE_vals)
            bins = np.linspace(log_tE.min(), log_tE.max(), 10)
            
            accs, centers, counts = [], [], []
            for i in range(len(bins) - 1):
                mask = (log_tE >= bins[i]) & (log_tE < bins[i+1])
                if mask.sum() > 0:
                    accs.append(pspl_correct[mask].mean())
                    centers.append((bins[i] + bins[i+1]) / 2)
                    counts.append(mask.sum())
            
            if not accs: return

            plt.figure(figsize=(7, 5))
            plt.plot(10**np.array(centers), accs, 'o-', color=COLORS[1])
            
            # Add bar plot for counts
            ax2 = plt.gca().twinx()
            ax2.bar(10**np.array(centers), counts, width=(10**bins[1]-10**bins[0])*0.8, color='gray', alpha=0.3)
            ax2.set_ylabel("Count per Bin (Log Scale)", color='gray')
            ax2.set_yscale('log')
            
            plt.title("PSPL Classification Accuracy vs. Einstein Time (t_E)")
            plt.xlabel("Einstein Radius Crossing Time (t_E) [Days]")
            plt.ylabel(f"Accuracy (PSPL Class)")
            plt.xscale('log')
            plt.ylim([0, 1.05])
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(self.output_dir / 'diag_pspl_tE.png')
            plt.close()
        except Exception as e:
            print(f"  Error plotting PSPL tE accuracy: {e}")

    def plot_temporal_bias(self):
        """Analyze temporal bias using Kolmogorov-Smirnov test."""
        
        y_true = self.y
        y_score = self.confs # Using confidence as score

        try:
            # 1. Compare confidence for correct vs incorrect
            correct_mask = self.preds == y_true
            
            if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
                print("  Cannot run temporal bias analysis: Zero correct or incorrect predictions.")
                return

            t_correct = self.timestamps[correct_mask]
            t_incorrect = self.timestamps[~correct_mask]
            
            # Use the time of the *last valid data point* for analysis
            time_indices_correct = self.lengths[correct_mask] - 1
            time_indices_incorrect = self.lengths[~correct_mask] - 1
            
            t_correct = np.array([t_correct[i, idx] for i, idx in enumerate(time_indices_correct)])
            t_incorrect = np.array([t_incorrect[i, idx] for i, idx in enumerate(time_indices_incorrect)])
            
            # KS Test: null hypothesis is that the two samples are drawn from the same distribution
            ks_stat, p_value = ks_2samp(t_correct, t_incorrect)
            
            print(f"\nTemporal Bias Diagnostics (KS Test on Last Timestep):")
            print(f"  KS Statistic: {ks_stat:.4f}")
            print(f"  P-Value: {p_value:.4e}")
            
            if p_value < 0.05:
                print("  Temporal bias detected (p < 0.05): Last data time for correct vs. incorrect events are significantly different.")
            else:
                print("  No significant temporal bias detected (p >= 0.05).")
            
            # Plot the distributions
            plt.figure(figsize=(7, 5))
            sns.histplot(t_correct, bins=30, kde=True, label='Correct Predictions', color='green', alpha=0.5)
            sns.histplot(t_incorrect, bins=30, kde=True, label='Incorrect Predictions', color='red', alpha=0.5)
            plt.title("Distribution of Last Data Timestep for Correct vs. Incorrect Predictions")
            plt.xlabel("Time (Days)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_bias_last_timestep.png')
            plt.close()
            
        except Exception as e:
            print(f" Error in temporal bias analysis: {e}")

    def plot_evolution_examples(self):
        """Plot probability evolution for sample events."""
        print(f"\nGenerating {self.n_evolution_per_type} evolution plots per class...")
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        for i, cls_name in enumerate(CLASS_NAMES):
            # Find correctly classified events
            candidates = np.where((self.y == i) & (self.preds == i))[0]
            if len(candidates) == 0:
                print(f"  No correct predictions for {cls_name}")
                continue
            
            # Random selection
            n_plot = min(len(candidates), self.n_evolution_per_type)
            selection = np.random.choice(candidates, n_plot, replace=False)
            
            for idx in tqdm(selection, desc=f"  {cls_name}"):
                self._plot_single_evolution(idx, cls_name, i)

    def _plot_single_evolution(self, idx: int, cls_name: str, cls_idx: int):
        """Plot single event evolution."""
        # Use full tensors for sequence prediction
        f = torch.tensor(self.flux[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        d = torch.tensor(self.delta_t[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long, device=self.device)
        
        length = self.lengths[idx]
        
        probs_seq = []
        with torch.no_grad():
            for t in range(1, length + 1):
                f_trunc = f[:, :t]
                d_trunc = d[:, :t]
                l_trunc = torch.tensor([t], dtype=torch.long, device=self.device)
                
                # Forward pass for truncated sequence
                logits_final = self.model(f_trunc, d_trunc, lengths=l_trunc)
                
                # --- START OF FIX: Robust Dimension Check for Evolution ---
                
                # Convert to probabilities
                probs_tensor = torch.softmax(logits_final, dim=-1)
                
                # Apply dimension fix: the model output for a truncated sequence 
                # might still return a 3D tensor if the last output is extracted 
                # outside the main forward pass (though it shouldn't for this architecture).
                if probs_tensor.ndim == 3:
                    probs = probs_tensor[:, -1, :]
                elif probs_tensor.ndim == 2:
                    probs = probs_tensor
                else:
                    raise ValueError(f"Unexpected tensor dimension in evolution plot: {probs_tensor.ndim}")
                    
                probs_seq.append(probs.squeeze(0).cpu().numpy())
                # --- END OF FIX ---
        
        if not probs_seq: return
        probs_full = np.array(probs_seq)
        time_axis = self.timestamps[idx, :length]
        
        plt.figure(figsize=(10, 6))
        
        # Plot raw flux in the background
        ax1 = plt.gca()
        ax1.plot(time_axis, self.raw_flux[idx, :length], 'o', color='lightgray', markersize=2, label='Raw Flux')
        ax1.set_ylabel('Raw Flux (mag or Jy)', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        ax1.set_xlim(time_axis.min(), time_axis.max())
        
        # Plot probabilities
        ax2 = ax1.twinx()
        for i, name in enumerate(CLASS_NAMES):
            ax2.plot(
                time_axis, 
                probs_full[:, i], 
                label=f'P({name})', 
                color=COLORS[i]
            )
            
        ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Predicted Probability')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y')

        plt.title(f'Probability Evolution for Event (True: {cls_name})')
        ax1.set_xlabel('Time (Days)')
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'evolution_{cls_name.lower()}_{idx}.png')
        plt.close()

    def run_early_detection_analysis(self):
        """
        Run early detection analysis by computing metrics at sequence length quantiles.
        Note: This is computationally expensive as it requires re-running inference.
        """
        print("\nRunning early detection analysis (re-running inference)...")
        
        # Define quantiles of sequence lengths to test (e.g., 25%, 50%, 75%, 100%)
        time_fractions = np.linspace(0.2, 1.0, 5) # Test 20%, 40%, 60%, 80%, 100%
        results = []
        
        max_len = self.flux.shape[1]
        
        for frac in time_fractions:
            print(f"  Inference at {frac*100:.0f}% of full length...")
            
            # --- Prepare truncated data ---
            new_lengths = np.clip((self.lengths * frac).astype(np.int64), a_min=1, a_max=max_len)
            
            # Only use up to the new max length
            current_max_len = new_lengths.max()
            
            n = len(self.flux)
            all_probs = []
            
            with torch.no_grad():
                for i in tqdm(range(0, n, self.batch_size), desc="    Inference"):
                    batch_end = min(i + self.batch_size, n)
                    
                    # Truncate flux and delta_t to the maximum length needed for this fraction
                    flux_batch = torch.tensor(
                        self.flux[i:batch_end, :current_max_len], 
                        dtype=torch.float32
                    , device=self.device)
                    
                    dt_batch = torch.tensor(
                        self.delta_t[i:batch_end, :current_max_len], 
                        dtype=torch.float32
                    , device=self.device)
                    
                    # Use the dynamically calculated lengths for this batch
                    len_batch = torch.tensor(
                        new_lengths[i:batch_end], 
                        dtype=torch.long
                    , device=self.device)
                    
                    # Forward pass: model returns logits (a Tensor)
                    logits = self.model(flux_batch, dt_batch, lengths=len_batch)

                    # --- Robust Dimension Check (Same fix as above) ---
                    
                    # 1. Convert logits to probabilities
                    probs_tensor = torch.softmax(logits, dim=-1)

                    # 2. Check and handle tensor dimensions
                    if probs_tensor.ndim == 3:
                        probs = probs_tensor[:, -1, :]
                    elif probs_tensor.ndim == 2:
                        probs = probs_tensor
                    else:
                        raise ValueError(f"Unexpected 'probs' tensor dimension: {probs_tensor.ndim}")
                    
                    probs = probs.cpu().numpy()
                    all_probs.append(probs)
                    # --- End Robust Dimension Check ---

            probs_frac = np.concatenate(all_probs, axis=0)
            preds_frac = probs_frac.argmax(axis=1)
            
            # Compute accuracy for this fraction
            acc = accuracy_score(self.y, preds_frac)
            f1 = f1_score(self.y, preds_frac, average='macro', zero_division=0)
            
            # --- START FIX: Cast numpy types to native Python types ---
            results.append({
                'fraction': float(frac),
                'max_sequence_length': int(current_max_len), 
                'accuracy': float(acc),
                'f1_macro': float(f1)
            })
            # --- END FIX ---


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
        
        # Save early detection results
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
    
    parser.add_argument('--n_evolution_per_type', type=int, default=5,
                       help="Number of evolution plots per class")
    
    args = parser.parse_args()
    
    evaluator = RomanEvaluator(
        experiment_name=args.experiment_name,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples, early_detection=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type
    )
    
    evaluator.run_all_analysis()
