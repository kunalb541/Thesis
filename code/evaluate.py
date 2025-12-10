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
    # Set current_dir to the directory of this script, handling environments 
    # where __file__ might not be defined (like interactive sessions)
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import CausalHybridModel, CausalConfig
    
    def count_parameters(model):
        """Returns the number of trainable parameters in a PyTorch model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
except NameError:
    # Fallback for environments without __file__
    current_dir = Path(os.getcwd())
    sys.path.insert(0, str(current_dir))
    try:
        from model import CausalHybridModel, CausalConfig
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError as e:
        print(f"\nCRITICAL ERROR: Could not import 'model.py'")
        print(f"Ensure model.py is in: {current_dir}")
        print(f"Error: {e}\n")
        sys.exit(1)
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'model.py'")
    print(f"Ensure model.py is in: {current_dir}")
    print(f"Error: {e}\n")
    sys.exit(1)

# =============================================================================
# PLOTTING SETUP
# =============================================================================
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
def compute_lengths_from_flux(flux: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Compute valid sequence lengths from padded flux."""
    valid_mask = (flux != pad_value)
    lengths = np.sum(valid_mask, axis=1).astype(np.int64)
    return np.maximum(lengths, 1)

# =============================================================================
# CORE EVALUATOR CLASS
# =============================================================================
class ComprehensiveEvaluator:
    """
    Complete evaluation suite for CausalHybridModel.
    Includes all diagnostic and visualization tools for Roman Telescope readiness.
    
    CRITICAL: Requires test data with pre-computed causal delta_t from simulate.py.
    """
    
    def __init__(self, model_path: str, data_path: str, output_dir: str, 
                 device: str = 'cuda', batch_size: int = 128, n_samples: Optional[int] = None,
                 run_early_detection: bool = False, n_evolution_per_type: int = 0): # <-- FIX: ADDED ARGUMENTS
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.run_early_detection = run_early_detection # <-- ADDED ATTRIBUTE
        self.n_evolution_per_type = n_evolution_per_type # <-- ADDED ATTRIBUTE
        
        # Setup output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f'eval_{self.timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ROMAN SPACE TELESCOPE - CAUSAL MODEL EVALUATION")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Device:    {self.device}")
        print(f"Output:    {self.output_dir}")
        if n_samples:
            print(f"Sampling:  {n_samples} events (subset)")
        
        # Load model
        print("\n[1/5] Loading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load data
        print("\n[2/5] Loading data...")
        self.flux, self.delta_t, self.y, self.params, self.timestamps, \
            self.n_classes, self.n_points = self._load_data(data_path)
        
        # Compute lengths
        print("\n[3/5] Computing sequence lengths...")
        self.lengths = compute_lengths_from_flux(self.flux)
        print(f"  Seq lengths: min={self.lengths.min()}, max={self.lengths.max()}, mean={self.lengths.mean():.1f}")
        
        # Run inference
        print("\n[4/5] Running inference...")
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # Compute metrics
        print("\n[5/5] Computing metrics...")
        self.metrics = self._compute_metrics()
        self._print_summary()
    
    def _load_model(self, model_path: str) -> Tuple[CausalHybridModel, CausalConfig]:
        """Load CausalHybridModel with robust config handling."""
        print(f"  Reading checkpoint: {Path(model_path).name}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"  ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Extract config
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = CausalConfig(**config_dict)
        else:
            print("  WARNING: Config not found in checkpoint, using default")
            config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        print(f"  Architecture: d_model={config.d_model}, heads={config.n_heads}, layers={config.n_transformer_layers}")
        
        # Initialize model
        model = CausalHybridModel(config)
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            return checkpoint, config
        
        # Strip DDP prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        
        # Load weights
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"  WARNING: Strict loading failed, trying strict=False")
            model.load_state_dict(new_state_dict, strict=False)
        
        print(f"  Model loaded: {count_parameters(model):,} parameters")
        # Ensure the model has the get_receptive_field method defined
        try:
             print(f"  Receptive field: {model.get_receptive_field()} timesteps")
        except AttributeError:
             print("  Receptive field information not available on model.")
        
        return model, config
    
    def _load_data(self, data_path: str) -> Tuple:
        """
        Load and preprocess evaluation data.
        
        CRITICAL: Requires pre-computed causal delta_t from simulate.py.
        This ensures observation gaps are correctly encoded.
        """
        print(f"  Reading {data_path}...")
        
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            print(f"  ERROR: Failed to load data: {e}")
            sys.exit(1)
        
        # Load flux
        flux = data.get('flux', data.get('X'))
        if flux is None:
            raise KeyError("Data missing 'flux' or 'X'")
        if flux.ndim == 3:
            flux = flux.squeeze(1)
        
        # Load labels
        y = data.get('labels', data.get('y'))
        if y is None:
            raise KeyError("Data missing 'labels' or 'y'")
        
        # CRITICAL: Load pre-computed causal delta_t
        if 'delta_t' not in data:
            print("\n" + "=" * 80)
            print("CRITICAL ERROR: Missing causal delta_t in test data")
            print("=" * 80)
            print("\nThe test dataset MUST contain pre-computed 'delta_t' array.")
            print("This ensures correct causal encoding of observation gaps.")
            print("\nTo fix:")
            print("  1. Regenerate test data using simulate.py")
            print("  2. Ensure the output NPZ contains 'delta_t' field")
            print("\nExample:")
            print("  python simulate.py --preset quick_test --output test_data.npz")
            print("\nExiting now to prevent incorrect evaluation.")
            print("=" * 80)
            sys.exit(1)
        
        delta_t = data['delta_t']
        if delta_t.ndim == 3:
            delta_t = delta_t.squeeze(1)
        print("  Using pre-computed causal delta_t (observation gaps preserved)")
        
        # Load timestamps for plotting
        timestamps = data.get('timestamps')
        if timestamps is None or timestamps.ndim == 1:
            n_points = flux.shape[1]
            # Create synthetic timestamps if missing
            timestamps = np.linspace(0, 100, n_points) 
            timestamps = np.tile(timestamps, (len(flux), 1))
        
        n_points = flux.shape[1]
        n_classes = len(np.unique(y))
        
        # Load parameters with JSON unwrapping
        params_dict = {}
        target_keys = ['params_binary_json', 'params_pspl_json', 'params_flat_json']
        
        for key in target_keys:
            if key in data:
                try:
                    raw = data[key]
                    
                    # Unwrap numpy array
                    if isinstance(raw, np.ndarray):
                        raw = raw.item() if raw.size == 1 else raw[0]
                    
                    # Decode bytes
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    
                    # Parse JSON
                    cat = key.split('_')[1]
                    params_dict[cat] = json.loads(str(raw))
                    print(f"  Loaded {len(params_dict[cat])} {cat} parameters")
                except Exception as e:
                    print(f"  WARNING: Failed to load {key}: {e}")
        
        params = params_dict if params_dict else None
        
        # Subsample if requested
        if self.n_samples is not None and self.n_samples < len(flux):
            print(f"  Subsampling to {self.n_samples} events...")
            idx = np.random.choice(len(flux), self.n_samples, replace=False)
            flux = flux[idx]
            delta_t = delta_t[idx]
            y = y[idx]
            timestamps = timestamps[idx]
            
            # Subsample parameters if they were loaded
            if params is not None:
                new_params = {}
                for cat, param_list in params.items():
                    # This assumes params are ordered consistently with the data, which is standard practice
                    new_params[cat] = [param_list[i] for i in idx if y[idx][i] == {'flat': 0, 'pspl': 1, 'binary': 2}.get(cat, -1)]
                params = new_params
        
        return flux, delta_t, y, params, timestamps, n_classes, n_points
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched inference using optimized model path."""
        predictions = []
        confidences = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.flux), self.batch_size), desc="  Batches"):
                end = min(i + self.batch_size, len(self.flux))
                
                # Convert to tensor
                f_b = torch.tensor(self.flux[i:end], dtype=torch.float32).to(self.device)
                d_b = torch.tensor(self.delta_t[i:end], dtype=torch.float32).to(self.device)
                l_b = torch.tensor(self.lengths[i:end], dtype=torch.int64).to(self.device)
                
                # Model inference - use optimized path
                out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=False)
                
                # Extract final predictions
                probs_np = out['probs'].cpu().numpy()  # (B, n_classes)
                preds_np = probs_np.argmax(axis=1)
                confs_np = probs_np.max(axis=1)
                
                predictions.extend(preds_np)
                confidences.extend(confs_np)
                all_probs.append(probs_np)
        
        return np.array(predictions), np.array(confidences), np.vstack(all_probs)
    
    def _compute_metrics(self) -> Dict:
        """Compute comprehensive classification metrics."""
        accuracy = accuracy_score(self.y, self.predictions)
        
        # Determine target names based on number of classes
        if self.n_classes == 3:
             target_names = ['Flat', 'PSPL', 'Binary']
             # Ensure labels are 0, 1, 2
             if sorted(np.unique(self.y).tolist()) != [0, 1, 2]:
                 print("WARNING: 3-class mode detected, but labels are not 0, 1, 2. Using 0, 1, 2 names.")
        elif self.n_classes == 2:
            target_names = ['PSPL', 'Binary']
            # Ensure labels are 0, 1
            if sorted(np.unique(self.y).tolist()) != [0, 1]:
                 print("WARNING: 2-class mode detected, but labels are not 0, 1. Using 0, 1 names.")
        else:
             target_names = [f'Class {i}' for i in range(self.n_classes)]
        
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
        
        # AUROC
        try:
            if self.n_classes > 1 and len(np.unique(self.y)) > 1:
                metrics['auroc_macro'] = roc_auc_score(
                    self.y, self.probs, multi_class='ovr', average='macro'
                )
                metrics['auroc_weighted'] = roc_auc_score(
                    self.y, self.probs, multi_class='ovr', average='weighted'
                )
            else:
                metrics['auroc_macro'] = 0.0
                metrics['auroc_weighted'] = 0.0
        except Exception as e:
            print(f"  WARNING: AUROC calculation failed: {e}")
            metrics['auroc_macro'] = 0.0
            metrics['auroc_weighted'] = 0.0
        
        # Per-class metrics
        for i, name in enumerate(target_names):
            # Check if the class name exists in the report keys (handles zero-division/missing class in predictions)
            if name in report:
                metrics[f'{name.lower()}_precision'] = report[name]['precision']
                metrics[f'{name.lower()}_recall'] = report[name]['recall']
                metrics[f'{name.lower()}_f1'] = report[name]['f1-score']
            else:
                 metrics[f'{name.lower()}_precision'] = 0.0
                 metrics[f'{name.lower()}_recall'] = 0.0
                 metrics[f'{name.lower()}_f1'] = 0.0
        
        return metrics
    
    def _print_summary(self):
        """Print evaluation summary."""
        print(f"\n{'=' * 80}")
        print(f"EVALUATION RESULTS ({self.n_classes} classes)")
        print(f"{'=' * 80}")
        print(f"Accuracy:    {self.metrics['accuracy']*100:.2f}%")
        print(f"AUROC (macro): {self.metrics['auroc_macro']:.4f}")
        
        names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        print(f"\nPer-Class Performance:")
        for name in names:
            try:
                prec = self.metrics[f'{name.lower()}_precision']
                rec = self.metrics[f'{name.lower()}_recall']
                f1 = self.metrics[f'{name.lower()}_f1']
                print(f"  {name:8s}: Prec={prec*100:5.1f}% | Rec={rec*100:5.1f}% | F1={f1*100:5.1f}%")
            except KeyError:
                print(f"  {name:8s}: Metrics unavailable")

        print(f"{'=' * 80}\n")
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    def plot_roc_curve(self):
        """Generate One-vs-Rest ROC curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        class_names = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'darkred', 'darkblue'] if self.n_classes == 3 else ['darkred', 'darkblue']
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (self.y == i).astype(int)
            # Check if the class is actually present and has variance in true labels
            if len(np.unique(y_true_binary)) > 1: 
                fpr, tpr, _ = roc_curve(y_true_binary, self.probs[:, i])
                auc = roc_auc_score(y_true_binary, self.probs[:, i])
                ax.plot(fpr, tpr, linewidth=3, color=color, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves - Roman Telescope Readiness', fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300)
        plt.close()
        # 
    
    def plot_confusion_matrix(self):
        """Generate confusion matrix heatmap."""
        cm = np.array(self.metrics['confusion_matrix'])
        labels = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        
        ax.set_title(f'Confusion Matrix ({self.n_classes}-Class)', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        # 
    
    def plot_calibration_curve(self):
        """Generate reliability diagram and confidence histograms."""
        correct = self.predictions == self.y
        bins = np.linspace(0, 1, 11)
        accs, centers, counts = [], [], []
        
        for i in range(len(bins)-1):
            mask = (self.confidences >= bins[i]) & (self.confidences < bins[i+1])
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                centers.append((bins[i] + bins[i+1]) / 2)
                counts.append(mask.sum())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
        ax1.plot(centers, accs, 'o-', label='Model', color='blue', linewidth=2, markersize=8)
        ax1.bar(centers, accs, width=0.08, alpha=0.2, color='blue')
        ax1.set_xlabel('Confidence', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.hist(self.confidences[correct], bins=20, alpha=0.6, color='green', label='Correct')
        ax2.hist(self.confidences[~correct], bins=20, alpha=0.6, color='red', label='Incorrect')
        ax2.set_xlabel('Confidence', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Confidence Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.png', dpi=300)
        plt.close()
    
    def plot_high_res_evolution(self, event_idx: Optional[int] = None, event_type: str = 'binary'):
        """
        Generate evolution plot showing flux, probability trajectory, and confidence over time.
        Demonstrates real-time classification capability for Roman Telescope.
        
        CRITICAL: Ensures time-prediction alignment for scientific validity.
        """
        if event_idx is None:
            # Safely get target class index
            target_map = {'flat': 0, 'pspl': 1}
            target_map['binary'] = 2 if self.n_classes == 3 else 1
            target_class = target_map.get(event_type, 2)
            
            candidates = np.where(
                (self.y == target_class) & 
                (self.predictions == target_class) & 
                (self.confidences > 0.8)
            )[0]
            
            if len(candidates) == 0:
                candidates = np.where(
                    (self.y == target_class) & 
                    (self.predictions == target_class)
                )[0]
            
            if len(candidates) == 0:
                print(f"  WARNING: Could not find a correctly classified high-confidence {event_type} event to plot.")
                return
            
            event_idx = np.random.choice(candidates)
        
        # Get original index if data was subsampled
        # Note: This is complex; assume local index for now as self.flux is the subsampled array
        
        flux = self.flux[event_idx]
        delta_t = self.delta_t[event_idx]
        full_len = self.lengths[event_idx]
        
        # Prepare batch
        f_in = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).to(self.device)
        d_in = torch.tensor(delta_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        l_in = torch.tensor([full_len], dtype=torch.int64).to(self.device)
        
        # Get full sequence output
        with torch.no_grad():
            try:
                out = self.model(f_in, d_in, lengths=l_in, return_all_timesteps=True)
                probs_seq = out['probs'][0, :full_len].cpu().numpy()
            except Exception as e:
                print(f"  WARNING: Error during full sequence inference for event {event_idx}: {e}")
                return
        
        # Extract valid time points
        # Assuming flux padded with 0.0 means invalid
        valid_mask = (flux != 0.0) & (np.arange(len(flux)) < full_len)
        times = self.timestamps[event_idx][valid_mask]
        fluxes = flux[valid_mask]
        
        # CRITICAL ALIGNMENT CHECK
        plot_len = min(len(times), len(probs_seq))
        if len(times) != len(probs_seq):
            print(f"  WARNING: Length mismatch in event {event_idx}. Times: {len(times)}, Predictions: {len(probs_seq)}")
        
        # Create 3-panel plot
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 1], hspace=0.3)
        
        class_labels = ['Flat', 'PSPL', 'Binary'] if self.n_classes == 3 else ['PSPL', 'Binary']
        class_colors = ['gray', 'red', 'blue'] if self.n_classes == 3 else ['red', 'blue']

        # Panel 1: Light curve
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(times[:plot_len], fluxes[:plot_len], c='black', s=15, alpha=0.7, label='Flux')
        ax1.set_title(
            f'Event {event_idx} (True: {event_type.upper()}, Pred: {class_labels[self.predictions[event_idx]]}) - Real-Time Evolution',
            fontweight='bold'
        )
        ax1.set_ylabel('Flux', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Panel 2: Class probabilities
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        for i in range(self.n_classes):
             ax2.plot(times[:plot_len], probs_seq[:plot_len, i], label=class_labels[i], color=class_colors[i], alpha=0.7, linewidth=2)
        
        ax2.axhline(0.5, color='k', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Class Probability', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Confidence
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        confidence_seq = probs_seq[:plot_len].max(axis=1)
        ax3.plot(times[:plot_len], confidence_seq, color='purple', linewidth=2, label='Confidence')
        ax3.axhline(0.9, color='green', linestyle='--', label='90% Threshold', linewidth=2)
        ax3.set_ylabel('Confidence', fontweight='bold')
        ax3.set_xlabel('Time (Arbitrary Units)', fontweight='bold')
        ax3.set_ylim(0.2, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'evolution_{event_type}_{event_idx}.png', dpi=300)
        plt.close()
    
    def plot_fine_early_detection(self):
        """
        Fine-grained early detection analysis.
        Measures accuracy as function of observation completion percentage.
        """
        if not self.run_early_detection:
             return
             
        print("  Running fine-grained early detection analysis...")
        
        # Subsample for efficiency
        n_test = min(len(self.flux), 1000)
        indices = np.random.choice(len(self.flux), n_test, replace=False)
        
        f_sub = self.flux[indices]
        d_sub = self.delta_t[indices]
        l_sub = self.lengths[indices]
        y_sub = self.y[indices]
        
        # Run inference once to get all timesteps
        all_probs_seq = []
        
        with torch.no_grad():
            for i in tqdm(range(0, n_test, self.batch_size), desc="  Early Detection Batches"):
                end = min(i + self.batch_size, n_test)
                f_b = torch.tensor(f_sub[i:end], dtype=torch.float32).to(self.device)
                d_b = torch.tensor(d_sub[i:end], dtype=torch.float32).to(self.device)
                l_b = torch.tensor(l_sub[i:end], dtype=torch.int64).to(self.device)
                
                # Check for zero length
                if l_b.max() == 0:
                     continue
                     
                out = self.model(f_b, d_b, lengths=l_b, return_all_timesteps=True)
                all_probs_seq.append(out['probs'].cpu().numpy())
        
        # Calculate accuracy at different fractions
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
                
                # Calculate the index of the observation corresponding to the fraction
                target_indices = (batch_lengths * frac).astype(int)
                target_indices = np.maximum(target_indices - 1, 0)
                
                preds = []
                for b in range(batch_size):
                    # Ensure index is within the actual sequence length and max dimension of the output tensor
                    valid_idx = min(target_indices[b], batch_probs.shape[1] - 1)
                    p = batch_probs[b, valid_idx]
                    preds.append(np.argmax(p))
                
                correct_count += np.sum(np.array(preds) == batch_y)
                total_count += batch_size
                current_batch_start += batch_size
            
            accuracies.append(correct_count / total_count if total_count > 0 else 0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(fractions * 100, accuracies, 'o-', linewidth=2, color='purple', markersize=4)
        plt.axhline(0.8, color='green', linestyle=':', linewidth=2, label='80% Accuracy')
        plt.axhline(0.9, color='blue', linestyle=':', linewidth=2, label='90% Accuracy')
        plt.xlabel('Percentage of Light Curve Observed', fontweight='bold')
        plt.ylabel('Classification Accuracy', fontweight='bold')
        plt.title('Early Detection Performance - Roman Telescope Readiness', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fine_early_detection.png', dpi=300)
        plt.close()
        # 
    
    def diagnose_temporal_bias(self):
        """Diagnose potential temporal bias using t0 distribution KS-test."""
        if not self.params:
            return
        if self.n_samples:
            print("  Skipping temporal bias check (subsampled data)")
            return
        
        print("  Running temporal bias diagnostic...")
        
        try:
            # Need to align params with self.y
            pspl_t0, binary_t0 = [], []
            
            # Map index to class label for easier lookup
            y_to_cat = {0: 'flat', 1: 'pspl', 2: 'binary'} if self.n_classes == 3 else {0: 'pspl', 1: 'binary'}

            for i, p_list in self.params.items():
                if i in y_to_cat.values():
                    t0s = [p.get('t_0', None) for p in p_list if p.get('t_0') is not None]
                    if i == 'pspl':
                        pspl_t0.extend(t0s)
                    elif i == 'binary':
                        binary_t0.extend(t0s)
            
            pspl_t0 = np.array(pspl_t0)
            binary_t0 = np.array(binary_t0)

            if pspl_t0.size < 2 or binary_t0.size < 2:
                print("  Insufficient parameter data for temporal bias check")
                return
            
            plt.figure(figsize=(10, 6))
            plt.hist(pspl_t0, bins=30, alpha=0.5, label='PSPL t0', density=True, color='red')
            plt.hist(binary_t0, bins=30, alpha=0.5, label='Binary t0', density=True, color='blue')
            plt.legend()
            plt.title('Temporal Bias Diagnostic (Peak Time Distribution)', fontweight='bold')
            plt.xlabel('Peak Time (t0)', fontweight='bold')
            plt.ylabel('Density', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_bias_check.png', dpi=300)
            plt.close()
            
            # Check if t0 distributions are significantly different
            stat, pval = ks_2samp(pspl_t0, binary_t0)
            if pval < 0.05:
                print(f"  WARNING: Significant t0 bias detected (p={pval:.4f} < 0.05). Model may be using peak time as a shortcut feature.")
            else:
                print(f"  PASSED: t0 distributions are statistically similar (p={pval:.4f})")
        except Exception as e:
            print(f"  Temporal bias check failed: {e}")
    
    def analyze_u0_dependency(self, n_bins: int = 10):
        """Analyze accuracy vs impact parameter (u0) for binary events."""
        if not self.params or 'binary' not in self.params:
            return
        if self.n_samples:
            print("  Skipping u0 dependency check (subsampled data)")
            return
        
        print("  Running u0 dependency analysis...")
        
        try:
            # Identify binary events index
            bin_label = 2 if self.n_classes == 3 else 1
            bin_mask = (self.y == bin_label)
            
            # Subselect predictions for binary events only
            bin_preds = self.predictions[bin_mask]
            bin_y = self.y[bin_mask]
            
            # Extract u0 values for the original binary events
            u0s = np.array([p.get('u_0', -1) for p in self.params['binary']])
            
            if len(u0s) != len(self.params['binary']):
                 print("  Parameter list length mismatch, skipping.")
                 return
            
            # Filter u0s to only include the events in the test set (if not subsampled)
            # This step is critical if params list contains all event parameters, but data_path only contains test subset
            # Assuming for now self.params['binary'] is exactly the parameters for self.y[bin_mask]
            
            if len(u0s) != len(bin_y):
                 print("  Binary event count mismatch between labels and parameters, skipping.")
                 return

            # Remove invalid u0s (u0 < 0 or missing, though simulated data should be fine)
            valid_u0_mask = u0s >= 0
            u0s = u0s[valid_u0_mask]
            bin_preds = bin_preds[valid_u0_mask]
            bin_y = bin_y[valid_u0_mask]

            if u0s.size < 10: # Minimum size for analysis
                 print("  Insufficient valid binary event parameters for u0 dependency analysis.")
                 return
            
            # Log-space binning
            # Use max(u0s.min(), 1e-4) to avoid log(0) if u0s contains very small values
            bins = np.logspace(np.log10(max(1e-4, u0s.min())), np.log10(u0s.max()), n_bins + 1)
            accs, centers, counts = [], [], []
            
            for i in range(len(bins)-1):
                m = (u0s >= bins[i]) & (u0s < bins[i+1])
                if m.sum() > 0:
                    accs.append((bin_preds[m] == bin_y[m]).mean())
                    # Geometric mean for center in log-space
                    centers.append(np.sqrt(bins[i] * bins[i+1])) 
                    counts.append(m.sum())
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Accuracy line
            ax1.semilogx(centers, accs, 'o-', color='tab:blue', linewidth=2, markersize=8, label='Accuracy')
            ax1.set_xlabel('Impact Parameter ($u_0$)', fontweight='bold')
            ax1.set_ylabel('Classification Accuracy', color='tab:blue', fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, which="both", alpha=0.3)
            
            # Count histogram
            ax2 = ax1.twinx()
            # Calculate widths for the bar plot based on the log bins
            bar_widths = np.diff(bins)
            ax2.bar(centers, counts, width=bar_widths, alpha=0.1, color='black', label='Count', log=True)
            ax2.set_ylabel('Event Count (Log Scale)', color='black', fontweight='bold')
            
            plt.title('Binary Event Accuracy vs Impact Parameter ($u_0$)', fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'u0_dependency.png', dpi=300)
            plt.close()
            # 

        except Exception as e:
            print(f"  u0 dependency analysis failed: {e}")
    
    def generate_all_plots(self):
        
        print("\n[Visualizations] Generating diagnostic plots...")
        
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_calibration_curve()
        
        if self.run_early_detection:
            self.plot_fine_early_detection()
        
        if self.n_evolution_per_type > 0: # Check the new attribute
            print(f"  Generating {self.n_evolution_per_type} evolution examples per type...")
            # Loop n times
            for _ in range(self.n_evolution_per_type): # <-- FIX: Use self.n_evolution_per_type
                if self.n_classes == 3:
                    self.plot_high_res_evolution(event_type='flat')
                self.plot_high_res_evolution(event_type='pspl')
                self.plot_high_res_evolution(event_type='binary')
        else:
             print("  Skipping evolution plots (n_evolution_per_type=0)")
            
        self.diagnose_temporal_bias()
        self.analyze_u0_dependency()
        
        # Save JSON summary
        summary = {
            'metrics': {
                # Convert numpy types to standard Python float/int for JSON serialization
                k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                for k, v in self.metrics.items()
                if 'matrix' not in k and 'report' not in k
            },
            'timestamp': self.timestamp,
            # Ensure config is serializable
            'model_config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'run_flags': {
                'early_detection': self.run_early_detection,
                'n_evolution_plots_per_type': self.n_evolution_per_type
            },
            'data_info': {
                'n_samples': len(self.flux),
                'n_classes': self.n_classes,
                'sequence_length': self.n_points,
                'causal_delta_t': True  # Confirms causal encoding used
            }
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n{'=' * 80}")
        print(f"Evaluation complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'=' * 80}\n")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite for Causal Hybrid Model (Roman Telescope Readiness).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # CRITICAL ARGUMENTS
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained PyTorch model checkpoint.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the test data NPZ file (must contain the "delta_t" array).')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Base directory to save evaluation results and plots.')
    
    # HARDWARE/PERFORMANCE ARGUMENTS
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference (e.g., "cuda" or "cpu").')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for parallel inference.')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of events to sample from the test data for quick runs (optional).')
    
    # PLOTTING/DIAGNOSTIC FLAGS (from user's original intent)
    parser.add_argument('--early_detection', action='store_true',
                        help='Run early detection analysis (accuracy at partial lengths).')
    parser.add_argument('--n_evolution_per_type', type=int, default=0,
                        help='Number of evolution plots to generate per class. Set > 0 to enable.')
    
    args = parser.parse_args()
    
    try:
        evaluator = ComprehensiveEvaluator(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            run_early_detection=args.early_detection,
            n_evolution_per_type=args.n_evolution_per_type
        )
        
        evaluator.generate_all_plots()
        
    except Exception as e:
        print(f"\n--- FATAL EVALUATION ERROR ---")
        print(f"An unhandled error occurred during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
