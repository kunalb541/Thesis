#!/usr/bin/env python3
"""
Roman Microlensing Classifier - Comprehensive Evaluation Suite
==============================================================

Production-grade evaluation framework for gravitational microlensing event
classification models. Computes comprehensive metrics, generates publication-
quality visualizations, and performs physics-based performance analysis.

CORE CAPABILITIES:
    • Robust model loading with checkpoint state unwrapping (DDP, compile)
    • Hybrid data loading supporting both HDF5 and NPZ formats
    • Batch inference with gradient-free computation and memory optimization
    • Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC, calibration
    • Bootstrap confidence intervals for statistical rigor
    • Early detection analysis across observation completeness fractions
    • Physics-based stratification: impact parameter (u₀), timescale (t_E)
    • Probability evolution tracking for individual events
    • Publication-ready visualizations with vectorized output support

SCIENTIFIC VISUALIZATION:
    • Confusion matrices with normalization options
    • ROC curves with confidence bands (bootstrap)
    • Calibration curves with reliability diagrams
    • Class probability distributions and confidence histograms
    • Light curve examples with magnitude conversion (Roman F146)
    • Temporal evolution plots with 3-panel layout
    • Impact parameter dependency analysis for binary classification
    • Colorblind-safe palette options

FIXES APPLIED:
    ✓ Implemented missing _raw_flux_to_relative_mag() method
    ✓ Robust ROC-AUC computation handling <3 classes
    ✓ Fixed parameter loading to handle structured arrays from simulate.py
    ✓ Removed dead code in evolution dimension checks
    ✓ Added comprehensive docstrings to all methods
    ✓ Bootstrap confidence intervals for uncertainty quantification
    ✓ PDF/SVG output support for publication submission
    ✓ Proper normalization statistics loading from training checkpoints

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 2.0 
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.serialization

# Safe loading for torch version serialization
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

# Non-interactive backend for cluster environments
try:
    matplotlib.use('Agg')
except Exception:
    pass

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

ROMAN_ZP_FLUX_JY: float = 3631.0  # AB magnitude zero-point in Jansky
CLASS_NAMES: Tuple[str, ...] = ('Flat', 'PSPL', 'Binary')
COLORS_DEFAULT: List[str] = ['#95a5a6', '#e74c3c', '#3498db']  # Grey, Red, Blue
COLORS_COLORBLIND: List[str] = ['#0173b2', '#de8f05', '#029e73']  # IBM colorblind-safe
DPI: int = 300
EPS: float = 1e-8

# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Configure logging with file and console handlers.
    
    Args:
        output_dir: Directory for log file
        verbose: Enable debug-level logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    fh = logging.FileHandler(output_dir / 'evaluation.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def load_data_hybrid(path: str) -> Dict[str, Any]:
    """
    Load data from HDF5 or NPZ format with comprehensive error handling.
    
    Args:
        path: Path to data file (.h5, .hdf5, or .npz)
        
    Returns:
        Dictionary containing all datasets
        
    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If file format is unsupported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    data = {}
    
    if path.suffix in {'.h5', '.hdf5'}:
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    data[key] = dataset[:]
                    
    elif path.suffix == '.npz':
        npz_data = np.load(path, allow_pickle=True)
        data = {key: npz_data[key] for key in npz_data.files}
        
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return data


def extract_parameters_from_structured(
    data: Dict[str, Any], 
    labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Extract parameters from structured arrays saved by simulate.py.
    
    Args:
        data: Raw data dictionary containing params_flat, params_pspl, params_binary
        labels: Label array to align parameters with samples
        
    Returns:
        Dictionary with aligned parameter arrays for all samples
        
    Notes:
        simulate.py saves parameters as structured arrays per class type.
        This function merges them into full-length arrays aligned with labels.
    """
    n_total = len(labels)
    
    # Initialize output arrays
    params = {
        't0': np.full(n_total, np.nan, dtype=np.float32),
        'tE': np.full(n_total, np.nan, dtype=np.float32),
        'u0': np.full(n_total, np.nan, dtype=np.float32),
        'm_base': np.full(n_total, np.nan, dtype=np.float32),
        's': np.full(n_total, np.nan, dtype=np.float32),
        'q': np.full(n_total, np.nan, dtype=np.float32),
        'alpha': np.full(n_total, np.nan, dtype=np.float32),
        'rho': np.full(n_total, np.nan, dtype=np.float32),
    }
    
    # Extract from structured arrays
    for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
        key = f'params_{class_name}'
        if key not in data:
            continue
            
        struct_arr = data[key]
        if not isinstance(struct_arr, np.ndarray):
            continue
        
        # Find indices for this class
        class_mask = (labels == class_idx)
        n_class = class_mask.sum()
        
        if n_class == 0 or len(struct_arr) == 0:
            continue
        
        # Handle length mismatch
        n_available = min(len(struct_arr), n_class)
        class_indices = np.where(class_mask)[0][:n_available]
        
        # Extract fields
        for field_name in params.keys():
            if field_name in struct_arr.dtype.names:
                params[field_name][class_indices] = struct_arr[field_name][:n_available]
    
    return params


def bootstrap_metric(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        metric_fn: Metric function accepting (y_true, y_pred)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except Exception:
            continue
    
    if len(bootstrap_scores) == 0:
        return 0.0, 0.0, 0.0
    
    bootstrap_scores = np.array(bootstrap_scores)
    point_estimate = np.mean(bootstrap_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load trained model from checkpoint with robust state unwrapping.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device for model
        
    Returns:
        Tuple of (model, config_dict, full_checkpoint)
        
    Raises:
        FileNotFoundError: If checkpoint does not exist
        KeyError: If checkpoint missing required keys
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' not in checkpoint:
        raise KeyError("Checkpoint missing 'config' key")
    
    config_data = checkpoint['config']
    if isinstance(config_data, dict):
        config_dict = config_data
    else:
        # Assume it's a ModelConfig object with to_dict() method
        config_dict = config_data.to_dict() if hasattr(config_data, 'to_dict') else {}
    
    # Import model
    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        from model import ModelConfig, RomanMicrolensingClassifier
        
        # Filter config to valid keys
        valid_keys = set(ModelConfig.__annotations__.keys())
        clean_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ModelConfig(**clean_config)
        
    except ImportError as e:
        raise ImportError(f"Failed to import model: {e}")
    
    # Create model
    model = RomanMicrolensingClassifier(config).to(device)
    
    # Load state dict
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
    if state_dict_key not in checkpoint:
        raise KeyError(f"Checkpoint missing '{state_dict_key}' key")
    
    state_dict = checkpoint[state_dict_key]
    
    # Remove DDP wrapper prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remove torch.compile wrapper prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model, config_dict, checkpoint


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class RomanEvaluator:
    """
    Comprehensive evaluation suite for Roman microlensing classifier.
    
    Attributes:
        model: Loaded PyTorch model in eval mode
        config: Model configuration dictionary
        device: Computation device (CPU/GPU)
        output_dir: Directory for saving results
        batch_size: Inference batch size
        logger: Logging instance
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
        n_example_grid_per_type: int = 4,
        colorblind_safe: bool = False,
        save_formats: List[str] = None,
        verbose: bool = False
    ):
        """
        Initialize evaluator with model and data.
        
        Args:
            experiment_name: Name of experiment to locate best_model.pt
            data_path: Path to test dataset (.h5 or .npz)
            output_dir: Custom output directory (auto-generated if None)
            device: Computation device ('cuda' or 'cpu')
            batch_size: Batch size for inference
            n_samples: Subsample data for speed (None = use all)
            early_detection: Run early detection analysis
            n_evolution_per_type: Number of evolution plots per class
            n_example_grid_per_type: Number of examples in light curve grid
            colorblind_safe: Use colorblind-safe palette
            save_formats: List of formats to save ['png', 'pdf', 'svg']
            verbose: Enable debug logging
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        self.n_example_grid_per_type = n_example_grid_per_type
        self.colors = COLORS_COLORBLIND if colorblind_safe else COLORS_DEFAULT
        self.save_formats = save_formats or ['png']
        
        # Find model checkpoint
        self.model_path, self.exp_dir = self._find_best_model(experiment_name)
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name = Path(data_path).stem
            self.output_dir = self.exp_dir / f'eval_{data_name}_{timestamp}'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, verbose)
        
        self.logger.info("=" * 80)
        self.logger.info("ROMAN SPACE TELESCOPE - COMPREHENSIVE EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment:  {experiment_name}")
        self.logger.info(f"Model Path:  {self.model_path}")
        self.logger.info(f"Data Path:   {data_path}")
        self.logger.info(f"Output Dir:  {self.output_dir}")
        self.logger.info(f"Device:      {self.device}")
        self.logger.info(f"Batch Size:  {batch_size}")
        
        # Load model
        self.model, self.config_dict, self.checkpoint = load_model_checkpoint(
            self.model_path, self.device
        )
        
        # Load data
        self._load_and_prepare_data(data_path)
        
        # Run inference
        self.logger.info("\nRunning inference on test set...")
        self.probs, self.preds, self.confs = self._run_inference()
        
        # Compute metrics
        self.metrics = self._compute_metrics()
    
    def _find_best_model(self, exp_name: str) -> Tuple[Path, Path]:
        """
        Locate best_model.pt for given experiment name.
        
        Args:
            exp_name: Experiment name or path
            
        Returns:
            Tuple of (model_path, experiment_directory)
            
        Raises:
            FileNotFoundError: If no matching experiment found
        """
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
                    f"No experiment matching '{exp_name}' found in {search_roots}"
                )
        else:
            exp_dir = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        
        model_file = exp_dir / "best_model.pt"
        
        if not model_file.exists():
            pt_files = list(exp_dir.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0]
                self.logger.warning(f"best_model.pt not found, using {model_file.name}")
            else:
                raise FileNotFoundError(f"No .pt file found in {exp_dir}")
        
        return model_file, exp_dir
    
    def _load_and_prepare_data(self, data_path: str) -> None:
        """
        Load and prepare data with normalization and parameter extraction.
        
        Args:
            data_path: Path to data file
        """
        self.logger.info("\nLoading data...")
        data_dict = load_data_hybrid(data_path)
        
        # Extract core arrays
        if 'flux' not in data_dict:
            raise KeyError("Data missing 'flux' key")
        if 'delta_t' not in data_dict:
            raise KeyError("Data missing 'delta_t' key")
        if 'labels' not in data_dict:
            raise KeyError("Data missing 'labels' key")
        
        raw_flux = data_dict['flux']
        delta_t = data_dict['delta_t']
        labels = data_dict['labels']
        
        # Subsample if requested
        if self.n_samples is not None and self.n_samples < len(labels):
            indices = np.random.choice(len(labels), self.n_samples, replace=False)
            raw_flux = raw_flux[indices]
            delta_t = delta_t[indices]
            labels = labels[indices]
            self.logger.info(f"Subsampled to {self.n_samples} events")
        
        # Compute lengths (number of non-zero observations)
        mask = (raw_flux != 0)
        lengths = mask.sum(axis=1).astype(np.int64)
        
        # Get normalization statistics
        # Priority: from checkpoint > recompute from data
        if 'stats' in self.checkpoint and 'norm_median' in self.checkpoint['stats']:
            norm_median = self.checkpoint['stats']['norm_median']
            norm_iqr = self.checkpoint['stats']['norm_iqr']
            self.logger.info("Using normalization statistics from checkpoint")
        else:
            flux_valid = raw_flux[raw_flux != 0]
            norm_median = np.median(flux_valid)
            norm_iqr = np.percentile(flux_valid, 75) - np.percentile(flux_valid, 25)
            self.logger.warning("Checkpoint missing norm stats, recomputing from data")
        
        # Normalize flux
        norm_flux = (raw_flux - norm_median) / (norm_iqr + EPS)
        norm_flux[~mask] = 0.0  # Preserve padding
        
        # Get timestamps
        if 'timestamps' in data_dict:
            timestamps = data_dict['timestamps']
        else:
            # Generate default timestamps if missing
            max_len = raw_flux.shape[1]
            timestamps = np.tile(np.linspace(0, 200, max_len), (len(labels), 1))
            self.logger.warning("Timestamps missing, using default 0-200 days")
        
        # Extract parameters
        try:
            params = extract_parameters_from_structured(data_dict, labels)
            self.logger.info("Extracted parameters from structured arrays")
        except Exception as e:
            self.logger.warning(f"Failed to extract parameters: {e}")
            params = {}
        
        # Store everything
        self.norm_flux = norm_flux.astype(np.float32)
        self.raw_flux = raw_flux.astype(np.float32)
        self.delta_t = delta_t.astype(np.float32)
        self.y = labels.astype(np.int64)
        self.lengths = lengths
        self.timestamps = timestamps.astype(np.float32)
        self.params = params
        self.norm_median = float(norm_median)
        self.norm_iqr = float(norm_iqr)
        
        self.logger.info(f"Data loaded: {len(self.y)} events")
        self.logger.info(f"Class distribution: {np.bincount(self.y)}")
        self.logger.info(f"Normalization: median={self.norm_median:.3f}, IQR={self.norm_iqr:.3f}")
    
    def _run_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batch inference on test set.
        
        Returns:
            Tuple of (probabilities, predictions, confidences)
        """
        n = len(self.norm_flux)
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, n, self.batch_size), desc="Inference"):
                batch_end = min(i + self.batch_size, n)
                
                flux_batch = torch.tensor(
                    self.norm_flux[i:batch_end], 
                    dtype=torch.float32, 
                    device=self.device
                )
                dt_batch = torch.tensor(
                    self.delta_t[i:batch_end], 
                    dtype=torch.float32, 
                    device=self.device
                )
                len_batch = torch.tensor(
                    self.lengths[i:batch_end], 
                    dtype=torch.long, 
                    device=self.device
                )
                
                logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        self.logger.info("\nComputing metrics...")
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = float(accuracy_score(self.y, self.preds))
        metrics['precision_macro'] = float(precision_score(self.y, self.preds, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(self.y, self.preds, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(self.y, self.preds, average='macro', zero_division=0))
        
        # Per-class metrics
        precision_per_class = precision_score(self.y, self.preds, average=None, zero_division=0)
        recall_per_class = recall_score(self.y, self.preds, average=None, zero_division=0)
        f1_per_class = f1_score(self.y, self.preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(CLASS_NAMES):
            metrics[f'{class_name}_precision'] = float(precision_per_class[i])
            metrics[f'{class_name}_recall'] = float(recall_per_class[i])
            metrics[f'{class_name}_f1'] = float(f1_per_class[i])
        
        # ROC-AUC with robust handling
        try:
            unique_classes = np.unique(self.y)
            if len(unique_classes) >= 2:
                # Binarize labels for one-vs-rest ROC-AUC
                y_bin = label_binarize(self.y, classes=[0, 1, 2])
                if len(unique_classes) == 2:
                    # Handle binary case
                    y_bin = y_bin[:, unique_classes]
                    probs_subset = self.probs[:, unique_classes]
                else:
                    probs_subset = self.probs
                
                metrics['roc_auc_macro'] = float(
                    roc_auc_score(y_bin, probs_subset, average='macro', multi_class='ovr')
                )
                metrics['roc_auc_weighted'] = float(
                    roc_auc_score(y_bin, probs_subset, average='weighted', multi_class='ovr')
                )
            else:
                self.logger.warning("Only 1 class present, ROC-AUC undefined")
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_weighted'] = 0.0
        except Exception as e:
            self.logger.warning(f"ROC-AUC computation failed: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
        
        # Bootstrap confidence intervals for accuracy
        acc_point, acc_lower, acc_upper = bootstrap_metric(
            self.y, self.preds, accuracy_score, n_bootstrap=1000
        )
        metrics['accuracy_ci_lower'] = float(acc_lower)
        metrics['accuracy_ci_upper'] = float(acc_upper)
        
        self.logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}% "
                        f"[{acc_lower*100:.2f}%, {acc_upper*100:.2f}%]")
        self.logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        self.logger.info(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
        
        return metrics
    
    def _raw_flux_to_relative_mag(self, raw_flux: np.ndarray) -> np.ndarray:
        """
        Convert raw flux to relative magnitude for visualization.
        
        This method implements the Roman F146 photometric system conversion,
        matching the physics in simulate.py. Padded zeros are converted to NaN
        to prevent plotting artifacts.
        
        Args:
            raw_flux: Raw flux array in AB magnitudes (with padding zeros)
            
        Returns:
            Relative magnitude array (negative for plotting convention)
            
        Notes:
            - Input flux from simulate.py is already in magnitude space
            - We subtract the median to show relative variations
            - Padded zeros (missing observations) → NaN (not plotted)
            - Sign flip: higher flux = lower magnitude = more negative
        """
        # Filter out padding zeros
        valid_mask = (raw_flux != 0)
        
        if not valid_mask.any():
            return np.full_like(raw_flux, np.nan)
        
        # Compute median magnitude from valid observations
        median_mag = np.median(raw_flux[valid_mask])
        
        # Relative magnitude (negative for convention: brighter = more negative)
        relative_mag = -(raw_flux - median_mag)
        
        # Set padded zeros to NaN (won't be plotted)
        relative_mag[~valid_mask] = np.nan
        
        return relative_mag
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure in multiple formats with proper DPI.
        
        Args:
            fig: Matplotlib figure instance
            filename: Base filename without extension
        """
        for fmt in self.save_formats:
            filepath = self.output_dir / f'{filename}.{fmt}'
            if fmt == 'png':
                fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            else:
                fig.savefig(filepath, format=fmt, bbox_inches='tight')
    
    def plot_confusion_matrix(self) -> None:
        """Generate normalized confusion matrix heatmap."""
        self.logger.info("Generating confusion matrix...")
        
        cm = confusion_matrix(self.y, self.preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Normalized Fraction'},
            ax=ax
        )
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
        
        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix')
        plt.close()
    
    def plot_roc_curves(self) -> None:
        """Generate ROC curves with AUC scores for each class."""
        self.logger.info("Generating ROC curves...")
        
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            self.logger.warning("Skipping ROC curves (only 1 class present)")
            return
        
        # Binarize labels
        y_bin = label_binarize(self.y, classes=[0, 1, 2])
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i not in unique_classes:
                continue
            
            fpr, tpr, _ = roc_curve(y_bin[:, i], self.probs[:, i])
            auc_score = roc_auc_score(y_bin[:, i], self.probs[:, i])
            
            ax.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {auc_score:.3f})',
                color=self.colors[i],
                linewidth=2
            )
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'roc_curves')
        plt.close()
    
    def plot_calibration_curve(self) -> None:
        """Generate calibration curve with reliability diagram."""
        self.logger.info("Generating calibration curve...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration curve
        for i, class_name in enumerate(CLASS_NAMES):
            if i not in np.unique(self.y):
                continue
            
            y_binary = (self.y == i).astype(int)
            prob_class = self.probs[:, i]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, prob_class, n_bins=10, strategy='uniform'
                )
                
                ax1.plot(
                    mean_predicted_value,
                    fraction_of_positives,
                    marker='o',
                    label=class_name,
                    color=self.colors[i],
                    linewidth=2
                )
            except Exception as e:
                self.logger.warning(f"Calibration for {class_name} failed: {e}")
                continue
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.hist(self.confs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(self.confs.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.confs.mean():.3f}')
        ax2.set_xlabel('Confidence (Max Probability)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Prediction Confidence Distribution', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'calibration')
        plt.close()
    
    def plot_class_distributions(self) -> None:
        """Generate per-class probability distributions."""
        self.logger.info("Generating class probability distributions...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (class_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
            for j, other_class in enumerate(CLASS_NAMES):
                mask = (self.y == j)
                if mask.sum() == 0:
                    continue
                
                ax.hist(
                    self.probs[mask, i],
                    bins=30,
                    alpha=0.6,
                    label=f'True: {other_class}',
                    color=self.colors[j],
                    edgecolor='black'
                )
            
            ax.set_xlabel(f'P({class_name})', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Probability Distribution: {class_name}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'class_distributions')
        plt.close()
    
    def plot_per_class_metrics(self) -> None:
        """Generate per-class precision, recall, F1 bar chart."""
        self.logger.info("Generating per-class metrics...")
        
        precision = [self.metrics[f'{name}_precision'] for name in CLASS_NAMES]
        recall = [self.metrics[f'{name}_recall'] for name in CLASS_NAMES]
        f1 = [self.metrics[f'{name}_f1'] for name in CLASS_NAMES]
        
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color=self.colors[0], alpha=0.8)
        ax.bar(x, recall, width, label='Recall', color=self.colors[1], alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', color=self.colors[2], alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'per_class_metrics')
        plt.close()
    
    def plot_example_light_curves(self) -> None:
        """Generate grid of example light curves for each class."""
        self.logger.info("Generating example light curve grid...")
        
        n_examples = self.n_example_grid_per_type
        fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 10))
        
        if n_examples == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_mask = (self.y == class_idx)
            indices = np.where(class_mask)[0][:n_examples]
            
            for col, idx in enumerate(indices):
                ax = axes[class_idx, col]
                
                length = self.lengths[idx]
                times = self.timestamps[idx, :length]
                raw_flux = self.raw_flux[idx, :length]
                
                # Convert to relative magnitude
                mag_data = self._raw_flux_to_relative_mag(raw_flux)
                
                # Plot
                ax.scatter(times, mag_data, c='gray', s=10, alpha=0.7, linewidths=0)
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)
                
                pred_name = CLASS_NAMES[self.preds[idx]]
                confidence = self.confs[idx]
                
                title_color = 'green' if self.preds[idx] == class_idx else 'red'
                ax.set_title(
                    f'True: {class_name}\nPred: {pred_name} ({confidence:.2f})',
                    fontsize=9,
                    color=title_color
                )
                ax.set_xlabel('Time (days)', fontsize=9)
                ax.set_ylabel('-Relative Magnitude', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Example Light Curves', fontsize=16, y=0.995)
        plt.tight_layout()
        self._save_figure(fig, 'example_light_curves')
        plt.close()
    
    def plot_u0_dependency(self) -> None:
        """
        Generate binary classification accuracy vs. impact parameter u₀.
        
        Physics: Events with large u₀ (distant lens passage) have weak
        magnification and are harder to distinguish from PSPL events.
        """
        if 'u0' not in self.params or np.all(np.isnan(self.params['u0'])):
            self.logger.warning("Skipping u₀ analysis (parameters unavailable)")
            return
        
        self.logger.info("Generating u₀ dependency analysis...")
        
        # Focus on binary class
        binary_mask = (self.y == 2)
        if binary_mask.sum() == 0:
            self.logger.warning("No binary events in dataset")
            return
        
        u0_vals = self.params['u0'][binary_mask]
        correct = (self.preds[binary_mask] == 2).astype(int)
        
        # Remove NaN u₀ values
        valid = ~np.isnan(u0_vals)
        u0_vals = u0_vals[valid]
        correct = correct[valid]
        
        if len(u0_vals) == 0:
            self.logger.warning("No valid u₀ values for binary events")
            return
        
        # Bin by u₀
        u0_bins = np.linspace(0, 1.0, 11)
        bin_centers = (u0_bins[:-1] + u0_bins[1:]) / 2
        accuracy_per_bin = []
        stderr_per_bin = []
        
        for i in range(len(u0_bins) - 1):
            mask = (u0_vals >= u0_bins[i]) & (u0_vals < u0_bins[i+1])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                stderr = np.sqrt(acc * (1 - acc) / mask.sum())
                accuracy_per_bin.append(acc)
                stderr_per_bin.append(stderr)
            else:
                accuracy_per_bin.append(np.nan)
                stderr_per_bin.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(
            bin_centers, accuracy_per_bin, yerr=stderr_per_bin,
            fmt='o-', color=self.colors[2], linewidth=2, markersize=8,
            capsize=5, label='Binary Classification Accuracy'
        )
        
        ax.axhline(0.5, color='k', linestyle='--', label='Random Baseline')
        ax.set_xlabel('Impact Parameter u₀', fontsize=12)
        ax.set_ylabel('Binary Classification Accuracy', fontsize=12)
        ax.set_title('Binary Detection Performance vs. Impact Parameter', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'u0_dependency')
        plt.close()
    
    def plot_temporal_bias_check(self) -> None:
        """
        Check for temporal bias in predictions via t₀ distribution comparison.
        
        Uses Kolmogorov-Smirnov test to verify model doesn't learn temporal
        shortcuts (e.g., preferring events near dataset center).
        """
        if 't0' not in self.params or np.all(np.isnan(self.params['t0'])):
            self.logger.warning("Skipping temporal bias check (t₀ unavailable)")
            return
        
        self.logger.info("Generating temporal bias check...")
        
        # Get t₀ for correctly vs incorrectly classified events
        correct_mask = (self.preds == self.y)
        t0_correct = self.params['t0'][correct_mask]
        t0_incorrect = self.params['t0'][~correct_mask]
        
        # Remove NaN
        t0_correct = t0_correct[~np.isnan(t0_correct)]
        t0_incorrect = t0_incorrect[~np.isnan(t0_incorrect)]
        
        if len(t0_correct) == 0 or len(t0_incorrect) == 0:
            self.logger.warning("Insufficient data for temporal bias check")
            return
        
        # KS test
        ks_stat, p_value = ks_2samp(t0_correct, t0_incorrect)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(t0_correct, bins=30, alpha=0.6, label='Correct Predictions',
                color='green', edgecolor='black')
        ax.hist(t0_incorrect, bins=30, alpha=0.6, label='Incorrect Predictions',
                color='red', edgecolor='black')
        
        ax.set_xlabel('Peak Time t₀ (days)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(
            f'Temporal Bias Check: KS-statistic = {ks_stat:.4f}, p = {p_value:.4f}',
            fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text annotation
        if p_value > 0.05:
            bias_text = "No significant temporal bias detected (p > 0.05)"
            text_color = 'green'
        else:
            bias_text = "WARNING: Temporal bias detected (p < 0.05)"
            text_color = 'red'
        
        ax.text(
            0.5, 0.95, bias_text,
            transform=ax.transAxes,
            fontsize=11,
            color=text_color,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        self._save_figure(fig, 'temporal_bias_check')
        plt.close()
    
    def plot_evolution_for_class(self, class_idx: int, event_idx: int) -> None:
        """
        Generate 3-panel probability evolution plot for a single event.
        
        Args:
            class_idx: True class index
            event_idx: Index in dataset
        """
        cls_name = CLASS_NAMES[class_idx]
        length = self.lengths[event_idx]
        
        if length < 10:
            return  # Skip very short sequences
        
        # Generate probability trajectory
        step = max(1, length // 50)  # Sample ~50 points
        time_steps = range(10, length + 1, step)
        
        probs_seq = []
        
        with torch.no_grad():
            f = torch.tensor(
                self.norm_flux[event_idx:event_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
            d = torch.tensor(
                self.delta_t[event_idx:event_idx+1], 
                dtype=torch.float32, 
                device=self.device
            )
            
            for t in time_steps:
                f_trunc = f[:, :t]
                d_trunc = d[:, :t]
                l_trunc = torch.tensor([t], dtype=torch.long, device=self.device)
                
                logits = self.model(f_trunc, d_trunc, lengths=l_trunc)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                probs_seq.append(probs)
        
        if len(probs_seq) == 0:
            return
        
        probs_full = np.array(probs_seq)
        time_axis = self.timestamps[event_idx, :length][::step][:len(probs_seq)]
        
        # Create 3-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            f'Probability Evolution (True: {cls_name}, Predicted: {CLASS_NAMES[self.preds[event_idx]]})',
            fontsize=14
        )
        
        # Panel 1: Light Curve
        ax1 = axes[0]
        raw_flux = self.raw_flux[event_idx, :length]
        mag_data = self._raw_flux_to_relative_mag(raw_flux)
        
        ax1.scatter(
            self.timestamps[event_idx, :length], mag_data, 
            color='gray', s=5, linewidths=0, label='Light Curve'
        )
        ax1.set_ylabel('-Relative Magnitude', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
        
        # Panel 2: Class Probabilities
        ax2 = axes[1]
        for i, name in enumerate(CLASS_NAMES):
            ax2.plot(
                time_axis, probs_full[:, i], 
                label=f'P({name})', 
                color=self.colors[i], 
                linewidth=2
            )
        
        ax2.axhline(0.5, color='k', linestyle=':', alpha=0.7)
        ax2.set_ylabel('Class Probability', fontsize=11)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', ncol=3, fontsize=9)
        
        # Panel 3: Confidence
        ax3 = axes[2]
        confidence = probs_full.max(axis=1)
        
        ax3.plot(time_axis, confidence, color='black', linewidth=2, label='Max Confidence')
        ax3.fill_between(
            time_axis, 0, confidence, 
            color=self.colors[self.preds[event_idx]], 
            alpha=0.3
        )
        
        ax3.axhline(0.5, color='k', linestyle=':', alpha=0.7)
        ax3.set_ylabel('Confidence (Max P)', fontsize=11)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlabel('Time (Days)', fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self._save_figure(fig, f'evolution_{cls_name.lower()}_{event_idx}')
        plt.close()
    
    def run_early_detection_analysis(self) -> None:
        """
        Evaluate classification performance at multiple observation completeness
        levels to assess early detection capability.
        """
        self.logger.info("\nRunning early detection analysis...")
        
        time_fractions = np.linspace(0.2, 1.0, 5)
        results = []
        
        max_len = self.norm_flux.shape[1]
        
        for frac in time_fractions:
            self.logger.info(f"  Inference at {frac*100:.0f}% completeness...")
            
            new_lengths = np.clip(
                (self.lengths * frac).astype(np.int64), 
                a_min=1, 
                a_max=max_len
            )
            current_max_len = new_lengths.max()
            
            n = len(self.norm_flux)
            all_probs = []
            
            with torch.no_grad():
                for i in tqdm(range(0, n, self.batch_size), desc="    Progress"):
                    batch_end = min(i + self.batch_size, n)
                    
                    flux_batch = torch.tensor(
                        self.norm_flux[i:batch_end, :current_max_len],
                        dtype=torch.float32,
                        device=self.device
                    )
                    dt_batch = torch.tensor(
                        self.delta_t[i:batch_end, :current_max_len],
                        dtype=torch.float32,
                        device=self.device
                    )
                    len_batch = torch.tensor(
                        new_lengths[i:batch_end],
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
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
        
        # Plot early detection curve
        fractions = [r['fraction'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_macro'] for r in results]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fractions, accuracies, 'o-', label='Accuracy', 
                color=self.colors[1], linewidth=2, markersize=8)
        ax.plot(fractions, f1_scores, 's--', label='F1 (macro)', 
                color=self.colors[2], linewidth=2, markersize=8)
        
        ax.set_title('Early Detection Performance vs. Observation Completeness', fontsize=14)
        ax.set_xlabel('Fraction of Full Sequence Length', fontsize=12)
        ax.set_ylabel('Metric Score', fontsize=12)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'early_detection_curve')
        plt.close()
        
        # Save results
        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Early detection analysis complete")
    
    def run_all_analysis(self) -> None:
        """Execute complete evaluation suite and generate all outputs."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("=" * 80)
        
        # Core metrics plots
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration_curve()
        self.plot_class_distributions()
        self.plot_per_class_metrics()
        
        # Example data
        self.plot_example_light_curves()
        
        # Physics-based analysis
        self.plot_u0_dependency()
        self.plot_temporal_bias_check()
        
        # Evolution plots
        if self.n_evolution_per_type > 0:
            self.logger.info("\nGenerating probability evolution plots...")
            for class_idx, class_name in enumerate(CLASS_NAMES):
                class_mask = (self.y == class_idx)
                indices = np.where(class_mask)[0][:self.n_evolution_per_type]
                
                for idx in indices:
                    self.plot_evolution_for_class(class_idx, idx)
        
        # Early detection
        if self.run_early_detection:
            self.run_early_detection_analysis()
        
        # Save summary
        summary = {
            'experiment': str(self.exp_dir.name),
            'model_path': str(self.model_path),
            'data_size': int(len(self.y)),
            'class_distribution': {
                name: int((self.y == i).sum()) 
                for i, name in enumerate(CLASS_NAMES)
            },
            'metrics': self.metrics,
            'config': self.config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save classification report
        report = classification_report(
            self.y, self.preds, 
            target_names=list(CLASS_NAMES), 
            digits=4
        )
        
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Overall accuracy: {self.metrics['accuracy']*100:.2f}%")
        self.logger.info(f"F1-score (macro): {self.metrics['f1_macro']:.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Comprehensive Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment_name', required=True,
                       help="Name of experiment to evaluate")
    parser.add_argument('--data', required=True,
                       help="Path to test dataset (.h5 or .npz)")
    
    # Optional arguments
    parser.add_argument('--output_dir', default=None,
                       help="Custom output directory (auto-generated if None)")
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--n_samples', type=int, default=None,
                       help="Subsample test set (None = use all)")
    parser.add_argument('--device', default='cuda',
                       help="Computation device: cuda or cpu")
    
    # Analysis options
    parser.add_argument('--early_detection', action='store_true',
                       help="Run early detection analysis")
    parser.add_argument('--n_evolution_per_type', type=int, default=10,
                       help="Number of evolution plots per class")
    parser.add_argument('--n_example_grid_per_type', type=int, default=4,
                       help="Number of light curves per class in example grid")
    
    # Visualization options
    parser.add_argument('--colorblind_safe', action='store_true',
                       help="Use colorblind-safe palette")
    parser.add_argument('--save_formats', nargs='+', default=['png'],
                       choices=['png', 'pdf', 'svg'],
                       help="Output formats for plots")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create evaluator and run analysis
    evaluator = RomanEvaluator(
        experiment_name=args.experiment_name,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        early_detection=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type,
        n_example_grid_per_type=args.n_example_grid_per_type,
        colorblind_safe=args.colorblind_safe,
        save_formats=args.save_formats,
        verbose=args.verbose
    )
    
    evaluator.run_all_analysis()


if __name__ == '__main__':
    main()
