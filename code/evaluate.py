#!/usr/bin/env python3
"""
Roman Microlensing Classifier - Comprehensive Evaluation Suite
==============================================================

Production-grade evaluation framework for gravitational microlensing event
classification models. Computes comprehensive metrics, generates publication-
quality visualizations, and performs physics-based performance analysis.

Core Capabilities
-----------------
    * Robust model loading with checkpoint state unwrapping (DDP, compile)
    * Hybrid data loading supporting both HDF5 and NPZ formats
    * Batch inference with gradient-free computation and memory optimization
    * Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC, calibration
    * Bootstrap confidence intervals for statistical rigor
    * Early detection analysis across observation completeness fractions
    * Physics-based stratification: impact parameter (u0), timescale (t_E)
    * Probability evolution tracking for individual events
    * Publication-ready visualizations with vectorized output support

Scientific Visualization
------------------------
    * Confusion matrices with normalization options
    * ROC curves with confidence bands (bootstrap)
    * Calibration curves with reliability diagrams
    * Class probability distributions and confidence histograms
    * Light curve examples with magnitude conversion (Roman F146)
    * Temporal evolution plots with 3-panel layout
    * Impact parameter dependency analysis for binary classification
    * Colorblind-safe palette options

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization

# Safe loading for torch version serialization
try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except (AttributeError, TypeError):
    pass

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
    
    Creates a logger that writes to both a file in the output directory
    and to the console. The file receives all messages (DEBUG+), while
    the console shows INFO+ by default.
    
    Parameters
    ----------
    output_dir : Path
        Directory where the log file will be created.
    verbose : bool, optional
        If True, set console logging level to DEBUG (default: False).
        
    Returns
    -------
    logging.Logger
        Configured logger instance with both file and console handlers.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - captures all log levels
    fh = logging.FileHandler(output_dir / 'evaluation.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler - INFO level by default
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def load_data_hybrid(path: str) -> Dict[str, Any]:
    """
    Load data from HDF5 or NPZ format with comprehensive error handling.
    
    Supports both HDF5 files (.h5, .hdf5) created by simulate.py and
    NPZ files (.npz) for alternative data formats. All datasets are
    loaded into memory as NumPy arrays.
    
    Parameters
    ----------
    path : str
        Path to the data file. Must have extension .h5, .hdf5, or .npz.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping dataset names to NumPy arrays. For HDF5 files,
        this includes 'flux', 'delta_t', 'labels', 'timestamps', and
        parameter arrays like 'params_flat', 'params_pspl', 'params_binary'.
        
    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file format is not supported.
        
    Examples
    --------
    >>> data = load_data_hybrid('test_data.h5')
    >>> print(data.keys())
    dict_keys(['flux', 'delta_t', 'labels', 'timestamps', 'params_pspl', ...])
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
    Extract physical parameters from structured arrays saved by simulate.py.
    
    The simulation script saves parameters as separate structured arrays
    for each class type (params_flat, params_pspl, params_binary). This
    function merges them into full-length arrays aligned with the labels,
    enabling physics-based analysis across the entire dataset.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Raw data dictionary from load_data_hybrid(), expected to contain
        keys like 'params_flat', 'params_pspl', 'params_binary'.
    labels : np.ndarray
        Label array of shape (n_samples,) with values in {0, 1, 2}.
        Used to align parameters with their corresponding samples.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with parameter arrays, each of shape (n_samples,):
        - 't0': Peak time in days
        - 'tE': Einstein crossing time in days  
        - 'u0': Impact parameter in Einstein radii
        - 'm_base': Baseline magnitude
        - 's': Binary separation (binary only)
        - 'q': Mass ratio (binary only)
        - 'alpha': Source trajectory angle (binary only)
        - 'rho': Source radius (binary only)
        
        Values are NaN for samples where the parameter is not applicable.
        
    Notes
    -----
    The structured arrays from simulate.py have dtype with named fields.
    This function handles the mapping from class-specific arrays to the
    full dataset indices based on the label array.
    
    Examples
    --------
    >>> data = load_data_hybrid('simulation.h5')
    >>> labels = data['labels']
    >>> params = extract_parameters_from_structured(data, labels)
    >>> u0_binary = params['u0'][labels == 2]  # u0 for binary events only
    """
    n_total = len(labels)
    
    # Initialize output arrays with NaN (not applicable by default)
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
    
    # Extract from structured arrays for each class
    for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
        key = f'params_{class_name}'
        if key not in data:
            continue
            
        struct_arr = data[key]
        if not isinstance(struct_arr, np.ndarray):
            continue
        
        # Find indices for this class in the full dataset
        class_mask = (labels == class_idx)
        n_class = class_mask.sum()
        
        if n_class == 0 or len(struct_arr) == 0:
            continue
        
        # Handle potential length mismatch (shouldn't happen but be safe)
        n_available = min(len(struct_arr), n_class)
        class_indices = np.where(class_mask)[0][:n_available]
        
        # Extract fields from structured array
        if hasattr(struct_arr, 'dtype') and struct_arr.dtype.names is not None:
            for field_name in params.keys():
                if field_name in struct_arr.dtype.names:
                    params[field_name][class_indices] = struct_arr[field_name][:n_available]
    
    return params


def bootstrap_metric(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a classification metric.
    
    Uses the percentile bootstrap method to estimate uncertainty in
    metric estimates. This is essential for reporting results with
    proper statistical rigor in publications.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels or probabilities of shape (n_samples,) or
        (n_samples, n_classes).
    metric_fn : Callable
        Metric function with signature metric_fn(y_true, y_pred) -> float.
        Examples: accuracy_score, f1_score, roc_auc_score.
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 1000).
    confidence : float, optional
        Confidence level for the interval, e.g., 0.95 for 95% CI (default: 0.95).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
        
    Returns
    -------
    Tuple[float, float, float]
        - point_estimate: Mean of bootstrap distribution
        - lower_bound: Lower confidence bound
        - upper_bound: Upper confidence bound
        
    Notes
    -----
    Invalid bootstrap samples (e.g., single class present) are skipped.
    If all samples are invalid, returns (0.0, 0.0, 0.0).
    
    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 2, 0, 2, 2])
    >>> point, lower, upper = bootstrap_metric(y_true, y_pred, accuracy_score)
    >>> print(f"Accuracy: {point:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            if np.isfinite(score):
                bootstrap_scores.append(score)
        except (ValueError, ZeroDivisionError):
            # Skip invalid samples (e.g., only one class present)
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
    Load trained model from checkpoint with robust state dictionary unwrapping.
    
    Handles checkpoints saved from various training configurations:
    - Standard single-GPU training
    - DistributedDataParallel (DDP) with 'module.' prefix
    - torch.compile() with '_orig_mod.' prefix
    - Combinations of the above
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to the .pt checkpoint file containing 'model_state_dict'
        and 'config' keys.
    device : torch.device
        Target device for model (cuda or cpu).
        
    Returns
    -------
    Tuple[Any, Dict[str, Any], Dict[str, Any]]
        - model: Loaded RomanMicrolensingClassifier in eval mode
        - config_dict: Model configuration dictionary
        - checkpoint: Full checkpoint dictionary for accessing stats
        
    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    KeyError
        If checkpoint is missing required keys ('config', 'model_state_dict').
    ImportError
        If model.py cannot be imported.
        
    Notes
    -----
    The checkpoint 'stats' key contains normalization statistics that
    MUST be used during evaluation to match training normalization.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    if 'config' not in checkpoint:
        raise KeyError("Checkpoint missing 'config' key")
    
    config_data = checkpoint['config']
    if isinstance(config_data, dict):
        config_dict = config_data
    else:
        # Handle ModelConfig objects saved directly
        config_dict = config_data.to_dict() if hasattr(config_data, 'to_dict') else {}
    
    # Import model architecture
    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        from model import ModelConfig, RomanMicrolensingClassifier
        
        # Filter to valid ModelConfig keys only
        valid_keys = set(ModelConfig.__annotations__.keys())
        clean_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ModelConfig(**clean_config)
        
    except ImportError as e:
        raise ImportError(f"Failed to import model: {e}")
    
    # Create model instance
    model = RomanMicrolensingClassifier(config).to(device)
    
    # Load state dictionary
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
    if state_dict_key not in checkpoint:
        raise KeyError(f"Checkpoint missing '{state_dict_key}' key")
    
    state_dict = checkpoint[state_dict_key]
    
    # Remove DDP wrapper prefix ('module.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remove torch.compile wrapper prefix ('_orig_mod.')
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
    
    This class orchestrates the complete evaluation workflow:
    1. Model loading with proper checkpoint handling
    2. Data loading with normalization matching training
    3. Batch inference with GPU acceleration
    4. Comprehensive metric computation
    5. Publication-quality visualization generation
    6. Physics-based performance analysis
    
    Parameters
    ----------
    experiment_name : str
        Name or path of experiment to evaluate. Used to locate best_model.pt
        in the results directory structure.
    data_path : str
        Path to test dataset in HDF5 (.h5) or NPZ (.npz) format.
    output_dir : str, optional
        Custom output directory. If None, creates timestamped directory
        under the experiment folder (default: None).
    device : str, optional
        Computation device, 'cuda' or 'cpu' (default: 'cuda').
    batch_size : int, optional
        Batch size for inference (default: 128).
    n_samples : int, optional
        Subsample dataset to this many samples for faster evaluation.
        If None, uses entire dataset (default: None).
    early_detection : bool, optional
        Run early detection analysis at multiple completeness levels
        (default: False).
    n_evolution_per_type : int, optional
        Number of probability evolution plots per class (default: 0).
    n_example_grid_per_type : int, optional
        Number of example light curves per class in grid (default: 4).
    colorblind_safe : bool, optional
        Use IBM colorblind-safe palette (default: False).
    save_formats : List[str], optional
        Output formats for plots, e.g., ['png', 'pdf', 'svg'] (default: ['png']).
    verbose : bool, optional
        Enable debug-level logging (default: False).
    
    Attributes
    ----------
    model : torch.nn.Module
        Loaded classifier model in eval mode.
    config_dict : Dict[str, Any]
        Model configuration from checkpoint.
    checkpoint : Dict[str, Any]
        Full checkpoint including normalization statistics.
    norm_flux : np.ndarray
        Normalized flux data of shape (n_samples, seq_len).
    norm_delta_t : np.ndarray
        Normalized delta_t data of shape (n_samples, seq_len).
    raw_flux : np.ndarray
        Raw flux data for visualization.
    y : np.ndarray
        Ground truth labels of shape (n_samples,).
    probs : np.ndarray
        Model output probabilities of shape (n_samples, 3).
    preds : np.ndarray
        Model predictions of shape (n_samples,).
    confs : np.ndarray
        Prediction confidences (max probability) of shape (n_samples,).
    metrics : Dict[str, float]
        Computed evaluation metrics.
    params : Dict[str, np.ndarray]
        Physical parameters for each sample.
    
    Examples
    --------
    >>> evaluator = RomanEvaluator(
    ...     experiment_name='baseline_20241201',
    ...     data_path='test_data.h5',
    ...     early_detection=True,
    ...     save_formats=['png', 'pdf']
    ... )
    >>> evaluator.run_all_analysis()
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
        save_formats: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """Initialize evaluator with model and data."""
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
        self.logger.info("ROMAN SPACE TELESCOPE - MICROLENSING CLASSIFIER EVALUATION")
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
        
        # Load and prepare data with proper normalization
        self._load_and_prepare_data(data_path)
        
        # Run inference
        self.logger.info("\nRunning inference on test set...")
        self.probs, self.preds, self.confs = self._run_inference()
        
        # Compute metrics
        self.metrics = self._compute_metrics()
    
    def _find_best_model(self, exp_name: str) -> Tuple[Path, Path]:
        """
        Locate best_model.pt for the given experiment name.
        
        Searches common result directory locations for experiment folders
        matching the provided name. Returns the most recently modified
        matching directory.
        
        Parameters
        ----------
        exp_name : str
            Experiment name, partial name, or direct path to experiment directory.
            
        Returns
        -------
        Tuple[Path, Path]
            - model_path: Path to best_model.pt (or first .pt file found)
            - exp_dir: Path to experiment directory
            
        Raises
        ------
        FileNotFoundError
            If no matching experiment or checkpoint file is found.
        """
        search_roots = [Path('../results'), Path('results'), Path('.')]
        candidates = []
        
        for root in search_roots:
            if root.exists():
                candidates.extend(list(root.glob(f"*{exp_name}*")))
        
        if not candidates:
            # Try direct path
            if Path(exp_name).exists() and Path(exp_name).is_dir():
                exp_dir = Path(exp_name)
            else:
                raise FileNotFoundError(
                    f"No experiment matching '{exp_name}' found in {search_roots}"
                )
        else:
            # Use most recently modified match
            exp_dir = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        
        model_file = exp_dir / "best_model.pt"
        
        if not model_file.exists():
            # Fall back to any .pt file
            pt_files = list(exp_dir.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt file found in {exp_dir}")
        
        return model_file, exp_dir
    
    def _load_and_prepare_data(self, data_path: str) -> None:
        """
        Load and prepare data with proper normalization for both flux and delta_t.
        
        This method implements the CRITICAL normalization matching between
        training and evaluation. The checkpoint contains statistics computed
        during training, and these MUST be used to normalize evaluation data
        identically.
        
        Parameters
        ----------
        data_path : str
            Path to data file (.h5 or .npz).
            
        Raises
        ------
        KeyError
            If required datasets ('flux', 'delta_t', 'labels') are missing.
            
        Notes
        -----
        If checkpoint statistics are missing, the method falls back to
        computing statistics from the evaluation data. This is logged as
        a warning since it may cause distribution mismatch.
        """
        self.logger.info("\nLoading data...")
        data_dict = load_data_hybrid(data_path)
        
        # Validate required datasets
        for key in ['flux', 'delta_t', 'labels']:
            if key not in data_dict:
                raise KeyError(f"Data missing '{key}' key")
        
        raw_flux = data_dict['flux']
        raw_delta_t = data_dict['delta_t']
        labels = data_dict['labels']
        
        # Optional subsampling for faster evaluation
        if self.n_samples is not None and self.n_samples < len(labels):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(labels), self.n_samples, replace=False)
            raw_flux = raw_flux[indices]
            raw_delta_t = raw_delta_t[indices]
            labels = labels[indices]
            self.logger.info(f"Subsampled to {self.n_samples} events")
        
        # Compute sequence lengths (non-zero observations)
        mask = (raw_flux != 0)
        lengths = mask.sum(axis=1).astype(np.int64)
        
        # =====================================================================
        # CRITICAL: Load normalization statistics from checkpoint
        # Training normalizes flux and delta_t with SEPARATE statistics.
        # Evaluation MUST use the SAME statistics for consistent inference.
        # =====================================================================
        
        stats = self.checkpoint.get('stats', {})
        
        # Flux normalization
        if 'norm_median' in stats and 'norm_iqr' in stats:
            flux_median = float(stats['norm_median'])
            flux_iqr = float(stats['norm_iqr'])
            self.logger.info("Using flux normalization from checkpoint")
        else:
            # Fallback: compute from data (may cause mismatch)
            flux_valid = raw_flux[raw_flux != 0]
            flux_median = float(np.median(flux_valid))
            flux_iqr = float(np.percentile(flux_valid, 75) - np.percentile(flux_valid, 25))
            self.logger.warning("Checkpoint missing flux stats, computing from data")
        
        # Delta_t normalization (CRITICAL FIX in v2.1)
        if 'delta_t_median' in stats and 'delta_t_iqr' in stats:
            delta_t_median = float(stats['delta_t_median'])
            delta_t_iqr = float(stats['delta_t_iqr'])
            self.logger.info("Using delta_t normalization from checkpoint")
        else:
            # Fallback: compute from data
            delta_t_valid = raw_delta_t[raw_delta_t > 0]
            if len(delta_t_valid) > 0:
                delta_t_median = float(np.median(delta_t_valid))
                delta_t_iqr = float(np.percentile(delta_t_valid, 75) - 
                                   np.percentile(delta_t_valid, 25))
            else:
                delta_t_median = 0.0
                delta_t_iqr = 1.0
            self.logger.warning("Checkpoint missing delta_t stats, computing from data")
        
        # Ensure IQR is not zero
        flux_iqr = max(flux_iqr, EPS)
        delta_t_iqr = max(delta_t_iqr, EPS)
        
        # Apply normalization
        norm_flux = (raw_flux - flux_median) / flux_iqr
        norm_flux[~mask] = 0.0  # Preserve padding
        
        norm_delta_t = (raw_delta_t - delta_t_median) / delta_t_iqr
        norm_delta_t[~mask] = 0.0  # Preserve padding
        
        # Load timestamps
        if 'timestamps' in data_dict:
            timestamps = data_dict['timestamps']
        else:
            # Generate default timestamps
            max_len = raw_flux.shape[1]
            timestamps = np.tile(np.linspace(0, 200, max_len), (len(labels), 1))
            self.logger.warning("Timestamps missing, using default 0-200 days")
        
        # Extract physical parameters
        try:
            params = extract_parameters_from_structured(data_dict, labels)
            self.logger.info("Extracted parameters from structured arrays")
        except Exception as e:
            self.logger.warning(f"Parameter extraction failed: {e}")
            params = {}
        
        # Store processed data
        self.norm_flux = norm_flux.astype(np.float32)
        self.norm_delta_t = norm_delta_t.astype(np.float32)
        self.raw_flux = raw_flux.astype(np.float32)
        self.raw_delta_t = raw_delta_t.astype(np.float32)
        self.y = labels.astype(np.int64)
        self.lengths = lengths
        self.timestamps = timestamps.astype(np.float32)
        self.params = params
        
        # Store normalization parameters for reference
        self.flux_median = flux_median
        self.flux_iqr = flux_iqr
        self.delta_t_median = delta_t_median
        self.delta_t_iqr = delta_t_iqr
        
        self.logger.info(f"Data loaded: {len(self.y)} events")
        self.logger.info(f"Class distribution: {np.bincount(self.y, minlength=3)}")
        self.logger.info(f"Flux normalization: median={flux_median:.4f}, IQR={flux_iqr:.4f}")
        self.logger.info(f"Delta_t normalization: median={delta_t_median:.4f}, IQR={delta_t_iqr:.4f}")
    
    def _run_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batch inference on the test set with optimized memory handling.
        
        Uses torch.inference_mode() for maximum performance and non_blocking
        transfers for GPU pipelining.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - probs: Class probabilities of shape (n_samples, 3)
            - preds: Predictions of shape (n_samples,)
            - confs: Confidences of shape (n_samples,)
        """
        n = len(self.norm_flux)
        all_probs = []
        
        self.model.eval()
        
        with torch.inference_mode():
            for i in tqdm(range(0, n, self.batch_size), desc="Inference"):
                batch_end = min(i + self.batch_size, n)
                
                flux_batch = torch.from_numpy(
                    self.norm_flux[i:batch_end]
                ).to(self.device, non_blocking=True)
                
                dt_batch = torch.from_numpy(
                    self.norm_delta_t[i:batch_end]
                ).to(self.device, non_blocking=True)
                
                len_batch = torch.from_numpy(
                    self.lengths[i:batch_end]
                ).to(self.device, non_blocking=True)
                
                logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        
        return probs, preds, confs
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics with robust edge-case handling.
        
        Computes accuracy, precision, recall, F1 scores (per-class and macro),
        ROC-AUC scores, and bootstrap confidence intervals for accuracy.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - accuracy: Overall accuracy
            - precision_macro, recall_macro, f1_macro: Macro-averaged metrics
            - {Class}_precision, {Class}_recall, {Class}_f1: Per-class metrics
            - roc_auc_macro, roc_auc_weighted: Multi-class ROC-AUC
            - accuracy_ci_lower, accuracy_ci_upper: Bootstrap 95% CI
        """
        self.logger.info("\nComputing metrics...")
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = float(accuracy_score(self.y, self.preds))
        metrics['precision_macro'] = float(
            precision_score(self.y, self.preds, average='macro', zero_division=0)
        )
        metrics['recall_macro'] = float(
            recall_score(self.y, self.preds, average='macro', zero_division=0)
        )
        metrics['f1_macro'] = float(
            f1_score(self.y, self.preds, average='macro', zero_division=0)
        )
        
        # Per-class metrics
        precision_per_class = precision_score(
            self.y, self.preds, average=None, zero_division=0, labels=[0, 1, 2]
        )
        recall_per_class = recall_score(
            self.y, self.preds, average=None, zero_division=0, labels=[0, 1, 2]
        )
        f1_per_class = f1_score(
            self.y, self.preds, average=None, zero_division=0, labels=[0, 1, 2]
        )
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(precision_per_class):
                metrics[f'{class_name}_precision'] = float(precision_per_class[i])
                metrics[f'{class_name}_recall'] = float(recall_per_class[i])
                metrics[f'{class_name}_f1'] = float(f1_per_class[i])
            else:
                metrics[f'{class_name}_precision'] = 0.0
                metrics[f'{class_name}_recall'] = 0.0
                metrics[f'{class_name}_f1'] = 0.0
        
        # ROC-AUC with edge case handling
        try:
            unique_classes = np.unique(self.y)
            n_unique = len(unique_classes)
            
            if n_unique >= 2:
                y_bin = label_binarize(self.y, classes=[0, 1, 2])
                
                if n_unique == 2:
                    # Only 2 classes: compute for available classes
                    valid_cols = [i for i in range(3) if i in unique_classes]
                    y_bin_valid = y_bin[:, valid_cols]
                    probs_valid = self.probs[:, valid_cols]
                    probs_valid = probs_valid / (probs_valid.sum(axis=1, keepdims=True) + EPS)
                    
                    metrics['roc_auc_macro'] = float(
                        roc_auc_score(y_bin_valid, probs_valid, average='macro')
                    )
                    metrics['roc_auc_weighted'] = float(
                        roc_auc_score(y_bin_valid, probs_valid, average='weighted')
                    )
                else:
                    # All 3 classes present
                    metrics['roc_auc_macro'] = float(
                        roc_auc_score(y_bin, self.probs, average='macro', multi_class='ovr')
                    )
                    metrics['roc_auc_weighted'] = float(
                        roc_auc_score(y_bin, self.probs, average='weighted', multi_class='ovr')
                    )
            else:
                self.logger.warning("Only 1 class present, ROC-AUC undefined")
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_weighted'] = 0.0
                
        except ValueError as e:
            self.logger.warning(f"ROC-AUC computation failed: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
        
        # Bootstrap confidence intervals
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
        
        The simulation outputs flux in magnitude space. This method
        computes relative magnitude (deviation from baseline) with
        proper sign convention for astronomical plots.
        
        Parameters
        ----------
        raw_flux : np.ndarray
            Raw flux array from simulation (in magnitude units).
            Padding zeros are converted to NaN.
            
        Returns
        -------
        np.ndarray
            Relative magnitude array where:
            - Negative values = brighter than baseline
            - Positive values = fainter than baseline
            - NaN = padded/masked observations
            
        Notes
        -----
        Sign convention: -Δm so that magnification events appear as
        positive peaks (brighter = more negative mag = positive on plot).
        """
        valid_mask = (raw_flux != 0)
        
        if not valid_mask.any():
            return np.full_like(raw_flux, np.nan)
        
        # Baseline magnitude from valid observations
        median_mag = np.median(raw_flux[valid_mask])
        
        # Relative magnitude with sign flip (brighter = more negative)
        relative_mag = -(raw_flux - median_mag)
        
        # Mark padding as NaN
        relative_mag[~valid_mask] = np.nan
        
        return relative_mag
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure in multiple formats with appropriate settings.
        
        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to save.
        filename : str
            Base filename without extension.
        """
        for fmt in self.save_formats:
            filepath = self.output_dir / f'{filename}.{fmt}'
            if fmt == 'png':
                fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
            else:
                fig.savefig(filepath, format=fmt, bbox_inches='tight')
    
    def plot_confusion_matrix(self) -> None:
        """
        Generate row-normalized confusion matrix heatmap.
        
        Saves a heatmap showing classification performance with both
        normalized fractions and raw counts annotated.
        """
        self.logger.info("Generating confusion matrix...")
        
        cm = confusion_matrix(self.y, self.preds, labels=[0, 1, 2])
        
        # Row normalization with zero handling
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype('float') / row_sums
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Fraction'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix (Row-Normalized)', fontsize=14)
        
        # Add raw counts as secondary annotation
        for i in range(3):
            for j in range(3):
                ax.text(j + 0.5, i + 0.75, f'n={cm[i, j]}',
                       ha='center', va='center', fontsize=8, color='gray')
        
        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix')
        plt.close()
    
    def plot_roc_curves(self) -> None:
        """
        Generate one-vs-rest ROC curves with AUC scores.
        
        Creates ROC curves for each class showing true positive rate
        vs false positive rate, with AUC values in the legend.
        """
        self.logger.info("Generating ROC curves...")
        
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            self.logger.warning("Skipping ROC curves (only 1 class present)")
            return
        
        y_bin = label_binarize(self.y, classes=[0, 1, 2])
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i not in unique_classes:
                continue
            
            if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
                continue
            
            fpr, tpr, _ = roc_curve(y_bin[:, i], self.probs[:, i])
            auc_score = roc_auc_score(y_bin[:, i], self.probs[:, i])
            
            ax.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {auc_score:.3f})',
                color=self.colors[i],
                linewidth=2
            )
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        plt.tight_layout()
        self._save_figure(fig, 'roc_curves')
        plt.close()
    
    def plot_calibration_curve(self) -> None:
        """
        Generate calibration curve and confidence distribution.
        
        Left panel shows reliability diagram (calibration curve).
        Right panel shows histogram of prediction confidences.
        """
        self.logger.info("Generating calibration curve...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration curve
        for i, class_name in enumerate(CLASS_NAMES):
            if i not in np.unique(self.y):
                continue
            
            y_binary = (self.y == i).astype(int)
            prob_class = self.probs[:, i]
            
            if y_binary.sum() == 0:
                continue
            
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
                    linewidth=2,
                    markersize=6
                )
            except ValueError as e:
                self.logger.warning(f"Calibration for {class_name} failed: {e}")
                continue
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)
        
        # Confidence histogram
        ax2.hist(self.confs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(self.confs.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.confs.mean():.3f}')
        ax2.axvline(np.median(self.confs), color='orange', linestyle=':', 
                   linewidth=2, label=f'Median: {np.median(self.confs):.3f}')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'calibration')
        plt.close()
    
    def plot_class_distributions(self) -> None:
        """
        Generate per-class probability distributions.
        
        Shows how probability for each class is distributed across
        samples from each true class, revealing confusion patterns.
        """
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
                    edgecolor='black',
                    linewidth=0.5
                )
            
            ax.set_xlabel(f'P({class_name})', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{class_name} Probability', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim(-0.02, 1.02)
        
        plt.tight_layout()
        self._save_figure(fig, 'class_distributions')
        plt.close()
    
    def plot_per_class_metrics(self) -> None:
        """
        Generate per-class precision, recall, F1 bar chart.
        
        Grouped bar chart comparing metrics across classes.
        """
        self.logger.info("Generating per-class metrics...")
        
        precision = [self.metrics[f'{name}_precision'] for name in CLASS_NAMES]
        recall = [self.metrics[f'{name}_recall'] for name in CLASS_NAMES]
        f1 = [self.metrics[f'{name}_f1'] for name in CLASS_NAMES]
        
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', 
                       color=self.colors[0], alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', 
                       color=self.colors[1], alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                       color=self.colors[2], alpha=0.8, edgecolor='black')
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'per_class_metrics')
        plt.close()
    
    def plot_example_light_curves(self) -> None:
        """
        Generate grid of example light curves for each class.
        
        Shows representative light curves with true and predicted labels.
        """
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
                mag_data = self._raw_flux_to_relative_mag(raw_flux)
                
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
                ax.set_ylabel(r'$-\Delta m$ (mag)', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Example Light Curves', fontsize=16, y=0.995)
        plt.tight_layout()
        self._save_figure(fig, 'example_light_curves')
        plt.close()
    
    def plot_u0_dependency(self) -> None:
        """
        Generate binary classification accuracy vs impact parameter u0.
        
        Physics: Events with large u0 have weak magnification and are
        harder to distinguish from PSPL events.
        """
        if 'u0' not in self.params or np.all(np.isnan(self.params['u0'])):
            self.logger.warning("Skipping u0 analysis (parameters unavailable)")
            return
        
        self.logger.info("Generating u0 dependency analysis...")
        
        binary_mask = (self.y == 2)
        if binary_mask.sum() == 0:
            self.logger.warning("No binary events in dataset")
            return
        
        u0_vals = self.params['u0'][binary_mask]
        correct = (self.preds[binary_mask] == 2).astype(int)
        
        valid = ~np.isnan(u0_vals)
        u0_vals = u0_vals[valid]
        correct = correct[valid]
        
        if len(u0_vals) == 0:
            self.logger.warning("No valid u0 values for binary events")
            return
        
        # Bin by u0
        u0_max = min(1.0, np.percentile(u0_vals, 99))
        u0_bins = np.linspace(0, u0_max, 11)
        bin_centers = (u0_bins[:-1] + u0_bins[1:]) / 2
        accuracy_per_bin = []
        stderr_per_bin = []
        counts_per_bin = []
        
        for i in range(len(u0_bins) - 1):
            mask = (u0_vals >= u0_bins[i]) & (u0_vals < u0_bins[i+1])
            n_bin = mask.sum()
            counts_per_bin.append(n_bin)
            
            if n_bin > 0:
                acc = correct[mask].mean()
                stderr = np.sqrt(acc * (1 - acc) / n_bin) if n_bin > 1 else 0
                accuracy_per_bin.append(acc)
                stderr_per_bin.append(stderr)
            else:
                accuracy_per_bin.append(np.nan)
                stderr_per_bin.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(
            bin_centers, accuracy_per_bin, yerr=stderr_per_bin,
            fmt='o-', color=self.colors[2], linewidth=2, markersize=8,
            capsize=5, label='Binary Classification Accuracy'
        )
        
        ax.axhline(0.5, color='k', linestyle='--', label='Random')
        ax.set_xlabel(r'Impact Parameter $u_0$', fontsize=12)
        ax.set_ylabel('Binary Classification Accuracy', fontsize=12)
        ax.set_title(r'Binary Detection vs Impact Parameter', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for i, (x, y, n) in enumerate(zip(bin_centers, accuracy_per_bin, counts_per_bin)):
            if not np.isnan(y) and n > 0:
                ax.annotate(f'n={n}', (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=7, color='gray')
        
        plt.tight_layout()
        self._save_figure(fig, 'u0_dependency')
        plt.close()
    
    def plot_temporal_bias_check(self) -> None:
        """
        Check for temporal bias via t0 distribution comparison.
        
        Uses Kolmogorov-Smirnov test to verify model doesn't learn
        temporal shortcuts.
        """
        if 't0' not in self.params or np.all(np.isnan(self.params['t0'])):
            self.logger.warning("Skipping temporal bias check (t0 unavailable)")
            return
        
        self.logger.info("Generating temporal bias check...")
        
        correct_mask = (self.preds == self.y)
        t0_correct = self.params['t0'][correct_mask]
        t0_incorrect = self.params['t0'][~correct_mask]
        
        t0_correct = t0_correct[~np.isnan(t0_correct)]
        t0_incorrect = t0_incorrect[~np.isnan(t0_incorrect)]
        
        if len(t0_correct) == 0 or len(t0_incorrect) == 0:
            self.logger.warning("Insufficient data for temporal bias check")
            return
        
        ks_stat, p_value = ks_2samp(t0_correct, t0_incorrect)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(t0_correct, bins=30, alpha=0.6, label='Correct',
                color='green', edgecolor='black', density=True)
        ax.hist(t0_incorrect, bins=30, alpha=0.6, label='Incorrect',
                color='red', edgecolor='black', density=True)
        
        ax.set_xlabel(r'Peak Time $t_0$ (days)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(
            f'Temporal Bias: KS={ks_stat:.4f}, p={p_value:.4f}',
            fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        if p_value > 0.05:
            bias_text = "No significant temporal bias (p > 0.05)"
            text_color = 'green'
        else:
            bias_text = "Temporal bias detected (p < 0.05)"
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
        
        Parameters
        ----------
        class_idx : int
            True class index (0, 1, or 2).
        event_idx : int
            Index of event in dataset.
        """
        cls_name = CLASS_NAMES[class_idx]
        length = int(self.lengths[event_idx])
        
        if length < 10:
            return
        
        # Sample ~50 points along the sequence
        step = max(1, length // 50)
        time_steps = list(range(10, length + 1, step))
        if time_steps[-1] != length:
            time_steps.append(length)
        
        # Pre-allocate and batch inference
        probs_seq = []
        
        with torch.inference_mode():
            f_full = torch.from_numpy(
                self.norm_flux[event_idx:event_idx+1]
            ).to(self.device)
            d_full = torch.from_numpy(
                self.norm_delta_t[event_idx:event_idx+1]
            ).to(self.device)
            
            for t in time_steps:
                f_trunc = f_full[:, :t]
                d_trunc = d_full[:, :t]
                l_trunc = torch.tensor([t], dtype=torch.long, device=self.device)
                
                logits = self.model(f_trunc, d_trunc, lengths=l_trunc)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                probs_seq.append(probs)
        
        if len(probs_seq) == 0:
            return
        
        probs_full = np.array(probs_seq)
        time_axis = np.array([self.timestamps[event_idx, t-1] for t in time_steps])
        
        # 3-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            f'Evolution (True: {cls_name}, Pred: {CLASS_NAMES[self.preds[event_idx]]})',
            fontsize=14
        )
        
        # Panel 1: Light Curve
        ax1 = axes[0]
        raw_flux = self.raw_flux[event_idx, :length]
        mag_data = self._raw_flux_to_relative_mag(raw_flux)
        
        ax1.scatter(
            self.timestamps[event_idx, :length], mag_data, 
            color='gray', s=5, linewidths=0
        )
        ax1.set_ylabel(r'$-\Delta m$', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
        
        # Panel 2: Probabilities
        ax2 = axes[1]
        for i, name in enumerate(CLASS_NAMES):
            ax2.plot(time_axis, probs_full[:, i], label=f'P({name})', 
                    color=self.colors[i], linewidth=2)
        
        ax2.axhline(0.5, color='k', linestyle=':', alpha=0.7)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left', ncol=3, fontsize=9)
        
        # Panel 3: Confidence
        ax3 = axes[2]
        confidence = probs_full.max(axis=1)
        
        ax3.plot(time_axis, confidence, color='black', linewidth=2)
        ax3.fill_between(
            time_axis, 0, confidence, 
            color=self.colors[self.preds[event_idx]], 
            alpha=0.3
        )
        
        ax3.axhline(0.5, color='k', linestyle=':', alpha=0.7)
        ax3.set_ylabel('Confidence', fontsize=11)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlabel('Time (Days)', fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self._save_figure(fig, f'evolution_{cls_name.lower()}_{event_idx}')
        plt.close()
    
    def run_early_detection_analysis(self) -> None:
        """
        Evaluate classification at multiple observation completeness levels.
        
        Tests model performance with truncated sequences to assess
        early detection capability.
        """
        self.logger.info("\nRunning early detection analysis...")
        
        time_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        results = []
        
        max_len = self.norm_flux.shape[1]
        
        for frac in time_fractions:
            self.logger.info(f"  Evaluating at {frac*100:.0f}% completeness...")
            
            new_lengths = np.clip(
                (self.lengths * frac).astype(np.int64), 
                a_min=1, 
                a_max=max_len
            )
            current_max_len = int(new_lengths.max())
            
            n = len(self.norm_flux)
            all_probs = []
            
            with torch.inference_mode():
                for i in tqdm(range(0, n, self.batch_size), desc="    Progress", leave=False):
                    batch_end = min(i + self.batch_size, n)
                    
                    flux_batch = torch.from_numpy(
                        self.norm_flux[i:batch_end, :current_max_len]
                    ).to(self.device)
                    
                    dt_batch = torch.from_numpy(
                        self.norm_delta_t[i:batch_end, :current_max_len]
                    ).to(self.device)
                    
                    len_batch = torch.from_numpy(
                        new_lengths[i:batch_end]
                    ).to(self.device)
                    
                    logits = self.model(flux_batch, dt_batch, lengths=len_batch)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
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
        
        # Plot
        fractions = [r['fraction'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_macro'] for r in results]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fractions, accuracies, 'o-', label='Accuracy', 
                color=self.colors[1], linewidth=2, markersize=8)
        ax.plot(fractions, f1_scores, 's--', label='F1 (macro)', 
                color=self.colors[2], linewidth=2, markersize=8)
        
        ax.set_title('Early Detection Performance', fontsize=14)
        ax.set_xlabel('Sequence Completeness', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'early_detection_curve')
        plt.close()
        
        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Early detection analysis complete")
    
    def run_all_analysis(self) -> None:
        """Execute complete evaluation suite and generate all outputs."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("=" * 80)
        
        # Core metrics
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration_curve()
        self.plot_class_distributions()
        self.plot_per_class_metrics()
        
        # Examples
        self.plot_example_light_curves()
        
        # Physics-based
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
            'normalization': {
                'flux_median': self.flux_median,
                'flux_iqr': self.flux_iqr,
                'delta_t_median': self.delta_t_median,
                'delta_t_iqr': self.delta_t_iqr
            },
            'metrics': self.metrics,
            'config': self.config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        report = classification_report(
            self.y, self.preds, 
            target_names=list(CLASS_NAMES), 
            digits=4,
            labels=[0, 1, 2],
            zero_division=0
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
# MAIN
# =============================================================================

def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--experiment_name', required=True,
                       help="Name of experiment to evaluate")
    parser.add_argument('--data', required=True,
                       help="Path to test dataset (.h5 or .npz)")
    
    parser.add_argument('--output_dir', default=None,
                       help="Custom output directory")
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--n_samples', type=int, default=None,
                       help="Subsample test set")
    parser.add_argument('--device', default='cuda',
                       help="Device: cuda or cpu")
    
    parser.add_argument('--early_detection', action='store_true',
                       help="Run early detection analysis")
    parser.add_argument('--n_evolution_per_type', type=int, default=10,
                       help="Evolution plots per class")
    parser.add_argument('--n_example_grid_per_type', type=int, default=4,
                       help="Examples per class in grid")
    
    parser.add_argument('--colorblind_safe', action='store_true',
                       help="Use colorblind-safe palette")
    parser.add_argument('--save_formats', nargs='+', default=['png'],
                       choices=['png', 'pdf', 'svg'],
                       help="Output formats")
    parser.add_argument('--verbose', action='store_true',
                       help="Debug logging")
    
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
        n_example_grid_per_type=args.n_example_grid_per_type,
        colorblind_safe=args.colorblind_safe,
        save_formats=args.save_formats,
        verbose=args.verbose
    )
    
    evaluator.run_all_analysis()


if __name__ == '__main__':
    main()
