#!/usr/bin/env python3
"""
Roman Microlensing Classifier - Evaluation Suite v4.1.0
=======================================================

Production-grade evaluation framework for gravitational microlensing event
classification models. Computes metrics, generates publication-quality
visualizations, and performs physics-based performance analysis.

VERSION 4.1.0 FIXES:
--------------------
    * CRITICAL FIX: Subsampling now returns file indices; m_base and params
      extraction uses correct file indices instead of post-subsample indices
    * CRITICAL FIX: torch.serialization import guarded to prevent crash on
      some PyTorch versions
    * CRITICAL FIX: torch.load() wrapped for weights_only compatibility with
      older PyTorch versions
    * CRITICAL FIX: torch.cuda.empty_cache() guarded with is_available() check
    * HIGH IMPACT FIX: O(N^2) params extraction replaced with O(N) precomputed
      within-class index mapping
    * FIX: Plotting now uses times >= 0 instead of times > 0 (Roman cadence
      starts at t=0.0 by design)
    * FIX: Removed unused seaborn import; guarded matplotlib style selection
    * FIX: ROC curve per-class wrapped in try/except to handle missing classes
    * FIX: Comment corrected for EVOLUTION_OBS_COUNTS (~45 checkpoints)

VERSION 4.0.0 FIXES:
--------------------
    * CRITICAL FIX: EVOLUTION_OBS_COUNTS reduced from 1364 to ~45 checkpoints
      (was range(100, 6920, 5) causing extreme slowdown in evolution plots)
    * CRITICAL FIX: Added ROMAN_CADENCE_DAYS constant for v4 simulate.py sync
    * CLEANUP: Marked unused functions as legacy utilities
    * CLEANUP: Shortened docstrings throughout
    * SYNC: Compatible with simulate.py v4.0.0 and model.py v4.0.0

VERSION 3.1.0 FIXES:
--------------------
    * m_base loading from global array (simulate.py v3.1.0+)
    * extract_baseline_magnitudes() checks global 'm_base' dataset FIRST

VERSION 3.0.2 FIXES:
--------------------
    * plot_evolution_for_class padding bug (normalized padding != 0.0)
    * run_early_detection_analysis padding bug
    * get_valid_lengths() broken for normalized data (deprecated)
    * run_inference() now uses masked pooling via valid_lengths

DATA FORMAT NOTE:
-----------------
HDF5 'flux' key contains MAGNIFICATION (A):
  - A = 1.0: baseline (unmagnified)
  - A > 1.0: magnified
  - A = 0.0: masked/invalid

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 4.1.0
Date: January 2025
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
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

# v4.1.0 FIX: Guard torch.serialization import to prevent crash on some PyTorch versions
try:
    import torch.serialization  # type: ignore
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except (AttributeError, TypeError):
        pass
except ImportError:
    pass

# Non-interactive backend for cluster environments
try:
    matplotlib.use('Agg')
except Exception:
    pass

import matplotlib.pyplot as plt
# v4.1.0 FIX: Removed unused seaborn import
from scipy import stats
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

__version__: Final[str] = "4.1.0"

# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES: Final[Tuple[str, ...]] = ('Flat', 'PSPL', 'Binary')
NUM_CLASSES: Final[int] = 3
INVALID_TIMESTAMP: Final[float] = -999.0

# v4.0.0: Sync with simulate.py v4.0.0 cadence constants
ROMAN_CADENCE_MINUTES: Final[float] = 15.0
ROMAN_CADENCE_DAYS: Final[float] = ROMAN_CADENCE_MINUTES / (24.0 * 60.0)
ROMAN_SEASON_DURATION_DAYS: Final[float] = 72.0

# =============================================================================
# LEGACY UTILITIES (kept for diagnostic/post-processing use)
# These functions are NOT used in the main evaluation pipeline but are
# preserved for users who may need flux-to-magnitude conversions.
# =============================================================================

# AB magnitude system zero-point flux (Oke & Gunn 1983)
ROMAN_ZP_FLUX_JY: Final[float] = 3631.0


def flux_to_mag(flux_jy: np.ndarray) -> np.ndarray:
    """
    LEGACY: Convert flux (Jansky) to AB magnitude.
    
    NOTE: Not used in main pipeline. Main pipeline uses magnification_to_mag().
    Kept for users needing Jansky flux conversions.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux_jy / ROMAN_ZP_FLUX_JY)
    return mag


def magnification_to_delta_mag(A: np.ndarray) -> np.ndarray:
    """
    LEGACY: Convert magnification to delta magnitude.
    
    NOTE: Not used in main pipeline. Kept for diagnostic use.
    Returns delta_mag = -2.5 * log10(A), where A=1 gives delta_mag=0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_mag = -2.5 * np.log10(A)
    delta_mag = np.where((np.isfinite(delta_mag)) & (A > 0), delta_mag, 0.0)
    return delta_mag

# =============================================================================
# END LEGACY UTILITIES
# =============================================================================

# Color palettes
COLORS_DEFAULT: Final[List[str]] = ['#7f8c8d', '#c0392b', '#2980b9']
COLORS_COLORBLIND: Final[List[str]] = ['#0173b2', '#de8f05', '#029e73']

# Publication settings
DPI: Final[int] = 600
DPI_SCREEN: Final[int] = 120
EPS: Final[float] = 1e-8

# Figure sizes (inches)
FIG_SINGLE_COL: Final[Tuple[float, float]] = (3.5, 3.0)
FIG_DOUBLE_COL: Final[Tuple[float, float]] = (7.0, 5.0)
FIG_FULL_PAGE: Final[Tuple[float, float]] = (7.0, 9.0)
FIG_CONFUSION_MATRIX: Final[Tuple[float, float]] = (4.5, 4.0)
FIG_ROC_CURVES: Final[Tuple[float, float]] = (5.0, 4.0)
FIG_CALIBRATION: Final[Tuple[float, float]] = (8.0, 4.0)
FIG_U0_DEPENDENCY: Final[Tuple[float, float]] = (5.0, 4.0)
FIG_TEMPORAL_BIAS: Final[Tuple[float, float]] = (5.0, 4.0)
FIG_PER_CLASS_METRICS: Final[Tuple[float, float]] = (5.0, 4.0)
FIG_EVOLUTION: Final[Tuple[float, float]] = (8, 10)

# Bootstrap settings
DEFAULT_N_BOOTSTRAP: Final[int] = 1000
ROC_N_BOOTSTRAP: Final[int] = 200
MIN_SAMPLES_FOR_BOOTSTRAP: Final[int] = 100

# Evolution plot settings
EVOLUTION_N_STEPS: Final[int] = 20
EVOLUTION_MIN_VALID_POINTS: Final[int] = 100

# Early detection settings
EARLY_DETECTION_MIN_REQUIRED: Final[int] = 10
EARLY_DETECTION_FRACTIONS: Final[List[float]] = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Histogram bins
DEFAULT_HIST_BINS: Final[int] = 30
CALIBRATION_DEFAULT_BINS: Final[int] = 10

# ROC interpolation points
ROC_INTERP_POINTS: Final[int] = 100

# Confidence interval percentiles (95% CI)
CI_LOWER_PERCENTILE: Final[float] = 2.5
CI_UPPER_PERCENTILE: Final[float] = 97.5

# u0 dependency analysis
U0_BINS: Final[np.ndarray] = np.linspace(0, 1.0, 50)
U0_REFERENCE_LINE: Final[float] = 0.3

# Probability threshold for random classifier
RANDOM_CLASSIFIER_PROB: Final[float] = 1.0 / NUM_CLASSES

# Minimum valid points for plotting
MIN_VALID_POINTS_PLOT: Final[int] = 3

# Cache clear frequency (batches)
CACHE_CLEAR_FREQ: Final[int] = 100

# Synthetic timestamps (days)
SYNTHETIC_TIME_MAX: Final[float] = ROMAN_SEASON_DURATION_DAYS

# Roman baseline magnitude range
ROMAN_SOURCE_MAG_MIN: Final[float] = 18.0
ROMAN_SOURCE_MAG_MAX: Final[float] = 24.0
ROMAN_DEFAULT_BASELINE_MAG: Final[float] = 22.0

# v4.2.0: Restored dense sampling for detailed evolution plots
# For 72-day Roman season with 6912 observations (15-min cadence)
# Sample every 5 observations = ~1.25 hour resolution in evolution plots
EVOLUTION_OBS_COUNTS: Final[List[int]] = list(range(100, 6913, 5))  # 1363 checkpoints

# Font sizes for plots
FONT_SIZE_TITLE: Final[int] = 12
FONT_SIZE_LABEL: Final[int] = 10
FONT_SIZE_TICK: Final[int] = 9
FONT_SIZE_LEGEND: Final[int] = 8
FONT_SIZE_ANNOTATION: Final[int] = 7
FONT_SIZE_CONFUSION_CELL: Final[int] = 9

# Legend positioning
LEGEND_BBOX_ROC: Final[Tuple[float, float]] = (1.02, 0.5)
LEGEND_BBOX_CALIBRATION: Final[Tuple[float, float]] = (0.02, 0.98)
LEGEND_BBOX_U0: Final[Tuple[float, float]] = (0.98, 0.98)

# Annotation offsets
U0_ANNOTATION_Y_OFFSET: Final[float] = -0.12


# =============================================================================
# FLUX TO MAGNITUDE CONVERSION
# =============================================================================

def magnification_to_mag(A: np.ndarray, baseline_mag: Union[float, np.ndarray] = 22.0) -> np.ndarray:
    """
    Convert magnification to apparent AB magnitude.

    Parameters
    ----------
    A : np.ndarray
        Magnification values (A=1.0 is baseline).
    baseline_mag : float or np.ndarray
        Baseline magnitude when A=1.0.

    Returns
    -------
    np.ndarray
        Apparent magnitude (brighter = smaller value).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = baseline_mag - 2.5 * np.log10(np.maximum(A, EPS))
    mag = np.where(np.isfinite(mag) & (A > 0), mag, np.nan)
    return mag


# =============================================================================
# DEPRECATED FUNCTION
# =============================================================================

def get_valid_lengths(flux_norm: np.ndarray) -> np.ndarray:
    """
    DEPRECATED: Compute valid sequence lengths from normalized flux.

    WARNING: This function is BROKEN for normalized data because it checks
    for != 0.0, but after normalization padding = -flux_mean/flux_std != 0.0.
    Use valid_lengths from load_and_prepare_data() instead.

    Kept for backward compatibility but issues DeprecationWarning.
    """
    warnings.warn(
        "get_valid_lengths() is DEPRECATED and broken for normalized data. "
        "Use valid_lengths from load_and_prepare_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    n_samples, seq_len = flux_norm.shape
    # WARNING: This is WRONG for normalized data!
    valid_lengths = np.sum(flux_norm != 0.0, axis=1).astype(np.int32)
    valid_lengths = np.maximum(valid_lengths, 1)
    return valid_lengths


# =============================================================================
# v4.1.0: PYTORCH COMPATIBILITY HELPERS
# =============================================================================

def torch_load_compat(
    path: Path,
    map_location: Union[str, torch.device],
    weights_only: bool = False
) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint with compatibility for older PyTorch versions.
    
    v4.1.0 FIX: The weights_only parameter is not universally supported
    in older PyTorch versions. This wrapper handles the TypeError gracefully.
    
    Parameters
    ----------
    path : Path
        Path to checkpoint file.
    map_location : str or torch.device
        Device to map tensors to.
    weights_only : bool
        If True, only load weights (safer). Default is False.
    
    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older PyTorch version without weights_only parameter
        return torch.load(path, map_location=map_location)


# =============================================================================
# M_BASE AND EXPERIMENT FINDING
# =============================================================================

def extract_baseline_magnitudes(
    data_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Extract or generate baseline magnitudes (m_base) for events.

    Strategy (in order):
    1. Load from global 'm_base' dataset (v3.1.0+ format)
    2. Load from params_{class} structured arrays (legacy)
    3. Generate random m_base in Roman range [18, 24] mag
    
    Parameters
    ----------
    indices : np.ndarray
        File indices (original HDF5 row indices) for selected events.
    labels : np.ndarray
        Class labels for selected events.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_events = len(indices)
    m_base = np.full(n_events, ROMAN_DEFAULT_BASELINE_MAG, dtype=np.float32)

    def _generate_random_m_base(n: int, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed=42 + seed_offset)
        return rng.uniform(ROMAN_SOURCE_MAG_MIN, ROMAN_SOURCE_MAG_MAX, size=n).astype(np.float32)

    try:
        with h5py.File(data_path, 'r') as f:
            # Check for global m_base dataset FIRST (v3.1.0+ format)
            if 'm_base' in f:
                logger.info("Found global m_base dataset (v3.1.0+ format)")
                global_m_base = f['m_base'][:]
                
                valid_mask = (indices >= 0) & (indices < len(global_m_base))
                
                if valid_mask.all():
                    m_base = global_m_base[indices].astype(np.float32)
                    logger.info(f"Loaded m_base: min={m_base.min():.2f}, max={m_base.max():.2f}")
                    return m_base
                else:
                    n_invalid = (~valid_mask).sum()
                    logger.warning(f"{n_invalid}/{len(indices)} indices out of bounds")
                    m_base[valid_mask] = global_m_base[indices[valid_mask]].astype(np.float32)
                    return m_base

            # LEGACY: Fall back to params_{class} arrays
            logger.warning("Global m_base not found, falling back to params_{class} (legacy)")
            
            has_m_base = False
            for class_idx, class_name in enumerate(CLASS_NAMES):
                param_key = f'params_{class_name.lower()}'
                if param_key in f:
                    param_dataset = f[param_key]
                    if 'm_base' in param_dataset.dtype.names:
                        has_m_base = True
                        break

            if has_m_base:
                logger.info("Found m_base in HDF5 parameters")
                file_labels = f['labels'][:]
                
                class_counts = [(file_labels == c).sum() for c in range(NUM_CLASSES)]
                class_offsets = [0]
                for c in range(NUM_CLASSES - 1):
                    class_offsets.append(class_offsets[-1] + class_counts[c])

                for class_idx, class_name in enumerate(CLASS_NAMES):
                    class_mask = (labels == class_idx)
                    if not class_mask.any():
                        continue

                    param_key = f'params_{class_name.lower()}'
                    if param_key not in f:
                        continue

                    param_dataset = f[param_key]
                    if 'm_base' not in param_dataset.dtype.names:
                        continue

                    n_params = len(param_dataset)
                    file_indices = indices[class_mask]
                    class_specific_indices = file_indices - class_offsets[class_idx]
                    
                    valid_mask = (class_specific_indices >= 0) & (class_specific_indices < n_params)
                    
                    if valid_mask.any():
                        output_positions = np.where(class_mask)[0][valid_mask]
                        valid_class_indices = class_specific_indices[valid_mask]
                        m_base_values = param_dataset['m_base'][valid_class_indices]
                        m_base[output_positions] = m_base_values

                logger.info(f"Loaded m_base: min={m_base.min():.2f}, max={m_base.max():.2f}")

            else:
                logger.warning("m_base not found, generating random values")
                seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
                m_base = _generate_random_m_base(n_events, seed_offset)

    except Exception as e:
        logger.error(f"Error loading m_base: {e}")
        seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
        m_base = _generate_random_m_base(n_events, seed_offset)

    return m_base


def find_experiment_checkpoint(
    experiment_name: str,
    base_dir: Path = Path('../results/checkpoints'),
    logger: Optional[logging.Logger] = None
) -> Tuple[Path, Path]:
    """
    Find experiment directory and checkpoint file.

    Strategy:
    1. Check for .current_experiment file
    2. Match partial experiment names
    3. Find best.pt, best_model.pt, or latest checkpoint
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base_dir = Path(base_dir)

    # Case 1: Full path to checkpoint
    if '/' in experiment_name or experiment_name.endswith('.pt'):
        checkpoint_path = Path(experiment_name)
        if checkpoint_path.exists():
            exp_dir = checkpoint_path.parent
            logger.info(f"Using checkpoint: {checkpoint_path}")
            return exp_dir, checkpoint_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Case 2: Check .current_experiment file
    current_exp_file = base_dir / '.current_experiment'
    if current_exp_file.exists():
        with open(current_exp_file, 'r') as f:
            saved_exp_name = f.read().strip()

        exp_dir = base_dir / saved_exp_name
        if exp_dir.exists():
            logger.info(f"Found current experiment: {exp_dir.name}")
            checkpoint_path = _find_checkpoint_in_dir(exp_dir, logger)
            if checkpoint_path:
                return exp_dir, checkpoint_path

    # Case 3: Match partial name
    if base_dir.exists():
        matching_dirs = []
        for candidate in base_dir.iterdir():
            if candidate.is_dir() and experiment_name.lower() in candidate.name.lower():
                matching_dirs.append(candidate)

        if matching_dirs:
            exp_dir = sorted(matching_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            logger.info(f"Matched experiment: {exp_dir.name}")

            checkpoint_path = _find_checkpoint_in_dir(exp_dir, logger)
            if checkpoint_path:
                return exp_dir, checkpoint_path

    # Case 4: Try ../results/
    results_dir = Path('../results')
    if results_dir.exists():
        exp_dir = results_dir / experiment_name
        if exp_dir.exists() and exp_dir.is_dir():
            logger.info(f"Found experiment in ../results/: {exp_dir.name}")

            checkpoint_path = _find_checkpoint_in_dir(exp_dir, logger)
            if checkpoint_path:
                return exp_dir, checkpoint_path

    raise FileNotFoundError(
        f"Could not find experiment '{experiment_name}' in:\n"
        f" - {base_dir}\n"
        f" - {results_dir}\n"
    )


def _find_checkpoint_in_dir(exp_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Find best checkpoint in experiment directory."""
    if logger is None:
        logger = logging.getLogger(__name__)

    # Priority order
    for filename in ['best.pt', 'best_model.pt', 'checkpoint_latest.pt']:
        checkpoint = exp_dir / filename
        if checkpoint.exists():
            logger.info(f"Using checkpoint: {filename}")
            return checkpoint

    # Most recent checkpoint_*.pt
    checkpoints = list(exp_dir.glob('checkpoint_*.pt'))
    if checkpoints:
        checkpoint = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info(f"Using recent checkpoint: {checkpoint.name}")
        return checkpoint

    # Check checkpoints subdirectory
    checkpoints_subdir = exp_dir / 'checkpoints'
    if checkpoints_subdir.exists():
        for filename in ['best.pt', 'best_model.pt', 'checkpoint_latest.pt']:
            checkpoint = checkpoints_subdir / filename
            if checkpoint.exists():
                logger.info(f"Using checkpoint from subdirectory: {checkpoint.name}")
                return checkpoint

    logger.warning(f"No checkpoint found in {exp_dir}")
    return None


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def configure_matplotlib(use_latex: bool = False) -> None:
    """Configure matplotlib for publication-quality figures."""
    # v4.1.0 FIX: Guard style selection to prevent crash if style not available
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            try:
                plt.style.use('ggplot')
            except Exception:
                pass  # Fall back to default style

    plt.rcParams.update({
        'figure.dpi': DPI_SCREEN,
        'savefig.dpi': DPI,
        'figure.figsize': FIG_DOUBLE_COL,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'font.size': 10,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'patch.linewidth': 0.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.2,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'errorbar.capsize': 4,
    })

    if use_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })


# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger('RomanEvaluator')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(output_dir / 'evaluation.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types and PyTorch tensors."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (Path, os.PathLike)):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    point_estimate = statistic(data)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


# =============================================================================
# MODEL LOADING
# =============================================================================

def unwrap_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Unwrap model state dictionary from DDP and torch.compile wrappers."""
    unwrapped = {}

    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        if key.startswith('_orig_mod.'):
            key = key[10:]
        unwrapped[key] = value

    return unwrapped


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model from checkpoint with robust wrapper handling."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # v4.1.0 FIX: Use compatibility wrapper for torch.load
    checkpoint = torch_load_compat(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if 'model_config' not in checkpoint:
        raise KeyError(f"Checkpoint missing 'model_config'. Keys: {list(checkpoint.keys())}")

    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint missing 'model_state_dict'. Keys: {list(checkpoint.keys())}")

    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from model import ModelConfig, RomanMicrolensingClassifier
    except ImportError as e:
        raise RuntimeError(f"Failed to import model architecture: {e}")

    config_dict = checkpoint['model_config']
    config = ModelConfig.from_dict(config_dict)
    model = RomanMicrolensingClassifier(config)

    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = unwrap_model_state_dict(state_dict)

    try:
        model.load_state_dict(unwrapped_state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load state dict (architecture mismatch?): {e}")

    model.to(device)
    model.eval()

    return model, config_dict


def load_normalization_stats(checkpoint_path: Path) -> Dict[str, float]:
    """Load normalization statistics from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # v4.1.0 FIX: Use compatibility wrapper for torch.load
    checkpoint = torch_load_compat(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )

    if 'stats' not in checkpoint:
        raise ValueError(
            f"Checkpoint missing 'stats' dictionary. Keys: {list(checkpoint.keys())}. "
            f"Ensure model was trained with train.py v2.4+"
        )

    stats_dict = checkpoint['stats']
    required_keys = {'flux_mean', 'flux_std', 'delta_t_mean', 'delta_t_std'}
    missing_keys = required_keys - set(stats_dict.keys())

    if missing_keys:
        raise ValueError(f"Stats missing required keys: {missing_keys}")

    stats = {
        'flux_mean': float(stats_dict['flux_mean']),
        'flux_std': float(stats_dict['flux_std']),
        'delta_t_mean': float(stats_dict['delta_t_mean']),
        'delta_t_std': float(stats_dict['delta_t_std'])
    }

    for key, value in stats.items():
        if not np.isfinite(value):
            raise ValueError(f"Stat '{key}' has invalid value: {value}")
        if 'std' in key and value <= 0:
            raise ValueError(f"Std stat '{key}' must be positive, got {value}")

    return stats


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(
    data_path: Path,
    stats: Dict[str, float],
    n_samples: Optional[int] = None,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load and normalize data from HDF5 or NPZ file WITH SEQUENCE COMPACTION.

    Compaction moves all valid (non-zero) observations to a contiguous prefix,
    matching what the model was trained on.

    Returns valid_lengths computed from RAW data (before normalization).
    
    v4.1.0 FIX: Now returns selected_indices (original file row indices) as the
    6th return value. This is critical for correct m_base and params extraction
    when subsampling is used.
    
    Returns
    -------
    flux_norm : np.ndarray
        Normalized flux/magnification [n_samples, seq_len]
    delta_t_norm : np.ndarray
        Normalized delta_t [n_samples, seq_len]
    labels : np.ndarray
        Class labels [n_samples]
    timestamps : np.ndarray
        Timestamps [n_samples, seq_len]
    valid_lengths : np.ndarray
        Valid sequence lengths [n_samples]
    selected_indices : np.ndarray
        Original file row indices [n_samples] - for m_base/params lookup
    data_format : str
        'hdf5' or 'npz'
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if logger:
        logger.info(f"Loading data from: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix == '.h5':
        data_format = 'hdf5'
        with h5py.File(data_path, 'r') as f:
            if 'flux' in f:
                flux = f['flux'][:]
            elif 'mag' in f:
                flux = f['mag'][:]
            else:
                raise ValueError(f"HDF5 missing flux data. Keys: {list(f.keys())}")

            if 'delta_t' not in f:
                raise ValueError(f"HDF5 missing delta_t. Keys: {list(f.keys())}")
            delta_t = f['delta_t'][:]

            if 'labels' not in f:
                raise ValueError(f"HDF5 missing labels. Keys: {list(f.keys())}")
            labels = f['labels'][:]

            if 'timestamps' in f:
                timestamps = f['timestamps'][:]
            else:
                n_samples_total, seq_len = flux.shape
                timestamps = np.tile(
                    np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                    (n_samples_total, 1)
                )
                if logger:
                    logger.warning("Timestamps not found, using synthetic times")

    elif suffix == '.npz':
        data_format = 'npz'
        data = np.load(data_path)

        if 'flux' in data:
            flux = data['flux']
        elif 'mag' in data:
            flux = data['mag']
        else:
            raise ValueError(f"NPZ missing flux data. Keys: {list(data.keys())}")

        if 'delta_t' in data:
            delta_t = data['delta_t']
        else:
            raise ValueError(f"NPZ missing delta_t. Keys: {list(data.keys())}")

        if 'labels' in data:
            labels = data['labels']
        elif 'y' in data:
            labels = data['y']
        else:
            raise ValueError(f"NPZ missing labels. Keys: {list(data.keys())}")

        if 'timestamps' in data:
            timestamps = data['timestamps']
        elif 'times' in data:
            timestamps = data['times']
        else:
            n_samples_total, seq_len = flux.shape
            timestamps = np.tile(
                np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                (n_samples_total, 1)
            )
            if logger:
                logger.warning("Timestamps not found, using synthetic times")

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .h5 or .npz")

    if flux.shape != delta_t.shape:
        raise ValueError(f"Shape mismatch: flux {flux.shape} vs delta_t {delta_t.shape}")

    if len(labels) != len(flux):
        raise ValueError(f"Length mismatch: labels {len(labels)} vs flux {len(flux)}")

    # v4.1.0 FIX: Track selected file indices for correct m_base/params extraction
    # Default: identity mapping (all rows selected in order)
    selected_indices = np.arange(len(flux), dtype=np.int64)

    # Subsample if requested
    if n_samples is not None and n_samples < len(flux):
        if logger:
            logger.info(f"Subsampling {n_samples} from {len(flux)} samples (seed={seed})")

        rng = np.random.RandomState(seed)
        n_classes = len(np.unique(labels))
        samples_per_class = n_samples // n_classes

        indices = []
        for class_idx in range(n_classes):
            class_mask = (labels == class_idx)
            class_indices = np.where(class_mask)[0]

            if len(class_indices) > 0:
                n_take = min(samples_per_class, len(class_indices))
                selected = rng.choice(class_indices, size=n_take, replace=False)
                indices.extend(selected)

        remainder = n_samples - len(indices)
        if remainder > 0:
            all_indices = np.arange(len(flux))
            available = np.setdiff1d(all_indices, indices)
            if len(available) >= remainder:
                extra = rng.choice(available, size=remainder, replace=False)
                indices.extend(extra)

        indices = np.array(indices[:n_samples], dtype=np.int64)
        
        # v4.1.0 FIX: Store the original file indices before slicing
        selected_indices = indices.copy()

        flux = flux[indices]
        delta_t = delta_t[indices]
        labels = labels[indices]
        timestamps = timestamps[indices]

    # SEQUENCE COMPACTION (matching training pipeline)
    if logger:
        logger.info("Applying sequence compaction...")

    n_total, seq_len = flux.shape
    flux_mean = stats['flux_mean']
    flux_std = stats['flux_std']
    delta_t_mean = stats['delta_t_mean']
    delta_t_std = stats['delta_t_std']

    flux_compacted = np.zeros_like(flux)
    delta_t_compacted = np.zeros_like(delta_t)
    timestamps_compacted = np.full_like(timestamps, INVALID_TIMESTAMP)
    valid_lengths = np.zeros(n_total, dtype=np.int32)

    for i in range(n_total):
        valid_mask = (flux[i] != 0.0)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            n_valid = 1
            flux_compacted[i, 0] = flux_mean
            delta_t_compacted[i, 0] = 0.0
            timestamps_compacted[i, 0] = timestamps[i, 0] if timestamps[i, 0] != INVALID_TIMESTAMP else 0.0
        else:
            flux_compacted[i, :n_valid] = flux[i, valid_mask]
            delta_t_compacted[i, :n_valid] = delta_t[i, valid_mask]
            timestamps_compacted[i, :n_valid] = timestamps[i, valid_mask]

        valid_lengths[i] = n_valid

    if logger:
        logger.info(f"Compaction complete. Valid lengths: min={valid_lengths.min()}, "
                   f"max={valid_lengths.max()}, mean={valid_lengths.mean():.1f}")

    # Normalize
    flux_norm = (flux_compacted - flux_mean) / (flux_std + EPS)
    delta_t_norm = (delta_t_compacted - delta_t_mean) / (delta_t_std + EPS)

    if logger:
        logger.info(f"Loaded {len(flux_norm)} samples")
        flux_valid = flux_compacted[flux_compacted != 0]
        flux_norm_valid = flux_norm[flux_compacted != 0]
        logger.info(f"Flux range (valid): [{flux_valid.min():.2f}, {flux_valid.max():.2f}] -> "
                   f"[{flux_norm_valid.min():.2f}, {flux_norm_valid.max():.2f}]")

        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            logger.info(f" Class {CLASS_NAMES[cls]}: {cnt} ({100*cnt/len(labels):.1f}%)")

    # v4.1.0 FIX: Return selected_indices as 6th element
    return flux_norm, delta_t_norm, labels, timestamps_compacted, valid_lengths, selected_indices, data_format


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    model: torch.nn.Module,
    flux: np.ndarray,
    delta_t: np.ndarray,
    device: torch.device,
    valid_lengths: Optional[np.ndarray] = None,
    batch_size: int = 128,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run batch inference with memory-efficient chunked processing."""
    model.eval()

    n_samples = len(flux)
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_logits = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    all_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    if logger:
        logger.info(f"Running inference on {n_samples} samples ({n_batches} batches)")

    with torch.no_grad(), torch.inference_mode():
        for i in tqdm(range(0, n_samples, batch_size),
                     desc="Inference",
                     disable=(logger is None),
                     ncols=80):

            end_idx = min(i + batch_size, n_samples)

            flux_batch = torch.from_numpy(flux[i:end_idx]).to(device)
            delta_t_batch = torch.from_numpy(delta_t[i:end_idx]).to(device)

            if valid_lengths is not None:
                lengths_batch = torch.from_numpy(valid_lengths[i:end_idx]).to(device)
            else:
                lengths_batch = None

            logits = model(flux_batch, delta_t_batch, lengths=lengths_batch)

            is_hierarchical = (hasattr(model, 'config') and model.config.hierarchical)
            if is_hierarchical:
                probs = torch.exp(logits)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits[i:end_idx] = logits.cpu().numpy()
            all_probs[i:end_idx] = probs.cpu().numpy()

            # v4.1.0 FIX: Guard torch.cuda.empty_cache() for CPU-only builds
            if (i // batch_size) % CACHE_CLEAR_FREQ == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    predictions = all_probs.argmax(axis=1)
    confidences = all_probs.max(axis=1)

    if logger:
        logger.info(f"Inference complete. Mean confidence: {confidences.mean():.4f}")

    return predictions, all_probs, confidences, all_logits


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def extract_parameters_from_file(
    data_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    data_format: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract physical parameters (u0, tE, etc.) for specified indices.
    
    Parameters
    ----------
    indices : np.ndarray
        File indices (original HDF5 row indices) for selected events.
    labels : np.ndarray
        Class labels for selected events (post-subsample).
    
    v4.1.0 FIX: Uses O(N) precomputed within-class index mapping instead of
    O(N^2) per-event computation. This is critical for large files.
    """
    try:
        if data_format == 'hdf5':
            with h5py.File(data_path, 'r') as f:
                params = {}
                param_keys = [k for k in f.keys() if k.startswith('params_')]

                if not param_keys:
                    if logger:
                        logger.warning("No parameter datasets found in HDF5")
                    return None

                # v4.1.0 FIX: Precompute within-class indices for O(N) lookup
                all_labels = f['labels'][:].astype(np.int32)
                n_total = len(all_labels)
                
                # Build within-class index array once
                within_class_idx = np.full(n_total, -1, dtype=np.int32)
                for c in range(NUM_CLASSES):
                    mask = (all_labels == c)
                    within_class_idx[mask] = np.arange(mask.sum(), dtype=np.int32)

                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    param_key = f'params_{class_name}'

                    if param_key not in f:
                        continue

                    # Find which of our selected indices belong to this class
                    class_mask = (labels == class_idx)
                    file_indices_for_class = indices[class_mask]

                    if len(file_indices_for_class) == 0:
                        continue

                    param_data = f[param_key][:]
                    
                    # v4.1.0 FIX: Use precomputed within-class indices
                    # Ensure file indices are within bounds
                    valid_file_mask = (file_indices_for_class >= 0) & (file_indices_for_class < n_total)
                    valid_file_indices = file_indices_for_class[valid_file_mask]
                    
                    if len(valid_file_indices) == 0:
                        continue
                    
                    # Get within-class indices for these file indices
                    class_event_indices = within_class_idx[valid_file_indices]
                    
                    # Filter out any invalid within-class indices
                    valid_class_mask = (class_event_indices >= 0) & (class_event_indices < len(param_data))
                    final_class_indices = class_event_indices[valid_class_mask]
                    
                    if len(final_class_indices) > 0:
                        params[class_name] = param_data[final_class_indices]

        elif data_format == 'npz':
            data = np.load(data_path)

            if 'params' in data:
                all_params = data['params']
                params = {}
                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    class_mask = (labels == class_idx)
                    file_indices_for_class = indices[class_mask]
                    if len(file_indices_for_class) > 0:
                        # Ensure indices are within bounds
                        valid_mask = file_indices_for_class < len(all_params)
                        if valid_mask.any():
                            params[class_name] = all_params[file_indices_for_class[valid_mask]]
            else:
                params = {}
                all_labels = data['labels'] if 'labels' in data else data['y']
                n_total = len(all_labels)
                
                # v4.1.0 FIX: Precompute within-class indices
                within_class_idx = np.full(n_total, -1, dtype=np.int32)
                for c in range(NUM_CLASSES):
                    mask = (all_labels == c)
                    within_class_idx[mask] = np.arange(mask.sum(), dtype=np.int32)
                
                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    param_key = f'params_{class_name}'
                    if param_key in data:
                        class_mask = (labels == class_idx)
                        file_indices_for_class = indices[class_mask]
                        
                        if len(file_indices_for_class) == 0:
                            continue
                        
                        param_data = data[param_key]
                        
                        # Use precomputed within-class indices
                        valid_file_mask = (file_indices_for_class >= 0) & (file_indices_for_class < n_total)
                        valid_file_indices = file_indices_for_class[valid_file_mask]
                        
                        if len(valid_file_indices) == 0:
                            continue
                        
                        class_event_indices = within_class_idx[valid_file_indices]
                        valid_class_mask = (class_event_indices >= 0) & (class_event_indices < len(param_data))
                        final_class_indices = class_event_indices[valid_class_mask]
                        
                        if len(final_class_indices) > 0:
                            params[class_name] = param_data[final_class_indices]

                if not params:
                    return None
        else:
            return None

        return params if params else None

    except Exception as e:
        if logger:
            logger.warning(f"Failed to extract parameters: {e}")
        return None


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics with confidence intervals."""
    if logger:
        logger.info("Computing comprehensive metrics...")

    metrics = {}

    # Accuracy with CI
    acc, acc_lower, acc_upper = bootstrap_ci(
        np.arange(len(y_true)),
        lambda idx: accuracy_score(y_true[idx], y_pred[idx]),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed
    )

    metrics['accuracy'] = float(acc)
    metrics['accuracy_ci_lower'] = float(acc_lower)
    metrics['accuracy_ci_upper'] = float(acc_upper)

    # Macro metrics
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # Weighted metrics
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    for i, name in enumerate(CLASS_NAMES):
        metrics[f'precision_{name}'] = float(precision_per_class[i])
        metrics[f'recall_{name}'] = float(recall_per_class[i])
        metrics[f'f1_{name}'] = float(f1_per_class[i])

    # ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        roc_auc_macro = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['roc_auc_macro'] = float(roc_auc_macro)
        roc_auc_weighted = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
        metrics['roc_auc_weighted'] = float(roc_auc_weighted)

        for i, name in enumerate(CLASS_NAMES):
            try:
                roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                metrics[f'roc_auc_{name}'] = float(roc_auc)
            except ValueError:
                # Class not present or only one label
                metrics[f'roc_auc_{name}'] = 0.0

    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute ROC-AUC: {e}")
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_weighted'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + EPS)

    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_normalized'] = cm_normalized.tolist()

    if logger:
        logger.info(f" Accuracy: {metrics['accuracy']*100:.2f}% "
                   f"[{metrics['accuracy_ci_lower']*100:.2f}%, {metrics['accuracy_ci_upper']*100:.2f}%]")
        logger.info(f" F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f" ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

    return metrics


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================

class RomanEvaluator:
    """
    Comprehensive evaluation suite for Roman microlensing classifier.

    Handles model loading, data preprocessing, batch inference, metrics
    computation, and publication-quality visualization generation.
    """

    def __init__(
        self,
        experiment_name: str,
        data_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        device: str = 'cuda',
        batch_size: int = 128,
        n_samples: Optional[int] = None,
        early_detection: bool = False,
        n_evolution_per_type: int = 10,
        n_example_grid_per_type: int = 4,
        colorblind_safe: bool = False,
        save_formats: Optional[List[str]] = None,
        use_latex: bool = False,
        verbose: bool = False,
        seed: int = 42,
        calibration_n_bins: int = CALIBRATION_DEFAULT_BINS,
        roc_bootstrap_ci: bool = True
    ):
        """Initialize evaluator and load all required data."""

        # Set random seed
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Configuration
        self.batch_size = batch_size
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        self.n_example_grid_per_type = n_example_grid_per_type
        self.calibration_n_bins = calibration_n_bins
        self.roc_bootstrap_ci = roc_bootstrap_ci

        # Device setup
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)

        # Find experiment/checkpoint
        self.exp_dir, self.model_path = find_experiment_checkpoint(
            experiment_name,
            base_dir=Path('../results/checkpoints')
        )

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = Path(data_path).stem
            output_dir = self.exp_dir / f'eval_{dataset_name}_{timestamp}'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir, verbose=verbose)

        # Configure matplotlib
        configure_matplotlib(use_latex=use_latex)

        # Color palette
        self.colors = COLORS_COLORBLIND if colorblind_safe else COLORS_DEFAULT

        # Save formats
        if save_formats is None:
            save_formats = ['png']
        self.save_formats = save_formats

        # Log configuration
        self.logger.info("=" * 80)
        self.logger.info("ROMAN MICROLENSING CLASSIFIER EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Evaluator version: {__version__}")
        self.logger.info(f"Experiment: {self.exp_dir.name}")
        self.logger.info(f"Checkpoint: {self.model_path.name}")
        self.logger.info(f"Data: {data_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("-" * 80)

        # Load model
        self.logger.info("Loading model...")
        self.model, self.config_dict = load_model_from_checkpoint(
            self.model_path, self.device
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded: {total_params:,} parameters")

        # Check for auxiliary head
        if hasattr(self.model, 'head_aux') and self.model.head_aux is not None:
            self.logger.info("Auxiliary 3-class head detected")
            self.has_aux_head = True
        else:
            self.has_aux_head = False

        # Load normalization stats
        self.logger.info("Loading normalization statistics...")
        stats = load_normalization_stats(self.model_path)

        self.flux_mean = stats['flux_mean']
        self.flux_std = stats['flux_std']
        self.delta_t_mean = stats['delta_t_mean']
        self.delta_t_std = stats['delta_t_std']

        self.logger.info(f" Flux: mean={self.flux_mean:.4f}, std={self.flux_std:.4f}")
        self.logger.info(f" Delta_t: mean={self.delta_t_mean:.6f}, std={self.delta_t_std:.6f}")

        # Load data
        self.logger.info("-" * 80)
        self.data_path = Path(data_path)

        # v4.1.0 FIX: Capture file_indices for correct m_base/params extraction
        self.flux_norm, self.delta_t_norm, self.y, self.timestamps, self.valid_lengths, self.file_indices, self.data_format = \
            load_and_prepare_data(
                self.data_path, stats, n_samples=n_samples,
                seed=seed, logger=self.logger
            )

        # Run inference
        self.logger.info("-" * 80)
        self.preds, self.probs, self.confs, self.logits = run_inference(
            self.model, self.flux_norm, self.delta_t_norm,
            self.device, valid_lengths=self.valid_lengths,
            batch_size=batch_size, logger=self.logger
        )

        # Compute metrics
        self.logger.info("-" * 80)
        self.metrics = compute_comprehensive_metrics(
            self.y, self.preds, self.probs,
            n_bootstrap=DEFAULT_N_BOOTSTRAP, confidence=0.95,
            seed=seed, logger=self.logger
        )

        # Extract/generate baseline magnitudes
        # v4.1.0 FIX: Use file_indices instead of np.arange(len(self.y))
        self.logger.info("-" * 80)
        self.logger.info("Loading/generating baseline magnitudes...")
        self.baseline_mags = extract_baseline_magnitudes(
            self.data_path,
            self.file_indices,  # v4.1.0 FIX: correct file indices
            self.y,
            logger=self.logger
        )

        # Load parameters
        # v4.1.0 FIX: Use file_indices instead of np.arange(len(self.y))
        self.logger.info("-" * 80)
        self.logger.info("Attempting to load physical parameters...")
        self.params = extract_parameters_from_file(
            self.data_path,
            self.file_indices,  # v4.1.0 FIX: correct file indices
            self.y,
            self.data_format,
            logger=self.logger
        )

        if self.params is not None:
            self.logger.info("Parameters loaded successfully")
        else:
            self.logger.info("Parameters not available (u0 analysis will be skipped)")

        self.logger.info("=" * 80)
        self.logger.info("INITIALIZATION COMPLETE")
        self.logger.info("=" * 80)

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in specified formats."""
        for fmt in self.save_formats:
            path = self.output_dir / f'{name}.{fmt}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
            self.logger.debug(f"Saved: {path}")

    def plot_confusion_matrix(self) -> None:
        """Generate normalized confusion matrix heatmap."""
        cm = np.array(self.metrics['confusion_matrix'])
        cm_norm = np.array(self.metrics['confusion_matrix_normalized'])

        fig, ax = plt.subplots(figsize=FIG_CONFUSION_MATRIX)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        cbar.set_label('Fraction', rotation=270, labelpad=15, fontsize=FONT_SIZE_LABEL)
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                pct_text = f'{cm_norm[i, j]*100:.1f}%'
                count_text = f'({cm[i, j]:,})'
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'

                ax.text(j, i - 0.12, pct_text, ha='center', va='center',
                       color=color, fontsize=FONT_SIZE_CONFUSION_CELL, fontweight='bold')
                ax.text(j, i + 0.18, count_text, ha='center', va='center',
                       color=color, fontsize=FONT_SIZE_ANNOTATION)

        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=FONT_SIZE_TICK)
        ax.set_yticklabels(CLASS_NAMES, fontsize=FONT_SIZE_TICK)
        ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('True Class', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=FONT_SIZE_TITLE)

        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix')
        plt.close()

        self.logger.info("Generated: confusion_matrix")

    def plot_roc_curves(self) -> None:
        """Generate ROC curves with bootstrap confidence intervals."""
        y_true_bin = label_binarize(self.y, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=FIG_ROC_CURVES)

        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            # v4.1.0 FIX: Wrap per-class ROC computation in try/except
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.probs[:, i])
                auc = self.metrics.get(f'roc_auc_{name}', 0.0)

                ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

                if self.roc_bootstrap_ci and len(self.y) > MIN_SAMPLES_FOR_BOOTSTRAP:
                    rng = np.random.RandomState(self.seed)
                    tpr_bootstrap = []
                    fpr_common = np.linspace(0, 1, ROC_INTERP_POINTS)

                    for _ in range(ROC_N_BOOTSTRAP):
                        idx = rng.choice(len(self.y), size=len(self.y), replace=True)
                        y_boot = y_true_bin[idx, i]
                        p_boot = self.probs[idx, i]

                        try:
                            fpr_b, tpr_b, _ = roc_curve(y_boot, p_boot)
                            tpr_interp = np.interp(fpr_common, fpr_b, tpr_b)
                            tpr_bootstrap.append(tpr_interp)
                        except:
                            continue

                    if tpr_bootstrap:
                        tpr_bootstrap = np.array(tpr_bootstrap)
                        tpr_lower = np.percentile(tpr_bootstrap, CI_LOWER_PERCENTILE, axis=0)
                        tpr_upper = np.percentile(tpr_bootstrap, CI_UPPER_PERCENTILE, axis=0)
                        ax.fill_between(fpr_common, tpr_lower, tpr_upper, color=color, alpha=0.2)
            
            except ValueError as e:
                # Class may not be present or only one label exists
                self.logger.warning(f"Could not compute ROC for {name}: {e}")
                continue

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=FONT_SIZE_LEGEND)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        plt.tight_layout()
        self._save_figure(fig, 'roc_curves')
        plt.close()

        self.logger.info("Generated: roc_curves")

    def plot_calibration_curve(self) -> None:
        """Generate calibration reliability diagram."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_CALIBRATION)

        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            y_binary = (self.y == i).astype(int)
            p_class = self.probs[:, i]

            try:
                prob_true, prob_pred = calibration_curve(
                    y_binary, p_class,
                    n_bins=self.calibration_n_bins,
                    strategy='uniform'
                )
                ax1.plot(prob_pred, prob_true, 'o-', color=color, linewidth=2, markersize=5, label=name)
            except Exception as e:
                self.logger.warning(f"Calibration curve failed for {name}: {e}")

        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        ax1.set_xlabel('Predicted Probability', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Observed Frequency', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax1.set_title('Calibration Curve', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left', bbox_to_anchor=LEGEND_BBOX_CALIBRATION)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')
        ax1.tick_params(labelsize=FONT_SIZE_TICK)

        ax2.hist(self.confs, bins=DEFAULT_HIST_BINS, color='gray', alpha=0.7, edgecolor='black')
        ax2.axvline(self.confs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={self.confs.mean():.3f}')
        ax2.set_xlabel('Prediction Confidence', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('Count', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax2.set_title('Confidence Distribution', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left')
        ax2.tick_params(labelsize=FONT_SIZE_TICK)

        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        self._save_figure(fig, 'calibration')
        plt.close()

        self.logger.info("Generated: calibration")

    def plot_class_distributions(self) -> None:
        """Generate class probability distribution plots."""
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=FIG_FULL_PAGE)

        for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, self.colors)):
            p_class = self.probs[:, i]
            correct = (self.y == i) & (self.preds == i)
            incorrect = (self.y == i) & (self.preds != i)

            if correct.sum() > 0:
                ax.hist(p_class[correct], bins=DEFAULT_HIST_BINS, alpha=0.7,
                       color=color, label='Correct', edgecolor='black')
            if incorrect.sum() > 0:
                ax.hist(p_class[incorrect], bins=DEFAULT_HIST_BINS, alpha=0.7,
                       color='red', label='Incorrect', edgecolor='black')

            ax.set_xlabel('Predicted Probability', fontweight='bold', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('Count', fontweight='bold', fontsize=FONT_SIZE_LABEL)
            ax.set_title(f'{name}', fontsize=FONT_SIZE_TITLE, fontweight='bold')
            ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper center')
            ax.set_xlim([0, 1])
            ax.tick_params(labelsize=FONT_SIZE_TICK)

        plt.tight_layout()
        self._save_figure(fig, 'class_distributions')
        plt.close()

        self.logger.info("Generated: class_distributions")

    def plot_per_class_metrics(self) -> None:
        """Generate per-class metrics bar chart."""
        metrics_names = ['Precision', 'Recall', 'F1']
        class_metrics = np.zeros((len(CLASS_NAMES), len(metrics_names)))

        for i, name in enumerate(CLASS_NAMES):
            class_metrics[i, 0] = self.metrics[f'precision_{name}']
            class_metrics[i, 1] = self.metrics[f'recall_{name}']
            class_metrics[i, 2] = self.metrics[f'f1_{name}']

        fig, ax = plt.subplots(figsize=FIG_PER_CLASS_METRICS)

        x = np.arange(len(CLASS_NAMES))
        width = 0.22
        metric_colors = ['#3498db', '#e74c3c', '#2ecc71']

        for i, (metric_name, metric_color) in enumerate(zip(metrics_names, metric_colors)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, class_metrics[:, i], width,
                         label=metric_name, alpha=0.85, color=metric_color,
                         edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars, class_metrics[:, i]):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION)

        ax.set_xlabel('Class', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Score', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=FONT_SIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right', ncol=3)
        ax.set_ylim([0, 1.15])

        plt.tight_layout()
        self._save_figure(fig, 'per_class_metrics')
        plt.close()

        self.logger.info("Generated: per_class_metrics")

    def plot_example_light_curves(self) -> None:
        """Generate grid of example light curves."""
        n_per_class = self.n_example_grid_per_type

        fig_width = n_per_class * 2.8
        fig_height = len(CLASS_NAMES) * 2.5

        fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                                figsize=(fig_width, fig_height), squeeze=False)

        for class_idx, class_name in enumerate(CLASS_NAMES):
            correct_mask = (self.y == class_idx) & (self.preds == class_idx)
            indices = np.where(correct_mask)[0][:n_per_class]

            if len(indices) < n_per_class:
                class_mask = (self.y == class_idx)
                all_indices = np.where(class_mask)[0][:n_per_class]
                if len(all_indices) > 0:
                    indices = all_indices

            for col_idx in range(n_per_class):
                ax = axes[class_idx, col_idx]

                if col_idx >= len(indices):
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=FONT_SIZE_TICK)
                    ax.set_xlabel('Time (days)', fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_ylabel('Mag (AB)', fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_title(f'{class_name}', fontsize=FONT_SIZE_TICK, fontweight='bold')
                    continue

                idx = indices[col_idx]
                flux_norm = self.flux_norm[idx]
                times = self.timestamps[idx]

                flux = flux_norm * (self.flux_std + EPS) + self.flux_mean
                # v4.1.0 FIX: Use times >= 0 instead of times > 0
                # Roman cadence starts at t=0.0 by design
                valid_mask = (times != INVALID_TIMESTAMP) & (times >= 0) & (flux != 0)

                if valid_mask.sum() < MIN_VALID_POINTS_PLOT:
                    ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center',
                           transform=ax.transAxes, fontsize=FONT_SIZE_ANNOTATION)
                    prob = self.probs[idx, class_idx]
                    ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=FONT_SIZE_TICK, fontweight='bold')
                    continue

                times_valid = times[valid_mask]
                flux_valid = flux[valid_mask]

                m_base = self.baseline_mags[idx]
                mag_valid = magnification_to_mag(flux_valid, m_base)

                ax.scatter(times_valid, mag_valid, s=3, alpha=0.7)
                ax.invert_yaxis()
                ax.set_xlabel('Time (days)', fontsize=FONT_SIZE_ANNOTATION)
                ax.set_ylabel('Mag (AB)', fontsize=FONT_SIZE_ANNOTATION)

                prob = self.probs[idx, class_idx]
                ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=FONT_SIZE_TICK, fontweight='bold')
                ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
                ax.grid(alpha=0.2)

        plt.subplots_adjust(hspace=0.4, wspace=0.35)
        plt.tight_layout()
        self._save_figure(fig, 'example_light_curves')
        plt.close()

        self.logger.info("Generated: example_light_curves")

    def plot_u0_dependency(self) -> None:
        """Analyze binary classification accuracy vs impact parameter."""
        if self.params is None or 'binary' not in self.params:
            self.logger.info("Skipping u0 dependency (parameters not available)")
            return

        binary_mask = (self.y == 2)
        binary_params = self.params['binary']

        if 'u0' not in binary_params.dtype.names:
            self.logger.warning("u0 field not found in binary parameters")
            return

        u0_values = binary_params['u0']
        binary_preds = self.preds[binary_mask]

        bin_centers = (U0_BINS[:-1] + U0_BINS[1:]) / 2

        accuracies = []
        errors = []
        counts = []

        for i in range(len(U0_BINS) - 1):
            mask = (u0_values >= U0_BINS[i]) & (u0_values < U0_BINS[i+1])

            if mask.sum() > 0:
                acc = (binary_preds[mask] == 2).mean()
                n = mask.sum()
                err = np.sqrt(acc * (1 - acc) / max(n, 1))

                accuracies.append(acc)
                errors.append(err)
                counts.append(n)
            else:
                accuracies.append(np.nan)
                errors.append(0)
                counts.append(0)

        fig, ax = plt.subplots(figsize=FIG_U0_DEPENDENCY)

        valid = ~np.isnan(accuracies)

        ax.errorbar(bin_centers[valid],
                   np.array(accuracies)[valid],
                   yerr=np.array(errors)[valid],
                   fmt='o-', color=self.colors[2],
                   capsize=4, linewidth=2, markersize=6,
                   label='Binary Accuracy')

        ax.axvline(U0_REFERENCE_LINE, color='gray', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=rf'$u_0 = {U0_REFERENCE_LINE}$')

        ax.set_xlabel(r'Impact Parameter $u_0$', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Binary Classification Accuracy', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Binary Detection vs Impact Parameter', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1.05])
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right', bbox_to_anchor=LEGEND_BBOX_U0)

        valid_bin_centers = bin_centers[valid]
        valid_counts = np.array(counts)[valid]

        for bc, cnt in zip(valid_bin_centers, valid_counts):
            ax.annotate(f'n={cnt}',
                       xy=(bc, 0),
                       xytext=(bc, U0_ANNOTATION_Y_OFFSET),
                       textcoords=('data', 'axes fraction'),
                       ha='center', fontsize=FONT_SIZE_ANNOTATION,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        self._save_figure(fig, 'u0_dependency')
        plt.close()

        self.logger.info("Generated: u0_dependency")

    def plot_temporal_bias_check(self) -> None:
        """Check for temporal selection bias using Kolmogorov-Smirnov test (correctly aligned)."""
        if self.params is None:
            self.logger.info("Skipping temporal bias check (parameters not available)")
            return

        correct = (self.preds == self.y)

        t0_correct_list: List[np.ndarray] = []
        t0_incorrect_list: List[np.ndarray] = []

        # Each params[class_key] array is extracted in the same order as samples of that class
        for class_idx, class_key in enumerate(['flat', 'pspl', 'binary']):
            if class_key not in self.params:
                continue

            p = self.params[class_key]
            if (not hasattr(p, "dtype")) or (p.dtype.names is None) or ('t0' not in p.dtype.names):
                self.logger.warning(f"No t0 field found in parameters for class '{class_key}'")
                continue

            # Correctness mask for samples of this class, in dataset sample order
            class_correct = correct[self.y == class_idx]

            # Guard against rare mismatch if extraction dropped invalid indices
            n = min(len(p), len(class_correct))
            if n == 0:
                continue

            t0 = np.asarray(p['t0'][:n])
            cc = np.asarray(class_correct[:n], dtype=bool)

            t0_correct_list.append(t0[cc])
            t0_incorrect_list.append(t0[~cc])

        if not t0_correct_list or not t0_incorrect_list:
            self.logger.warning("Insufficient t0 data for temporal bias check")
            return

        t0_correct = np.concatenate(t0_correct_list) if t0_correct_list else np.array([])
        t0_incorrect = np.concatenate(t0_incorrect_list) if t0_incorrect_list else np.array([])

        if (len(t0_correct) == 0) or (len(t0_incorrect) == 0):
            self.logger.warning("Insufficient data for temporal bias check (empty correct/incorrect set)")
            return

        ks_stat, p_value = ks_2samp(t0_correct, t0_incorrect)

        fig, ax = plt.subplots(figsize=FIG_TEMPORAL_BIAS)

        ax.hist(
            t0_correct, bins=DEFAULT_HIST_BINS, alpha=0.7,
            color='green', label=f'Correct (n={len(t0_correct):,})',
            density=True, edgecolor='black'
        )
        ax.hist(
            t0_incorrect, bins=DEFAULT_HIST_BINS, alpha=0.7,
            color='red', label=f'Incorrect (n={len(t0_incorrect):,})',
            density=True, edgecolor='black'
        )

        ax.set_xlabel(r'Peak Time $t_0$ (days)', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Normalized Density', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Temporal Bias Check', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        result = "BIAS DETECTED" if p_value < 0.05 else "NO BIAS"
        result_color = 'red' if p_value < 0.05 else 'green'
        ax.text(
            0.02, 0.98,
            f'KS statistic: D={ks_stat:.3f}\np-value: {p_value:.3f}\nResult: {result}',
            transform=ax.transAxes, fontsize=FONT_SIZE_LEGEND,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.4', facecolor='wheat',
                alpha=0.9, edgecolor=result_color, linewidth=2
            )
        )

        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')

        plt.tight_layout()
        self._save_figure(fig, 'temporal_bias_check')
        plt.close()

        self.logger.info(f"Generated: temporal_bias_check (KS p={p_value:.4f})")
      
   

    def plot_evolution_for_class(self, class_idx: int, sample_idx: int) -> None:
        """
        Generate probability evolution plot for specific sample.

        Creates three-panel visualization showing light curve,
        class probability evolution, and prediction confidence.
        """
        class_name = CLASS_NAMES[class_idx]

        flux_norm = self.flux_norm[sample_idx]
        delta_t_norm = self.delta_t_norm[sample_idx]
        times = self.timestamps[sample_idx]
        true_label = self.y[sample_idx]

        n_valid = int(self.valid_lengths[sample_idx])

        if n_valid < EVOLUTION_MIN_VALID_POINTS:
            self.logger.warning(f"Skipping evolution for {class_name}_{sample_idx} (too few points: {n_valid})")
            return

        is_hierarchical = (hasattr(self.model, 'config') and self.model.config.hierarchical)

        # v4.0.0 FIX: Use reduced observation checkpoints
        obs_counts = [n for n in EVOLUTION_OBS_COUNTS if n <= n_valid]

        if not obs_counts:
            obs_counts = [n_valid]
        elif obs_counts[-1] != n_valid:
            obs_counts.append(n_valid)

        n_steps = len(obs_counts)

        probs_evolution = np.zeros((n_steps, NUM_CLASSES))
        times_evolution = np.zeros(n_steps)

        # Correct normalized padding value
        padding_flux = (0.0 - self.flux_mean) / (self.flux_std + EPS)
        padding_delta_t = (0.0 - self.delta_t_mean) / (self.delta_t_std + EPS)

        with torch.no_grad(), torch.inference_mode():
            for i, n_obs in enumerate(obs_counts):
                flux_subset = flux_norm[:n_obs]
                delta_t_subset = delta_t_norm[:n_obs]
                time_at_step = times[n_obs - 1]

                times_evolution[i] = time_at_step

                max_len = len(flux_norm)
                flux_padded = np.full(max_len, padding_flux, dtype=np.float32)
                delta_t_padded = np.full(max_len, padding_delta_t, dtype=np.float32)

                flux_padded[:n_obs] = flux_subset
                delta_t_padded[:n_obs] = delta_t_subset

                flux_tensor = torch.from_numpy(flux_padded[None, :]).to(self.device)
                delta_t_tensor = torch.from_numpy(delta_t_padded[None, :]).to(self.device)
                lengths_tensor = torch.tensor([n_obs], dtype=torch.long, device=self.device)

                logits = self.model(flux_tensor, delta_t_tensor, lengths=lengths_tensor)

                if is_hierarchical:
                    probs = torch.exp(logits)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)

                probs_evolution[i] = probs.cpu().numpy()[0]

        # Denormalize for plotting
        flux_denorm = flux_norm * (self.flux_std + EPS) + self.flux_mean

        times_valid = times[:n_valid]
        flux_valid = flux_denorm[:n_valid]

        # v4.1.0 FIX: Use times >= 0 instead of times > 0
        plot_mask = (times_valid >= 0) & (flux_valid > 0) & np.isfinite(flux_valid)
        times_plot = times_valid[plot_mask]
        flux_plot = flux_valid[plot_mask]

        if len(times_plot) < MIN_VALID_POINTS_PLOT:
            self.logger.warning(f"Skipping evolution plot for {class_name}_{sample_idx}")
            return

        m_base = self.baseline_mags[sample_idx]
        mag_plot = magnification_to_mag(flux_plot, m_base)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIG_EVOLUTION, sharex=True)

        # Panel 1: Light curve
        ax1.scatter(times_plot, mag_plot, s=1, alpha=0.7)
        ax1.invert_yaxis()
        ax1.set_ylabel('AB Magnitude', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax1.set_title(f'Probability Evolution: {class_name} (True={CLASS_NAMES[true_label]})',
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax1.grid(alpha=0.2)
        ax1.tick_params(labelsize=FONT_SIZE_TICK)

        # Panel 2: Probability evolution
        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            ax2.plot(times_evolution, probs_evolution[:, i],
                    'o-', color=color, label=name, linewidth=1, markersize=0.1)

        ax2.axhline(RANDOM_CLASSIFIER_PROB, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Class Probability', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best', ncol=3)
        ax2.grid(alpha=0.2)
        ax2.tick_params(labelsize=FONT_SIZE_TICK)

        # Panel 3: Confidence
        confidence = probs_evolution.max(axis=1)

        ax3.plot(times_evolution, confidence, 'o-', color='black', linewidth=1, markersize=1, label='Confidence')
        ax3.fill_between(times_evolution, 0, confidence, alpha=0.3, color='gray')

        ax3.set_xlabel('Time (days)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax3.set_ylabel('Max Probability', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.set_xlim([times_plot.min(), times_plot.max()])
        ax3.legend(fontsize=FONT_SIZE_LEGEND)
        ax3.grid(alpha=0.2)
        ax3.tick_params(labelsize=FONT_SIZE_TICK)

        plt.subplots_adjust(hspace=0.30)
        plt.tight_layout()
        self._save_figure(fig, f'evolution_{class_name}_{sample_idx}')
        plt.close()

        self.logger.debug(f"Generated: evolution_{class_name}_{sample_idx}")

    def run_early_detection_analysis(self) -> None:
        """Analyze classification performance vs observation completeness."""
        self.logger.info("\nRunning early detection analysis...")

        fractions = EARLY_DETECTION_FRACTIONS
        n_valid_per_sample = self.valid_lengths
        min_valid = n_valid_per_sample.min()

        fractions_filtered = [f for f in fractions
                             if int(min_valid * f) >= EARLY_DETECTION_MIN_REQUIRED]

        if not fractions_filtered:
            self.logger.warning(f"Sequences too short for early detection (min_valid={min_valid})")
            return

        results = []

        # Compute correct normalized padding
        padding_flux = (0.0 - self.flux_mean) / (self.flux_std + EPS)
        padding_delta_t = (0.0 - self.delta_t_mean) / (self.delta_t_std + EPS)

        for frac in fractions_filtered:
            self.logger.info(f" Testing {frac*100:.0f}% completeness...")

            predictions_trunc = []

            with torch.no_grad(), torch.inference_mode():
                for i in range(len(self.flux_norm)):
                    n_valid = n_valid_per_sample[i]
                    n_use = max(int(n_valid * frac), EARLY_DETECTION_MIN_REQUIRED)

                    flux_trunc = self.flux_norm[i, :n_use]
                    delta_t_trunc = self.delta_t_norm[i, :n_use]

                    max_len = self.flux_norm.shape[1]
                    flux_padded = np.full(max_len, padding_flux, dtype=np.float32)
                    delta_t_padded = np.full(max_len, padding_delta_t, dtype=np.float32)

                    flux_padded[:n_use] = flux_trunc
                    delta_t_padded[:n_use] = delta_t_trunc

                    flux_tensor = torch.from_numpy(flux_padded[None, :]).to(self.device)
                    delta_t_tensor = torch.from_numpy(delta_t_padded[None, :]).to(self.device)
                    lengths_tensor = torch.tensor([n_use], dtype=torch.long, device=self.device)

                    logits = self.model(flux_tensor, delta_t_tensor, lengths=lengths_tensor)
                    pred = logits.argmax(dim=-1).cpu().item()

                    predictions_trunc.append(pred)

            predictions_trunc = np.array(predictions_trunc)

            acc = accuracy_score(self.y, predictions_trunc)
            f1 = f1_score(self.y, predictions_trunc, average='macro', zero_division=0)

            _, acc_lower, acc_upper = bootstrap_ci(
                np.arange(len(self.y)),
                lambda idx: accuracy_score(self.y[idx], predictions_trunc[idx]),
                n_bootstrap=DEFAULT_N_BOOTSTRAP,
                confidence=0.95,
                seed=self.seed
            )

            results.append({
                'fraction': float(frac),
                'accuracy': float(acc),
                'accuracy_ci_lower': float(acc_lower),
                'accuracy_ci_upper': float(acc_upper),
                'f1_macro': float(f1)
            })

            self.logger.info(f"    Accuracy: {acc*100:.2f}% [{acc_lower*100:.2f}%, {acc_upper*100:.2f}%]")

        # Plot
        fractions_plot = [r['fraction'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        acc_lower = [r['accuracy_ci_lower'] for r in results]
        acc_upper = [r['accuracy_ci_upper'] for r in results]
        f1_scores = [r['f1_macro'] for r in results]

        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)

        acc_err_lower = [a - l for a, l in zip(accuracies, acc_lower)]
        acc_err_upper = [u - a for a, u in zip(accuracies, acc_upper)]

        ax.errorbar(np.array(fractions_plot) * 100, accuracies,
                   yerr=[acc_err_lower, acc_err_upper],
                   fmt='o-', label='Accuracy', color=self.colors[1],
                   capsize=4, linewidth=1.5, markersize=5)

        ax.plot(np.array(fractions_plot) * 100, f1_scores, 's--',
               label='F1 (macro)', color=self.colors[2], linewidth=1.5, markersize=5)

        ax.set_xlabel('Sequence Completeness (%)', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Score', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Early Detection Performance', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(5, 105)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='lower right')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        plt.tight_layout()
        self._save_figure(fig, 'early_detection_curve')
        plt.close()

        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyJSONEncoder)

        self.logger.info("Early detection analysis complete")

    def run_all_analysis(self) -> None:
        """Execute complete evaluation suite."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("=" * 80)

        # Core metrics
        try:
            self.plot_confusion_matrix()
        except Exception as e:
            self.logger.error(f"Failed to plot confusion matrix: {e}", exc_info=True)

        try:
            self.plot_roc_curves()
        except Exception as e:
            self.logger.error(f"Failed to plot ROC curves: {e}", exc_info=True)

        try:
            self.plot_calibration_curve()
        except Exception as e:
            self.logger.error(f"Failed to plot calibration curve: {e}", exc_info=True)

        try:
            self.plot_class_distributions()
        except Exception as e:
            self.logger.error(f"Failed to plot class distributions: {e}", exc_info=True)

        try:
            self.plot_per_class_metrics()
        except Exception as e:
            self.logger.error(f"Failed to plot per-class metrics: {e}", exc_info=True)

        try:
            self.plot_example_light_curves()
        except Exception as e:
            self.logger.error(f"Failed to plot example light curves: {e}", exc_info=True)

        try:
            self.plot_u0_dependency()
        except Exception as e:
            self.logger.error(f"Failed to plot u0 dependency: {e}", exc_info=True)

        try:
            self.plot_temporal_bias_check()
        except Exception as e:
            self.logger.error(f"Failed to plot temporal bias: {e}", exc_info=True)

        # Evolution plots
        if self.n_evolution_per_type > 0:
            self.logger.info("\nGenerating probability evolution plots...")
            for class_idx, class_name in enumerate(CLASS_NAMES):
                class_mask = (self.y == class_idx)
                indices = np.where(class_mask)[0][:self.n_evolution_per_type]

                for idx in indices:
                    try:
                        self.plot_evolution_for_class(class_idx, idx)
                    except Exception as e:
                        self.logger.warning(f"Failed evolution plot for {class_name}_{idx}: {e}")

        # Early detection
        if self.run_early_detection:
            try:
                self.run_early_detection_analysis()
            except Exception as e:
                self.logger.error(f"Failed early detection analysis: {e}", exc_info=True)

        # Save summary
        summary = {
            'experiment': str(self.exp_dir.name),
            'model_path': str(self.model_path),
            'data_path': str(self.data_path),
            'data_size': int(len(self.y)),
            'class_distribution': {
                name: int((self.y == i).sum())
                for i, name in enumerate(CLASS_NAMES)
            },
            'normalization': {
                'flux_mean': float(self.flux_mean),
                'flux_std': float(self.flux_std),
                'delta_t_mean': float(self.delta_t_mean),
                'delta_t_std': float(self.delta_t_std)
            },
            'metrics': self.metrics,
            'config': self.config_dict,
            'parameters': {
                'batch_size': self.batch_size,
                'calibration_n_bins': self.calibration_n_bins,
                'roc_bootstrap_ci': self.roc_bootstrap_ci,
                'seed': self.seed
            },
            'has_aux_head': self.has_aux_head,
            'timestamp': datetime.now().isoformat(),
            'version': __version__
        }

        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

        np.savez(
            self.output_dir / 'predictions.npz',
            y_true=self.y,
            y_pred=self.preds,
            probabilities=self.probs,
            confidences=self.confs,
            logits=self.logits
        )

        report = classification_report(
            self.y, self.preds,
            target_names=list(CLASS_NAMES),
            digits=4,
            labels=[0, 1, 2],
            zero_division=0
        )

        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

        cm = confusion_matrix(self.y, self.preds, labels=[0, 1, 2])
        np.save(self.output_dir / 'confusion_matrix.npy', cm)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Overall accuracy: {self.metrics['accuracy']*100:.2f}%")
        self.logger.info(f"F1-score (macro): {self.metrics['f1_macro']:.4f}")
        self.logger.info(f"ROC-AUC (macro): {self.metrics['roc_auc_macro']:.4f}")

        self.logger.info("\nPer-class performance:")
        for i, name in enumerate(CLASS_NAMES):
            n_samples = (self.y == i).sum()
            prec = self.metrics[f'precision_{name}']
            rec = self.metrics[f'recall_{name}']
            f1 = self.metrics[f'f1_{name}']
            self.logger.info(
                f" {name:6s} (n={n_samples:5d}): P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--experiment-name', required=True, help="Experiment name")
    parser.add_argument('--data', required=True, help="Path to test dataset (.h5 or .npz)")

    parser.add_argument('--output-dir', default=None, help="Custom output directory")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--n-samples', type=int, default=None, help="Subsample test set")
    parser.add_argument('--device', default='cuda', help="Device: cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    parser.add_argument('--early-detection', action='store_true', help="Run early detection analysis")
    parser.add_argument('--n-evolution-per-type', type=int, default=10, help="Evolution plots per class")
    parser.add_argument('--n-example-grid-per-type', type=int, default=4, help="Examples per class in grid")
    parser.add_argument('--calibration-n-bins', type=int, default=CALIBRATION_DEFAULT_BINS,
                       help="Calibration curve bins")
    parser.add_argument('--no-roc-bootstrap-ci', action='store_true', help="Disable ROC bootstrap CI")

    parser.add_argument('--colorblind-safe', action='store_true', help="Use colorblind-safe palette")
    parser.add_argument('--use-latex', action='store_true', help="Enable LaTeX rendering")
    parser.add_argument('--save-formats', nargs='+', default=['png'],
                       choices=['png', 'pdf', 'svg'], help="Output formats")
    parser.add_argument('--verbose', action='store_true', help="Debug logging")

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
        use_latex=args.use_latex,
        verbose=args.verbose,
        seed=args.seed,
        calibration_n_bins=args.calibration_n_bins,
        roc_bootstrap_ci=(not args.no_roc_bootstrap_ci)
    )

    evaluator.run_all_analysis()


if __name__ == '__main__':
    main()
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err
#SBATCH --job-name=ml_pipeline
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

# =============================================================================
# ML TRAINING PIPELINE v7.0
# =============================================================================
#
# FIXES from v6.0:
# - Resume logic now passes --output-dir to force train.py to use existing dir
# - Tracking files created immediately after training starts
# - Evaluation summary job requests GPU
# - Added cleanup for /tmp and /dev/shm
# - Better error handling and validation
#
# Directory structure:
#   ../results/checkpoints/
#   ├── baseline/
#   │   └── d32_l4_hier_TIMESTAMP/
#   │       ├── checkpoints/checkpoint_latest.pt
#   │       ├── best.pt
#   │       └── config.json
#   ├── distinct/
#   │   └── d32_l4_hier_TIMESTAMP/
#   ├── .exp_baseline    (contains: d32_l4_hier_TIMESTAMP)
#   ├── .exp_distinct    (contains: d32_l4_hier_TIMESTAMP)
#   └── .pipeline_progress
#
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR="$HOME/Thesis/data"
BASELINE_TRAIN="${DATA_DIR}/raw/baseline_train.h5"
BASELINE_TEST="${DATA_DIR}/test/baseline_test.h5"
DISTINCT_TRAIN="${DATA_DIR}/raw/distinct_train.h5"
DISTINCT_TEST="${DATA_DIR}/test/distinct_test.h5"

OUTPUT_BASE="../results/checkpoints"
mkdir -p logs "${OUTPUT_BASE}"

PROGRESS_FILE="${OUTPUT_BASE}/.pipeline_progress"

# Training hyperparameters
BATCH_SIZE=256
NUM_WORKERS=0
ACCUMULATION_STEPS=2
MAX_EPOCHS=150

# Model architecture
D_MODEL=32
N_LAYERS=4
DROPOUT=0.3
WINDOW_SIZE=5

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=2.0
AUX_WEIGHT=0.5
STAGE2_TEMPERATURE=1.0

USE_CLASS_WEIGHTS="false"

# =============================================================================
# ENVIRONMENT
# =============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

export LOCAL_WORLD_SIZE=4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

get_checkpoint_epoch() {
    local checkpoint_path="$1"
    if [ -f "${checkpoint_path}" ]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('${checkpoint_path}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

mark_stage_complete() {
    local stage="$1"
    echo "${stage}:complete:$(date +%s)" >> "${PROGRESS_FILE}"
    log_info "Stage complete: ${stage}"
}

is_stage_complete() {
    local stage="$1"
    if [ -f "${PROGRESS_FILE}" ]; then
        grep -q "^${stage}:complete:" "${PROGRESS_FILE}" && return 0
    fi
    return 1
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup_tmp() {
    log_info "Cleaning up /tmp..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
        rm -f /tmp/train_*_'${USER}'.h5 2>/dev/null || true
        echo "Node $(hostname): /tmp cleaned"
    '
}

cleanup_shm() {
    log_info "Cleaning up /dev/shm..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
        rm -f /dev/shm/*_'${USER}'_*.h5 2>/dev/null || true
        rm -f /dev/shm/*_job'${SLURM_JOB_ID}'_*.h5 2>/dev/null || true
        echo "Node $(hostname): /dev/shm cleaned"
    '
}

cleanup_all() {
    log_info "Running full cleanup..."
    cleanup_tmp
    cleanup_shm
    log_info "Cleanup complete"
}

# Cleanup on exit
trap cleanup_all EXIT

# =============================================================================
# FIND LATEST EXPERIMENT DIRECTORY
# =============================================================================

find_latest_experiment() {
    local base_dir="$1"
    
    if [ ! -d "${base_dir}" ]; then
        echo ""
        return
    fi
    
    # Find most recently modified directory matching pattern d*_l*_*
    local latest=$(ls -td "${base_dir}"/d*_l*_* 2>/dev/null | head -1)
    
    if [ -n "${latest}" ] && [ -d "${latest}" ]; then
        basename "${latest}"
    else
        echo ""
    fi
}

# =============================================================================
# FIX CHECKPOINT STATS (magnification -> flux)
# =============================================================================

fix_checkpoint_stats() {
    local checkpoint_path="$1"
    
    if [ ! -f "${checkpoint_path}" ]; then
        return 1
    fi
    
    python3 << EOF
import torch

ckpt_path = "${checkpoint_path}"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

if 'stats' in ckpt:
    stats = ckpt['stats']
    modified = False
    
    # Add flux_mean/std as aliases if missing
    if 'magnification_mean' in stats and 'flux_mean' not in stats:
        stats['flux_mean'] = stats['magnification_mean']
        modified = True
    if 'magnification_std' in stats and 'flux_std' not in stats:
        stats['flux_std'] = stats['magnification_std']
        modified = True
    
    if modified:
        torch.save(ckpt, ckpt_path)
        print(f"Fixed stats in {ckpt_path}")
    else:
        print(f"Stats already OK in {ckpt_path}")
else:
    print(f"No stats in {ckpt_path}")
EOF
}

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

train_model() {
    local train_data="$1"
    local preset_name="$2"
    
    log_info "========================================"
    log_info "Training: ${preset_name}"
    log_info "========================================"
    
    # Output directory for this preset
    local preset_output="${OUTPUT_BASE}/${preset_name}"
    mkdir -p "${preset_output}"
    
    # Tracking file for this preset's experiment
    local exp_file="${OUTPUT_BASE}/.exp_${preset_name}"
    
    # =========================================================================
    # DATA CACHING TO /tmp
    # =========================================================================
    
    local local_train="/tmp/train_${preset_name}_${USER}.h5"
    
    if [ -f "${local_train}" ]; then
        log_info "Found cached data in /tmp"
        local source_size=$(stat -c%s "${train_data}" 2>/dev/null || echo "0")
        local cached_size=$(stat -c%s "${local_train}" 2>/dev/null || echo "0")
        
        if [ "$source_size" = "$cached_size" ] && [ "$source_size" != "0" ]; then
            log_info "Cache valid, skipping copy"
        else
            log_info "Cache invalid, re-copying..."
            srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
        fi
    else
        log_info "Copying training data to /tmp..."
        srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
    fi
    
    # =========================================================================
    # RESUME LOGIC - Find existing experiment directory
    # =========================================================================
    
    local resume_flag=""
    local output_dir_flag=""
    local current_epoch=0
    local exp_name=""
    
    # Check if we have a tracked experiment from previous run
    if [ -f "${exp_file}" ]; then
        exp_name=$(cat "${exp_file}" | tr -d '[:space:]')
        if [ -n "${exp_name}" ]; then
            log_info "Found tracked experiment: ${exp_name}"
        fi
    fi
    
    # If no tracked experiment, look for latest directory
    if [ -z "${exp_name}" ]; then
        exp_name=$(find_latest_experiment "${preset_output}")
        if [ -n "${exp_name}" ]; then
            log_info "Found existing experiment directory: ${exp_name}"
            # Save it immediately to tracking file
            echo "${exp_name}" > "${exp_file}"
        fi
    fi
    
    # Check for checkpoint in experiment directory
    if [ -n "${exp_name}" ]; then
        local exp_dir="${preset_output}/${exp_name}"
        local latest_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"
        
        if [ -f "${latest_ckpt}" ]; then
            current_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
            log_info "Found checkpoint at epoch ${current_epoch}"
            
            if [ ${current_epoch} -ge ${MAX_EPOCHS} ]; then
                log_info "Training already complete (${current_epoch} >= ${MAX_EPOCHS})"
                # Fix stats before returning
                fix_checkpoint_stats "${exp_dir}/best.pt"
                return 0
            else
                log_info "RESUMING from epoch ${current_epoch}"
                resume_flag="--resume ${latest_ckpt}"
                # CRITICAL: Tell train.py to use the SAME output directory
                output_dir_flag="--output-dir ${exp_dir}"
            fi
        else
            log_info "No checkpoint found in ${exp_dir}, starting fresh in same dir"
            output_dir_flag="--output-dir ${exp_dir}"
        fi
    else
        log_info "No existing experiment found, starting fresh"
    fi
    
    # =========================================================================
    # BUILD COMMAND FLAGS
    # =========================================================================
    
    local class_weights_flag=""
    if [ "${USE_CLASS_WEIGHTS}" = "false" ]; then
        class_weights_flag="--no-class-weights"
    fi
    
    # =========================================================================
    # LAUNCH DISTRIBUTED TRAINING
    # =========================================================================
    
    log_info "Launching training..."
    log_info "  GPUs: $((SLURM_NNODES * 4))"
    log_info "  Batch size: ${BATCH_SIZE}"
    log_info "  Output base: ${preset_output}"
    log_info "  Output dir: ${output_dir_flag:-'(auto)'}"
    log_info "  Resume: ${resume_flag:-'(none)'}"
    
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --rdzv_id="train-${preset_name}-${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        train.py \
        --data "${local_train}" \
        --output "${preset_output}" \
        ${output_dir_flag} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --accumulation-steps ${ACCUMULATION_STEPS} \
        --epochs ${MAX_EPOCHS} \
        --lr ${LEARNING_RATE} \
        --weight-decay ${WEIGHT_DECAY} \
        --warmup-epochs ${WARMUP_EPOCHS} \
        --clip-norm ${CLIP_NORM} \
        --d-model ${D_MODEL} \
        --n-layers ${N_LAYERS} \
        --dropout ${DROPOUT} \
        --window-size ${WINDOW_SIZE} \
        --stage1-weight ${STAGE1_WEIGHT} \
        --stage2-weight ${STAGE2_WEIGHT} \
        --aux-weight ${AUX_WEIGHT} \
        --stage2-temperature ${STAGE2_TEMPERATURE} \
        --hierarchical \
        --use-aux-head \
        --attention-pooling \
        --save-every 3 \
        ${class_weights_flag} \
        ${resume_flag}
    
    local exit_code=$?
    
    # =========================================================================
    # UPDATE TRACKING FILE
    # =========================================================================
    # train.py writes .current_experiment in the output directory
    # Read it to get the actual experiment name
    
    local current_exp_file="${preset_output}/.current_experiment"
    if [ -f "${current_exp_file}" ]; then
        exp_name=$(cat "${current_exp_file}" | tr -d '[:space:]')
        echo "${exp_name}" > "${exp_file}"
        log_info "Updated tracking file: ${exp_name}"
    else
        # Fallback: find latest directory
        exp_name=$(find_latest_experiment "${preset_output}")
        if [ -n "${exp_name}" ]; then
            echo "${exp_name}" > "${exp_file}"
            log_info "Updated tracking file (fallback): ${exp_name}"
        fi
    fi
    
    # =========================================================================
    # CHECK COMPLETION STATUS
    # =========================================================================
    
    if [ $exit_code -eq 0 ]; then
        local exp_dir="${preset_output}/${exp_name}"
        local final_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"
        
        if [ -f "${final_ckpt}" ]; then
            local final_epoch=$(get_checkpoint_epoch "${final_ckpt}")
            
            if [ ${final_epoch} -ge ${MAX_EPOCHS} ]; then
                log_info "Training COMPLETE: ${preset_name} (${final_epoch} epochs)"
                # Fix stats in best.pt for evaluation compatibility
                fix_checkpoint_stats "${exp_dir}/best.pt"
                return 0
            else
                log_info "Training incomplete: ${final_epoch}/${MAX_EPOCHS} epochs"
                return 1
            fi
        fi
    fi
    
    log_error "Training exited with code ${exit_code}"
    return 1
}

# =============================================================================
# EVALUATION SUBMISSION
# =============================================================================

submit_evaluations() {
    local baseline_exp="$1"
    local distinct_exp="$2"
    
    log_info "Submitting evaluation jobs..."
    log_info "  Baseline experiment: ${baseline_exp}"
    log_info "  Distinct experiment: ${distinct_exp}"
    
    # Full paths to experiment directories
    local baseline_dir="${OUTPUT_BASE}/baseline/${baseline_exp}"
    local distinct_dir="${OUTPUT_BASE}/distinct/${distinct_exp}"
    
    # Validate directories exist
    if [ ! -d "${baseline_dir}" ]; then
        log_error "Baseline directory not found: ${baseline_dir}"
        return 1
    fi
    if [ ! -d "${distinct_dir}" ]; then
        log_error "Distinct directory not found: ${distinct_dir}"
        return 1
    fi
    
    # Fix checkpoint stats before evaluation
    log_info "Fixing checkpoint stats for evaluation compatibility..."
    fix_checkpoint_stats "${baseline_dir}/best.pt"
    fix_checkpoint_stats "${distinct_dir}/best.pt"
    
    # Create evaluation script
    cat > "${OUTPUT_BASE}/eval_job.sh" << 'EVALSCRIPT'
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

EXPERIMENT_DIR="$1"
TEST_DATA="$2"
EVAL_NAME="$3"

echo "=========================================="
echo "Evaluation: ${EVAL_NAME}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="

# Check if experiment directory exists
if [ ! -d "${EXPERIMENT_DIR}" ]; then
    echo "ERROR: Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# Check for best.pt
if [ ! -f "${EXPERIMENT_DIR}/best.pt" ]; then
    echo "ERROR: best.pt not found in ${EXPERIMENT_DIR}"
    exit 1
fi

# Run evaluation
timeout 28m python evaluate.py \
    --experiment-name "${EXPERIMENT_DIR}/best.pt" \
    --data "${TEST_DATA}" \
    --batch-size 128 \
    --n-evolution-per-type 10 \
    --save-formats png \
    --colorblind-safe \
    --device cuda

echo "Evaluation complete: ${EVAL_NAME}"
EVALSCRIPT

    chmod +x "${OUTPUT_BASE}/eval_job.sh"
    
    # Submit 4 evaluation jobs
    JOB1=$(sbatch --parsable --job-name=eval_b2b \
        --output=logs/eval_b2b_%j.out --error=logs/eval_b2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_dir}" "${BASELINE_TEST}" "baseline_on_baseline")
    log_info "Submitted eval 1/4: baseline->baseline (${JOB1})"
    
    JOB2=$(sbatch --parsable --job-name=eval_b2d \
        --output=logs/eval_b2d_%j.out --error=logs/eval_b2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_dir}" "${DISTINCT_TEST}" "baseline_on_distinct")
    log_info "Submitted eval 2/4: baseline->distinct (${JOB2})"
    
    JOB3=$(sbatch --parsable --job-name=eval_d2d \
        --output=logs/eval_d2d_%j.out --error=logs/eval_d2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_dir}" "${DISTINCT_TEST}" "distinct_on_distinct")
    log_info "Submitted eval 3/4: distinct->distinct (${JOB3})"
    
    JOB4=$(sbatch --parsable --job-name=eval_d2b \
        --output=logs/eval_d2b_%j.out --error=logs/eval_d2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_dir}" "${BASELINE_TEST}" "distinct_on_baseline")
    log_info "Submitted eval 4/4: distinct->baseline (${JOB4})"
    
    # Submit summary job - WITH GPU REQUEST
    sbatch --dependency=afterok:${JOB1}:${JOB2}:${JOB3}:${JOB4} \
        --job-name=summary \
        --output=logs/summary_%j.out --error=logs/summary_%j.err \
        --time=00:10:00 \
        -p gpu_a100_short \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:1 \
        --wrap="cd ~/Thesis/code && python3 << 'PYEOF'
import json
from pathlib import Path

output_base = Path('../results/checkpoints')

# Read experiment names
baseline_exp = (output_base / '.exp_baseline').read_text().strip()
distinct_exp = (output_base / '.exp_distinct').read_text().strip()

baseline_dir = output_base / 'baseline' / baseline_exp
distinct_dir = output_base / 'distinct' / distinct_exp

evaluations = [
    (baseline_dir, 'baseline_test', 'Baseline -> Baseline'),
    (baseline_dir, 'distinct_test', 'Baseline -> Distinct'),
    (distinct_dir, 'distinct_test', 'Distinct -> Distinct'),
    (distinct_dir, 'baseline_test', 'Distinct -> Baseline'),
]

print()
print('='*80)
print('PIPELINE COMPLETE - FINAL RESULTS')
print('='*80)

results = []

for exp_dir, test_set, description in evaluations:
    # Find eval directory
    eval_dirs = list(exp_dir.glob(f'eval_{test_set}*'))
    if not eval_dirs:
        print(f'[X] {description}: NOT FOUND')
        continue
    
    eval_dir = sorted(eval_dirs)[-1]
    summary_file = eval_dir / 'evaluation_summary.json'
    
    if not summary_file.exists():
        print(f'[X] {description}: NO SUMMARY')
        continue
    
    with open(summary_file) as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    acc = metrics.get('accuracy', 0) * 100
    flat_r = metrics.get('recall_Flat', 0) * 100
    pspl_r = metrics.get('recall_PSPL', 0) * 100
    binary_r = metrics.get('recall_Binary', 0) * 100
    
    print(f'[OK] {description}')
    print(f'     Accuracy: {acc:.2f}%')
    print(f'     Recall - Flat: {flat_r:.1f}%  PSPL: {pspl_r:.1f}%  Binary: {binary_r:.1f}%')
    print()
    
    results.append({
        'evaluation': description,
        'accuracy': acc,
        'recall_flat': flat_r,
        'recall_pspl': pspl_r,
        'recall_binary': binary_r
    })

print('='*80)

# Save combined results
combined_results = output_base / 'final_results.json'
with open(combined_results, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to: {combined_results}')
PYEOF"
    
    log_info "Summary job submitted"
}

# =============================================================================
# AUTO-CONTINUATION
# =============================================================================

setup_continuation() {
    log_info "Setting up auto-continuation..."
    
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --kill-on-invalid-dep=yes \
        "$0")
    
    if [ -n "${CONT_JOB}" ]; then
        echo "${CONT_JOB}" > "${OUTPUT_BASE}/.continuation_job_id"
        log_info "Continuation job: ${CONT_JOB}"
    fi
}

cancel_continuation() {
    if [ -f "${OUTPUT_BASE}/.continuation_job_id" ]; then
        local cont_job=$(cat "${OUTPUT_BASE}/.continuation_job_id")
        scancel ${cont_job} 2>/dev/null || true
        rm -f "${OUTPUT_BASE}/.continuation_job_id"
        log_info "Cancelled continuation job: ${cont_job}"
    fi
}

# =============================================================================
# SIGNAL HANDLER FOR PREEMPTION
# =============================================================================

handle_preemption() {
    log_info "Received preemption signal, saving state..."
    # Training script handles its own checkpointing
    # Just ensure continuation is set up
    exit 0
}

trap handle_preemption USR1

# =============================================================================
# CLEAN START OPTION
# =============================================================================

clean_start() {
    log_info "========================================"
    log_info "CLEANING PREVIOUS RUN"
    log_info "========================================"
    
    # Remove progress file
    rm -f "${PROGRESS_FILE}"
    log_info "Removed progress file"
    
    # Remove tracking files
    rm -f "${OUTPUT_BASE}/.exp_baseline"
    rm -f "${OUTPUT_BASE}/.exp_distinct"
    rm -f "${OUTPUT_BASE}/.current_experiment"
    rm -f "${OUTPUT_BASE}/.continuation_job_id"
    log_info "Removed tracking files"
    
    # Optionally remove old experiment directories
    if [ "${CLEAN_EXPERIMENTS:-false}" = "true" ]; then
        rm -rf "${OUTPUT_BASE}/baseline"
        rm -rf "${OUTPUT_BASE}/distinct"
        log_info "Removed old experiment directories"
    fi
    
    # Cleanup cached data
    cleanup_all
    
    log_info "Clean start complete"
}

# Check for clean start flag
if [ "${1:-}" = "--clean" ] || [ "${CLEAN_START:-false}" = "true" ]; then
    clean_start
fi

# =============================================================================
# MAIN PIPELINE
# =============================================================================

log_info "========================================"
log_info "ML TRAINING PIPELINE v7.0"
log_info "========================================"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Nodes: ${SLURM_NNODES}"
log_info "GPUs: $((SLURM_NNODES * 4))"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Max epochs: ${MAX_EPOCHS}"
log_info "Model: d${D_MODEL}_l${N_LAYERS}"
log_info "========================================"

# Determine current phase
PHASE="unknown"
if ! is_stage_complete "train_baseline"; then
    PHASE="train_baseline"
elif ! is_stage_complete "train_distinct"; then
    PHASE="train_distinct"
elif ! is_stage_complete "evaluations"; then
    PHASE="evaluations"
else
    PHASE="complete"
fi

log_info "Current phase: ${PHASE}"

# =============================================================================
# PHASE 1: BASELINE TRAINING
# =============================================================================

if [ "${PHASE}" = "train_baseline" ]; then
    log_info "========================================"
    log_info "PHASE 1: BASELINE TRAINING"
    log_info "========================================"
    
    setup_continuation
    
    if train_model "${BASELINE_TRAIN}" "baseline"; then
        mark_stage_complete "train_baseline"
        cancel_continuation
        
        log_info "Baseline complete, submitting next phase..."
        sbatch "$0"
        exit 0
    else
        log_info "Baseline incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 2: DISTINCT TRAINING
# =============================================================================

if [ "${PHASE}" = "train_distinct" ]; then
    log_info "========================================"
    log_info "PHASE 2: DISTINCT TRAINING"
    log_info "========================================"
    
    setup_continuation
    
    if train_model "${DISTINCT_TRAIN}" "distinct"; then
        mark_stage_complete "train_distinct"
        cancel_continuation
        PHASE="evaluations"
    else
        log_info "Distinct incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 3: EVALUATIONS
# =============================================================================

if [ "${PHASE}" = "evaluations" ]; then
    log_info "========================================"
    log_info "PHASE 3: EVALUATIONS"
    log_info "========================================"
    
    # Read experiment names from tracking files
    if [ ! -f "${OUTPUT_BASE}/.exp_baseline" ]; then
        log_error "Missing tracking file: .exp_baseline"
        # Try to find it
        baseline_exp=$(find_latest_experiment "${OUTPUT_BASE}/baseline")
        if [ -n "${baseline_exp}" ]; then
            echo "${baseline_exp}" > "${OUTPUT_BASE}/.exp_baseline"
            log_info "Recovered baseline experiment: ${baseline_exp}"
        else
            log_error "Cannot find baseline experiment"
            exit 1
        fi
    fi
    
    if [ ! -f "${OUTPUT_BASE}/.exp_distinct" ]; then
        log_error "Missing tracking file: .exp_distinct"
        # Try to find it
        distinct_exp=$(find_latest_experiment "${OUTPUT_BASE}/distinct")
        if [ -n "${distinct_exp}" ]; then
            echo "${distinct_exp}" > "${OUTPUT_BASE}/.exp_distinct"
            log_info "Recovered distinct experiment: ${distinct_exp}"
        else
            log_error "Cannot find distinct experiment"
            exit 1
        fi
    fi
    
    BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline" | tr -d '[:space:]')
    DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct" | tr -d '[:space:]')
    
    log_info "Baseline experiment: ${BASELINE_EXP}"
    log_info "Distinct experiment: ${DISTINCT_EXP}"
    
    submit_evaluations "${BASELINE_EXP}" "${DISTINCT_EXP}"
    mark_stage_complete "evaluations"
    
    log_info "========================================"
    log_info "EVALUATION JOBS SUBMITTED"
    log_info "Monitor with: squeue -u \$USER"
    log_info "========================================"
    exit 0
fi

# =============================================================================
# PHASE 4: COMPLETE
# =============================================================================

if [ "${PHASE}" = "complete" ]; then
    log_info "========================================"
    log_info "PIPELINE ALREADY COMPLETE"
    log_info "========================================"
    
    # Show final summary
    if [ -f "${OUTPUT_BASE}/.exp_baseline" ] && [ -f "${OUTPUT_BASE}/.exp_distinct" ]; then
        BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline" | tr -d '[:space:]')
        DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct" | tr -d '[:space:]')
        log_info "Baseline: ${OUTPUT_BASE}/baseline/${BASELINE_EXP}"
        log_info "Distinct: ${OUTPUT_BASE}/distinct/${DISTINCT_EXP}"
    fi
    
    exit 0
fi

exit 0
