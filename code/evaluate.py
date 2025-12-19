#!/usr/bin/env python3
"""
Roman Microlensing Classifier - Comprehensive Evaluation Suite v3.0.1
=====================================================================

Production-grade evaluation framework for gravitational microlensing event
classification models. Computes comprehensive metrics, generates publication-
quality visualizations, and performs physics-based performance analysis.

VERSION 3.1.0 CRITICAL FIXES:
-----------------------------
    * CRITICAL FIX: m_base loading from global array
      - simulate.py v3.1.0 now saves global m_base dataset aligned with shuffled data
      - extract_baseline_magnitudes() now checks for global 'm_base' dataset FIRST
      - Falls back to params_{class} only for older data files
      - This fixes the bug where all events showed m_base=22.0 in plots
    * SYNC: Compatible with simulate.py v3.1.0 (72-day season, 15-min cadence)

VERSION 3.0.2 CRITICAL FIXES:
-----------------------------
    * CRITICAL FIX: plot_evolution_for_class padding bug
      - Was using 0.0 for padding, but normalized padding is -flux_mean/flux_std
      - Now uses np.full() with correct normalized padding value
      - Fixed valid_indices detection to use self.valid_lengths
    * CRITICAL FIX: run_early_detection_analysis padding bug
      - Same issue: was using 0.0, now uses correct normalized padding
    * CRITICAL FIX: get_valid_lengths() was broken for normalized data
      - It checked flux_norm != 0.0, but after normalization padding ≠ 0.0
      - Now load_and_prepare_data() returns valid_lengths computed from raw data
      - Deprecated get_valid_lengths() with warning
    * CRITICAL FIX: run_inference() was not using masked pooling
      - Training uses lengths for masked pooling (only valid positions)
      - Evaluation was using lengths=None (averaged over padding too!)
      - Now run_inference() accepts valid_lengths and passes to model
    * These bugs caused evaluation to differ from training behavior

VERSION 3.0.1 COSMETIC FIXES:
-----------------------------
    * FIX: Confusion matrix text sizing and overlap prevention
    * FIX: ROC curves legend positioning outside plot area
    * FIX: u0 dependency plot count annotations repositioned
    * FIX: Temporal bias check legend/text box overlap resolved
    * FIX: Calibration curve legend positioning improved
    * FIX: Per-class metrics bar chart label overlap prevention
    * FIX: Evolution plots panel spacing increased
    * FIX: Example light curves grid spacing improved

VERSION 3.0.0 CRITICAL FIXES:
-----------------------------
    * CRITICAL: Proper m_base handling - loads from HDF5 if available, generates if not
    * CRITICAL: Fixed evolution plots - progressive truncation (not flat lines)
    * CRITICAL: Proper AB magnitude display using per-event m_base
    * MAJOR: Streamlined experiment loading with automatic checkpoint discovery
    * MAJOR: All light curves show realistic Roman magnitudes (18-24 mag range)
    * Complete type hints (100% coverage)
    * All constants moved to module level (no magic numbers)

DATA FORMAT NOTE:
----------------
HDF5 'flux' key contains MAGNIFICATION (A):
  - A = 1.0: baseline (unmagnified)
  - A > 1.0: magnified
  - A = 0.0: masked/invalid

Parameters stored in params_{class} structured arrays MAY contain:
  - m_base: baseline AB magnitude (18-24 mag for Roman) [OPTIONAL - generated if not present]
  - t0, tE, u0: physical parameters
  - s, q, alpha, rho: binary-specific parameters

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
    * ROC curves with bootstrap confidence bands
    * Calibration curves with reliability diagrams
    * Class probability distributions and confidence histograms
    * Light curve examples with magnitude conversion (Roman F146)
    * Temporal evolution plots with 3-panel layout
    * Impact parameter dependency analysis for binary classification
    * Colorblind-safe palette options (IBM/Wong standard)

VERSION HISTORY
---------------
v3.0.1 (Current):
    * COSMETIC: Fixed all plot text/legend overlap issues
    * COSMETIC: Improved figure sizing and spacing
    * COSMETIC: Better annotation positioning

v3.0.0:
    * CRITICAL: m_base auto-generation if not in HDF5
    * CRITICAL: Fixed evolution plots (progressive truncation)
    * MAJOR: Smart experiment/checkpoint finding
    * All components synchronized to v3.0.0

Fixes Applied (v2.7.0 - Comprehensive Update)
---------------------------------------------
    * CRITICAL FIX: Added missing get_valid_lengths() function (was NameError)
    * CRITICAL FIX: Added missing ROMAN_ZP_FLUX_JY constant (was NameError)
    * CRITICAL FIX: Fixed plot_evolution_for_class indentation (was at module level)
    * MAJOR FIX: Added auxiliary head evaluation support for v2.9 train.py compatibility
    * MAJOR FIX: Updated docstrings to correctly say "mean/std" not "median/IQR"
    * MAJOR FIX: Probability computation correctly handles hierarchical mode
    * MINOR FIX: Version updated to 2.7.0 for consistency with model.py

Fixes Applied (v2.6 - Hierarchical Mode Compatibility)
------------------------------------------------------
    * CRITICAL FIX: Probability computation in run_inference() now correctly handles
      hierarchical mode by using torch.exp() instead of F.softmax() (S0-2)
    * Hierarchical mode outputs log-probabilities, not logits; softmax was incorrect

Fixes Applied (v2.5 - Complete Documentation & Robustness)
-----------------------------------------------------------
    * CRITICAL: Complete docstring coverage (100%) for all methods
    * CRITICAL: Enhanced error handling in parameter extraction with detailed messages
    * CRITICAL: Robust statistics loading with comprehensive validation
    * MAJOR: Complete type hint coverage throughout codebase
    * MAJOR: Improved error messages with actionable guidance
    * MAJOR: Enhanced defensive programming in all plotting methods
    * MINOR: Better handling of edge cases (empty data, malformed HDF5)
    * MINOR: Validation of all numerical values (NaN, inf checking)

    Previous fixes (v2.4):
    * CRITICAL: Fixed evolution plot x-axis to use time (days)
    * CRITICAL: All three panels share consistent x-axis (time in days)
    * MAJOR: Improved spacing and formatting to prevent label overlap
    * MAJOR: Explicit filtering of padded observations in evolution plots

    Previous fixes (v2.3):
    * CRITICAL: Fixed DDP/compile wrapper handling in checkpoint loading
    * CRITICAL: Hard failure on missing normalization statistics
    * CRITICAL: Reproducible seeding for subsampling
    * MAJOR: Complete docstrings for core methods
    * MAJOR: Fixed parameter extraction edge cases
    * MINOR: ROC confidence bands via bootstrap
    * Publication-quality matplotlib settings (A&A/MNRAS standard)

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 3.0.2
Date: December 2024
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

__version__: Final[str] = "3.1.0"

# =============================================================================
# CONSTANTS
# =============================================================================

# NOTE: Data format changed in simulate.py v2.7+
# Data now contains MAGNIFICATIONS (A), not Jansky flux
# A = 1.0 = baseline, A = 2.0 = 2x brighter, A = 10.0 = 10x brighter

CLASS_NAMES: Final[Tuple[str, ...]] = ('Flat', 'PSPL', 'Binary')
NUM_CLASSES: Final[int] = 3
INVALID_TIMESTAMP: Final[float] = -999.0 # Explicit padding value for invalid observations

# v2.7.0 FIX: Added missing constant (was causing NameError)
# AB magnitude system zero-point flux
# Reference: Oke & Gunn (1983), ApJ 266, 713
ROMAN_ZP_FLUX_JY: Final[float] = 3631.0

# Color palettes
COLORS_DEFAULT: Final[List[str]] = ['#7f8c8d', '#c0392b', '#2980b9'] # Grey, Red, Blue
COLORS_COLORBLIND: Final[List[str]] = ['#0173b2', '#de8f05', '#029e73'] # IBM colorblind-safe

# Publication settings
DPI: Final[int] = 600 # Publication standard
DPI_SCREEN: Final[int] = 120 # For quick preview
EPS: Final[float] = 1e-8

# Figure sizes (inches) - optimized for A&A/MNRAS single/double column
FIG_SINGLE_COL: Final[Tuple[float, float]] = (3.5, 3.0) # ~8.9cm
FIG_DOUBLE_COL: Final[Tuple[float, float]] = (7.0, 5.0) # ~17.8cm
FIG_FULL_PAGE: Final[Tuple[float, float]] = (7.0, 9.0)

# v3.0.1: Adjusted figure sizes for better cosmetics
FIG_CONFUSION_MATRIX: Final[Tuple[float, float]] = (4.5, 4.0) # Larger for text clarity
FIG_ROC_CURVES: Final[Tuple[float, float]] = (5.0, 4.0) # Wider for legend
FIG_CALIBRATION: Final[Tuple[float, float]] = (8.0, 4.0) # Wider two-panel
FIG_U0_DEPENDENCY: Final[Tuple[float, float]] = (5.0, 4.0) # More height for annotations
FIG_TEMPORAL_BIAS: Final[Tuple[float, float]] = (5.0, 4.0) # More space for legend
FIG_PER_CLASS_METRICS: Final[Tuple[float, float]] = (5.0, 4.0) # Wider for bars
FIG_EVOLUTION: Final[Tuple[float, float]] = (8, 10) # Taller for 3 panels

# =============================================================================
# v3.0.0: ADDITIONAL CONSTANTS (previously magic numbers)
# =============================================================================

# Bootstrap settings
DEFAULT_N_BOOTSTRAP: Final[int] = 1000
ROC_N_BOOTSTRAP: Final[int] = 200 # Reduced for speed in ROC CI
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

# Confidence interval percentiles (for 95% CI)
CI_LOWER_PERCENTILE: Final[float] = 2.5
CI_UPPER_PERCENTILE: Final[float] = 97.5

# u0 dependency analysis bins
U0_BINS: Final[np.ndarray] = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
U0_REFERENCE_LINE: Final[float] = 0.3 # Typical detectability threshold

# Probability threshold for random classifier
RANDOM_CLASSIFIER_PROB: Final[float] = 1.0 / NUM_CLASSES

# Minimum valid points for plotting
MIN_VALID_POINTS_PLOT: Final[int] = 3

# Cache clear frequency (batches)
CACHE_CLEAR_FREQ: Final[int] = 100

# Synthetic timestamps (days)
# v3.1.0: Updated for 72-day Roman season
SYNTHETIC_TIME_MAX: Final[float] = 72.0

# =============================================================================
# v3.0.0: NEW CONSTANTS FOR M_BASE AND EVOLUTION
# =============================================================================

# Roman Space Telescope baseline magnitude range
ROMAN_SOURCE_MAG_MIN: Final[float] = 18.0
ROMAN_SOURCE_MAG_MAX: Final[float] = 24.0
ROMAN_DEFAULT_BASELINE_MAG: Final[float] = 22.0

# Evolution plot observation checkpoints (progressive truncation)
# v3.1.0: Updated for 72-day season with 6912 observations
EVOLUTION_OBS_COUNTS: Final[List[int]] = list(range(100, 6920, 100)) 

# =============================================================================
# v3.0.1: COSMETIC CONSTANTS
# =============================================================================

# Font sizes for plots
FONT_SIZE_TITLE: Final[int] = 12
FONT_SIZE_LABEL: Final[int] = 10
FONT_SIZE_TICK: Final[int] = 9
FONT_SIZE_LEGEND: Final[int] = 8
FONT_SIZE_ANNOTATION: Final[int] = 7
FONT_SIZE_CONFUSION_CELL: Final[int] = 9

# Legend positioning
LEGEND_BBOX_ROC: Final[Tuple[float, float]] = (1.02, 0.5) # Outside right
LEGEND_BBOX_CALIBRATION: Final[Tuple[float, float]] = (0.02, 0.98) # Upper left inside
LEGEND_BBOX_U0: Final[Tuple[float, float]] = (0.98, 0.98) # Upper right inside

# Annotation offsets
U0_ANNOTATION_Y_OFFSET: Final[float] = -0.12 # Below axis for count annotations

# =============================================================================
# FLUX TO MAGNITUDE CONVERSION FOR PLOTTING
# =============================================================================

def magnification_to_mag(A: np.ndarray, baseline_mag: Union[float, np.ndarray] = 22.0) -> np.ndarray:
    """
    Convert magnification to apparent magnitude.

    v3.0.0: Now supports per-event baseline magnitudes (array input).

    Parameters
    ----------
    A : np.ndarray
        Magnification values (A = 1.0 is baseline, A > 1 is brighter)
    baseline_mag : float or np.ndarray
        Baseline magnitude when A = 1.0. Can be:
        - Scalar: same baseline for all events (default: 22.0)
        - Array: per-event baseline magnitudes (shape must broadcast with A)

    Returns
    -------
    np.ndarray
        Apparent magnitude (m = baseline_mag - 2.5*log10(A))
        Brighter objects have smaller (more negative) magnitude values.

    Notes
    -----
    When plotting, use ax.invert_yaxis() to follow astronomical convention
    where brighter objects appear higher on the plot.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = baseline_mag - 2.5 * np.log10(np.maximum(A, EPS))
    mag = np.where(np.isfinite(mag) & (A > 0), mag, np.nan)
    return mag
def magnification_to_delta_mag(A: np.ndarray) -> np.ndarray:
    """
    Convert magnification to delta magnitude for plotting.

    NOTE: With simulate.py v2.7+/v3.0.0, data contains MAGNIFICATIONS (A), not Jansky flux.

    Parameters
    ----------
    A : np.ndarray or float
        Magnification values (A = flux_lensed / flux_unlensed)
        A = 1.0 means baseline (no magnification)
        A = 2.0 means 2x brighter

    Returns
    -------
    np.ndarray or float
        Delta magnitude (Δm = -2.5 log10(A))
        Δm = 0 means baseline
        Δm < 0 means brighter (magnified)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_mag = -2.5 * np.log10(A)
    # Handle masked values (A=0) and keep valid values
    delta_mag = np.where((np.isfinite(delta_mag)) & (A > 0), delta_mag, 0.0)
    return delta_mag
def flux_to_mag(flux_jy: np.ndarray) -> np.ndarray:
    """
    Convert flux in Jansky to AB magnitude.

    Uses the standard AB magnitude system with zero-point at 3631 Jy.

    Parameters
    ----------
    flux_jy : np.ndarray
        Flux array in Jansky units.

    Returns
    -------
    np.ndarray
        AB magnitude array. Invalid fluxes (<=0) return NaN.

    Notes
    -----
    Formula: m_AB = -2.5 * log10(f_ν / 3631 Jy)

    References
    ----------
    Oke & Gunn (1983): "Secondary standard stars for absolute spectrophotometry"
    ApJ, 266, 713
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux_jy / ROMAN_ZP_FLUX_JY)
    return mag
# =============================================================================
# v2.7.0 FIX: MISSING FUNCTION - get_valid_lengths
# =============================================================================

def get_valid_lengths(flux_norm: np.ndarray) -> np.ndarray:
    """
    Compute valid sequence lengths from normalized flux array.

    .. deprecated:: 3.0.2
        This function is BROKEN for normalized data because it checks for != 0.0,
        but after normalization, padding = -flux_mean/flux_std != 0.0.
        Use the valid_lengths returned from load_and_prepare_data() instead.

    After compaction, valid observations form a contiguous prefix [0, n_valid).
    This function counts non-zero values assuming compaction has been applied.

    v2.7.0 FIX: This function was missing, causing NameError during evaluation.

    WARNING (v3.0.2): This function only works correctly on RAW (unnormalized) data!
    After normalization, padding values become -flux_mean/flux_std, NOT 0.0.
    For normalized data, use valid_lengths from load_and_prepare_data() instead.

    Parameters
    ----------
    flux_norm : np.ndarray
        Normalized flux array, shape (n_samples, seq_len).
        After compaction, valid observations are at indices [0, n_valid)
        and padding zeros are at indices [n_valid, seq_len).

    Returns
    -------
    np.ndarray
        Valid lengths for each sample, shape (n_samples,).
        Each element indicates the number of valid observations in the
        contiguous prefix for that sample.

    Notes
    -----
    This function assumes that the data has been compacted (as done in
    load_and_prepare_data), meaning:
    - Valid observations occupy indices [0, n_valid)
    - Padding zeros occupy indices [n_valid, seq_len)

    If data is NOT compacted (zeros scattered throughout), this function
    will still work but may overcount valid observations.

    WARNING: This function is deprecated. After normalization, padding != 0.0!

    Examples
    --------
    >>> flux = np.array([[1.0, 2.0, 0.0, 0.0],
    ...                  [1.0, 2.0, 3.0, 0.0]])
    >>> get_valid_lengths(flux)
    array([2, 3], dtype=int32)
    """
    n_samples, seq_len = flux_norm.shape

    # v3.0.2: Add runtime deprecation warning
    import warnings
    warnings.warn(
        "get_valid_lengths() is DEPRECATED: it doesn't work on normalized data. "
        "Use valid_lengths from load_and_prepare_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Fast vectorized version: count non-zeros per row
    # WARNING: This is WRONG for normalized data where padding = -mean/std != 0.0!
    valid_lengths = np.sum(flux_norm != 0.0, axis=1).astype(np.int32)

    # Ensure at least 1 valid observation (edge case protection)
    valid_lengths = np.maximum(valid_lengths, 1)

    return valid_lengths
# =============================================================================
# v3.0.0: NEW FUNCTIONS FOR M_BASE HANDLING AND EXPERIMENT FINDING
# =============================================================================

def extract_baseline_magnitudes(
    data_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Extract or generate baseline magnitudes (m_base) for events.

    v3.1.0 FIX: Now checks for global 'm_base' dataset FIRST. This dataset
    is aligned with the shuffled data order (saved by simulate.py v3.1.0+).
    Falls back to per-class params_{class} arrays only for older data files.

    Strategy (in order):
    1. Try to load from global 'm_base' dataset (v3.1.0+ format)
    2. Try to load from params_{class} structured arrays (legacy format)
    3. Generate random m_base in Roman range [18, 24] mag as last resort

    Parameters
    ----------
    data_path : Path
        Path to HDF5 dataset.
    indices : np.ndarray
        Global indices of events to load (shape: [N]).
    labels : np.ndarray
        Class labels (0=Flat, 1=PSPL, 2=Binary) for the selected events (shape: [N]).
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    np.ndarray
        Baseline magnitudes (shape: [N]).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_events = len(indices)
    m_base = np.full(n_events, ROMAN_DEFAULT_BASELINE_MAG, dtype=np.float32)

    def _generate_random_m_base(n: int, seed_offset: int = 0) -> np.ndarray:
        """Generate random baseline magnitudes in Roman range."""
        seed_val = 42 + seed_offset
        rng = np.random.RandomState(seed=seed_val)
        return rng.uniform(
            ROMAN_SOURCE_MAG_MIN,
            ROMAN_SOURCE_MAG_MAX,
            size=n
        ).astype(np.float32)

    try:
        with h5py.File(data_path, 'r') as f:
            # =================================================================
            # v3.1.0 FIX: Check for global m_base dataset FIRST
            # This is aligned with the shuffled data order
            # =================================================================
            if 'm_base' in f:
                logger.info("Found global m_base dataset (v3.1.0+ format), loading directly...")
                global_m_base = f['m_base'][:]
                
                # Validate indices are within bounds
                valid_mask = (indices >= 0) & (indices < len(global_m_base))
                
                if valid_mask.all():
                    m_base = global_m_base[indices].astype(np.float32)
                    logger.info(f"Loaded m_base: min={m_base.min():.2f}, max={m_base.max():.2f}, mean={m_base.mean():.2f}")
                    return m_base
                else:
                    n_invalid = (~valid_mask).sum()
                    logger.warning(f"{n_invalid}/{len(indices)} indices out of bounds for global m_base")
                    # Load valid ones, keep default for invalid
                    m_base[valid_mask] = global_m_base[indices[valid_mask]].astype(np.float32)
                    logger.info(f"Loaded m_base (partial): min={m_base.min():.2f}, max={m_base.max():.2f}")
                    return m_base

            # =================================================================
            # LEGACY: Fall back to params_{class} arrays
            # This path is for older data files without global m_base
            # WARNING: This may not work correctly if data is shuffled!
            # =================================================================
            logger.warning("Global m_base dataset not found, falling back to params_{class} arrays (legacy)")
            
            # Check if any parameter dataset has m_base
            has_m_base = False
            for class_idx, class_name in enumerate(CLASS_NAMES):
                param_key = f'params_{class_name.lower()}'
                if param_key in f:
                    param_dataset = f[param_key]
                    if 'm_base' in param_dataset.dtype.names:
                        has_m_base = True
                        break

            if has_m_base:
                logger.info("Found m_base in HDF5 parameters, loading per-event values...")

                # =============================================================
                # v3.0.3 FIX: Load FILE labels to compute correct class offsets
                # =============================================================
                # HDF5 structure: data is sorted by class
                #   - Flat events: indices 0 to n_flat-1
                #   - PSPL events: indices n_flat to n_flat+n_pspl-1
                #   - Binary events: indices n_flat+n_pspl to end
                # params_flat, params_pspl, params_binary are 0-indexed within each class
                #
                # To map a global file index to a class-specific param index:
                #   class_specific_idx = global_file_idx - class_offset
                #
                # WARNING: This logic assumes data is NOT shuffled, which may
                # not be true for simulate.py v2.8+. Use global m_base instead!
                # =============================================================

                file_labels = f['labels'][:]
                
                # Compute class counts and offsets from FILE structure
                class_counts = [(file_labels == c).sum() for c in range(NUM_CLASSES)]
                class_offsets = [0]
                for c in range(NUM_CLASSES - 1):
                    class_offsets.append(class_offsets[-1] + class_counts[c])
                
                logger.debug(f"File class counts: {dict(zip(CLASS_NAMES, class_counts))}")
                logger.debug(f"File class offsets: {class_offsets}")

                # Load m_base for each event based on its class
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    # Find which of our selected events belong to this class
                    class_mask = (labels == class_idx)
                    if not class_mask.any():
                        continue

                    param_key = f'params_{class_name.lower()}'
                    if param_key not in f:
                        logger.warning(f"Parameter dataset '{param_key}' not found, using default")
                        continue

                    param_dataset = f[param_key]
                    if 'm_base' not in param_dataset.dtype.names:
                        logger.warning(f"'m_base' not in {param_key}, using default")
                        continue

                    n_params = len(param_dataset)
                    
                    # Get the global file indices for events of this class
                    file_indices = indices[class_mask]
                    
                    # Map global file indices to class-specific parameter indices
                    # global_file_idx = class_offset + class_specific_idx
                    # => class_specific_idx = global_file_idx - class_offset
                    class_specific_indices = file_indices - class_offsets[class_idx]
                    
                    # Validate indices are within bounds
                    valid_mask = (class_specific_indices >= 0) & (class_specific_indices < n_params)
                    
                    if not valid_mask.all():
                        n_invalid = (~valid_mask).sum()
                        logger.warning(
                            f"{class_name}: {n_invalid}/{len(file_indices)} indices out of bounds "
                            f"(range: [{class_specific_indices.min()}, {class_specific_indices.max()}], "
                            f"n_params: {n_params})"
                        )
                    
                    if valid_mask.any():
                        # Get positions in output array where we'll store values
                        output_positions = np.where(class_mask)[0][valid_mask]
                        valid_class_indices = class_specific_indices[valid_mask]
                        
                        # Load m_base values
                        m_base_values = param_dataset['m_base'][valid_class_indices]
                        m_base[output_positions] = m_base_values

                logger.info(f"Loaded m_base: min={m_base.min():.2f}, max={m_base.max():.2f}, mean={m_base.mean():.2f}")

            else:
                # Generate random m_base values in Roman range
                logger.warning("m_base not found in HDF5 parameters")
                logger.info(f"Generating random m_base in range [{ROMAN_SOURCE_MAG_MIN}, {ROMAN_SOURCE_MAG_MAX}] mag")

                seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
                m_base = _generate_random_m_base(n_events, seed_offset)

                logger.info(f"Generated m_base: min={m_base.min():.2f}, max={m_base.max():.2f}, mean={m_base.mean():.2f}")

    except Exception as e:
        logger.error(f"Error loading/generating m_base: {e}")
        logger.warning(f"Falling back to random m_base generation")
        
        seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
        m_base = _generate_random_m_base(n_events, seed_offset)

    return m_base

def find_experiment_checkpoint(
    experiment_name: str,
    base_dir: Path = Path('../results/checkpoints'),
    logger: Optional[logging.Logger] = None
) -> Tuple[Path, Path]:
    """
    Find experiment directory and checkpoint file automatically.

    v3.0.0: NEW FUNCTION - smart experiment/checkpoint discovery.

    Strategy:
    1. Check for .current_experiment file
    2. Match partial experiment names (e.g., "d32_l2" matches "d32_l2_hier_20241217_012345")
    3. Find best.pt, best_model.pt, or latest checkpoint
    4. Return paths

    Parameters
    ----------
    experiment_name : str
        Full or partial experiment name.
    base_dir : Path
        Base directory to search for experiments.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    exp_dir : Path
        Experiment directory.
    checkpoint_path : Path
        Path to checkpoint file.

    Raises
    ------
    FileNotFoundError
        If no matching experiment or checkpoint found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base_dir = Path(base_dir)

    # Case 1: User provided full path to checkpoint
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

            # Look for checkpoint in this directory
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
            # Sort by modification time, take most recent
            exp_dir = sorted(matching_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            logger.info(f"Matched experiment: {exp_dir.name}")

            checkpoint_path = _find_checkpoint_in_dir(exp_dir, logger)
            if checkpoint_path:
                return exp_dir, checkpoint_path

    # Case 4: Try treating as directory name in ../results/
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
        f"Available experiments:\n" +
        '\n'.join(f" - {d.name}" for d in base_dir.iterdir() if d.is_dir())
        if base_dir.exists() else " (none found)"
    )
def _find_checkpoint_in_dir(exp_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """
    Find best checkpoint in experiment directory.

    v3.0.0: Helper function for find_experiment_checkpoint().

    Priority: best.pt > best_model.pt > checkpoint_latest.pt > most recent checkpoint_*.pt

    Parameters
    ----------
    exp_dir : Path
        Experiment directory to search.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    Path or None
        Checkpoint path if found, None otherwise.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Priority 1: best.pt
    best_pt = exp_dir / 'best.pt'
    if best_pt.exists():
        logger.info(f"Using best checkpoint: {best_pt.name}")
        return best_pt

    # Priority 2: best_model.pt
    best_model_pt = exp_dir / 'best_model.pt'
    if best_model_pt.exists():
        logger.info(f"Using best model: {best_model_pt.name}")
        return best_model_pt

    # Priority 3: checkpoint_latest.pt
    latest_pt = exp_dir / 'checkpoint_latest.pt'
    if latest_pt.exists():
        logger.info(f"Using latest checkpoint: {latest_pt.name}")
        return latest_pt

    # Priority 4: Most recent checkpoint_*.pt
    checkpoints = list(exp_dir.glob('checkpoint_*.pt'))
    if checkpoints:
        checkpoint = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info(f"Using recent checkpoint: {checkpoint.name}")
        return checkpoint

    # Also check checkpoints subdirectory
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
# MATPLOTLIB CONFIGURATION FOR PUBLICATION
# =============================================================================

def configure_matplotlib(use_latex: bool = False) -> None:
    """
    Configure matplotlib for publication-quality figures.

    Sets rendering parameters to match A&A and MNRAS journal standards
    including font sizes, line widths, and grid transparency.

    Parameters
    ----------
    use_latex : bool, optional
        Enable LaTeX rendering for text. Requires LaTeX installation.
        Default is False for compatibility.

    Notes
    -----
    The configuration prioritizes:
    - 600 DPI output for publication
    - Computer Modern fonts matching LaTeX documents
    - Appropriate sizing for single/double column layouts
    - Astronomy-standard grid transparency (alpha=0.2)
    - Inward-pointing ticks on all axes

    References
    ----------
    A&A: https://www.aanda.org/for-authors/latex-issues
    MNRAS: https://academic.oup.com/mnras/pages/General_Instructions
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Base configuration
    plt.rcParams.update({
        # Figure
        'figure.dpi': DPI_SCREEN,
        'savefig.dpi': DPI,
        'figure.figsize': FIG_DOUBLE_COL,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'font.size': 10,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'patch.linewidth': 0.5,

        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.2, # Astronomy publication standard
        'grid.linewidth': 0.5,

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,

        # Error bars
        'errorbar.capsize': 4, # Increased for 600 DPI visibility
    })

    # LaTeX configuration (optional)
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
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
        Directory where log file will be saved.
    verbose : bool, optional
        If True, console shows DEBUG messages. Default is False.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Log file is named 'evaluation.log' and uses append mode to preserve
    logs across multiple runs.
    """
    logger = logging.getLogger('RomanEvaluator')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler (DEBUG+)
    fh = logging.FileHandler(output_dir / 'evaluation.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler (INFO+ or DEBUG+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy types and PyTorch tensors.

    Handles conversion of NumPy arrays, scalars, and other special
    types to JSON-serializable formats.

    Examples
    --------
    >>> data = {'array': np.array([1, 2, 3]), 'scalar': np.float64(3.14)}
    >>> json.dumps(data, cls=NumpyJSONEncoder)
    '{"array": [1, 2, 3], "scalar": 3.14}'
    """

    def default(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable type.

        Parameters
        ----------
        obj : Any
            Object to convert.

        Returns
        -------
        Any
            JSON-serializable representation.
        """
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
    """
    Compute bootstrap confidence interval for a statistic.

    Uses percentile method for confidence interval estimation with
    optional random seeding for reproducibility.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    statistic : callable
        Function to compute statistic (e.g., np.mean, np.median).
    n_bootstrap : int, optional
        Number of bootstrap samples. Default is DEFAULT_N_BOOTSTRAP.
    confidence : float, optional
        Confidence level (0-1). Default is 0.95 for 95% CI.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    point_estimate : float
        Point estimate of the statistic.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.

    Notes
    -----
    Uses percentile bootstrap method (Efron & Tibshirani, 1993).
    For n < 30, consider exact methods instead of bootstrap.

    References
    ----------
    Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
    Chapman & Hall/CRC

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mean, lower, upper = bootstrap_ci(data, np.mean, n_bootstrap=1000)
    >>> print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
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
    """
    Unwrap model state dictionary from DDP and torch.compile wrappers.

    Handles state dictionaries saved from models wrapped in:
    - DistributedDataParallel (DDP): removes 'module.' prefix
    - torch.compile: removes '_orig_mod.' prefix
    - Combinations of both wrappers

    Parameters
    ----------
    state_dict : dict
        State dictionary potentially containing wrapper prefixes.

    Returns
    -------
    dict
        Clean state dictionary with wrapper prefixes removed.

    Notes
    -----
    This function is critical for checkpoint compatibility when models
    are trained with different wrapper configurations than evaluation.

    Examples
    --------
    >>> state_dict = {'module._orig_mod.conv.weight': tensor(...)}
    >>> clean_dict = unwrap_model_state_dict(state_dict)
    >>> list(clean_dict.keys())
    ['conv.weight']
    """
    unwrapped = {}

    for key, value in state_dict.items():
        # Remove 'module.' prefix (DDP wrapper)
        if key.startswith('module.'):
            key = key[7:] # len('module.') = 7

        # Remove '_orig_mod.' prefix (torch.compile wrapper)
        if key.startswith('_orig_mod.'):
            key = key[10:] # len('_orig_mod.') = 10

        unwrapped[key] = value

    return unwrapped
def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint with robust wrapper handling.

    Loads a RomanMicrolensingClassifier from a checkpoint file, handling
    various wrapper states (DDP, compile) and extracting configuration.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file (.pt or .pth).
    device : torch.device
        Device to load model onto.

    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode.
    config_dict : dict
        Configuration dictionary from checkpoint.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    RuntimeError
        If model architecture cannot be reconstructed from config.
    KeyError
        If required keys missing from checkpoint.

    Notes
    -----
    Checkpoint must contain:
    - 'model_config': Configuration dictionary for model reconstruction
    - 'model_state_dict': Model weights (possibly with wrapper prefixes)

    The function automatically unwraps DDP and compile wrappers before
    loading the state dict.

    v3.0.0: Compatible with train.py v3.0.0 and model.py v3.0.0 checkpoints.

    Examples
    --------
    >>> model, config = load_model_from_checkpoint(
    ...     Path('best_model.pt'),
    ...     torch.device('cuda')
    ... )
    >>> model.eval()
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Validate required keys
    if 'model_config' not in checkpoint:
        raise KeyError(
            f"Checkpoint missing 'model_config'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    if 'model_state_dict' not in checkpoint:
        raise KeyError(
            f"Checkpoint missing 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    # Import model (must be in path)
    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from model import ModelConfig, RomanMicrolensingClassifier
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import model architecture. Ensure model.py is in path. "
            f"Error: {e}"
        )

    # Reconstruct model
    config_dict = checkpoint['model_config']
    config = ModelConfig.from_dict(config_dict)
    model = RomanMicrolensingClassifier(config)

    # Unwrap state dict and load
    state_dict = checkpoint['model_state_dict']
    unwrapped_state_dict = unwrap_model_state_dict(state_dict)

    try:
        model.load_state_dict(unwrapped_state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load state dict. This may indicate architecture mismatch. "
            f"Error: {e}"
        )

    model.to(device)
    model.eval()

    return model, config_dict
def load_normalization_stats(checkpoint_path: Path) -> Dict[str, float]:
    """
    Load normalization statistics from checkpoint.

    Extracts flux and delta_t normalization statistics required for
    proper data preprocessing during evaluation.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file containing stats.

    Returns
    -------
    stats : dict
        Dictionary with keys:
        - 'flux_mean': Mean flux value
        - 'flux_std': Standard deviation of flux
        - 'delta_t_mean': Mean delta_t value
        - 'delta_t_std': Standard deviation of delta_t

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    ValueError
        If stats are missing or contain invalid values (NaN, inf, zero).

    Notes
    -----
    This function enforces HARD FAILURE if normalization stats are missing
    or invalid. Using default values (0.0, 1.0) would cause silent prediction
    failures.

    v2.7.0 FIX: Docstring updated to correctly say "mean/std" instead of
    "median/IQR" to match the actual implementation in train.py v2.9+.

    v3.0.0: Compatible with train.py v3.0.0 checkpoint format.

    Normalization formula:
        normalized = (value - mean) / (std + eps)

    Examples
    --------
    >>> stats = load_normalization_stats(Path('best_model.pt'))
    >>> print(stats)
    {'flux_mean': 1.05, 'flux_std': 0.23, 'delta_t_mean': 0.0084, 'delta_t_std': 0.015}
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )

    # Check for stats dictionary
    if 'stats' not in checkpoint:
        raise ValueError(
            f"CRITICAL: Checkpoint missing 'stats' dictionary. "
            f"Cannot proceed without normalization statistics. "
            f"Available keys: {list(checkpoint.keys())}. "
            f"Ensure the model was trained with train.py v2.4+ which saves stats."
        )

    stats_dict = checkpoint['stats']

    # Validate stats dictionary structure
    required_keys = {'flux_mean', 'flux_std', 'delta_t_mean', 'delta_t_std'}
    missing_keys = required_keys - set(stats_dict.keys())

    if missing_keys:
        raise ValueError(
            f"CRITICAL: Stats dictionary missing required keys: {missing_keys}. "
            f"Available keys: {list(stats_dict.keys())}. "
            f"Retrain model with complete normalization statistics."
        )

    # Extract and validate stats
    stats = {
        'flux_mean': float(stats_dict['flux_mean']),
        'flux_std': float(stats_dict['flux_std']),
        'delta_t_mean': float(stats_dict['delta_t_mean']),
        'delta_t_std': float(stats_dict['delta_t_std'])
    }

    # Validate numerical values
    for key, value in stats.items():
        if not np.isfinite(value):
            raise ValueError(
                f"CRITICAL: Stat '{key}' has invalid value: {value}. "
                f"Must be finite (not NaN or inf). Retrain model."
            )

        if 'std' in key and value <= 0:
            raise ValueError(
                f"CRITICAL: Std stat '{key}' must be positive, got {value}. "
                f"This indicates degenerate data distribution. Check training data."
            )

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load and normalize data from HDF5 or NPZ file WITH SEQUENCE COMPACTION.

    CRITICAL FIX (v2.7+): Implements the same sequence compaction as train.py's
    RAMLensingDataset to ensure training/evaluation consistency.

    Compaction moves all valid (non-zero) observations to a contiguous prefix
    [0, n_valid), matching what the model was trained on.

    v3.0.2 FIX: Now returns valid_lengths computed from RAW data (before normalization).
    Previously, get_valid_lengths() was called on normalized data which was WRONG
    because after normalization, padding = -flux_mean/flux_std != 0.0.

    Parameters
    ----------
    data_path : Path
        Path to data file (.h5 or .npz).
    stats : dict
        Normalization statistics from training checkpoint.
    n_samples : int, optional
        Number of samples to subsample. If None, use all data.
    seed : int, optional
        Random seed for subsampling reproducibility. Default is 42.
    logger : logging.Logger, optional
        Logger for progress messages.

    Returns
    -------
    flux_norm : np.ndarray
        Normalized AND COMPACTED flux array, shape (n_samples, seq_len).
    delta_t_norm : np.ndarray
        Normalized AND COMPACTED delta_t array, shape (n_samples, seq_len).
    labels : np.ndarray
        Class labels, shape (n_samples,).
    timestamps : np.ndarray
        COMPACTED observation timestamps, shape (n_samples, seq_len).
    data_format : str
        Format of loaded data ('hdf5' or 'npz').
    valid_lengths : np.ndarray
        Valid sequence lengths (computed from raw data), shape (n_samples,).

    Notes
    -----
    v3.0.0 FIX: Added 'mag' key check for HDF5 backward compatibility,
    matching the fix in train.py v3.0.0 (line 357).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if logger:
        logger.info(f"Loading data from: {data_path}")

    # Determine format
    suffix = data_path.suffix.lower()

    if suffix == '.h5':
        data_format = 'hdf5'
        with h5py.File(data_path, 'r') as f:
            # v3.0.0 FIX: Check for both 'flux' and 'mag' keys for backward compatibility
            if 'flux' in f:
                flux = f['flux'][:]
            elif 'mag' in f:
                flux = f['mag'][:]
            else:
                raise ValueError(
                    f"HDF5 file missing flux data. Available keys: {list(f.keys())}. "
                    f"Expected 'flux' or 'mag' key."
                )

            if 'delta_t' not in f:
                raise ValueError(
                    f"HDF5 file missing delta_t. Available keys: {list(f.keys())}"
                )
            delta_t = f['delta_t'][:]

            if 'labels' not in f:
                raise ValueError(
                    f"HDF5 file missing labels. Available keys: {list(f.keys())}"
                )
            labels = f['labels'][:]

            if 'timestamps' in f:
                timestamps = f['timestamps'][:]
            else:
                # Generate synthetic timestamps if missing
                n_samples_total, seq_len = flux.shape
                timestamps = np.tile(
                    np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                    (n_samples_total, 1)
                )
                if logger:
                    logger.warning("Timestamps not found in HDF5, using synthetic times")

    elif suffix == '.npz':
        data_format = 'npz'
        data = np.load(data_path)

        # Try common key names
        if 'flux' in data:
            flux = data['flux']
        elif 'mag' in data:
            flux = data['mag']
        else:
            raise ValueError(
                f"NPZ file missing flux data. Available keys: {list(data.keys())}"
            )

        if 'delta_t' in data:
            delta_t = data['delta_t']
        else:
            raise ValueError(
                f"NPZ file missing delta_t. Available keys: {list(data.keys())}"
            )

        if 'labels' in data:
            labels = data['labels']
        elif 'y' in data:
            labels = data['y']
        else:
            raise ValueError(
                f"NPZ file missing labels. Available keys: {list(data.keys())}"
            )

        if 'timestamps' in data:
            timestamps = data['timestamps']
        elif 'times' in data:
            timestamps = data['times']
        else:
            # Generate synthetic timestamps
            n_samples_total, seq_len = flux.shape
            timestamps = np.tile(
                np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                (n_samples_total, 1)
            )
            if logger:
                logger.warning("Timestamps not found in NPZ, using synthetic times")

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .h5 or .npz"
        )

    # Validate shapes
    if flux.shape != delta_t.shape:
        raise ValueError(
            f"Shape mismatch: flux {flux.shape} vs delta_t {delta_t.shape}"
        )

    if len(labels) != len(flux):
        raise ValueError(
            f"Length mismatch: labels {len(labels)} vs flux {len(flux)}"
        )

    # Subsample if requested
    if n_samples is not None and n_samples < len(flux):
        if logger:
            logger.info(f"Subsampling {n_samples} from {len(flux)} samples (seed={seed})")

        # Stratified sampling to maintain class balance
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

        # Add remainder randomly if needed
        remainder = n_samples - len(indices)
        if remainder > 0:
            all_indices = np.arange(len(flux))
            available = np.setdiff1d(all_indices, indices)
            if len(available) >= remainder:
                extra = rng.choice(available, size=remainder, replace=False)
                indices.extend(extra)

        indices = np.array(indices[:n_samples])

        flux = flux[indices]
        delta_t = delta_t[indices]
        labels = labels[indices]
        timestamps = timestamps[indices]

    # =========================================================================
    # CRITICAL FIX: SEQUENCE COMPACTION
    # Must match train.py's RAMLensingDataset.__getitem__ logic exactly
    # =========================================================================
    if logger:
        logger.info("Applying sequence compaction (matching training pipeline)...")

    n_total, seq_len = flux.shape
    flux_mean = stats['flux_mean']
    flux_std = stats['flux_std']
    delta_t_mean = stats['delta_t_mean']
    delta_t_std = stats['delta_t_std']

    # Pre-allocate compacted arrays
    flux_compacted = np.zeros_like(flux)
    delta_t_compacted = np.zeros_like(delta_t)
    timestamps_compacted = np.full_like(timestamps, INVALID_TIMESTAMP)
    valid_lengths = np.zeros(n_total, dtype=np.int32)

    for i in range(n_total):
        # Identify valid (non-zero/non-masked) observations
        # This matches train.py: valid_mask = flux_raw != 0.0
        valid_mask = (flux[i] != 0.0)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            # Edge case: entirely empty sequence
            # Match train.py behavior: set first element to mean
            n_valid = 1
            flux_compacted[i, 0] = flux_mean
            delta_t_compacted[i, 0] = 0.0
            timestamps_compacted[i, 0] = timestamps[i, 0] if timestamps[i, 0] != INVALID_TIMESTAMP else 0.0
        else:
            # Compact: move valid observations to contiguous prefix [0, n_valid)
            flux_compacted[i, :n_valid] = flux[i, valid_mask]
            delta_t_compacted[i, :n_valid] = delta_t[i, valid_mask]
            timestamps_compacted[i, :n_valid] = timestamps[i, valid_mask]

        valid_lengths[i] = n_valid

    if logger:
        logger.info(f" Compaction complete. Valid lengths: min={valid_lengths.min()}, "
                   f"max={valid_lengths.max()}, mean={valid_lengths.mean():.1f}")

    # =========================================================================
    # END COMPACTION FIX
    # =========================================================================

    # Normalize (using compacted data)
    flux_norm = (flux_compacted - flux_mean) / (flux_std + EPS)
    delta_t_norm = (delta_t_compacted - delta_t_mean) / (delta_t_std + EPS)

    if logger:
        logger.info(f"Loaded {len(flux_norm)} samples")

        # Report on actual (non-zero) values only
        flux_valid = flux_compacted[flux_compacted != 0]
        flux_norm_valid = flux_norm[flux_compacted != 0]

        logger.info(f"Flux range (valid only): [{flux_valid.min():.2f}, {flux_valid.max():.2f}] -> "
                   f"[{flux_norm_valid.min():.2f}, {flux_norm_valid.max():.2f}]")

        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            logger.info(f" Class {CLASS_NAMES[cls]}: {cnt} ({100*cnt/len(labels):.1f}%)")

    return flux_norm, delta_t_norm, labels, timestamps_compacted, valid_lengths, data_format
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
    """
    Run batch inference with memory-efficient chunked processing.

    Processes data in batches to avoid OOM errors on large datasets.
    Uses gradient-free computation for efficiency.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode.
    flux : np.ndarray
        Normalized flux array, shape (n_samples, seq_len).
    delta_t : np.ndarray
        Normalized delta_t array, shape (n_samples, seq_len).
    device : torch.device
        Device for computation.
    valid_lengths : np.ndarray, optional
        Valid sequence lengths for each sample, shape (n_samples,).
        If provided, model uses masked pooling (only valid positions).
        If None, model averages over all positions (may include padding).
        v3.0.2: Added to match training behavior (masked pooling).
    batch_size : int, optional
        Batch size for inference. Default is 128.
    logger : logging.Logger, optional
        Logger for progress messages.

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).
    probabilities : np.ndarray
        Class probabilities, shape (n_samples, n_classes).
    confidences : np.ndarray
        Prediction confidences (max probability), shape (n_samples,).
    logits : np.ndarray
        Raw logits, shape (n_samples, n_classes).

    Notes
    -----
    Uses torch.no_grad() and torch.inference_mode() for memory efficiency.
    Processes data in chunks to avoid OOM on large datasets (>1M samples).

    v2.6 FIX: Correctly handles hierarchical mode by using torch.exp()
    instead of F.softmax(), since hierarchical mode outputs log-probabilities.

    v3.0.0: Compatible with model.py v3.0.0 hierarchical output format.

    Examples
    --------
    >>> preds, probs, confs, logits = run_inference(
    ...     model, flux_norm, delta_t_norm,
    ...     device=torch.device('cuda'),
    ...     batch_size=256
    ... )
    >>> print(f"Accuracy: {(preds == labels).mean():.4f}")
    """
    model.eval()

    n_samples = len(flux)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Pre-allocate output arrays
    all_logits = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    all_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    if logger:
        logger.info(f"Running inference on {n_samples} samples "
                   f"({n_batches} batches of size {batch_size})")

    with torch.no_grad(), torch.inference_mode():
        for i in tqdm(range(0, n_samples, batch_size),
                     desc="Inference",
                     disable=(logger is None),
                     ncols=80):

            end_idx = min(i + batch_size, n_samples)

            # Prepare batch
            flux_batch = torch.from_numpy(flux[i:end_idx]).to(device)
            delta_t_batch = torch.from_numpy(delta_t[i:end_idx]).to(device)

            # v3.0.2 FIX: Pass valid_lengths to model for masked pooling
            # This matches training behavior where model only pools over valid positions
            if valid_lengths is not None:
                lengths_batch = torch.from_numpy(valid_lengths[i:end_idx]).to(device)
            else:
                lengths_batch = None

            # Forward pass
            logits = model(flux_batch, delta_t_batch, lengths=lengths_batch)

            # v2.6 FIX (S0-2): Correct probability computation for hierarchical mode
            # Check if model is in hierarchical mode
            is_hierarchical = (hasattr(model, 'config') and model.config.hierarchical)
            if is_hierarchical:
                # logits are log-probabilities, convert to probabilities
                probs = torch.exp(logits)
                # Renormalize to handle numerical errors
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)

            # Store results
            all_logits[i:end_idx] = logits.cpu().numpy()
            all_probs[i:end_idx] = probs.cpu().numpy()

            # Clear cache periodically
            if (i // batch_size) % CACHE_CLEAR_FREQ == 0:
                torch.cuda.empty_cache()

    # Derive predictions and confidences
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
    Extract physical parameters for specified indices.

    Loads microlensing parameters (u0, tE, q, s, etc.) from data file
    for physics-based analysis.

    Parameters
    ----------
    data_path : Path
        Path to data file containing parameters.
    indices : np.ndarray
        Indices of samples to extract parameters for.
    labels : np.ndarray
        Class labels for samples (to determine which param group to use).
    data_format : str
        Format of data file ('hdf5' or 'npz').
    logger : logging.Logger, optional
        Logger for warnings.

    Returns
    -------
    params : dict or None
        Dictionary with parameter arrays for each class:
        - 'flat': Parameters for flat events
        - 'pspl': Parameters for PSPL events (includes u0, tE)
        - 'binary': Parameters for binary events (includes u0, tE, q, s)
        Returns None if parameters not found.

    Notes
    -----
    HDF5 format expects structured arrays:
    - params_flat: Fields like [f_base, t0, ...]
    - params_pspl: Fields like [f_base, t0, tE, u0, ...]
    - params_binary: Fields like [f_base, t0, tE, u0, s, q, ...]

    NPZ format expects similar structure under 'params' key.

    Examples
    --------
    >>> indices = np.array([0, 1, 2, 100, 101, 102])
    >>> labels = np.array([0, 0, 1, 1, 2, 2])
    >>> params = extract_parameters_from_file(
    ...     Path('test.h5'), indices, labels, 'hdf5'
    ... )
    >>> print(params['binary']['u0'])
    array([0.1, 0.2])
    """
    try:
        if data_format == 'hdf5':
            with h5py.File(data_path, 'r') as f:
                params = {}

                # Check for parameter datasets
                param_keys = [k for k in f.keys() if k.startswith('params_')]

                if not param_keys:
                    if logger:
                        logger.warning("No parameter datasets found in HDF5 file")
                    return None

                # Load parameters for each class
                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    param_key = f'params_{class_name}'

                    if param_key not in f:
                        if logger:
                            logger.debug(f"Missing {param_key} in HDF5")
                        continue

                    # Get indices for this class
                    class_mask = (labels[indices] == class_idx)
                    class_indices = indices[class_mask]

                    if len(class_indices) == 0:
                        continue

                    # Load structured array
                    param_data = f[param_key][:]

                    # Extract fields for requested indices
                    # Note: params are stored per-class-event, not global index
                    # Need to map global indices to class-specific indices

                    # Count how many events of this class appear before each index
                    all_labels = f['labels'][:]
                    class_event_indices = []

                    for global_idx in class_indices:
                        # Count how many events of this class appear before global_idx
                        class_event_idx = (all_labels[:global_idx] == class_idx).sum()
                        class_event_indices.append(class_event_idx)

                    class_event_indices = np.array(class_event_indices)

                    # Validate indices
                    if len(param_data) > 0 and class_event_indices.max() < len(param_data):
                        params[class_name] = param_data[class_event_indices]
                    else:
                        if logger:
                            logger.warning(
                                f"Parameter index mismatch for {class_name}: "
                                f"max_idx={class_event_indices.max()}, "
                                f"param_len={len(param_data)}"
                            )

        elif data_format == 'npz':
            data = np.load(data_path)

            if 'params' in data:
                # Assume params is a structured array with all events
                all_params = data['params']

                params = {}
                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    class_mask = (labels[indices] == class_idx)
                    class_indices = indices[class_mask]

                    if len(class_indices) > 0:
                        params[class_name] = all_params[class_indices]

            else:
                # Try separate param files
                params = {}
                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    param_key = f'params_{class_name}'

                    if param_key in data:
                        class_mask = (labels[indices] == class_idx)
                        class_indices = indices[class_mask]

                        if len(class_indices) > 0:
                            all_labels = data['labels'] if 'labels' in data else data['y']

                            # Map global to class-specific indices
                            class_event_indices = []
                            for global_idx in class_indices:
                                class_event_idx = (all_labels[:global_idx] == class_idx).sum()
                                class_event_indices.append(class_event_idx)

                            class_event_indices = np.array(class_event_indices)
                            param_data = data[param_key]

                            if class_event_indices.max() < len(param_data):
                                params[class_name] = param_data[class_event_indices]

                if not params:
                    if logger:
                        logger.warning("No parameter data found in NPZ file")
                    return None

        else:
            if logger:
                logger.warning(f"Unknown data format: {data_format}")
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
    """
    Compute comprehensive classification metrics with confidence intervals.

    Calculates accuracy, precision, recall, F1, and ROC-AUC with bootstrap
    confidence intervals for statistical rigor.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).
    y_probs : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes).
    n_bootstrap : int, optional
        Number of bootstrap samples for CI. Default is DEFAULT_N_BOOTSTRAP.
    confidence : float, optional
        Confidence level (0-1). Default is 0.95.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    logger : logging.Logger, optional
        Logger for progress messages.

    Returns
    -------
    metrics : dict
        Comprehensive metrics dictionary containing:
        - Overall: accuracy, precision, recall, F1 (macro and weighted)
        - Per-class: precision, recall, F1 for each class
        - AUROC: macro and weighted averages
        - Confidence intervals for all metrics
        - Confusion matrix (raw and normalized)

    Notes
    -----
    Bootstrap confidence intervals use percentile method (Efron & Tibshirani, 1993).
    For sample sizes < 100, CI estimates may be unreliable.

    References
    ----------
    Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
    Fawcett (2006): "An introduction to ROC analysis", Pattern Recognition Letters

    Examples
    --------
    >>> metrics = compute_comprehensive_metrics(
    ...     y_true, y_pred, y_probs,
    ...     n_bootstrap=1000,
    ...     confidence=0.95
    ... )
    >>> print(f"Accuracy: {metrics['accuracy']:.4f} "
    ...       f"[{metrics['accuracy_ci_lower']:.4f}, "
    ...       f"{metrics['accuracy_ci_upper']:.4f}]")
    """
    if logger:
        logger.info("Computing comprehensive metrics...")

    metrics = {}

    # Overall accuracy with CI
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

    # Precision, Recall, F1 (macro)
    metrics['precision_macro'] = float(precision_score(
        y_true, y_pred, average='macro', zero_division=0
    ))
    metrics['recall_macro'] = float(recall_score(
        y_true, y_pred, average='macro', zero_division=0
    ))
    metrics['f1_macro'] = float(f1_score(
        y_true, y_pred, average='macro', zero_division=0
    ))

    # Precision, Recall, F1 (weighted)
    metrics['precision_weighted'] = float(precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    ))
    metrics['recall_weighted'] = float(recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    ))
    metrics['f1_weighted'] = float(f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    ))

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )

    for i, name in enumerate(CLASS_NAMES):
        metrics[f'precision_{name}'] = float(precision_per_class[i])
        metrics[f'recall_{name}'] = float(recall_per_class[i])
        metrics[f'f1_{name}'] = float(f1_per_class[i])

    # ROC-AUC
    try:
        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        # Macro average
        roc_auc_macro = roc_auc_score(
            y_true_bin, y_probs, average='macro', multi_class='ovr'
        )
        metrics['roc_auc_macro'] = float(roc_auc_macro)

        # Weighted average
        roc_auc_weighted = roc_auc_score(
            y_true_bin, y_probs, average='weighted', multi_class='ovr'
        )
        metrics['roc_auc_weighted'] = float(roc_auc_weighted)

        # Per-class
        for i, name in enumerate(CLASS_NAMES):
            roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            metrics[f'roc_auc_{name}'] = float(roc_auc)

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
                   f"[{metrics['accuracy_ci_lower']*100:.2f}%, "
                   f"{metrics['accuracy_ci_upper']*100:.2f}%]")
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

    Parameters
    ----------
    experiment_name : str
        Name of experiment (used to find checkpoint directory).
    data_path : str or Path
        Path to test data file (.h5 or .npz).
    output_dir : str or Path, optional
        Custom output directory. If None, creates timestamped dir.
    device : str, optional
        Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
    batch_size : int, optional
        Batch size for inference. Default is 128.
    n_samples : int, optional
        Number of samples to evaluate. If None, use all.
    early_detection : bool, optional
        Run early detection analysis. Default is False.
    n_evolution_per_type : int, optional
        Number of evolution plots per class. Default is 10.
    n_example_grid_per_type : int, optional
        Number of examples per class in grid. Default is 4.
    colorblind_safe : bool, optional
        Use colorblind-safe palette. Default is False.
    save_formats : list of str, optional
        Output formats for figures. Default is ['png'].
    use_latex : bool, optional
        Enable LaTeX rendering. Default is False.
    verbose : bool, optional
        Enable debug logging. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    calibration_n_bins : int, optional
        Number of bins for calibration curve. Default is CALIBRATION_DEFAULT_BINS.
    roc_bootstrap_ci : bool, optional
        Add bootstrap CI to ROC curves. Default is True.

    Attributes
    ----------
    model : torch.nn.Module
        Loaded classifier model.
    device : torch.device
        Computation device.
    logger : logging.Logger
        Configured logger.
    output_dir : Path
        Output directory for results.
    colors : list
        Color palette for plots.

    Methods
    -------
    run_all_analysis()
        Execute complete evaluation suite.
    plot_confusion_matrix()
        Generate confusion matrix heatmap.
    plot_roc_curves()
        Generate ROC curves with AUC scores.
    plot_calibration_curve()
        Generate calibration reliability diagram.
    plot_class_distributions()
        Generate probability distributions.
    plot_per_class_metrics()
        Generate per-class metrics bar chart.
    plot_example_light_curves()
        Generate example light curve grid.
    plot_u0_dependency()
        Analyze accuracy vs impact parameter.
    plot_temporal_bias_check()
        Check for temporal selection bias.
    plot_evolution_for_class()
        Generate probability evolution plot.
    run_early_detection_analysis()
        Analyze early detection performance.

    Notes
    -----
    v3.0.1: All plotting methods updated with cosmetic fixes for text/legend overlap.
    v3.0.0: Compatible with train.py v3.0.0, model.py v3.0.0, and simulate.py v3.0.0.

    Examples
    --------
    >>> evaluator = RomanEvaluator(
    ...     experiment_name='baseline_20241215_120000',
    ...     data_path='data/test.h5',
    ...     device='cuda',
    ...     early_detection=True,
    ...     colorblind_safe=True
    ... )
    >>> evaluator.run_all_analysis()
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

        # v3.0.0: Use smart experiment/checkpoint finder
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
        self.logger.info(f"Seed: {seed}")
        self.logger.info("-" * 80)

        # Load model
        self.logger.info("Loading model...")
        self.model, self.config_dict = load_model_from_checkpoint(
            self.model_path, self.device
        )

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded: {total_params:,} parameters")
        self.logger.info(f"Configuration: {self.config_dict}")

        # v2.7.0+: Check for auxiliary head (v2.9+ train.py feature)
        if hasattr(self.model, 'head_aux') and self.model.head_aux is not None:
            self.logger.info("Auxiliary 3-class head detected (v2.9+ model)")
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

        # v3.0.2 FIX: valid_lengths is now returned from load_and_prepare_data
        # (computed from raw data BEFORE normalization).
        # The old get_valid_lengths() was WRONG because it checked flux_norm != 0.0,
        # but after normalization, padding = -flux_mean/flux_std ≠ 0.0!
        self.flux_norm, self.delta_t_norm, self.y, self.timestamps, self.valid_lengths, self.data_format = \
            load_and_prepare_data(
                self.data_path, stats, n_samples=n_samples,
                seed=seed, logger=self.logger
            )

        # Run inference
        # v3.0.2 FIX: Pass valid_lengths for proper masked pooling (matches training)
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

        # v3.0.0: Extract or generate baseline magnitudes
        self.logger.info("-" * 80)
        self.logger.info("Loading/generating baseline magnitudes...")
        self.baseline_mags = extract_baseline_magnitudes(
            self.data_path,
            np.arange(len(self.y)),
            self.y,
            logger=self.logger
        )

        # Try to load parameters
        self.logger.info("-" * 80)
        self.logger.info("Attempting to load physical parameters...")
        self.params = extract_parameters_from_file(
            self.data_path,
            np.arange(len(self.y)),
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
        """
        Save figure in specified formats.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        name : str
            Base filename (without extension).

        Notes
        -----
        Saves figure in all formats specified in self.save_formats.
        Automatically handles DPI and bbox_inches settings.
        """
        for fmt in self.save_formats:
            path = self.output_dir / f'{name}.{fmt}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
            self.logger.debug(f"Saved: {path}")

    def plot_confusion_matrix(self) -> None:
        """
        Generate normalized confusion matrix heatmap.

        Creates a publication-quality confusion matrix visualization with
        percentages and sample counts. Uses astronomy-standard colormap.

        v3.0.1 FIX: Increased figure size and reduced font size to prevent
        text overlap in cells.

        Output
        ------
        confusion_matrix.{png,pdf,svg} : file
            Heatmap visualization in specified formats.

        Notes
        -----
        Confusion matrix is row-normalized (sum to 100% per true class).
        Annotated with both percentages and absolute counts.
        """
        cm = np.array(self.metrics['confusion_matrix'])
        cm_norm = np.array(self.metrics['confusion_matrix_normalized'])

        # v3.0.1: Use larger figure size to prevent text overlap
        fig, ax = plt.subplots(figsize=FIG_CONFUSION_MATRIX)

        # Plot normalized confusion matrix
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

        # Colorbar with adjusted size
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        cbar.set_label('Fraction', rotation=270, labelpad=15, fontsize=FONT_SIZE_LABEL)
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

        # v3.0.1: Annotate cells with smaller font and better formatting
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                # Use smaller font and split percentage/count on two lines
                pct_text = f'{cm_norm[i, j]*100:.1f}%'
                count_text = f'({cm[i, j]:,})'
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'

                # Draw percentage
                ax.text(j, i - 0.12, pct_text, ha='center', va='center',
                       color=color, fontsize=FONT_SIZE_CONFUSION_CELL, fontweight='bold')
                # Draw count below
                ax.text(j, i + 0.18, count_text, ha='center', va='center',
                       color=color, fontsize=FONT_SIZE_ANNOTATION)

        # Labels with proper sizing
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
        """
        Generate ROC curves with bootstrap confidence intervals.

        Creates one-vs-rest ROC curves for each class with optional
        bootstrap confidence bands for statistical rigor.

        v3.0.1 FIX: Legend moved outside plot area to prevent overlap with curves.

        Output
        ------
        roc_curves.{png,pdf,svg} : file
            ROC curve visualization in specified formats.

        Notes
        -----
        Uses one-vs-rest strategy for multiclass ROC.
        Bootstrap confidence intervals computed with ROC_N_BOOTSTRAP samples.
        Diagonal reference line represents random classifier.

        References
        ----------
        Fawcett (2006): "An introduction to ROC analysis"
        Pattern Recognition Letters, 27(8), 861-874
        """
        y_true_bin = label_binarize(self.y, classes=[0, 1, 2])

        # v3.0.1: Use wider figure for external legend
        fig, ax = plt.subplots(figsize=FIG_ROC_CURVES)

        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.probs[:, i])
            auc = self.metrics.get(f'roc_auc_{name}', 0.0)

            # Plot main curve
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{name} (AUC={auc:.3f})')

            # Bootstrap confidence interval
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

                    ax.fill_between(fpr_common, tpr_lower, tpr_upper,
                                   color=color, alpha=0.2)

        # Diagonal reference (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=FONT_SIZE_TITLE)

        # v3.0.1: Move legend outside plot to prevent overlap
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 fontsize=FONT_SIZE_LEGEND, framealpha=0.95, edgecolor='black')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        # Adjust layout to accommodate external legend
        plt.tight_layout()
        self._save_figure(fig, 'roc_curves')
        plt.close()

        self.logger.info("Generated: roc_curves")

    def plot_calibration_curve(self) -> None:
        """
        Generate calibration reliability diagram.

        Plots predicted probabilities vs observed frequencies to assess
        model calibration. Includes confidence histogram.

        v3.0.1 FIX: Improved legend positioning and panel spacing.

        Output
        ------
        calibration.{png,pdf,svg} : file
            Two-panel calibration diagnostic in specified formats.

        Notes
        -----
        Perfect calibration follows the diagonal (predicted = observed).
        Uses binning method with configurable number of bins.
        Includes confidence histogram to show prediction distribution.

        References
        ----------
        Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities
        with Supervised Learning", ICML
        """
        # v3.0.1: Use wider figure for two panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_CALIBRATION)

        # Calibration curve
        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            # Get binary labels and probabilities for this class
            y_binary = (self.y == i).astype(int)
            p_class = self.probs[:, i]

            # Compute calibration curve
            try:
                prob_true, prob_pred = calibration_curve(
                    y_binary, p_class,
                    n_bins=self.calibration_n_bins,
                    strategy='uniform'
                )

                ax1.plot(prob_pred, prob_true, 'o-',
                        color=color, linewidth=2, markersize=5,
                        label=name)
            except Exception as e:
                self.logger.warning(f"Calibration curve failed for {name}: {e}")

        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')

        ax1.set_xlabel('Predicted Probability', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Observed Frequency', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax1.set_title('Calibration Curve', fontweight='bold', fontsize=FONT_SIZE_TITLE)

        # v3.0.1: Position legend in upper left, inside plot
        ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left',
                  bbox_to_anchor=LEGEND_BBOX_CALIBRATION, framealpha=0.9)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')
        ax1.tick_params(labelsize=FONT_SIZE_TICK)

        # Confidence histogram
        ax2.hist(self.confs, bins=DEFAULT_HIST_BINS, color='gray', alpha=0.7, edgecolor='black')
        ax2.axvline(self.confs.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={self.confs.mean():.3f}')
        ax2.set_xlabel('Prediction Confidence', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('Count', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax2.set_title('Confidence Distribution', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left')
        ax2.tick_params(labelsize=FONT_SIZE_TICK)

        # v3.0.1: Increase spacing between panels
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        self._save_figure(fig, 'calibration')
        plt.close()

        self.logger.info("Generated: calibration")

    def plot_class_distributions(self) -> None:
        """
        Generate class probability distribution plots.

        Shows distribution of predicted probabilities for each class,
        separated by correct and incorrect predictions.

        Output
        ------
        class_distributions.{png,pdf,svg} : file
            Three-panel probability distribution in specified formats.

        Notes
        -----
        Correct predictions should show high probability mass near 1.0.
        Misclassifications show characteristic patterns useful for debugging.
        """
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=FIG_FULL_PAGE)

        for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, self.colors)):
            # Get probabilities for this class
            p_class = self.probs[:, i]

            # Separate correct vs incorrect
            correct = (self.y == i) & (self.preds == i)
            incorrect = (self.y == i) & (self.preds != i)

            # Plot distributions
            if correct.sum() > 0:
                ax.hist(p_class[correct], bins=DEFAULT_HIST_BINS, alpha=0.7,
                       color=color, label='Correct', edgecolor='black')

            if incorrect.sum() > 0:
                ax.hist(p_class[incorrect], bins=DEFAULT_HIST_BINS, alpha=0.7,
                       color='red', label='Incorrect', edgecolor='black')

            ax.set_xlabel('Predicted Probability', fontweight='bold', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel('Count', fontweight='bold', fontsize=FONT_SIZE_LABEL)
            ax.set_title(f'{name}', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
            ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper center')
            ax.set_xlim([0, 1])
            ax.tick_params(labelsize=FONT_SIZE_TICK)

        plt.tight_layout()
        self._save_figure(fig, 'class_distributions')
        plt.close()

        self.logger.info("Generated: class_distributions")

    def plot_per_class_metrics(self) -> None:
        """
        Generate per-class metrics bar chart.

        Displays precision, recall, and F1-score for each class as
        grouped bar chart for easy comparison.

        v3.0.1 FIX: Increased figure width and adjusted bar spacing to prevent overlap.

        Output
        ------
        per_class_metrics.{png,pdf,svg} : file
            Bar chart in specified formats.

        Notes
        -----
        Useful for identifying which classes are harder to classify.
        F1-score provides balanced view of precision-recall tradeoff.
        """
        metrics_names = ['Precision', 'Recall', 'F1']
        class_metrics = np.zeros((len(CLASS_NAMES), len(metrics_names)))

        for i, name in enumerate(CLASS_NAMES):
            class_metrics[i, 0] = self.metrics[f'precision_{name}']
            class_metrics[i, 1] = self.metrics[f'recall_{name}']
            class_metrics[i, 2] = self.metrics[f'f1_{name}']

        # v3.0.1: Use wider figure
        fig, ax = plt.subplots(figsize=FIG_PER_CLASS_METRICS)

        x = np.arange(len(CLASS_NAMES))
        width = 0.22 # v3.0.1: Slightly narrower bars

        # Define bar colors for metrics
        metric_colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue, Red, Green

        for i, (metric_name, metric_color) in enumerate(zip(metrics_names, metric_colors)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, class_metrics[:, i], width,
                         label=metric_name, alpha=0.85, color=metric_color,
                         edgecolor='black', linewidth=0.5)

            # v3.0.1: Add value labels on top of bars
            for bar, val in zip(bars, class_metrics[:, i]):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), # 2 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION,
                           rotation=0)

        ax.set_xlabel('Class', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Score', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=FONT_SIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)

        # v3.0.1: Position legend at top to avoid overlapping bars
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right', ncol=3)
        ax.set_ylim([0, 1.15]) # Extra space for value labels

        plt.tight_layout()
        self._save_figure(fig, 'per_class_metrics')
        plt.close()

        self.logger.info("Generated: per_class_metrics")

    def plot_example_light_curves(self) -> None:
        """
        Generate grid of example light curves.

        Shows n_example_grid_per_type correctly classified examples
        for each class with predicted probabilities.

        v3.0.1 FIX: Improved grid spacing and label sizing.

        Output
        ------
        example_light_curves.{png,pdf,svg} : file
            Multi-panel grid in specified formats.

        Notes
        -----
        Denormalized light curves shown in AB magnitudes.
        Predictions displayed with confidence scores.
        Invalid observations (padding) are masked.
        """
        n_per_class = self.n_example_grid_per_type

        # v3.0.1: Adjust figure size based on grid dimensions
        fig_width = n_per_class * 2.8
        fig_height = len(CLASS_NAMES) * 2.5

        fig, axes = plt.subplots(
            len(CLASS_NAMES), n_per_class,
            figsize=(fig_width, fig_height),
            squeeze=False
        )

        for class_idx, class_name in enumerate(CLASS_NAMES):
            # Get correctly classified examples, fallback to any example of this class
            correct_mask = (self.y == class_idx) & (self.preds == class_idx)
            indices = np.where(correct_mask)[0][:n_per_class]

            # Fallback: if insufficient correct examples, show any examples from this class
            if len(indices) < n_per_class:
                class_mask = (self.y == class_idx)
                all_indices = np.where(class_mask)[0][:n_per_class]
                if len(all_indices) > 0:
                    indices = all_indices

            for col_idx in range(n_per_class):
                ax = axes[class_idx, col_idx]

                if col_idx >= len(indices):
                    # No more examples for this class
                    ax.text(0.5, 0.5, 'No data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=FONT_SIZE_TICK)
                    ax.set_xlabel('Time (days)', fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_ylabel('Mag (AB)', fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_title(f'{class_name}', fontsize=FONT_SIZE_TICK, fontweight='bold')
                    continue

                idx = indices[col_idx]

                # Get data
                flux_norm = self.flux_norm[idx]
                times = self.timestamps[idx]

                # Denormalize
                flux = flux_norm * (self.flux_std + EPS) + self.flux_mean

                # Filter padded observations (explicit check for both conditions)
                valid_mask = (times != INVALID_TIMESTAMP) & (times > 0) & (flux != 0)

                if valid_mask.sum() < MIN_VALID_POINTS_PLOT:
                    # Not enough valid points, show empty plot
                    ax.text(0.5, 0.5, 'Insufficient\ndata',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_xlabel('Time (days)', fontsize=FONT_SIZE_ANNOTATION)
                    ax.set_ylabel('Mag (AB)', fontsize=FONT_SIZE_ANNOTATION)
                    prob = self.probs[idx, class_idx]
                    ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=FONT_SIZE_TICK, fontweight='bold')
                    continue

                # Extract valid observations
                times_valid = times[valid_mask]
                flux_valid = flux[valid_mask]

                # v3.0.0: Use per-event baseline magnitude
                m_base = self.baseline_mags[idx]
                mag_valid = magnification_to_mag(flux_valid, m_base)

                # Plot
                ax.scatter(times_valid, mag_valid, s=3, alpha=0.7)

                # Formatting
                ax.invert_yaxis()
                ax.set_xlabel('Time (days)', fontsize=FONT_SIZE_ANNOTATION)
                ax.set_ylabel('Mag (AB)', fontsize=FONT_SIZE_ANNOTATION)

                # Title with prediction - v3.0.1: more compact
                prob = self.probs[idx, class_idx]
                ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=FONT_SIZE_TICK, fontweight='bold')

                ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
                ax.grid(alpha=0.2)

        # v3.0.1: Adjust subplot spacing
        plt.subplots_adjust(hspace=0.4, wspace=0.35)
        plt.tight_layout()
        self._save_figure(fig, 'example_light_curves')
        plt.close()

        self.logger.info("Generated: example_light_curves")

    def plot_u0_dependency(self) -> None:
        """
        Analyze binary classification accuracy vs impact parameter.

        Stratifies binary events by impact parameter (u0) and computes
        accuracy to reveal physical limitations of classification.

        v3.0.1 FIX: Repositioned count annotations and improved legend placement.

        Output
        ------
        u0_dependency.{png,pdf,svg} : file
            Accuracy vs u0 plot in specified formats.

        Notes
        -----
        High u0 events are fundamentally harder to classify as binary
        due to weak caustic signatures. This analysis quantifies the
        transition from detectable to undetectable binary signals.

        References
        ----------
        Gaudi (2012): "Microlensing Surveys for Exoplanets"
        ARA&A, 50, 411-453
        """
        if self.params is None or 'binary' not in self.params:
            self.logger.info("Skipping u0 dependency (parameters not available)")
            return

        # Get binary events with u0 values
        binary_mask = (self.y == 2)
        binary_params = self.params['binary']

        if 'u0' not in binary_params.dtype.names:
            self.logger.warning("u0 field not found in binary parameters")
            return

        u0_values = binary_params['u0']
        binary_preds = self.preds[binary_mask]

        # Stratify by u0
        bin_centers = (U0_BINS[:-1] + U0_BINS[1:]) / 2

        accuracies = []
        errors = []
        counts = []

        for i in range(len(U0_BINS) - 1):
            mask = (u0_values >= U0_BINS[i]) & (u0_values < U0_BINS[i+1])

            if mask.sum() > 0:
                acc = (binary_preds[mask] == 2).mean()
                # Binomial error
                n = mask.sum()
                err = np.sqrt(acc * (1 - acc) / max(n, 1))

                accuracies.append(acc)
                errors.append(err)
                counts.append(n)
            else:
                accuracies.append(np.nan)
                errors.append(0)
                counts.append(0)

        # v3.0.1: Use adjusted figure size
        fig, ax = plt.subplots(figsize=FIG_U0_DEPENDENCY)

        # Filter valid points
        valid = ~np.isnan(accuracies)

        ax.errorbar(bin_centers[valid],
                   np.array(accuracies)[valid],
                   yerr=np.array(errors)[valid],
                   fmt='o-', color=self.colors[2],
                   capsize=4, linewidth=2, markersize=6,
                   label='Binary Accuracy')

        # Reference line at U0_REFERENCE_LINE (typical detectability threshold)
        ax.axvline(U0_REFERENCE_LINE, color='gray', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=rf'$u_0 = {U0_REFERENCE_LINE}$')

        ax.set_xlabel(r'Impact Parameter $u_0$', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Binary Classification Accuracy', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Binary Detection vs Impact Parameter', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1.05])
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        # v3.0.1: Position legend inside plot, upper right
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right',
                 bbox_to_anchor=LEGEND_BBOX_U0, framealpha=0.95)

        # v3.0.1: Add count annotations below x-axis using secondary axis technique
        # Create annotation text below the plot
        valid_bin_centers = bin_centers[valid]
        valid_counts = np.array(counts)[valid]

        # Add counts as text below each point, positioned at y=-0.08 in axes coordinates
        for bc, cnt in zip(valid_bin_centers, valid_counts):
            ax.annotate(f'n={cnt}',
                       xy=(bc, 0),
                       xytext=(bc, U0_ANNOTATION_Y_OFFSET),
                       textcoords=('data', 'axes fraction'),
                       ha='center', fontsize=FONT_SIZE_ANNOTATION,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        # Adjust bottom margin for annotations
        plt.subplots_adjust(bottom=0.18)
        self._save_figure(fig, 'u0_dependency')
        plt.close()

        self.logger.info("Generated: u0_dependency")

    def plot_temporal_bias_check(self) -> None:
        """
        Check for temporal selection bias using Kolmogorov-Smirnov test.

        Compares t0 distributions of correct vs incorrect predictions
        to detect temporal bias in classification performance.

        v3.0.1 FIX: Repositioned text box and legend to prevent overlap.

        Output
        ------
        temporal_bias_check.{png,pdf,svg} : file
            Distribution comparison in specified formats.

        Notes
        -----
        Temporal bias can arise from:
        - Insufficient observations near mission boundaries
        - Non-uniform cadence
        - Edge effects in time series features

        KS-test p < 0.05 indicates significant distributional difference
        (Massey, 1951).

        References
        ----------
        Massey (1951): "The Kolmogorov-Smirnov Test for Goodness of Fit"
        Journal of the American Statistical Association, 46(253), 68-78
        """
        if self.params is None:
            self.logger.info("Skipping temporal bias check (parameters not available)")
            return

        # Try to extract t0 from any parameter set
        t0_all = []

        for class_name in ['flat', 'pspl', 'binary']:
            if class_name in self.params:
                params = self.params[class_name]
                if 't0' in params.dtype.names:
                    t0_all.extend(params['t0'])

        if not t0_all:
            self.logger.warning("No t0 values found in parameters")
            return

        t0_all = np.array(t0_all)

        # Get correct vs incorrect predictions
        correct = (self.preds == self.y)

        # Make sure we have matching lengths
        n_params = len(t0_all)
        correct_subset = correct[:n_params]

        # KS test
        t0_correct = t0_all[correct_subset]
        t0_incorrect = t0_all[~correct_subset]

        if len(t0_correct) == 0 or len(t0_incorrect) == 0:
            self.logger.warning("Insufficient data for temporal bias check")
            return

        ks_stat, p_value = ks_2samp(t0_correct, t0_incorrect)

        # v3.0.1: Use adjusted figure size
        fig, ax = plt.subplots(figsize=FIG_TEMPORAL_BIAS)

        ax.hist(t0_correct, bins=DEFAULT_HIST_BINS, alpha=0.7,
               color='green', label=f'Correct (n={len(t0_correct):,})',
               density=True, edgecolor='black')
        ax.hist(t0_incorrect, bins=DEFAULT_HIST_BINS, alpha=0.7,
               color='red', label=f'Incorrect (n={len(t0_incorrect):,})',
               density=True, edgecolor='black')

        ax.set_xlabel(r'Peak Time $t_0$ (days)', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Normalized Density', fontweight='bold', fontsize=FONT_SIZE_LABEL)
        ax.set_title('Temporal Bias Check', fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        # KS test result - v3.0.1: Position in upper left, away from legend
        result = "BIAS DETECTED" if p_value < 0.05 else "NO BIAS"
        result_color = 'red' if p_value < 0.05 else 'green'
        ax.text(0.02, 0.98,
               f'KS statistic: D={ks_stat:.3f}\np-value: {p_value:.3f}\nResult: {result}',
               transform=ax.transAxes, fontsize=FONT_SIZE_LEGEND,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                        alpha=0.9, edgecolor=result_color, linewidth=2))

        # v3.0.1: Position legend in upper right
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right', framealpha=0.9)

        plt.tight_layout()
        self._save_figure(fig, 'temporal_bias_check')
        plt.close()

        self.logger.info(f"Generated: temporal_bias_check (KS p={p_value:.4f})")

    # v2.7.0 FIX: This method was incorrectly indented at module level
    # Now properly indented as a class method
    def plot_evolution_for_class(self, class_idx: int, sample_idx: int) -> None:
        """
        Generate probability evolution plot for specific sample.

        Creates three-panel visualization showing:
        1. Light curve with observation completeness
        2. Class probability evolution over time
        3. Prediction confidence evolution

        v3.0.1 FIX: Increased panel spacing to prevent y-label overlap.

        Parameters
        ----------
        class_idx : int
            True class index (0=Flat, 1=PSPL, 2=Binary).
        sample_idx : int
            Index of sample in dataset.

        Output
        ------
        evolution_{class}_{idx}.{png,pdf,svg} : file
            Three-panel evolution plot in specified formats.

        Notes
        -----
        FIXED (v2.7+): Now correctly handles both compacted and non-compacted data
        by identifying valid observations and truncating by observation count,
        not array index.
        """
        class_name = CLASS_NAMES[class_idx]

        # Get data
        flux_norm = self.flux_norm[sample_idx]
        delta_t_norm = self.delta_t_norm[sample_idx]
        times = self.timestamps[sample_idx]
        true_label = self.y[sample_idx]

        # =========================================================================
        # v3.0.2 FIX: Use pre-computed valid_lengths instead of flux_norm != 0.0
        # =========================================================================
        # The old code used: valid_indices = np.where(flux_norm != 0.0)[0]
        # This was WRONG because after normalization, padding value is 
        # -flux_mean/flux_std, NOT 0.0!
        # 
        # Use self.valid_lengths which was correctly computed during data loading
        # from the original (pre-normalized) data.
        n_valid = int(self.valid_lengths[sample_idx])

        if n_valid < EVOLUTION_MIN_VALID_POINTS:
            self.logger.warning(f"Skipping evolution for {class_name}_{sample_idx} (too few points: {n_valid})")
            return

        # Data is compacted: valid observations are at [0, n_valid)
        # This is guaranteed by the data loading pipeline

        is_hierarchical = (hasattr(self.model, 'config') and self.model.config.hierarchical)

        # =========================================================================
        # v3.0.0 FIX: Use explicit observation checkpoints for progressive truncation
        # =========================================================================
        # Use predefined observation counts instead of linspace
        obs_counts = [n for n in EVOLUTION_OBS_COUNTS if n <= n_valid]

        # Ensure we have minimum checkpoints
        if not obs_counts:
            obs_counts = [n_valid]
        elif obs_counts[-1] != n_valid:
            obs_counts.append(n_valid)

        n_steps = len(obs_counts)

        probs_evolution = np.zeros((n_steps, NUM_CLASSES))
        times_evolution = np.zeros(n_steps)

        # =========================================================================
        # v3.0.2 FIX: Compute correct normalized padding value
        # =========================================================================
        # The model was trained with padding = (0 - flux_mean) / flux_std
        # We must use the SAME padding value during inference!
        padding_flux = (0.0 - self.flux_mean) / (self.flux_std + EPS)
        padding_delta_t = (0.0 - self.delta_t_mean) / (self.delta_t_std + EPS)

        with torch.no_grad(), torch.inference_mode():
            for i, n_obs in enumerate(obs_counts):
                # =========================================================================
                # Extract first n_obs VALID observations (data is compacted)
                # =========================================================================
                flux_subset = flux_norm[:n_obs]
                delta_t_subset = delta_t_norm[:n_obs]
                time_at_step = times[n_obs - 1]

                # Record time at this step (for x-axis)
                times_evolution[i] = time_at_step

                # =========================================================================
                # v3.0.2 FIX: Use correct normalized padding value, not 0.0!
                # =========================================================================
                max_len = len(flux_norm)
                flux_padded = np.full(max_len, padding_flux, dtype=np.float32)
                delta_t_padded = np.full(max_len, padding_delta_t, dtype=np.float32)

                flux_padded[:n_obs] = flux_subset
                delta_t_padded[:n_obs] = delta_t_subset

                # Inference
                # v3.0.2 FIX: Pass lengths for proper masked pooling
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

        # =========================================================================
        # Denormalize for plotting
        # =========================================================================
        flux_denorm = flux_norm * (self.flux_std + EPS) + self.flux_mean

        # Get valid observations for light curve plot (data is compacted)
        times_valid = times[:n_valid]
        flux_valid = flux_denorm[:n_valid]

        # Filter any remaining invalid values
        plot_mask = (times_valid > 0) & (flux_valid > 0) & np.isfinite(flux_valid)
        times_plot = times_valid[plot_mask]
        flux_plot = flux_valid[plot_mask]

        if len(times_plot) < MIN_VALID_POINTS_PLOT:
            self.logger.warning(f"Skipping evolution plot for {class_name}_{sample_idx} (insufficient valid points for plot)")
            return

        # v3.0.0: Get per-event baseline magnitude
        m_base = self.baseline_mags[sample_idx]
        mag_plot = magnification_to_mag(flux_plot, m_base)

        # =========================================================================
        # Plot - v3.0.1: Improved spacing
        # =========================================================================
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIG_EVOLUTION, sharex=True)

        # Panel 1: Light curve in AB magnitudes
        ax1.scatter(times_plot, mag_plot, s=5, alpha=0.7)
        ax1.invert_yaxis()
        ax1.set_ylabel('AB Magnitude', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax1.set_title(f'Probability Evolution: {class_name} (True={CLASS_NAMES[true_label]})',
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax1.grid(alpha=0.2)
        ax1.tick_params(labelsize=FONT_SIZE_TICK)

        # Panel 2: Probability evolution
        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            ax2.plot(times_evolution, probs_evolution[:, i],
                    'o-', color=color, label=name, linewidth=0.5, markersize=1)

        ax2.axhline(RANDOM_CLASSIFIER_PROB, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Class Probability', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best', framealpha=0.9, ncol=3)
        ax2.grid(alpha=0.2)
        ax2.tick_params(labelsize=FONT_SIZE_TICK)

        # Panel 3: Confidence
        confidence = probs_evolution.max(axis=1)
        predicted_class = probs_evolution.argmax(axis=1)

        ax3.plot(times_evolution, confidence, 'o-', color='black',
                linewidth=2, markersize=4, label='Confidence')
        ax3.fill_between(times_evolution, 0, confidence, alpha=0.3, color='gray')

        ax3.set_xlabel('Time (days)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax3.set_ylabel('Max Probability', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.set_xlim([times_plot.min(), times_plot.max()])
        ax3.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
        ax3.grid(alpha=0.2)
        ax3.tick_params(labelsize=FONT_SIZE_TICK)

        # v3.0.1: Increase spacing between panels
        plt.subplots_adjust(hspace=0.30)
        plt.tight_layout()
        self._save_figure(fig, f'evolution_{class_name}_{sample_idx}')
        plt.close()

        self.logger.debug(f"Generated: evolution_{class_name}_{sample_idx}")

    def run_early_detection_analysis(self) -> None:
        """
        Analyze classification performance vs observation completeness.

        Evaluates how accuracy and F1-score change as a function of
        the fraction of observations used, quantifying early detection
        capabilities.

        Output
        ------
        early_detection_curve.{png,pdf,svg} : file
            Performance vs completeness plot in specified formats.
        early_detection_results.json : file
            Detailed results for all completeness fractions.

        Notes
        -----
        Useful for mission planning and real-time alert systems.
        Validates that classifier can detect events before peak.
        Fractions tested: EARLY_DETECTION_FRACTIONS

        Minimum of EARLY_DETECTION_MIN_REQUIRED valid observations required per fraction.
        """
        self.logger.info("\nRunning early detection analysis...")

        # Completeness fractions to test
        fractions = EARLY_DETECTION_FRACTIONS

        # Validate sequence lengths
        n_valid_per_sample = self.valid_lengths
        min_valid = n_valid_per_sample.min()

        # Filter fractions that would give too few points
        fractions_filtered = [f for f in fractions
                             if int(min_valid * f) >= EARLY_DETECTION_MIN_REQUIRED]

        if not fractions_filtered:
            self.logger.warning(
                f"Sequences too short for early detection "
                f"(min_valid={min_valid}, need >={EARLY_DETECTION_MIN_REQUIRED})"
            )
            return

        results = []

        for frac in fractions_filtered:
            self.logger.info(f" Testing {frac*100:.0f}% completeness...")

            # Truncate sequences
            predictions_trunc = []

            with torch.no_grad(), torch.inference_mode():
                for i in range(len(self.flux_norm)):
                    n_valid = n_valid_per_sample[i]
                    n_use = max(int(n_valid * frac), EARLY_DETECTION_MIN_REQUIRED)

                    # Create truncated version
                    flux_trunc = self.flux_norm[i, :n_use]
                    delta_t_trunc = self.delta_t_norm[i, :n_use]

                    # Pad to full length
                    # v3.0.2 FIX: Use correct normalized padding value, not 0.0!
                    padding_flux = (0.0 - self.flux_mean) / (self.flux_std + EPS)
                    padding_delta_t = (0.0 - self.delta_t_mean) / (self.delta_t_std + EPS)
                    max_len = self.flux_norm.shape[1]
                    flux_padded = np.full(max_len, padding_flux, dtype=np.float32)
                    delta_t_padded = np.full(max_len, padding_delta_t, dtype=np.float32)

                    flux_padded[:n_use] = flux_trunc
                    delta_t_padded[:n_use] = delta_t_trunc

                    # Inference
                    # v3.0.2 FIX: Pass lengths for proper masked pooling
                    flux_tensor = torch.from_numpy(flux_padded[None, :]).to(self.device)
                    delta_t_tensor = torch.from_numpy(delta_t_padded[None, :]).to(self.device)
                    lengths_tensor = torch.tensor([n_use], dtype=torch.long, device=self.device)

                    logits = self.model(flux_tensor, delta_t_tensor, lengths=lengths_tensor)
                    pred = logits.argmax(dim=-1).cpu().item()

                    predictions_trunc.append(pred)

            predictions_trunc = np.array(predictions_trunc)

            # Compute metrics
            acc = accuracy_score(self.y, predictions_trunc)
            f1 = f1_score(self.y, predictions_trunc, average='macro', zero_division=0)

            # Bootstrap CI for accuracy
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

        # Accuracy with error bars
        acc_err_lower = [a - l for a, l in zip(accuracies, acc_lower)]
        acc_err_upper = [u - a for a, u in zip(accuracies, acc_upper)]

        ax.errorbar(np.array(fractions_plot) * 100, accuracies,
                   yerr=[acc_err_lower, acc_err_upper],
                   fmt='o-', label='Accuracy', color=self.colors[1],
                   capsize=4, linewidth=1.5, markersize=5)

        ax.plot(np.array(fractions_plot) * 100, f1_scores, 's--',
               label='F1 (macro)', color=self.colors[2],
               linewidth=1.5, markersize=5)

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

        # Save results
        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyJSONEncoder)

        self.logger.info("Early detection analysis complete")

    def run_all_analysis(self) -> None:
        """
        Execute complete evaluation suite.

        Runs all visualization and analysis methods in sequence:
        1. Core metrics (confusion matrix, ROC, calibration)
        2. Example light curves
        3. Physics-based analysis (u0 dependency, temporal bias)
        4. Probability evolution plots (if requested)
        5. Early detection analysis (if requested)

        Saves comprehensive summary JSON with all metrics and metadata.

        Output
        ------
        All visualization files plus:
        - evaluation_summary.json: Complete metrics and configuration
        - predictions.npz: Raw predictions and probabilities
        - classification_report.txt: Detailed per-class metrics
        - confusion_matrix.npy: Raw confusion matrix
        - evaluation.log: Complete log file

        Notes
        -----
        Progress is logged to both console and file.
        Failures in individual plots do not stop the overall analysis.
        """
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

        # Examples
        try:
            self.plot_example_light_curves()
        except Exception as e:
            self.logger.error(f"Failed to plot example light curves: {e}", exc_info=True)

        # Physics-based
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

        # Save predictions
        np.savez(
            self.output_dir / 'predictions.npz',
            y_true=self.y,
            y_pred=self.preds,
            probabilities=self.probs,
            confidences=self.confs,
            logits=self.logits
        )

        # Classification report
        report = classification_report(
            self.y, self.preds,
            target_names=list(CLASS_NAMES),
            digits=4,
            labels=[0, 1, 2],
            zero_division=0
        )

        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

        # Confusion matrix raw
        cm = confusion_matrix(self.y, self.preds, labels=[0, 1, 2])
        np.save(self.output_dir / 'confusion_matrix.npy', cm)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Overall accuracy: {self.metrics['accuracy']*100:.2f}%")
        self.logger.info(f"F1-score (macro): {self.metrics['f1_macro']:.4f}")
        self.logger.info(f"ROC-AUC (macro): {self.metrics['roc_auc_macro']:.4f}")

        # Per-class summary
        self.logger.info("\nPer-class performance:")
        for i, name in enumerate(CLASS_NAMES):
            n_samples = (self.y == i).sum()
            prec = self.metrics[f'precision_{name}']
            rec = self.metrics[f'recall_{name}']
            f1 = self.metrics[f'f1_{name}']
            self.logger.info(
                f" {name:6s} (n={n_samples:5d}): "
                f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
            )
# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Parse arguments and run evaluation.

    Entry point for command-line execution. Parses all arguments,
    initializes the RomanEvaluator, and runs the complete analysis suite.

    Examples
    --------
    Basic evaluation:
    >>> python evaluate.py --experiment-name baseline_20241215_120000 \\
    ...                    --data test.h5

    With early detection and colorblind-safe palette:
    >>> python evaluate.py --experiment-name baseline_20241215_120000 \\
    ...                    --data test.h5 --early-detection --colorblind-safe

    Subsample and save multiple formats:
    >>> python evaluate.py --experiment-name baseline_20241215_120000 \\
    ...                    --data test.h5 --n-samples 10000 \\
    ...                    --save-formats png pdf svg
    """
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--experiment-name', required=True,
                       help="Name of experiment to evaluate")
    parser.add_argument('--data', required=True,
                       help="Path to test dataset (.h5 or .npz)")

    # Optional arguments
    parser.add_argument('--output-dir', default=None,
                       help="Custom output directory")
    parser.add_argument('--batch-size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--n-samples', type=int, default=None,
                       help="Subsample test set")
    parser.add_argument('--device', default='cuda',
                       help="Device: cuda or cpu")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")

    # Analysis options
    parser.add_argument('--early-detection', action='store_true',
                       help="Run early detection analysis")
    parser.add_argument('--n-evolution-per-type', type=int, default=10,
                       help="Evolution plots per class")
    parser.add_argument('--n-example-grid-per-type', type=int, default=4,
                       help="Examples per class in grid")
    parser.add_argument('--calibration-n-bins', type=int, default=CALIBRATION_DEFAULT_BINS,
                       help="Number of bins for calibration curve")
    parser.add_argument('--no-roc-bootstrap-ci', action='store_true',
                       help="Disable ROC bootstrap CI (faster)")

    # Output options
    parser.add_argument('--colorblind-safe', action='store_true',
                       help="Use colorblind-safe palette")
    parser.add_argument('--use-latex', action='store_true',
                       help="Enable LaTeX rendering")
    parser.add_argument('--save-formats', nargs='+', default=['png'],
                       choices=['png', 'pdf', 'svg'],
                       help="Output formats")
    parser.add_argument('--verbose', action='store_true',
                       help="Debug logging")

    args = parser.parse_args()

    # Convert hyphenated args to underscored for kwargs
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
