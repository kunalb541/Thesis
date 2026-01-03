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

matplotlib.use('Agg')

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

__version__: Final[str] = "7.1.0"  # Bumped for interface fix

# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES: Final[Tuple[str, ...]] = ('Flat', 'PSPL', 'Binary')
NUM_CLASSES: Final[int] = 3
INVALID_TIMESTAMP: Final[float] = -999.0

# Roman telescope parameters
ROMAN_CADENCE_MINUTES: Final[float] = 15.0
ROMAN_CADENCE_DAYS: Final[float] = ROMAN_CADENCE_MINUTES / (24.0 * 60.0)
ROMAN_SEASON_DURATION_DAYS: Final[float] = 72.0

# Baseline magnitude range
ROMAN_SOURCE_MAG_MIN: Final[float] = 18.0
ROMAN_SOURCE_MAG_MAX: Final[float] = 24.0
ROMAN_DEFAULT_BASELINE_MAG: Final[float] = 22.0

# Numerical constants
EPS: Final[float] = 1e-8

# Color palettes
PALETTE_DEFAULT: Final[List[str]] = ['#7f8c8d', '#c0392b', '#2980b9']
PALETTE_COLORBLIND: Final[List[str]] = ['#0173b2', '#de8f05', '#029e73']

# Plot settings
DPI: Final[int] = 300
DPI_SCREEN: Final[int] = 100

# Figure sizes (width, height in inches)
FIG_CONFUSION: Final[Tuple[float, float]] = (5.0, 4.5)
FIG_ROC: Final[Tuple[float, float]] = (6.0, 5.0)
FIG_CALIBRATION: Final[Tuple[float, float]] = (10.0, 4.5)
FIG_CLASS_DIST: Final[Tuple[float, float]] = (12.0, 4.0)
FIG_PER_CLASS: Final[Tuple[float, float]] = (6.0, 4.5)
FIG_EXAMPLES: Final[Tuple[float, float]] = (12.0, 8.0)
FIG_U0: Final[Tuple[float, float]] = (6.0, 5.0)
FIG_TEMPORAL: Final[Tuple[float, float]] = (6.0, 5.0)
FIG_EVOLUTION: Final[Tuple[float, float]] = (8.0, 10.0)
FIG_EARLY: Final[Tuple[float, float]] = (6.0, 4.5)

# Bootstrap settings
N_BOOTSTRAP: Final[int] = 1000
ROC_N_BOOTSTRAP: Final[int] = 200
MIN_SAMPLES_BOOTSTRAP: Final[int] = 100

# Evolution plot settings
EVOLUTION_MIN_VALID: Final[int] = 100
EVOLUTION_OBS_COUNTS: Final[List[int]] = list(range(100, 6913, 5))

# Early detection settings
EARLY_DETECTION_MIN: Final[int] = 10
EARLY_DETECTION_FRACTIONS: Final[List[float]] = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Histogram bins
HIST_BINS: Final[int] = 30
CALIBRATION_BINS: Final[int] = 10

# ROC interpolation
ROC_INTERP_POINTS: Final[int] = 100

# Confidence interval percentiles (95% CI)
CI_LOWER: Final[float] = 2.5
CI_UPPER: Final[float] = 97.5

# u0 analysis
U0_BINS: Final[np.ndarray] = np.linspace(0, 1.0, 50)
U0_REFERENCE: Final[float] = 0.3

# Random classifier baseline
RANDOM_PROB: Final[float] = 1.0 / NUM_CLASSES

# Minimum valid points
MIN_VALID_POINTS: Final[int] = 3

# Memory management
CACHE_CLEAR_FREQ: Final[int] = 100

# Synthetic timestamps
SYNTHETIC_TIME_MAX: Final[float] = ROMAN_SEASON_DURATION_DAYS


# =============================================================================
# MAGNITUDE CONVERSION
# =============================================================================

def magnification_to_mag(
    A: np.ndarray,
    baseline_mag: Union[float, np.ndarray] = 22.0
) -> np.ndarray:
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
# PYTORCH COMPATIBILITY
# =============================================================================

def torch_load_compat(
    path: Path,
    map_location: Union[str, torch.device],
    weights_only: bool = False
) -> Dict[str, Any]:
    """Load checkpoint with backwards-compatible weights_only parameter."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =============================================================================
# BASELINE MAGNITUDE EXTRACTION
# =============================================================================

def extract_baseline_magnitudes(
    data_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Extract baseline magnitudes (m_base) for events.

    Parameters
    ----------
    data_path : Path
        Path to HDF5 data file.
    indices : np.ndarray
        File indices for selected events.
    labels : np.ndarray
        Class labels for selected events.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    np.ndarray
        Baseline magnitudes for each event.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_events = len(indices)
    m_base = np.full(n_events, ROMAN_DEFAULT_BASELINE_MAG, dtype=np.float32)

    def _generate_random(n: int, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed=42 + seed_offset)
        return rng.uniform(ROMAN_SOURCE_MAG_MIN, ROMAN_SOURCE_MAG_MAX, size=n).astype(np.float32)

    try:
        with h5py.File(data_path, 'r') as f:
            if 'm_base' in f:
                logger.info("Loading m_base from global dataset")
                global_m_base = f['m_base'][:]
                valid_mask = (indices >= 0) & (indices < len(global_m_base))
                
                if valid_mask.all():
                    m_base = global_m_base[indices].astype(np.float32)
                else:
                    m_base[valid_mask] = global_m_base[indices[valid_mask]].astype(np.float32)
                
                logger.info(f"m_base range: [{m_base.min():.2f}, {m_base.max():.2f}]")
                return m_base

            logger.warning("Global m_base not found, checking class parameters")
            
            has_m_base = False
            for class_name in CLASS_NAMES:
                param_key = f'params_{class_name.lower()}'
                if param_key in f:
                    if 'm_base' in f[param_key].dtype.names:
                        has_m_base = True
                        break

            if has_m_base:
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
                    if param_key not in f or 'm_base' not in f[param_key].dtype.names:
                        continue

                    param_data = f[param_key]
                    n_params = len(param_data)
                    file_indices = indices[class_mask]
                    class_specific_indices = file_indices - class_offsets[class_idx]
                    
                    valid_mask = (class_specific_indices >= 0) & (class_specific_indices < n_params)
                    if valid_mask.any():
                        output_pos = np.where(class_mask)[0][valid_mask]
                        valid_idx = class_specific_indices[valid_mask]
                        m_base[output_pos] = param_data['m_base'][valid_idx]

                logger.info(f"m_base range: [{m_base.min():.2f}, {m_base.max():.2f}]")
            else:
                logger.warning("m_base not found, generating random values")
                seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
                m_base = _generate_random(n_events, seed_offset)

    except Exception as e:
        logger.error(f"Error loading m_base: {e}")
        seed_offset = int(indices[0]) % 1000 if len(indices) > 0 else 0
        m_base = _generate_random(n_events, seed_offset)

    return m_base


# =============================================================================
# EXPERIMENT FINDING
# =============================================================================

def find_experiment_checkpoint(
    experiment_name: str,
    base_dir: Path = Path('../results/checkpoints'),
    logger: Optional[logging.Logger] = None
) -> Tuple[Path, Path]:
    """
    Find experiment directory and checkpoint file.

    Parameters
    ----------
    experiment_name : str
        Experiment name, partial name, or full path to checkpoint.
    base_dir : Path
        Base directory for experiments.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    Tuple[Path, Path]
        (experiment_directory, checkpoint_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base_dir = Path(base_dir)

    # Full path to checkpoint
    if '/' in experiment_name or experiment_name.endswith('.pt'):
        checkpoint_path = Path(experiment_name)
        if checkpoint_path.exists():
            exp_dir = checkpoint_path.parent
            logger.info(f"Using checkpoint: {checkpoint_path}")
            return exp_dir, checkpoint_path
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check .current_experiment file
    current_exp_file = base_dir / '.current_experiment'
    if current_exp_file.exists():
        saved_name = current_exp_file.read_text().strip()
        exp_dir = base_dir / saved_name
        if exp_dir.exists():
            logger.info(f"Found current experiment: {exp_dir.name}")
            checkpoint = _find_checkpoint(exp_dir, logger)
            if checkpoint:
                return exp_dir, checkpoint

    # Match partial name
    if base_dir.exists():
        matching = []
        for candidate in base_dir.iterdir():
            if candidate.is_dir() and experiment_name.lower() in candidate.name.lower():
                matching.append(candidate)

        if matching:
            exp_dir = sorted(matching, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            logger.info(f"Matched experiment: {exp_dir.name}")
            checkpoint = _find_checkpoint(exp_dir, logger)
            if checkpoint:
                return exp_dir, checkpoint

    # Try ../results/
    results_dir = Path('../results')
    if results_dir.exists():
        exp_dir = results_dir / experiment_name
        if exp_dir.exists() and exp_dir.is_dir():
            logger.info(f"Found in ../results/: {exp_dir.name}")
            checkpoint = _find_checkpoint(exp_dir, logger)
            if checkpoint:
                return exp_dir, checkpoint

    raise FileNotFoundError(f"Experiment '{experiment_name}' not found")


def _find_checkpoint(exp_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Find best checkpoint in experiment directory."""
    for filename in ['best.pt', 'best_model.pt', 'checkpoint_latest.pt']:
        checkpoint = exp_dir / filename
        if checkpoint.exists():
            logger.info(f"Using: {filename}")
            return checkpoint

    checkpoints = list(exp_dir.glob('checkpoint_*.pt'))
    if checkpoints:
        checkpoint = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info(f"Using: {checkpoint.name}")
        return checkpoint

    subdir = exp_dir / 'checkpoints'
    if subdir.exists():
        for filename in ['best.pt', 'best_model.pt', 'checkpoint_latest.pt']:
            checkpoint = subdir / filename
            if checkpoint.exists():
                logger.info(f"Using: checkpoints/{filename}")
                return checkpoint

    logger.warning(f"No checkpoint in {exp_dir}")
    return None


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def configure_matplotlib() -> None:
    """Configure matplotlib and seaborn for publication-quality figures."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
    
    plt.rcParams.update({
        'figure.dpi': DPI_SCREEN,
        'savefig.dpi': DPI,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
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
    """JSON encoder for NumPy types."""

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
        return super().default(obj)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
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

def unwrap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_compat(checkpoint_path, map_location=device, weights_only=False)

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
        raise RuntimeError(f"Failed to import model: {e}")

    config_dict = checkpoint['model_config']
    config = ModelConfig.from_dict(config_dict)
    model = RomanMicrolensingClassifier(config)

    state_dict = unwrap_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, config_dict


def load_normalization_stats(checkpoint_path: Path) -> Dict[str, float]:
    """Load normalization statistics from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_compat(checkpoint_path, map_location='cpu', weights_only=False)

    if 'stats' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'stats'. Keys: {list(checkpoint.keys())}")

    stats_dict = checkpoint['stats']
    required = {'flux_mean', 'flux_std', 'delta_t_mean', 'delta_t_std'}
    missing = required - set(stats_dict.keys())

    if missing:
        raise ValueError(f"Stats missing keys: {missing}")

    stats = {
        'flux_mean': float(stats_dict['flux_mean']),
        'flux_std': float(stats_dict['flux_std']),
        'delta_t_mean': float(stats_dict['delta_t_mean']),
        'delta_t_std': float(stats_dict['delta_t_std'])
    }

    for key, value in stats.items():
        if not np.isfinite(value):
            raise ValueError(f"Stat '{key}' invalid: {value}")
        if 'std' in key and value <= 0:
            raise ValueError(f"Std '{key}' must be positive: {value}")

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
    Load and normalize data from HDF5 file.
    
    FIXED: Normalization now matches train.py - only valid observations are
    normalized, padded zeros remain as zeros so model mask inference works.

    Returns
    -------
    flux_norm : np.ndarray
        Normalized flux [n_samples, seq_len]. Padded positions = 0.0.
    delta_t_norm : np.ndarray
        Normalized delta_t [n_samples, seq_len]. Padded positions = 0.0.
    labels : np.ndarray
        Class labels [n_samples]
    timestamps : np.ndarray
        Timestamps [n_samples, seq_len]
    valid_lengths : np.ndarray
        Valid sequence lengths [n_samples]
    selected_indices : np.ndarray
        Original file row indices [n_samples]
    data_format : str
        'hdf5' or 'npz'
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    if logger:
        logger.info(f"Loading: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix == '.h5':
        data_format = 'hdf5'
        with h5py.File(data_path, 'r') as f:
            flux = f['flux'][:] if 'flux' in f else f['mag'][:]
            delta_t = f['delta_t'][:]
            labels = f['labels'][:]
            
            if 'timestamps' in f:
                timestamps = f['timestamps'][:]
            else:
                n_total, seq_len = flux.shape
                timestamps = np.tile(
                    np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                    (n_total, 1)
                )

    elif suffix == '.npz':
        data_format = 'npz'
        data = np.load(data_path)
        flux = data['flux'] if 'flux' in data else data['mag']
        delta_t = data['delta_t']
        labels = data['labels'] if 'labels' in data else data['y']
        
        if 'timestamps' in data:
            timestamps = data['timestamps']
        elif 'times' in data:
            timestamps = data['times']
        else:
            n_total, seq_len = flux.shape
            timestamps = np.tile(
                np.linspace(0, SYNTHETIC_TIME_MAX, seq_len, dtype=np.float32),
                (n_total, 1)
            )
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    selected_indices = np.arange(len(flux), dtype=np.int64)

    # Subsample if requested
    if n_samples is not None and n_samples < len(flux):
        if logger:
            logger.info(f"Subsampling {n_samples} from {len(flux)}")

        rng = np.random.RandomState(seed)
        n_classes = len(np.unique(labels))
        per_class = n_samples // n_classes

        indices = []
        for c in range(n_classes):
            class_idx = np.where(labels == c)[0]
            n_take = min(per_class, len(class_idx))
            if n_take > 0:
                indices.extend(rng.choice(class_idx, size=n_take, replace=False))

        remainder = n_samples - len(indices)
        if remainder > 0:
            available = np.setdiff1d(np.arange(len(flux)), indices)
            if len(available) >= remainder:
                indices.extend(rng.choice(available, size=remainder, replace=False))

        indices = np.array(indices[:n_samples], dtype=np.int64)
        selected_indices = indices.copy()

        flux = flux[indices]
        delta_t = delta_t[indices]
        labels = labels[indices]
        timestamps = timestamps[indices]

    # Compute valid lengths and normalize ONLY valid observations
    # This matches train.py's MicrolensingDataset behavior
    if logger:
        logger.info("Computing valid lengths and normalizing (train-compatible)")

    n_total, seq_len = flux.shape
    flux_mean = stats['flux_mean']
    flux_std = stats['flux_std']
    delta_t_mean = stats['delta_t_mean']
    delta_t_std = stats['delta_t_std']

    valid_lengths = np.zeros(n_total, dtype=np.int32)
    
    # Create output arrays - start with copies
    flux_norm = flux.astype(np.float32).copy()
    delta_t_norm = delta_t.astype(np.float32).copy()

    for i in range(n_total):
        # Identify valid observations (non-zero flux)
        valid_mask = (flux[i] != 0.0)
        n_valid = valid_mask.sum()
        valid_lengths[i] = max(n_valid, 1)  # At least 1 to avoid edge cases
        
        # Normalize ONLY valid observations, leave zeros as zeros
        # This matches train.py's MicrolensingDataset.__getitem__
        if n_valid > 0:
            flux_norm[i, valid_mask] = (flux[i, valid_mask] - flux_mean) / (flux_std + EPS)
            delta_t_norm[i, valid_mask] = (delta_t[i, valid_mask] - delta_t_mean) / (delta_t_std + EPS)
        # Padded positions remain 0.0 (already copied from original)

    if logger:
        logger.info(f"Valid lengths: min={valid_lengths.min()}, max={valid_lengths.max()}, "
                   f"mean={valid_lengths.mean():.1f}")

    if logger:
        logger.info(f"Loaded {len(flux_norm)} samples")
        for c, name in enumerate(CLASS_NAMES):
            cnt = (labels == c).sum()
            logger.info(f"  {name}: {cnt} ({100*cnt/len(labels):.1f}%)")

    return flux_norm, delta_t_norm, labels, timestamps, valid_lengths, selected_indices, data_format


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
    Run batch inference.
    
    FIXED: Now uses observation_mask parameter instead of lengths,
    matching model.py v7.1.0 forward signature.
    """
    model.eval()
    n_samples = len(flux)
    seq_len = flux.shape[1]
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_logits = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    all_probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    if logger:
        logger.info(f"Inference: {n_samples} samples, {n_batches} batches")

    is_hierarchical = hasattr(model, 'config') and model.config.hierarchical

    with torch.no_grad(), torch.inference_mode():
        for i in tqdm(range(0, n_samples, batch_size), desc="Inference", disable=(logger is None)):
            end = min(i + batch_size, n_samples)
            batch_len = end - i

            flux_batch = torch.from_numpy(flux[i:end]).to(device)
            dt_batch = torch.from_numpy(delta_t[i:end]).to(device)
            
            # Build observation mask from valid_lengths if provided
            # Otherwise, model will infer from flux != 0.0
            if valid_lengths is not None:
                # Create mask: True for positions < valid_length
                len_batch = torch.from_numpy(valid_lengths[i:end]).to(device)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
                obs_mask = positions < len_batch.unsqueeze(1)  # [batch, seq_len]
                
                # Zero out padded positions to be safe
                # (should already be zero from load_and_prepare_data, but defensive)
                flux_batch = torch.where(obs_mask, flux_batch, torch.zeros_like(flux_batch))
                dt_batch = torch.where(obs_mask, dt_batch, torch.zeros_like(dt_batch))
            else:
                obs_mask = None

            # Call model with correct signature: forward(flux, delta_t, observation_mask=None, ...)
            logits = model(flux_batch, dt_batch, observation_mask=obs_mask)

            if is_hierarchical:
                probs = torch.exp(logits)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)

            all_logits[i:end] = logits.cpu().numpy()
            all_probs[i:end] = probs.cpu().numpy()

            if (i // batch_size) % CACHE_CLEAR_FREQ == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = all_probs.argmax(axis=1)
    confidences = all_probs.max(axis=1)

    if logger:
        logger.info(f"Mean confidence: {confidences.mean():.4f}")

    return predictions, all_probs, confidences, all_logits


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def extract_parameters(
    data_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    data_format: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, np.ndarray]]:
    """Extract physical parameters for specified indices."""
    try:
        if data_format == 'hdf5':
            with h5py.File(data_path, 'r') as f:
                params = {}
                param_keys = [k for k in f.keys() if k.startswith('params_')]

                if not param_keys:
                    return None

                all_labels = f['labels'][:].astype(np.int32)
                n_total = len(all_labels)
                
                within_class_idx = np.full(n_total, -1, dtype=np.int32)
                for c in range(NUM_CLASSES):
                    mask = (all_labels == c)
                    within_class_idx[mask] = np.arange(mask.sum(), dtype=np.int32)

                for class_idx, class_name in enumerate(['flat', 'pspl', 'binary']):
                    param_key = f'params_{class_name}'
                    if param_key not in f:
                        continue

                    class_mask = (labels == class_idx)
                    file_indices = indices[class_mask]
                    if len(file_indices) == 0:
                        continue

                    param_data = f[param_key][:]
                    
                    valid_file = (file_indices >= 0) & (file_indices < n_total)
                    valid_indices = file_indices[valid_file]
                    if len(valid_indices) == 0:
                        continue
                    
                    class_event_idx = within_class_idx[valid_indices]
                    valid_class = (class_event_idx >= 0) & (class_event_idx < len(param_data))
                    final_idx = class_event_idx[valid_class]
                    
                    if len(final_idx) > 0:
                        params[class_name] = param_data[final_idx]

                return params if params else None

        elif data_format == 'npz':
            data = np.load(data_path)
            
            if 'params' in data:
                all_params = data['params']
                params = {}
                for c, name in enumerate(['flat', 'pspl', 'binary']):
                    mask = labels == c
                    idx = indices[mask]
                    valid = idx < len(all_params)
                    if valid.any():
                        params[name] = all_params[idx[valid]]
                return params if params else None

            return None

    except Exception as e:
        if logger:
            logger.warning(f"Parameter extraction failed: {e}")
        return None


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Compute classification metrics with confidence intervals."""
    if logger:
        logger.info("Computing metrics")

    metrics = {}

    # Accuracy with CI
    acc, acc_lower, acc_upper = bootstrap_ci(
        np.arange(len(y_true)),
        lambda idx: accuracy_score(y_true[idx], y_pred[idx]),
        n_bootstrap=n_bootstrap,
        seed=seed
    )

    metrics['accuracy'] = float(acc)
    metrics['accuracy_ci_lower'] = float(acc_lower)
    metrics['accuracy_ci_upper'] = float(acc_upper)

    # Macro/weighted metrics
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics
    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    for i, name in enumerate(CLASS_NAMES):
        metrics[f'precision_{name}'] = float(prec[i])
        metrics[f'recall_{name}'] = float(rec[i])
        metrics[f'f1_{name}'] = float(f1[i])

    # ROC-AUC
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics['roc_auc_macro'] = float(roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr'))
        metrics['roc_auc_weighted'] = float(roc_auc_score(y_bin, y_probs, average='weighted', multi_class='ovr'))

        for i, name in enumerate(CLASS_NAMES):
            try:
                metrics[f'roc_auc_{name}'] = float(roc_auc_score(y_bin[:, i], y_probs[:, i]))
            except ValueError:
                metrics[f'roc_auc_{name}'] = 0.0

    except Exception as e:
        if logger:
            logger.warning(f"ROC-AUC failed: {e}")
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_weighted'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + EPS)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_normalized'] = cm_norm.tolist()

    if logger:
        logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}% "
                   f"[{metrics['accuracy_ci_lower']*100:.2f}%, {metrics['accuracy_ci_upper']*100:.2f}%]")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

    return metrics


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================

class RomanEvaluator:
    """Evaluation suite for Roman microlensing classifier."""

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
        verbose: bool = False,
        seed: int = 42,
        calibration_n_bins: int = CALIBRATION_BINS,
        roc_bootstrap_ci: bool = True
    ):
        """Initialize evaluator."""
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.batch_size = batch_size
        self.run_early_detection = early_detection
        self.n_evolution_per_type = n_evolution_per_type
        self.n_example_grid_per_type = n_example_grid_per_type
        self.calibration_n_bins = calibration_n_bins
        self.roc_bootstrap_ci = roc_bootstrap_ci

        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)

        self.exp_dir, self.model_path = find_experiment_checkpoint(
            experiment_name,
            base_dir=Path('../results/checkpoints')
        )

        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = Path(data_path).stem
            output_dir = self.exp_dir / f'eval_{dataset_name}_{timestamp}'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(self.output_dir, verbose=verbose)
        configure_matplotlib()

        self.colors = PALETTE_COLORBLIND if colorblind_safe else PALETTE_DEFAULT
        self.save_formats = save_formats or ['png']

        self.logger.info("=" * 70)
        self.logger.info("ROMAN MICROLENSING CLASSIFIER EVALUATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Version: {__version__}")
        self.logger.info(f"Experiment: {self.exp_dir.name}")
        self.logger.info(f"Checkpoint: {self.model_path.name}")
        self.logger.info(f"Data: {data_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("-" * 70)

        # Load model
        self.logger.info("Loading model...")
        self.model, self.config_dict = load_model_from_checkpoint(self.model_path, self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Parameters: {total_params:,}")

        self.has_aux_head = hasattr(self.model, 'head_aux') and self.model.head_aux is not None

        # Load normalization stats
        self.logger.info("Loading normalization stats...")
        stats = load_normalization_stats(self.model_path)
        self.flux_mean = stats['flux_mean']
        self.flux_std = stats['flux_std']
        self.delta_t_mean = stats['delta_t_mean']
        self.delta_t_std = stats['delta_t_std']
        self.logger.info(f"  Flux: mean={self.flux_mean:.4f}, std={self.flux_std:.4f}")
        self.logger.info(f"  Delta_t: mean={self.delta_t_mean:.6f}, std={self.delta_t_std:.6f}")

        # Load data
        self.logger.info("-" * 70)
        self.data_path = Path(data_path)
        (self.flux_norm, self.delta_t_norm, self.y, self.timestamps,
         self.valid_lengths, self.file_indices, self.data_format) = load_and_prepare_data(
            self.data_path, stats, n_samples=n_samples, seed=seed, logger=self.logger
        )

        # Run inference
        self.logger.info("-" * 70)
        self.preds, self.probs, self.confs, self.logits = run_inference(
            self.model, self.flux_norm, self.delta_t_norm,
            self.device, valid_lengths=self.valid_lengths,
            batch_size=batch_size, logger=self.logger
        )

        # Compute metrics
        self.logger.info("-" * 70)
        self.metrics = compute_metrics(
            self.y, self.preds, self.probs,
            n_bootstrap=N_BOOTSTRAP, seed=seed, logger=self.logger
        )

        # Load baseline magnitudes
        self.logger.info("-" * 70)
        self.logger.info("Loading baseline magnitudes...")
        self.baseline_mags = extract_baseline_magnitudes(
            self.data_path, self.file_indices, self.y, logger=self.logger
        )

        # Load parameters
        self.logger.info("-" * 70)
        self.logger.info("Loading physical parameters...")
        self.params = extract_parameters(
            self.data_path, self.file_indices, self.y, self.data_format, logger=self.logger
        )
        if self.params:
            self.logger.info("Parameters loaded")
        else:
            self.logger.info("Parameters not available")

        self.logger.info("=" * 70)
        self.logger.info("INITIALIZATION COMPLETE")
        self.logger.info("=" * 70)

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in specified formats."""
        for fmt in self.save_formats:
            path = self.output_dir / f'{name}.{fmt}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')

    def plot_confusion_matrix(self) -> None:
        """Generate normalized confusion matrix."""
        cm = np.array(self.metrics['confusion_matrix'])
        cm_norm = np.array(self.metrics['confusion_matrix_normalized'])

        fig, ax = plt.subplots(figsize=FIG_CONFUSION)
        
        sns.heatmap(
            cm_norm, annot=False, cmap='Blues', vmin=0, vmax=1,
            square=True, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Fraction', 'shrink': 0.8}, ax=ax
        )

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                pct = f'{cm_norm[i, j]*100:.1f}%'
                cnt = f'({cm[i, j]:,})'
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.4, pct, ha='center', va='center',
                       color=color, fontsize=10, fontweight='bold')
                ax.text(j + 0.5, i + 0.65, cnt, ha='center', va='center',
                       color=color, fontsize=8)

        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES, rotation=0)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix')
        plt.close()
        self.logger.info("Generated: confusion_matrix")

    def plot_roc_curves(self) -> None:
        """Generate ROC curves with confidence intervals."""
        y_bin = label_binarize(self.y, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=FIG_ROC)

        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], self.probs[:, i])
                auc = self.metrics.get(f'roc_auc_{name}', 0.0)
                ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

                if self.roc_bootstrap_ci and len(self.y) > MIN_SAMPLES_BOOTSTRAP:
                    rng = np.random.RandomState(self.seed)
                    tpr_boot = []
                    fpr_common = np.linspace(0, 1, ROC_INTERP_POINTS)

                    for _ in range(ROC_N_BOOTSTRAP):
                        idx = rng.choice(len(self.y), size=len(self.y), replace=True)
                        try:
                            fpr_b, tpr_b, _ = roc_curve(y_bin[idx, i], self.probs[idx, i])
                            tpr_boot.append(np.interp(fpr_common, fpr_b, tpr_b))
                        except:
                            continue

                    if tpr_boot:
                        tpr_boot = np.array(tpr_boot)
                        tpr_lo = np.percentile(tpr_boot, CI_LOWER, axis=0)
                        tpr_hi = np.percentile(tpr_boot, CI_UPPER, axis=0)
                        ax.fill_between(fpr_common, tpr_lo, tpr_hi, color=color, alpha=0.15)

            except ValueError as e:
                self.logger.warning(f"ROC for {name} failed: {e}")

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_aspect('equal')

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
                    y_binary, p_class, n_bins=self.calibration_n_bins, strategy='uniform'
                )
                ax1.plot(prob_pred, prob_true, 'o-', color=color, linewidth=1.5,
                        markersize=5, label=name)
            except Exception as e:
                self.logger.warning(f"Calibration for {name} failed: {e}")

        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Observed Frequency')
        ax1.set_title('Calibration Curve')
        ax1.legend(loc='upper left')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')

        sns.histplot(self.confs, bins=HIST_BINS, color='steelblue', alpha=0.7,
                    edgecolor='black', linewidth=0.5, ax=ax2)
        ax2.axvline(self.confs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={self.confs.mean():.3f}')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        self._save_figure(fig, 'calibration')
        plt.close()
        self.logger.info("Generated: calibration")

    def plot_class_distributions(self) -> None:
        """Generate class probability distribution plots."""
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=FIG_CLASS_DIST)

        for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, self.colors)):
            p_class = self.probs[:, i]
            correct = (self.y == i) & (self.preds == i)
            incorrect = (self.y == i) & (self.preds != i)

            if correct.sum() > 0:
                ax.hist(p_class[correct], bins=HIST_BINS, alpha=0.7, color=color,
                       label='Correct', edgecolor='black', linewidth=0.5)
            if incorrect.sum() > 0:
                ax.hist(p_class[incorrect], bins=HIST_BINS, alpha=0.7, color='#e74c3c',
                       label='Incorrect', edgecolor='black', linewidth=0.5)

            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Count')
            ax.set_title(name)
            ax.legend(loc='upper center')
            ax.set_xlim([0, 1])

        plt.tight_layout()
        self._save_figure(fig, 'class_distributions')
        plt.close()
        self.logger.info("Generated: class_distributions")

    def plot_per_class_metrics(self) -> None:
        """Generate per-class metrics bar chart."""
        metric_names = ['Precision', 'Recall', 'F1']
        class_metrics = np.zeros((len(CLASS_NAMES), len(metric_names)))

        for i, name in enumerate(CLASS_NAMES):
            class_metrics[i, 0] = self.metrics[f'precision_{name}']
            class_metrics[i, 1] = self.metrics[f'recall_{name}']
            class_metrics[i, 2] = self.metrics[f'f1_{name}']

        fig, ax = plt.subplots(figsize=FIG_PER_CLASS)

        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for i, (mname, mcolor) in enumerate(zip(metric_names, colors)):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, class_metrics[:, i], width,
                         label=mname, alpha=0.85, color=mcolor,
                         edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars, class_metrics[:, i]):
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.legend(loc='upper right', ncol=3)
        ax.set_ylim([0, 1.15])

        plt.tight_layout()
        self._save_figure(fig, 'per_class_metrics')
        plt.close()
        self.logger.info("Generated: per_class_metrics")

    def plot_example_light_curves(self) -> None:
        """Generate grid of example light curves."""
        n_per_class = self.n_example_grid_per_type
        fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class, figsize=FIG_EXAMPLES, squeeze=False)

        for class_idx, class_name in enumerate(CLASS_NAMES):
            correct_mask = (self.y == class_idx) & (self.preds == class_idx)
            indices = np.where(correct_mask)[0][:n_per_class]

            if len(indices) < n_per_class:
                class_mask = (self.y == class_idx)
                indices = np.where(class_mask)[0][:n_per_class]

            for col in range(n_per_class):
                ax = axes[class_idx, col]

                if col >= len(indices):
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel('Time (days)', fontsize=8)
                    ax.set_ylabel('Mag', fontsize=8)
                    ax.set_title(class_name, fontsize=9)
                    continue

                idx = indices[col]
                flux_norm_sample = self.flux_norm[idx]
                times = self.timestamps[idx]
                n_valid = int(self.valid_lengths[idx])

                # CRITICAL: Detect padding from flux_norm BEFORE denormalization
                # After denorm, zeros become flux_mean, making them undetectable
                # Use valid_lengths for robust masking
                valid_mask = np.zeros(len(flux_norm_sample), dtype=bool)
                valid_mask[:n_valid] = (flux_norm_sample[:n_valid] != 0.0) & \
                                       (times[:n_valid] != INVALID_TIMESTAMP) & \
                                       (times[:n_valid] >= 0)

                flux = flux_norm_sample * (self.flux_std + EPS) + self.flux_mean

                if valid_mask.sum() < MIN_VALID_POINTS:
                    ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center',
                           transform=ax.transAxes, fontsize=8)
                    prob = self.probs[idx, class_idx]
                    ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=9)
                    continue

                times_v = times[valid_mask]
                flux_v = flux[valid_mask]
                m_base = self.baseline_mags[idx]
                mag_v = magnification_to_mag(flux_v, m_base)

                ax.scatter(times_v, mag_v, s=2, alpha=0.7, c=self.colors[class_idx])
                ax.invert_yaxis()
                ax.set_xlabel('Time (days)', fontsize=8)
                ax.set_ylabel('Mag', fontsize=8)
                prob = self.probs[idx, class_idx]
                ax.set_title(f'{class_name} (P={prob:.2f})', fontsize=9)
                ax.tick_params(labelsize=7)

        plt.tight_layout()
        self._save_figure(fig, 'example_light_curves')
        plt.close()
        self.logger.info("Generated: example_light_curves")

    def plot_u0_dependency(self) -> None:
        """Analyze binary classification accuracy vs impact parameter."""
        if self.params is None or 'binary' not in self.params:
            self.logger.info("Skipping u0 dependency (parameters unavailable)")
            return

        binary_mask = (self.y == 2)
        binary_params = self.params['binary']

        if 'u0' not in binary_params.dtype.names:
            self.logger.warning("u0 field not found")
            return

        u0_values = binary_params['u0']
        binary_preds = self.preds[binary_mask]

        # Ensure lengths match
        min_len = min(len(u0_values), len(binary_preds))
        u0_values = u0_values[:min_len]
        binary_preds = binary_preds[:min_len]

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

        accuracies = np.array(accuracies)
        errors = np.array(errors)
        counts = np.array(counts)
        valid = ~np.isnan(accuracies)

        fig, ax = plt.subplots(figsize=FIG_U0)

        ax.errorbar(bin_centers[valid], accuracies[valid], yerr=errors[valid],
                   fmt='o-', color=self.colors[2], capsize=3, linewidth=1.5,
                   markersize=5, label='Binary Accuracy')

        ax.axvline(U0_REFERENCE, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                  label=f'$u_0 = {U0_REFERENCE}$')

        ax.set_xlabel(r'Impact Parameter $u_0$')
        ax.set_ylabel('Binary Classification Accuracy')
        ax.set_title('Binary Detection vs Impact Parameter')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1.05])
        ax.legend(loc='lower right')

        plt.tight_layout()
        self._save_figure(fig, 'u0_dependency')
        plt.close()
        self.logger.info("Generated: u0_dependency")

    def plot_temporal_bias_check(self) -> None:
        """Check for temporal selection bias using KS test."""
        if self.params is None:
            self.logger.info("Skipping temporal bias (parameters unavailable)")
            return

        correct = (self.preds == self.y)
        t0_correct_list = []
        t0_incorrect_list = []

        for class_idx, class_key in enumerate(['flat', 'pspl', 'binary']):
            if class_key not in self.params:
                continue

            p = self.params[class_key]
            if not hasattr(p, 'dtype') or p.dtype.names is None or 't0' not in p.dtype.names:
                continue

            class_correct = correct[self.y == class_idx]
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

        if len(t0_correct) == 0 or len(t0_incorrect) == 0:
            self.logger.warning("Empty correct/incorrect sets for temporal bias")
            return

        ks_stat, p_value = ks_2samp(t0_correct, t0_incorrect)

        fig, ax = plt.subplots(figsize=FIG_TEMPORAL)

        ax.hist(t0_correct, bins=HIST_BINS, alpha=0.7, color='#27ae60',
               label=f'Correct (n={len(t0_correct):,})', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(t0_incorrect, bins=HIST_BINS, alpha=0.7, color='#e74c3c',
               label=f'Incorrect (n={len(t0_incorrect):,})', density=True, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(r'Peak Time $t_0$ (days)')
        ax.set_ylabel('Normalized Density')
        ax.set_title('Temporal Bias Check')

        result = "BIAS DETECTED" if p_value < 0.05 else "NO SIGNIFICANT BIAS"
        result_color = '#e74c3c' if p_value < 0.05 else '#27ae60'
        ax.text(0.02, 0.98, f'KS stat: D={ks_stat:.3f}\np-value: {p_value:.3f}\n{result}',
               transform=ax.transAxes, fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9,
                        edgecolor=result_color, linewidth=2))

        ax.legend(loc='upper right')
        plt.tight_layout()
        self._save_figure(fig, 'temporal_bias_check')
        plt.close()
        self.logger.info(f"Generated: temporal_bias_check (KS p={p_value:.4f})")

    def plot_evolution_for_class(self, class_idx: int, sample_idx: int) -> None:
        """Generate probability evolution plot for a sample."""
        class_name = CLASS_NAMES[class_idx]
        flux_norm = self.flux_norm[sample_idx]
        delta_t_norm = self.delta_t_norm[sample_idx]
        times = self.timestamps[sample_idx]
        true_label = self.y[sample_idx]
        n_valid = int(self.valid_lengths[sample_idx])

        if n_valid < EVOLUTION_MIN_VALID:
            return

        is_hierarchical = hasattr(self.model, 'config') and self.model.config.hierarchical
        obs_counts = [n for n in EVOLUTION_OBS_COUNTS if n <= n_valid]
        if not obs_counts:
            obs_counts = [n_valid]
        elif obs_counts[-1] != n_valid:
            obs_counts.append(n_valid)

        n_steps = len(obs_counts)
        probs_evolution = np.zeros((n_steps, NUM_CLASSES))
        times_evolution = np.zeros(n_steps)

        seq_len = len(flux_norm)

        with torch.no_grad(), torch.inference_mode():
            for i, n_obs in enumerate(obs_counts):
                times_evolution[i] = times[n_obs - 1]

                # Create truncated version: zero out positions >= n_obs
                flux_trunc = flux_norm.copy()
                dt_trunc = delta_t_norm.copy()
                flux_trunc[n_obs:] = 0.0
                dt_trunc[n_obs:] = 0.0

                flux_t = torch.from_numpy(flux_trunc[None, :]).to(self.device)
                dt_t = torch.from_numpy(dt_trunc[None, :]).to(self.device)
                
                # Build observation mask
                positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                obs_mask = positions < n_obs

                logits = self.model(flux_t, dt_t, observation_mask=obs_mask)

                if is_hierarchical:
                    probs = torch.exp(logits)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)

                probs_evolution[i] = probs.cpu().numpy()[0]

        flux_denorm = flux_norm * (self.flux_std + EPS) + self.flux_mean
        times_valid = times[:n_valid]
        flux_valid = flux_denorm[:n_valid]
        flux_norm_valid = flux_norm[:n_valid]
        
        # CRITICAL: Use flux_norm (before denorm) to detect true zeros/padding
        # After denorm, zeros become flux_mean and cannot be filtered
        plot_mask = (times_valid >= 0) & (flux_norm_valid != 0.0) & np.isfinite(flux_valid)
        times_plot = times_valid[plot_mask]
        flux_plot = flux_valid[plot_mask]

        if len(times_plot) < MIN_VALID_POINTS:
            return

        m_base = self.baseline_mags[sample_idx]
        mag_plot = magnification_to_mag(flux_plot, m_base)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIG_EVOLUTION, sharex=True)

        # Light curve
        ax1.scatter(times_plot, mag_plot, s=1, alpha=0.7, c='steelblue')
        ax1.invert_yaxis()
        ax1.set_ylabel('AB Magnitude')
        ax1.set_title(f'Probability Evolution: {class_name} (True={CLASS_NAMES[true_label]})')

        # Probability evolution
        for i, (name, color) in enumerate(zip(CLASS_NAMES, self.colors)):
            ax2.plot(times_evolution, probs_evolution[:, i], '-', color=color,
                    label=name, linewidth=1.5)

        ax2.axhline(RANDOM_PROB, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Class Probability')
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc='upper right', ncol=3)

        # Confidence
        confidence = probs_evolution.max(axis=1)
        ax3.plot(times_evolution, confidence, '-', color='black', linewidth=1.5)
        ax3.fill_between(times_evolution, 0, confidence, alpha=0.2, color='gray')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Max Probability')
        ax3.set_ylim([0, 1.05])
        ax3.set_xlim([times_plot.min(), times_plot.max()])

        plt.tight_layout()
        self._save_figure(fig, f'evolution_{class_name}_{sample_idx}')
        plt.close()

    def run_early_detection_analysis(self) -> None:
        """Analyze classification performance vs observation completeness."""
        self.logger.info("Running early detection analysis...")

        fractions = EARLY_DETECTION_FRACTIONS
        min_valid = self.valid_lengths.min()

        fractions_ok = [f for f in fractions if int(min_valid * f) >= EARLY_DETECTION_MIN]
        if not fractions_ok:
            self.logger.warning(f"Sequences too short (min={min_valid})")
            return

        results = []
        seq_len = self.flux_norm.shape[1]

        for frac in fractions_ok:
            self.logger.info(f"  Testing {frac*100:.0f}% completeness...")
            preds_trunc = []

            with torch.no_grad(), torch.inference_mode():
                for i in range(len(self.flux_norm)):
                    n_valid = self.valid_lengths[i]
                    n_use = max(int(n_valid * frac), EARLY_DETECTION_MIN)

                    # Create truncated version
                    flux_trunc = self.flux_norm[i].copy()
                    dt_trunc = self.delta_t_norm[i].copy()
                    flux_trunc[n_use:] = 0.0
                    dt_trunc[n_use:] = 0.0

                    flux_t = torch.from_numpy(flux_trunc[None, :]).to(self.device)
                    dt_t = torch.from_numpy(dt_trunc[None, :]).to(self.device)
                    
                    # Build observation mask
                    positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                    obs_mask = positions < n_use

                    logits = self.model(flux_t, dt_t, observation_mask=obs_mask)
                    preds_trunc.append(logits.argmax(dim=-1).cpu().item())

            preds_trunc = np.array(preds_trunc)
            acc = accuracy_score(self.y, preds_trunc)
            f1 = f1_score(self.y, preds_trunc, average='macro', zero_division=0)

            _, acc_lo, acc_hi = bootstrap_ci(
                np.arange(len(self.y)),
                lambda idx: accuracy_score(self.y[idx], preds_trunc[idx]),
                n_bootstrap=N_BOOTSTRAP, seed=self.seed
            )

            results.append({
                'fraction': float(frac),
                'accuracy': float(acc),
                'accuracy_ci_lower': float(acc_lo),
                'accuracy_ci_upper': float(acc_hi),
                'f1_macro': float(f1)
            })

            self.logger.info(f"    Accuracy: {acc*100:.2f}% [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%]")

        # Plot
        fracs = np.array([r['fraction'] for r in results]) * 100
        accs = [r['accuracy'] for r in results]
        acc_lo = [r['accuracy_ci_lower'] for r in results]
        acc_hi = [r['accuracy_ci_upper'] for r in results]
        f1s = [r['f1_macro'] for r in results]

        fig, ax = plt.subplots(figsize=FIG_EARLY)

        acc_err_lo = [a - l for a, l in zip(accs, acc_lo)]
        acc_err_hi = [h - a for a, h in zip(accs, acc_hi)]

        ax.errorbar(fracs, accs, yerr=[acc_err_lo, acc_err_hi],
                   fmt='o-', label='Accuracy', color=self.colors[1],
                   capsize=3, linewidth=1.5, markersize=5)
        ax.plot(fracs, f1s, 's--', label='F1 (macro)', color=self.colors[2],
               linewidth=1.5, markersize=5)

        ax.set_xlabel('Sequence Completeness (%)')
        ax.set_ylabel('Score')
        ax.set_title('Early Detection Performance')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(5, 105)
        ax.legend(loc='lower right')

        plt.tight_layout()
        self._save_figure(fig, 'early_detection_curve')
        plt.close()

        with open(self.output_dir / 'early_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info("Early detection analysis complete")

    def run_all_analysis(self) -> None:
        """Execute complete evaluation suite."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("=" * 70)

        for method_name, method in [
            ('confusion_matrix', self.plot_confusion_matrix),
            ('roc_curves', self.plot_roc_curves),
            ('calibration', self.plot_calibration_curve),
            ('class_distributions', self.plot_class_distributions),
            ('per_class_metrics', self.plot_per_class_metrics),
            ('example_light_curves', self.plot_example_light_curves),
            ('u0_dependency', self.plot_u0_dependency),
            ('temporal_bias', self.plot_temporal_bias_check),
        ]:
            try:
                method()
            except Exception as e:
                self.logger.error(f"Failed {method_name}: {e}", exc_info=True)

        # Evolution plots
        if self.n_evolution_per_type > 0:
            self.logger.info("Generating evolution plots...")
            for class_idx, class_name in enumerate(CLASS_NAMES):
                indices = np.where(self.y == class_idx)[0][:self.n_evolution_per_type]
                for idx in indices:
                    try:
                        self.plot_evolution_for_class(class_idx, idx)
                    except Exception as e:
                        self.logger.warning(f"Evolution {class_name}_{idx} failed: {e}")

        # Early detection
        if self.run_early_detection:
            try:
                self.run_early_detection_analysis()
            except Exception as e:
                self.logger.error(f"Early detection failed: {e}", exc_info=True)

        # Save summary
        summary = {
            'experiment': str(self.exp_dir.name),
            'model_path': str(self.model_path),
            'data_path': str(self.data_path),
            'data_size': int(len(self.y)),
            'class_distribution': {
                name: int((self.y == i).sum()) for i, name in enumerate(CLASS_NAMES)
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
            self.y, self.preds, target_names=list(CLASS_NAMES),
            digits=4, labels=[0, 1, 2], zero_division=0
        )
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

        cm = confusion_matrix(self.y, self.preds, labels=[0, 1, 2])
        np.save(self.output_dir / 'confusion_matrix.npy', cm)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Results: {self.output_dir}")
        self.logger.info(f"Accuracy: {self.metrics['accuracy']*100:.2f}%")
        self.logger.info(f"F1 (macro): {self.metrics['f1_macro']:.4f}")
        self.logger.info(f"ROC-AUC (macro): {self.metrics['roc_auc_macro']:.4f}")

        self.logger.info("\nPer-class performance:")
        for i, name in enumerate(CLASS_NAMES):
            n = (self.y == i).sum()
            p = self.metrics[f'precision_{name}']
            r = self.metrics[f'recall_{name}']
            f = self.metrics[f'f1_{name}']
            self.logger.info(f"  {name:6s} (n={n:5d}): P={p:.3f} R={r:.3f} F1={f:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Classifier Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--experiment-name', required=True, help="Experiment name or checkpoint path")
    parser.add_argument('--data', required=True, help="Path to test dataset (.h5 or .npz)")
    parser.add_argument('--output-dir', default=None, help="Output directory")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--n-samples', type=int, default=None, help="Subsample size")
    parser.add_argument('--device', default='cuda', help="Device (cuda/cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    parser.add_argument('--early-detection', action='store_true', help="Run early detection analysis")
    parser.add_argument('--n-evolution-per-type', type=int, default=10, help="Evolution plots per class")
    parser.add_argument('--n-example-grid-per-type', type=int, default=4, help="Examples per class")
    parser.add_argument('--calibration-n-bins', type=int, default=CALIBRATION_BINS, help="Calibration bins")
    parser.add_argument('--no-roc-bootstrap-ci', action='store_true', help="Disable ROC CI")

    parser.add_argument('--colorblind-safe', action='store_true', help="Colorblind-safe palette")
    parser.add_argument('--save-formats', nargs='+', default=['png'],
                       choices=['png', 'pdf', 'svg'], help="Output formats")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

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
        verbose=args.verbose,
        seed=args.seed,
        calibration_n_bins=args.calibration_n_bins,
        roc_bootstrap_ci=(not args.no_roc_bootstrap_ci)
    )

    evaluator.run_all_analysis()


if __name__ == '__main__':
    main()
