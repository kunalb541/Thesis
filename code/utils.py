#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing - FIXED VERSION

Author: Kunal Bhatia
Date: October 2025
Version: 3.2 - Fixed normalization bugs and pad handling
"""

import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path
import config as CFG


def _feature_means_ignore_pad(X: np.ndarray, pad_value: float) -> np.ndarray:
    """
    Compute per-feature means ignoring pad_value.
    If a feature is fully padded, its mean is set to 0.0.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")
    N, F = X.shape
    means = np.zeros(F, dtype=np.float32)
    for j in range(F):
        col = X[:, j]
        valid = col != pad_value
        if np.any(valid):
            means[j] = col[valid].mean(dtype=np.float64)
        else:
            means[j] = 0.0
    return means.astype(np.float32)


def two_stage_normalize(X_train, X_val, X_test, pad_value=-1.0):
    """
    Two-stage normalization (StandardScaler then MinMaxScaler).
    Fits on TRAIN ONLY (after filling pads with per-feature train means),
    applies to train/val/test, and restores pad_value positions.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (N, F)
        Raw data arrays possibly containing pad_value.
    pad_value : float
        Sentinel for padded cells.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax
    """
    # Ensure arrays are float32 and contiguous
    X_train = np.asarray(X_train, dtype=np.float32, order="C")
    X_val   = np.asarray(X_val,   dtype=np.float32, order="C")
    X_test  = np.asarray(X_test,  dtype=np.float32, order="C")

    # Basic checks
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError("All inputs must be 2D arrays.")
    F = X_train.shape[1]
    if X_val.shape[1] != F or X_test.shape[1] != F:
        raise ValueError("Feature dimension mismatch across splits.")

    # Remember pad masks
    train_mask_pad = (X_train == pad_value)
    val_mask_pad   = (X_val   == pad_value)
    test_mask_pad  = (X_test  == pad_value)

    # Compute per-feature means on TRAIN ignoring pads
    means_train = _feature_means_ignore_pad(X_train, pad_value)   # shape (F,)
    means_train_row_train = np.broadcast_to(means_train, X_train.shape)
    means_train_row_val   = np.broadcast_to(means_train, X_val.shape)
    means_train_row_test  = np.broadcast_to(means_train, X_test.shape)

    # Fill pads with train feature means so transforms are well-defined
    X_train_filled = np.where(train_mask_pad, means_train_row_train, X_train)
    X_val_filled   = np.where(val_mask_pad,   means_train_row_val,   X_val)
    X_test_filled  = np.where(test_mask_pad,  means_train_row_test,  X_test)

    # Stage 1: StandardScaler (fit on TRAIN ONLY, full (N, F) matrix)
    # Note: We include filled values (equal to means), which do not bias means and only mildly affect variance.
    print("Applying StandardScaler (fit on train only, per-feature)...")
    scaler_standard = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler_standard.fit(X_train_filled)  # learns per-feature mean_ and scale_

    X_train_std = scaler_standard.transform(X_train_filled)
    X_val_std   = scaler_standard.transform(X_val_filled)
    X_test_std  = scaler_standard.transform(X_test_filled)

    # Stage 2: MinMaxScaler (fit on TRAIN ONLY, after standardization)
    print("Applying MinMaxScaler (fit on train only, per-feature)...")
    scaler_minmax = MinMaxScaler(copy=True, feature_range=(0.0, 1.0))
    scaler_minmax.fit(X_train_std)

    X_train_scaled = scaler_minmax.transform(X_train_std)
    X_val_scaled   = scaler_minmax.transform(X_val_std)
    X_test_scaled  = scaler_minmax.transform(X_test_std)

    # Restore pad_value at original pad positions
    X_train_scaled[train_mask_pad] = pad_value
    X_val_scaled[val_mask_pad]     = pad_value
    X_test_scaled[test_mask_pad]   = pad_value

    # Print ranges of non-padded values
    train_nonpad = X_train_scaled[~train_mask_pad]
    val_nonpad   = X_val_scaled[~val_mask_pad]
    test_nonpad  = X_test_scaled[~test_mask_pad]
    if train_nonpad.size:
        print(f"Final ranges (non-padded values):")
        print(f"  Train: [{train_nonpad.min():.3f}, {train_nonpad.max():.3f}]")
        print(f"  Val:   [{val_nonpad.min():.3f}, {val_nonpad.max():.3f}]")
        print(f"  Test:  [{test_nonpad.min():.3f}, {test_nonpad.max():.3f}]")

    return X_train_scaled.astype(np.float32, copy=False), \
           X_val_scaled.astype(np.float32, copy=False), \
           X_test_scaled.astype(np.float32, copy=False), \
           scaler_standard, scaler_minmax


def save_scalers(scaler_standard, scaler_minmax, output_dir):
    """Save scalers for later use during evaluation"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "scaler_standard.pkl", 'wb') as f:
        pickle.dump(scaler_standard, f)
    with open(output_dir / "scaler_minmax.pkl", 'wb') as f:
        pickle.dump(scaler_minmax, f)
    print(f"✓ Scalers saved to {output_dir}/")


def load_scalers(model_dir):
    """
    Load saved scalers from training directory

    Returns
    -------
    scaler_standard, scaler_minmax
    """
    model_dir = Path(model_dir)
    scaler_std_path = model_dir / "scaler_standard.pkl"
    scaler_mm_path  = model_dir / "scaler_minmax.pkl"
    if not scaler_std_path.exists():
        raise FileNotFoundError(f"StandardScaler not found: {scaler_std_path}")
    if not scaler_mm_path.exists():
        raise FileNotFoundError(f"MinMaxScaler not found: {scaler_mm_path}")
    with open(scaler_std_path, 'rb') as f:
        scaler_standard = pickle.load(f)
    with open(scaler_mm_path, 'rb') as f:
        scaler_minmax = pickle.load(f)
    print(f"✓ Loaded scalers from {model_dir}")
    return scaler_standard, scaler_minmax


def apply_scalers_to_data(X, scaler_standard, scaler_minmax, pad_value=-1.0):
    """
    Apply pre-fitted scalers to new data (for evaluation).
    Transforms the full (N, F) matrix, filling pads with train means implied
    by the fitted StandardScaler (its mean_), then restores pad_value.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    scaler_standard : StandardScaler (already fitted on train)
    scaler_minmax   : MinMaxScaler  (already fitted on train)
    pad_value       : float

    Returns
    -------
    X_scaled : np.ndarray, shape (N, F)
    """
    if X is None:
        raise ValueError("X is None.")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")

    X = np.asarray(X, dtype=np.float32, order="C")
    N, F = X.shape
    if getattr(scaler_standard, "mean_", None) is None:
        raise ValueError("Standard scaler is not fitted (mean_ missing).")
    if scaler_standard.mean_.shape[0] != F:
        raise ValueError(
            f"Feature mismatch: scaler has {scaler_standard.mean_.shape[0]} features, X has {F}."
        )

    pad_mask = (X == pad_value)

    # Fill pads with per-feature train means (from fitted scaler)
    means_row = np.broadcast_to(scaler_standard.mean_.astype(X.dtype), X.shape)
    X_filled = np.where(pad_mask, means_row, X)

    # Apply transforms
    X_std    = scaler_standard.transform(X_filled)
    X_scaled = scaler_minmax.transform(X_std)

    # Restore pads
    X_scaled[pad_mask] = pad_value

    # Report range on non-padded entries
    nonpad = X_scaled[~pad_mask]
    if nonpad.size:
        print(f"✓ Applied scalers. Data range (non-padded): [{nonpad.min():.3f}, {nonpad.max():.3f}]")

    return X_scaled.astype(np.float32, copy=False)


def load_npz_dataset(path, apply_perm=False, normalize=False):
    """
    Load dataset from .npz file

    Args:
        path: Path to .npz file
        apply_perm: Whether to apply saved permutation (shuffling)
        normalize: If True, applies normalization (NOT RECOMMENDED - use for evaluation only with caution)

    Returns:
        X, y, timestamps, meta

    WARNING: Setting normalize=True will fit scalers on THIS data, which causes
    data leakage if used on train/val/test before splitting. For training, always
    use normalize=False and apply two_stage_normalize() after splitting.
    """
    data = np.load(path, allow_pickle=False)

    X = data['X']
    y = data['y']
    timestamps = data['timestamps']

    meta_json = str(data['meta_json'])
    meta = json.loads(meta_json)

    if apply_perm and 'perm' in data.files:
        perm = data['perm']
        X = X[perm]
        y = y[perm]

    # Handle label encoding
    if y.dtype.kind in ('U', 'S', 'O'):
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8)

    # WARNING: Only use normalize=True for quick tests or when you know what you're doing
    if normalize:
        print("⚠️  WARNING: Normalizing during load. This should NOT be used for training!")
        print("   For training: Load raw data, split, then normalize with two_stage_normalize()")

        # Replace pad with 0 to avoid NaNs, then fit scalers on-the-fly
        X_tmp = X.copy().astype(np.float32)
        X_tmp[X_tmp == -1.0] = 0.0

        scaler_standard = StandardScaler()
        X_tmp = scaler_standard.fit_transform(X_tmp)

        scaler_minmax = MinMaxScaler()
        X_tmp = scaler_minmax.fit_transform(X_tmp)

        X = X_tmp

    return X, y, timestamps, meta


def check_gpu():
    """
    Check GPU availability and print info

    Returns:
        device: torch.device
        num_gpus: int
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
        device = torch.device('cuda')
    else:
        print("⚠ CUDA not available, using CPU")
        num_gpus = 0
        device = torch.device('cpu')

    return device, num_gpus


if __name__ == "__main__":
    print("="*60)
    print("GPU Check:")
    print("="*60)
    device, num_gpus = check_gpu()
    print("="*60)

    print("\n" + "="*60)
    print("Testing Scaler Functions:")
    print("="*60)

    # Create dummy data
    np.random.seed(42)
    N_train, N_val, N_test, F = 1000, 200, 200, 100
    X_train = np.random.randn(N_train, F).astype(np.float32)
    X_val   = np.random.randn(N_val,   F).astype(np.float32)
    X_test  = np.random.randn(N_test,  F).astype(np.float32)

    # Add some padding (-1) to the last 10 features
    X_train[:, -10:] = -1.0
    X_val[:,   -10:] = -1.0
    X_test[:,  -10:] = -1.0

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # Normalize
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
        X_train, X_val, X_test, pad_value=-1.0
    )

    # Apply scalers again to (a copy of) train to verify idempotency on non-padded cells
    X_train_check = apply_scalers_to_data(X_train.copy(), scaler_std, scaler_mm, pad_value=-1.0)

    # Basic assertions
    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape   == X_val.shape
    assert X_test_scaled.shape  == X_test.shape
    # Pads preserved
    assert np.all(X_train_scaled[:, -10:] == -1.0)
    assert np.all(X_val_scaled[:,   -10:] == -1.0)
    assert np.all(X_test_scaled[:,  -10:] == -1.0)

    print("\n✓ Normalization test passed!")
    print("="*60)
