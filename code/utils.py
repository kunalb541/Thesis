#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing - FIXED VERSION (v5.2)

Author: Kunal Bhatia
Date: November 2025
Version: 5.2 - Fixed normalization functions to handle 3D [N, C, T] data.
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
    Assumes X is 2D [N, F].
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
    Two-stage normalization (StandardScaler then MinMaxScaler) for 3D data.
    Works on [N, C, T] data by flattening to [N, C*T].
    Fits on TRAIN ONLY.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (N, C, T)
        Raw data arrays possibly containing pad_value.
    pad_value : float
        Sentinel for padded cells.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax
    """
    # Get original shapes
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(f"All inputs must be 3D [N, C, T] arrays.")
    
    N_train, C, T = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]
    F = C * T # Total features
    
    print(f"Normalizing 3D data by flattening [N, {C}, {T}] -> [N, {F}]")

    # Flatten to [N, F]
    X_train_flat = X_train.reshape(N_train, F).astype(np.float32)
    X_val_flat = X_val.reshape(N_val, F).astype(np.float32)
    X_test_flat = X_test.reshape(N_test, F).astype(np.float32)

    # Remember pad masks
    train_mask_pad = (X_train_flat == pad_value)
    val_mask_pad   = (X_val_flat   == pad_value)
    test_mask_pad  = (X_test_flat  == pad_value)

    # Compute per-feature means on TRAIN ignoring pads
    means_train = _feature_means_ignore_pad(X_train_flat, pad_value) # [F,]
    
    # Fill pads with train feature means
    X_train_filled = np.where(train_mask_pad, means_train, X_train_flat)
    X_val_filled   = np.where(val_mask_pad,   means_train, X_val_flat)
    X_test_filled  = np.where(test_mask_pad,  means_train, X_test_flat)

    # Stage 1: StandardScaler (fit on TRAIN ONLY)
    print("Applying StandardScaler (fit on train only, per-feature)...")
    scaler_standard = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler_standard.fit(X_train_filled)  # learns [F,] mean_ and scale_

    X_train_std = scaler_standard.transform(X_train_filled)
    X_val_std   = scaler_standard.transform(X_val_filled)
    X_test_std  = scaler_standard.transform(X_test_filled)

    # Stage 2: MinMaxScaler (fit on TRAIN ONLY, after standardization)
    print("Applying MinMaxScaler (fit on train only, per-feature)...")
    scaler_minmax = MinMaxScaler(copy=True, feature_range=(0.0, 1.0))
    scaler_minmax.fit(X_train_std)

    X_train_scaled_flat = scaler_minmax.transform(X_train_std)
    X_val_scaled_flat   = scaler_minmax.transform(X_val_std)
    X_test_scaled_flat  = scaler_minmax.transform(X_test_std)

    # Restore pad_value at original pad positions
    X_train_scaled_flat[train_mask_pad] = pad_value
    X_val_scaled_flat[val_mask_pad]     = pad_value
    X_test_scaled_flat[test_mask_pad]   = pad_value
    
    # Reshape back to [N, C, T]
    X_train_scaled = X_train_scaled_flat.reshape(N_train, C, T)
    X_val_scaled = X_val_scaled_flat.reshape(N_val, C, T)
    X_test_scaled = X_test_scaled_flat.reshape(N_test, C, T)

    # Print ranges
    train_nonpad = X_train_scaled_flat[~train_mask_pad]
    if train_nonpad.size:
        print(f"Final ranges (non-padded): [{train_nonpad.min():.3f}, {train_nonpad.max():.3f}]")

    return X_train_scaled, X_val_scaled, X_test_scaled, \
           scaler_standard, scaler_minmax


def save_scalers(scaler_standard, scaler_minmax, output_dir):
    """
    Save scalers for later use during evaluation.
    This is now just a wrapper for the DDP-safe function in train.py
    """
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
    Apply pre-fitted scalers to new 3D data [N, C, T].
    Flattens to [N, C*T], applies 1D scalers, and reshapes back.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
    scaler_standard : StandardScaler (already fitted on train, 1D)
    scaler_minmax   : MinMaxScaler  (already fitted on train, 1D)
    pad_value       : float

    Returns
    -------
    X_scaled : np.ndarray, shape (N, C, T)
    """
    if X is None:
        raise ValueError("X is None.")
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array [N, C, T], got {X.shape}")

    X = np.asarray(X, dtype=np.float32, order="C")
    N, C, T = X.shape
    F_flat = C * T
    
    # --- FIX: Check scaler features against flattened features ---
    if getattr(scaler_standard, "mean_", None) is None:
        raise ValueError("Standard scaler is not fitted (mean_ missing).")
    F_scaler = scaler_standard.mean_.shape[0]
    if F_scaler != F_flat:
        raise ValueError(
            f"Feature mismatch: scaler has {F_scaler} features, "
            f"X has [N, {C}, {T}] = {F_flat} features."
        )

    # Flatten to [N, F]
    X_flat = X.reshape(N, F_flat)
    pad_mask = (X_flat == pad_value)

    # Fill pads with per-feature train means (from fitted scaler)
    means_row = np.broadcast_to(scaler_standard.mean_.astype(X.dtype), X_flat.shape)
    X_filled = np.where(pad_mask, means_row, X_flat)

    # Apply transforms
    X_std    = scaler_standard.transform(X_filled)
    X_scaled_flat = scaler_minmax.transform(X_std)

    # Restore pads
    X_scaled_flat[pad_mask] = pad_value

    # Reshape back to [N, C, T]
    X_scaled = X_scaled_flat.reshape(N, C, T)

    # Report range on non-padded entries
    nonpad = X_scaled_flat[~pad_mask]
    if nonpad.size:
        print(f"✓ Applied scalers. Data range (non-padded): [{nonpad.min():.3f}, {nonpad.max():.3f}]")

    return X_scaled.astype(np.float32, copy=False)


def load_npz_dataset(path, apply_perm=False, normalize=False):
    """
    Load dataset from .npz file
    """
    data = np.load(path, allow_pickle=False)

    X = data['X'] # [N, T]
    y = data['y']
    timestamps = data['timestamps']

    meta_json = str(data['meta_json'])
    meta = json.loads(meta_json)

    if apply_perm and 'perm' in data.files:
        perm = data['perm']
        X = X[perm]
        y = y[perm]

    if y.dtype.kind in ('U', 'S', 'O'):
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8)
    
    if normalize:
        raise ValueError("normalize=True is deprecated. "
                         "Load raw data, reshape to 3D, split, "
                         "then use normalization functions.")

    return X, y, timestamps, meta


def check_gpu():
    """
    Check GPU availability and print info
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
    print("Testing Scaler Functions (3D):")
    print("="*60)

    # --- FIX: Test with 3D data [N, C, T] ---
    np.random.seed(42)
    N_train, N_val, N_test, C, T = 1000, 200, 200, 1, 1500
    X_train = np.random.randn(N_train, C, T).astype(np.float32)
    X_val   = np.random.randn(N_val,   C, T).astype(np.float32)
    X_test  = np.random.randn(N_test,  C, T).astype(np.float32)

    # Add some padding (-1) to the last 10 features
    X_train[:, :, -10:] = -1.0
    X_val[:,   :, -10:] = -1.0
    X_test[:,  :, -10:] = -1.0

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    # Normalize
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
        X_train, X_val, X_test, pad_value=-1.0
    )
    
    print(f"Scaled Train shape: {X_train_scaled.shape}")

    # Apply scalers again
    X_train_check = apply_scalers_to_data(X_train.copy(), scaler_std, scaler_mm, pad_value=-1.0)

    # Basic assertions
    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape   == X_val.shape
    assert X_test_scaled.shape  == X_test.shape
    assert np.all(X_train_scaled[:, :, -10:] == -1.0)
    assert np.all(X_val_scaled[:,   :, -10:] == -1.0)
    assert np.all(X_test_scaled[:,  :, -10:] == -1.0)
    assert np.allclose(X_train_scaled, X_train_check)

    print("\n✓ 3D Normalization test passed!")
    print("="*60)