#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing - ENHANCED VERSION (v5.6.2)

Author: Kunal Bhatia
Date: November 2025
Version: 5.6.2 - Production-ready with comprehensive validation
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
        raise ValueError(f"Expected 2D array, got {X.ndim}D with shape {X.shape}")
    
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
    Fits on TRAIN ONLY to prevent data leakage.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (N, C, T)
        Raw data arrays possibly containing pad_value.
    pad_value : float
        Sentinel for padded cells.

    Returns
    -------
    X_train_scaled : np.ndarray
        Normalized training data
    X_val_scaled : np.ndarray
        Normalized validation data
    X_test_scaled : np.ndarray
        Normalized test data
    scaler_standard : StandardScaler
        Fitted StandardScaler (save this!)
    scaler_minmax : MinMaxScaler
        Fitted MinMaxScaler (save this!)
        
    Raises
    ------
    ValueError
        If input dimensions are incompatible or data quality issues detected
    """
    # Validate inputs
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"All inputs must be 3D [N, C, T] arrays. "
            f"Got shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
        )
    
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Channel dimension mismatch: "
            f"train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
        )
    
    if X_train.shape[2] != X_val.shape[2] or X_train.shape[2] != X_test.shape[2]:
        raise ValueError(
            f"Sequence length mismatch: "
            f"train={X_train.shape[2]}, val={X_val.shape[2]}, test={X_test.shape[2]}"
        )
    
    N_train, C, T = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]
    F = C * T  # Total features
    
    print(f"Normalizing 3D data: [{C} channels × {T} timesteps] = {F} features")
    print(f"  Train: {N_train} samples")
    print(f"  Val:   {N_val} samples")
    print(f"  Test:  {N_test} samples")

    # Flatten to [N, F]
    X_train_flat = X_train.reshape(N_train, F).astype(np.float32)
    X_val_flat = X_val.reshape(N_val, F).astype(np.float32)
    X_test_flat = X_test.reshape(N_test, F).astype(np.float32)

    # Check for data quality issues
    train_all_pad = np.all(X_train_flat == pad_value)
    if train_all_pad:
        raise ValueError(
            "Training data contains only padding values! "
            "Check your data generation or loading process."
        )

    # Remember pad masks
    train_mask_pad = (X_train_flat == pad_value)
    val_mask_pad   = (X_val_flat   == pad_value)
    test_mask_pad  = (X_test_flat  == pad_value)

    # Compute per-feature means on TRAIN ignoring pads
    print("Computing per-feature means (ignoring padding)...")
    means_train = _feature_means_ignore_pad(X_train_flat, pad_value)  # [F,]
    
    # Check if we have any valid data
    if np.all(means_train == 0):
        print("⚠️  Warning: All features have zero mean (fully padded or all zeros)")
    
    # Fill pads with train feature means
    X_train_filled = np.where(train_mask_pad, means_train, X_train_flat)
    X_val_filled   = np.where(val_mask_pad,   means_train, X_val_flat)
    X_test_filled  = np.where(test_mask_pad,  means_train, X_test_flat)

    # Stage 1: StandardScaler (fit on TRAIN ONLY)
    print("Stage 1: StandardScaler (zero mean, unit variance)...")
    scaler_standard = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler_standard.fit(X_train_filled)  # learns [F,] mean_ and scale_

    X_train_std = scaler_standard.transform(X_train_filled)
    X_val_std   = scaler_standard.transform(X_val_filled)
    X_test_std  = scaler_standard.transform(X_test_filled)

    # Stage 2: MinMaxScaler (fit on TRAIN ONLY, after standardization)
    print("Stage 2: MinMaxScaler (range [0, 1])...")
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

    # Report final ranges
    train_nonpad = X_train_scaled_flat[~train_mask_pad]
    if train_nonpad.size:
        print(f"✓ Normalization complete")
        print(f"  Final range (non-padded train): [{train_nonpad.min():.3f}, {train_nonpad.max():.3f}]")
        print(f"  Padding preserved at: {pad_value}")
    else:
        print("⚠️  Warning: No non-padded data in training set!")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax


def save_scalers(scaler_standard, scaler_minmax, output_dir):
    """
    Save scalers for later use during evaluation.
    
    Parameters
    ----------
    scaler_standard : StandardScaler
        Fitted StandardScaler
    scaler_minmax : MinMaxScaler
        Fitted MinMaxScaler
    output_dir : str or Path
        Directory to save scalers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    std_path = output_dir / "scaler_standard.pkl"
    mm_path = output_dir / "scaler_minmax.pkl"
    
    try:
        with open(std_path, 'wb') as f:
            pickle.dump(scaler_standard, f)
        with open(mm_path, 'wb') as f:
            pickle.dump(scaler_minmax, f)
        print(f"✓ Scalers saved to {output_dir}/")
    except Exception as e:
        raise IOError(f"Failed to save scalers: {e}")


def load_scalers(model_dir):
    """
    Load saved scalers from training directory with validation.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing saved scalers
        
    Returns
    -------
    scaler_standard : StandardScaler
        Loaded StandardScaler
    scaler_minmax : MinMaxScaler
        Loaded MinMaxScaler
        
    Raises
    ------
    FileNotFoundError
        If scaler files don't exist
    ValueError
        If scalers are not properly fitted
    """
    model_dir = Path(model_dir)
    scaler_std_path = model_dir / "scaler_standard.pkl"
    scaler_mm_path  = model_dir / "scaler_minmax.pkl"
    
    if not scaler_std_path.exists():
        raise FileNotFoundError(
            f"StandardScaler not found: {scaler_std_path}\n"
            f"Expected in experiment directory: {model_dir}\n"
            f"Was training completed successfully?"
        )
    if not scaler_mm_path.exists():
        raise FileNotFoundError(
            f"MinMaxScaler not found: {scaler_mm_path}\n"
            f"Expected in experiment directory: {model_dir}\n"
            f"Was training completed successfully?"
        )
    
    try:
        with open(scaler_std_path, 'rb') as f:
            scaler_standard = pickle.load(f)
        with open(scaler_mm_path, 'rb') as f:
            scaler_minmax = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load scalers: {e}")
    
    # Validate scalers
    if not hasattr(scaler_standard, 'mean_'):
        raise ValueError(
            "StandardScaler is not fitted (missing mean_ attribute). "
            "Training may have been interrupted."
        )
    if not hasattr(scaler_minmax, 'min_'):
        raise ValueError(
            "MinMaxScaler is not fitted (missing min_ attribute). "
            "Training may have been interrupted."
        )
    
    n_features = len(scaler_standard.mean_)
    print(f"✓ Loaded scalers from {model_dir}")
    print(f"  StandardScaler: {n_features} features")
    print(f"  MinMaxScaler: {len(scaler_minmax.min_)} features")
    
    if len(scaler_minmax.min_) != n_features:
        raise ValueError(
            f"Scaler dimension mismatch: "
            f"StandardScaler has {n_features} features, "
            f"MinMaxScaler has {len(scaler_minmax.min_)} features"
        )
    
    return scaler_standard, scaler_minmax


def apply_scalers_to_data(X, scaler_standard, scaler_minmax, pad_value=-1.0):
    """
    Apply pre-fitted scalers to new 3D data [N, C, T].
    Flattens to [N, C*T], applies 1D scalers, and reshapes back.
    
    This is used during evaluation to normalize test data using
    the scalers fitted on training data.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Data to normalize
    scaler_standard : StandardScaler
        Already fitted on training data
    scaler_minmax : MinMaxScaler
        Already fitted on training data
    pad_value : float
        Sentinel value for padding

    Returns
    -------
    X_scaled : np.ndarray, shape (N, C, T)
        Normalized data with padding preserved
        
    Raises
    ------
    ValueError
        If input is None, wrong shape, or incompatible with scalers
    """
    if X is None:
        raise ValueError("Input X is None")
    
    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D array [N, C, T], got {X.ndim}D with shape {X.shape}"
        )

    X = np.asarray(X, dtype=np.float32, order="C")
    N, C, T = X.shape
    F_flat = C * T
    
    # Validate scaler compatibility
    if not hasattr(scaler_standard, "mean_"):
        raise ValueError(
            "StandardScaler is not fitted (mean_ attribute missing). "
            "Did you load the scaler correctly?"
        )
    
    F_scaler = scaler_standard.mean_.shape[0]
    if F_scaler != F_flat:
        raise ValueError(
            f"Feature dimension mismatch!\n"
            f"  Data: [{N}, {C}, {T}] = {F_flat} features\n"
            f"  Scaler was fitted on: {F_scaler} features\n"
            f"  This usually means you're using test data with different dimensions than training.\n"
            f"  Solution: Ensure all datasets use the same n_points and n_channels."
        )

    # Flatten to [N, F]
    X_flat = X.reshape(N, F_flat)
    pad_mask = (X_flat == pad_value)

    # Fill pads with per-feature train means (from fitted scaler)
    means_row = np.broadcast_to(scaler_standard.mean_.astype(X.dtype), X_flat.shape)
    X_filled = np.where(pad_mask, means_row, X_flat)

    # Apply transforms
    X_std = scaler_standard.transform(X_filled)
    X_scaled_flat = scaler_minmax.transform(X_std)

    # Restore pads
    X_scaled_flat[pad_mask] = pad_value

    # Reshape back to [N, C, T]
    X_scaled = X_scaled_flat.reshape(N, C, T)

    # Report range on non-padded entries
    nonpad = X_scaled_flat[~pad_mask]
    if nonpad.size:
        print(f"✓ Applied scalers to {N} samples")
        print(f"  Data range (non-padded): [{nonpad.min():.3f}, {nonpad.max():.3f}]")
        print(f"  Padding preserved at: {pad_value}")
    else:
        print("⚠️  Warning: All data is padded!")

    return X_scaled.astype(np.float32, copy=False)


def load_npz_dataset(path, apply_perm=False, normalize=False):
    """
    Load dataset from .npz file with comprehensive validation.
    ALWAYS returns 3D data: [N, C, T]
    
    Parameters
    ----------
    path : str or Path
        Path to .npz file
    apply_perm : bool
        Whether to apply shuffle permutation if available
    normalize : bool
        Deprecated - always False (use two_stage_normalize instead)
        
    Returns
    -------
    X : np.ndarray, shape (N, C, T)
        Data array (3D)
    y : np.ndarray, shape (N,)
        Labels (0=PSPL, 1=Binary)
    timestamps : np.ndarray
        Time values
    meta : dict
        Metadata dictionary
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file is corrupted or has wrong format
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Expected path: {path.resolve()}\n"
            f"Have you run simulate.py to generate this dataset?"
        )
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loading dataset: {path.name} ({file_size_mb:.1f} MB)")
    
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(
            f"Failed to load NPZ file: {e}\n"
            f"The file may be corrupted. Try regenerating it with simulate.py"
        )

    # Validate required fields
    required_fields = ['X', 'y', 'timestamps', 'meta_json']
    missing = [f for f in required_fields if f not in data.files]
    if missing:
        raise ValueError(
            f"Missing required fields in NPZ: {missing}\n"
            f"Available fields: {data.files}\n"
            f"The dataset may be from an old version or corrupted."
        )

    X = data['X']
    y = data['y']
    timestamps = data['timestamps']

    # Parse metadata
    try:
        meta_json = str(data['meta_json'])
        meta = json.loads(meta_json)
    except Exception as e:
        raise ValueError(f"Failed to parse metadata: {e}")

    # Apply permutation if requested
    if apply_perm and 'perm' in data.files:
        perm = data['perm']
        if len(perm) != len(X):
            raise ValueError(
                f"Permutation length {len(perm)} doesn't match data length {len(X)}"
            )
        X = X[perm]
        y = y[perm]
        print(f"✓ Applied shuffle permutation")

    # Convert labels to numeric if needed
    if y.dtype.kind in ('U', 'S', 'O'):
        print("Converting string labels to numeric...")
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8)
    
    # ALWAYS return 3D data
    original_shape = X.shape
    if X.ndim == 2:
        X = X[:, None, :]  # [N, T] -> [N, 1, T]
        print(f"Converted 2D data {original_shape} to 3D: {X.shape}")
    elif X.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D X array, got {X.ndim}D with shape {X.shape}"
        )
    
    # Check for deprecated normalize flag
    if normalize:
        raise ValueError(
            "normalize=True is deprecated and will cause data leakage!\n"
            "Instead: Load raw data, split into train/val/test, then use two_stage_normalize()"
        )
    
    # Validate dataset quality
    try:
        validate_dataset(X, y, min_samples_per_class=100)
    except ValueError as e:
        print(f"\n⚠️  Warning: Dataset validation issue: {e}")
        print("Continuing anyway, but results may be unreliable.\n")
    
    # Print summary
    pspl_count = np.sum(y == 0)
    binary_count = np.sum(y == 1)
    print(f"✓ Loaded dataset: {X.shape}")
    print(f"  PSPL: {pspl_count} ({pspl_count/len(y)*100:.1f}%)")
    print(f"  Binary: {binary_count} ({binary_count/len(y)*100:.1f}%)")
    print(f"  Timestamps: {len(timestamps)} points")

    return X, y, timestamps, meta


def validate_dataset(X, y, min_samples_per_class=100):
    """
    Validate dataset has sufficient samples and is balanced.
    
    Parameters
    ----------
    X : np.ndarray
        Data array
    y : np.ndarray
        Labels
    min_samples_per_class : int
        Minimum required samples per class
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dataset is empty!")
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) < 2:
        raise ValueError(
            f"Need at least 2 classes for classification, got {len(unique)}"
        )
    
    for cls, count in zip(unique, counts):
        if count < min_samples_per_class:
            raise ValueError(
                f"Class {cls} has only {count} samples, "
                f"need at least {min_samples_per_class} for reliable training"
            )
    
    # Check for data quality issues
    if np.all(np.isnan(X)):
        raise ValueError("Dataset contains only NaN values!")
    
    if np.all(X == -1.0):
        raise ValueError("Dataset contains only padding values!")
    
    # Check class balance
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > 10:
        print(f"⚠️  Warning: High class imbalance (ratio: {imbalance_ratio:.1f})")
        print(f"   Consider rebalancing or using class weights in loss function")
    
    print(f"✓ Dataset validation passed:")
    for cls, count in zip(unique, counts):
        cls_name = "PSPL" if cls == 0 else "Binary"
        percentage = count / len(y) * 100
        print(f"  {cls_name}: {count} samples ({percentage:.1f}%)")


def check_gpu():
    """
    Check GPU availability and print detailed info.
    
    Returns
    -------
    device : torch.device
        Best available device
    num_gpus : int
        Number of GPUs available
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {memory_gb:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        device = torch.device('cuda')
    else:
        print("⚠️  CUDA not available, using CPU")
        print("   Training will be slow. Consider using a GPU.")
        num_gpus = 0
        device = torch.device('cpu')

    return device, num_gpus


if __name__ == "__main__":
    print("="*80)
    print("UTILS MODULE SELF-TEST")
    print("="*80)
    
    print("\n" + "="*80)
    print("GPU Check:")
    print("="*80)
    device, num_gpus = check_gpu()
    
    print("\n" + "="*80)
    print("Testing Scaler Functions (3D):")
    print("="*80)

    # Test with 3D data [N, C, T]
    np.random.seed(42)
    N_train, N_val, N_test, C, T = 1000, 200, 200, 1, 1500
    X_train = np.random.randn(N_train, C, T).astype(np.float32)
    X_val   = np.random.randn(N_val,   C, T).astype(np.float32)
    X_test  = np.random.randn(N_test,  C, T).astype(np.float32)

    # Add some padding (-1) to the last 10 features
    X_train[:, :, -10:] = -1.0
    X_val[:,   :, -10:] = -1.0
    X_test[:,  :, -10:] = -1.0

    print(f"\nTest data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # Normalize
    print("\nRunning two_stage_normalize...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
        X_train, X_val, X_test, pad_value=-1.0
    )
    
    print(f"\nScaled shapes:")
    print(f"  Train: {X_train_scaled.shape}")
    print(f"  Val:   {X_val_scaled.shape}")
    print(f"  Test:  {X_test_scaled.shape}")

    # Test apply_scalers_to_data
    print("\nTesting apply_scalers_to_data...")
    X_train_check = apply_scalers_to_data(X_train.copy(), scaler_std, scaler_mm, pad_value=-1.0)

    # Verify results
    print("\nRunning validation checks...")
    
    # Check shapes
    assert X_train_scaled.shape == X_train.shape, "Shape mismatch after normalization"
    assert X_val_scaled.shape   == X_val.shape, "Val shape mismatch"
    assert X_test_scaled.shape  == X_test.shape, "Test shape mismatch"
    
    # Check padding preserved
    assert np.all(X_train_scaled[:, :, -10:] == -1.0), "Padding not preserved in train"
    assert np.all(X_val_scaled[:,   :, -10:] == -1.0), "Padding not preserved in val"
    assert np.all(X_test_scaled[:,  :, -10:] == -1.0), "Padding not preserved in test"
    
    # Check consistency
    assert np.allclose(X_train_scaled, X_train_check, rtol=1e-5), "Inconsistent normalization"
    
    print("✓ All validation checks passed!")
    
    print("\n" + "="*80)
    print("✅ UTILS MODULE SELF-TEST COMPLETE - ALL TESTS PASSED")
    print("="*80)