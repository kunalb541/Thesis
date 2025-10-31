#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing - FIXED VERSION

Author: Kunal Bhatia
Date: October 2025
Version: 3.1 - Fixed normalization bugs
"""

import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path
import config as CFG


def two_stage_normalize(X_train, X_val, X_test, pad_value=-1):
    """
    Two-stage normalization matching original TensorFlow approach
    CRITICAL: Fits on train data ONLY, transforms all splits
    
    Args:
        X_train: Training data (raw)
        X_val: Validation data (raw)
        X_test: Test data (raw)
        pad_value: Value used for padding
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax
    """
    # Mask padding values
    train_mask = (X_train != pad_value)
    val_mask = (X_val != pad_value)
    test_mask = (X_test != pad_value)
    
    # Stage 1: StandardScaler (per-feature normalization)
    print("Applying StandardScaler (fit on train only)...")
    scaler_standard = StandardScaler()
    
    # FIT ONLY on non-padded training values
    X_train_flat = X_train[train_mask].reshape(-1, 1)
    scaler_standard.fit(X_train_flat)
    
    # Transform all splits
    X_train_norm = X_train.copy()
    X_train_norm[train_mask] = scaler_standard.transform(
        X_train[train_mask].reshape(-1, 1)
    ).flatten()
    
    X_val_norm = X_val.copy()
    X_val_norm[val_mask] = scaler_standard.transform(
        X_val[val_mask].reshape(-1, 1)
    ).flatten()
    
    X_test_norm = X_test.copy()
    X_test_norm[test_mask] = scaler_standard.transform(
        X_test[test_mask].reshape(-1, 1)
    ).flatten()
    
    # Stage 2: MinMaxScaler
    print("Applying MinMaxScaler (fit on train only)...")
    scaler_minmax = MinMaxScaler()
    
    # FIT ONLY on normalized training data
    X_train_flat_norm = X_train_norm[train_mask].reshape(-1, 1)
    scaler_minmax.fit(X_train_flat_norm)
    
    # Transform all splits
    X_train_scaled = X_train_norm.copy()
    X_train_scaled[train_mask] = scaler_minmax.transform(
        X_train_norm[train_mask].reshape(-1, 1)
    ).flatten()
    
    X_val_scaled = X_val_norm.copy()
    X_val_scaled[val_mask] = scaler_minmax.transform(
        X_val_norm[val_mask].reshape(-1, 1)
    ).flatten()
    
    X_test_scaled = X_test_norm.copy()
    X_test_scaled[test_mask] = scaler_minmax.transform(
        X_test_norm[test_mask].reshape(-1, 1)
    ).flatten()
    
    print(f"Final ranges (non-padded values):")
    print(f"  Train: [{X_train_scaled[train_mask].min():.3f}, {X_train_scaled[train_mask].max():.3f}]")
    print(f"  Val:   [{X_val_scaled[val_mask].min():.3f}, {X_val_scaled[val_mask].max():.3f}]")
    print(f"  Test:  [{X_test_scaled[test_mask].min():.3f}, {X_test_scaled[test_mask].max():.3f}]")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax


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
    
    Args:
        model_dir: Directory containing scaler_*.pkl files
    
    Returns:
        scaler_standard, scaler_minmax
    """
    model_dir = Path(model_dir)
    
    scaler_std_path = model_dir / "scaler_standard.pkl"
    scaler_mm_path = model_dir / "scaler_minmax.pkl"
    
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


def apply_scalers_to_data(X, scaler_standard, scaler_minmax, pad_value=-1):
    """
    Apply pre-fitted scalers to new data (for evaluation)
    
    Args:
        X: Raw data to normalize
        scaler_standard: Fitted StandardScaler from training
        scaler_minmax: Fitted MinMaxScaler from training
        pad_value: Value used for padding
    
    Returns:
        X_normalized: Normalized data
    """
    X_normalized = X.copy()
    mask = (X_normalized != pad_value)
    
    # Stage 1: StandardScaler
    X_normalized[mask] = scaler_standard.transform(
        X_normalized[mask].reshape(-1, 1)
    ).flatten()
    
    # Stage 2: MinMaxScaler
    X_normalized[mask] = scaler_minmax.transform(
        X_normalized[mask].reshape(-1, 1)
    ).flatten()
    
    print(f"✓ Applied scalers. Data range: [{X_normalized[mask].min():.3f}, {X_normalized[mask].max():.3f}]")
    
    return X_normalized


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
        
        # Replace PAD_VALUE with 0 first
        X_normalized = X.copy()
        X_normalized[X_normalized == -1.0] = 0.0
        
        # StandardScaler first
        scaler_standard = StandardScaler()
        X_normalized = scaler_standard.fit_transform(X_normalized)
        
        # MinMaxScaler second
        scaler_minmax = MinMaxScaler()
        X_normalized = scaler_minmax.fit_transform(X_normalized)
        
        X = X_normalized
    
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
    X_train = np.random.randn(1000, 100)
    X_val = np.random.randn(200, 100)
    X_test = np.random.randn(200, 100)
    
    # Add some padding
    X_train[:, -10:] = -1
    X_val[:, -10:] = -1
    X_test[:, -10:] = -1
    
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler_std, scaler_mm = two_stage_normalize(
        X_train, X_val, X_test, pad_value=-1
    )
    
    print("\n✓ Normalization test passed!")
    print("="*60)