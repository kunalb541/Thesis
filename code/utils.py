#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing

Author: Kunal Bhatia
Date: October 2025
"""

import json
import numpy as np
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import config as CFG
#!/usr/bin/env python3
"""
Two-stage normalization matching original TensorFlow approach
"""


def two_stage_normalize(X_train, X_val, X_test, pad_value=-1):
    """
    Reproduce exact normalization from original TensorFlow notebook
    
    Returns:
        Normalized arrays and fitted scalers
    """
    # Mask padding values
    train_mask = (X_train != pad_value)
    val_mask = (X_val != pad_value)
    test_mask = (X_test != pad_value)
    
    # Stage 1: StandardScaler (per-feature normalization)
    print("Applying StandardScaler...")
    scaler_standard = StandardScaler()
    
    # Fit only on non-padded values
    X_train_flat = X_train[train_mask].reshape(-1, 1)
    scaler_standard.fit(X_train_flat)
    
    # Transform
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
    print("Applying MinMaxScaler...")
    scaler_minmax = MinMaxScaler()
    
    # Fit on normalized train data
    X_train_flat_norm = X_train_norm[train_mask].reshape(-1, 1)
    scaler_minmax.fit(X_train_flat_norm)
    
    # Transform
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
    
    print(f"Final ranges: Train [{X_train_scaled[train_mask].min():.3f}, {X_train_scaled[train_mask].max():.3f}]")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax

def save_scalers(scaler_standard, scaler_minmax, output_dir):
    """Save scalers for later use"""
    with open(f"{output_dir}/scaler_standard.pkl", 'wb') as f:
        pickle.dump(scaler_standard, f)
    with open(f"{output_dir}/scaler_minmax.pkl", 'wb') as f:
        pickle.dump(scaler_minmax, f)


def load_npz_dataset(path, apply_perm=False, normalize=True):
    """Load dataset with proper normalization"""
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
    
    # 🔥 ADD THIS: Double normalization like TensorFlow
    if normalize:
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
    print("GPU Check:")
    print("="*60)
    device, num_gpus = check_gpu()
    print("="*60)
