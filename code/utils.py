#!/usr/bin/env python3
"""
Utility functions for data loading and preprocessing

Author: Kunal Bhatia
Date: October 2025
"""

import json
import numpy as np
import torch

import config as CFG



from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(X_train, X_val, X_test):
    """Two-stage normalization like original TensorFlow code"""
    # Stage 1: StandardScaler
    scaler_standard = StandardScaler()
    X_train_norm = scaler_standard.fit_transform(X_train)
    X_val_norm = scaler_standard.transform(X_val)
    X_test_norm = scaler_standard.transform(X_test)
    
    # Stage 2: MinMaxScaler
    scaler_minmax = MinMaxScaler()
    X_train_scaled = scaler_minmax.fit_transform(X_train_norm)
    X_val_scaled = scaler_minmax.transform(X_val_norm)
    X_test_scaled = scaler_minmax.transform(X_test_norm)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_standard, scaler_minmax

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
