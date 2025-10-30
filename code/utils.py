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


def load_npz_dataset(path, apply_perm=False):
    """
    Load dataset from NPZ file with optional permutation
    
    Args:
        path: Path to .npz file
        apply_perm: If True, apply saved permutation for shuffling
        
    Returns:
        X: Light curves [N, L]
        y: Labels [N] (0=PSPL, 1=Binary)
        timestamps: Time array [L]
        meta: Metadata dictionary
    """
    data = np.load(path, allow_pickle=False)
    
    X = data['X']
    y = data['y']
    timestamps = data['timestamps']
    
    # Parse metadata JSON
    meta_json = str(data['meta_json'])
    meta = json.loads(meta_json)
    
    # Apply permutation if requested and available
    if apply_perm and 'perm' in data.files:
        perm = data['perm']
        X = X[perm]
        y = y[perm]
    
    # Handle label encoding (in case labels are strings)
    if y.dtype.kind in ('U', 'S', 'O'):
        # Convert string labels to integers
        y = np.array([0 if 'PSPL' in str(v).upper() else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8)
    
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