#!/usr/bin/env python3
"""
Standardize existing dataset to remove statistical shortcuts
"""

import numpy as np
import argparse
from pathlib import Path
import json


def standardize_light_curve(flux):
    """
    Standardize to mean=0, std=1
    This removes ALL statistical differences between classes
    """
    valid = flux > 0
    if valid.sum() < 10:
        return flux
    
    flux_valid = flux[valid]
    
    mean = flux_valid.mean()
    std = flux_valid.std()
    
    if std > 1e-6:
        flux[valid] = (flux_valid - mean) / std
    else:
        flux[valid] = 0.0
    
    return flux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .npz file')
    parser.add_argument('--output', required=True, help='Output .npz file')
    args = parser.parse_args()
    
    print("="*80)
    print("STANDARDIZING DATASET")
    print("="*80)
    
    # Load
    data = np.load(args.input, allow_pickle=True)
    X = data['X'].copy()
    y = data['y']
    
    print(f"\nLoaded: {args.input}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {(y==0).sum():,} PSPL, {(y==1).sum():,} Binary")
    
    # Check before
    binary_X = X[y==1]
    pspl_X = X[y==0]
    
    binary_means_before = []
    pspl_means_before = []
    
    for lc in binary_X[:1000]:
        valid = lc > 0
        if valid.sum() > 0:
            binary_means_before.append(lc[valid].mean())
    
    for lc in pspl_X[:1000]:
        valid = lc > 0
        if valid.sum() > 0:
            pspl_means_before.append(lc[valid].mean())
    
    print(f"\nBEFORE standardization:")
    print(f"  Binary mean: {np.mean(binary_means_before):.4f}")
    print(f"  PSPL mean:   {np.mean(pspl_means_before):.4f}")
    print(f"  Difference:  {abs(np.mean(binary_means_before) - np.mean(pspl_means_before)):.4f}")
    
    # Standardize each light curve
    print(f"\nStandardizing {len(X):,} light curves...")
    
    for i in range(len(X)):
        X[i] = standardize_light_curve(X[i])
        
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i+1:,} / {len(X):,}")
    
    # Check after
    binary_X = X[y==1]
    pspl_X = X[y==0]
    
    binary_means_after = []
    pspl_means_after = []
    binary_stds_after = []
    pspl_stds_after = []
    
    for lc in binary_X[:1000]:
        valid = lc != 0  # Now valid means non-zero (standardized)
        if valid.sum() > 0:
            binary_means_after.append(lc[valid].mean())
            binary_stds_after.append(lc[valid].std())
    
    for lc in pspl_X[:1000]:
        valid = lc != 0
        if valid.sum() > 0:
            pspl_means_after.append(lc[valid].mean())
            pspl_stds_after.append(lc[valid].std())
    
    print(f"\nAFTER standardization:")
    print(f"  Binary: mean={np.mean(binary_means_after):.4f}, std={np.mean(binary_stds_after):.4f}")
    print(f"  PSPL:   mean={np.mean(pspl_means_after):.4f}, std={np.mean(pspl_stds_after):.4f}")
    print(f"  Mean difference:  {abs(np.mean(binary_means_after) - np.mean(pspl_means_after)):.4f}")
    print(f"  Std difference:   {abs(np.mean(binary_stds_after) - np.mean(pspl_stds_after)):.4f}")
    
    if abs(np.mean(binary_means_after) - np.mean(pspl_means_after)) < 0.1:
        print("\n✅ Standardization successful!")
    else:
        print("\n⚠️  Still some difference remaining")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Preserve metadata
    save_dict = {
        'X': X.astype(np.float32),
        'y': y,
        'timestamps': data['timestamps']
    }
    
    if 'params_binary_json' in data.files:
        save_dict['params_binary_json'] = data['params_binary_json']
    
    if 'meta_json' in data.files:
        meta = json.loads(data['meta_json'].item())
        meta['standardized'] = True
        save_dict['meta_json'] = json.dumps(meta)
    
    np.savez(output_path, **save_dict)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ STANDARDIZATION COMPLETE")
    print("="*80)
    print("\nNow the model MUST learn shapes, not statistics!")


if __name__ == "__main__":
    main()
