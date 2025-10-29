#!/usr/bin/env python3
"""
Reshuffle baseline dataset to ensure proper class mixing
"""

import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    print("="*80)
    print("RESHUFFLING BASELINE DATASET")
    print("="*80)
    
    # Load
    data = np.load(args.input, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"\nBefore shuffle:")
    print(f"  Total: {len(X):,}")
    print(f"  PSPL: {(y==0).sum():,}")
    print(f"  Binary: {(y==1).sum():,}")
    
    print(f"\n  First 100k: {(y[:100000]==0).sum():,} PSPL, {(y[:100000]==1).sum():,} Binary")
    print(f"  Last 100k:  {(y[-100000:]==0).sum():,} PSPL, {(y[-100000:]==1).sum():,} Binary")
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(X))
    
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]
    
    print(f"\nAfter shuffle:")
    print(f"  First 100k: {(y_shuffled[:100000]==0).sum():,} PSPL, {(y_shuffled[:100000]==1).sum():,} Binary")
    print(f"  Last 100k:  {(y_shuffled[-100000:]==0).sum():,} PSPL, {(y_shuffled[-100000:]==1).sum():,} Binary")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'X': X_shuffled,
        'y': y_shuffled,
    }
    
    if 'timestamps' in data.files:
        save_dict['timestamps'] = data['timestamps']
    
    if 'params_binary_json' in data.files:
        save_dict['params_binary_json'] = data['params_binary_json']
    
    if 'meta_json' in data.files:
        save_dict['meta_json'] = data['meta_json']
    
    np.savez(output_path, **save_dict)
    
    print(f"\n✓ Saved shuffled data to: {output_path}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
