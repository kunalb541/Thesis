#!/usr/bin/env python3
"""
Verify that cadence masking is actually applied
"""
import numpy as np
import sys

def check_masking(npz_file):
    print("="*80)
    print("MASKING VERIFICATION")
    print("="*80)
    
    data = np.load(npz_file, allow_pickle=False)
    X = data['X']
    
    # Count PAD_VALUE (-1.0) entries
    pad_count = (X == -1.0).sum()
    total_points = X.size
    pad_percentage = 100 * pad_count / total_points
    
    print(f"\nDataset: {npz_file}")
    print(f"Shape: {X.shape}")
    print(f"Total data points: {total_points:,}")
    print(f"PAD_VALUE (-1) count: {pad_count:,}")
    print(f"PAD percentage: {pad_percentage:.2f}%")
    
    # Check a few random light curves
    print(f"\nSample light curves:")
    for i in np.random.choice(len(X), min(10, len(X)), replace=False):
        lc = X[i]
        n_pad = (lc == -1.0).sum()
        n_points = len(lc)
        pct = 100 * n_pad / n_points
        print(f"  Event {i:6d}: {n_pad:4d}/{n_points} padded ({pct:5.1f}%)")
    
    # Expected vs actual
    if 'meta_json' in data.files:
        import json
        meta = json.loads(data['meta_json'].item())
        expected_prob = meta.get('cadence_mask_prob', 0.0)
        expected_pct = expected_prob * 100
        
        print(f"\n{'='*80}")
        print(f"Expected masking: {expected_pct:.1f}%")
        print(f"Actual masking:   {pad_percentage:.2f}%")
        
        if abs(pad_percentage - expected_pct) < 2.0:
            print(f"✅ Masking is working correctly!")
            return True
        elif pad_percentage < 1.0:
            print(f"❌ MASKING NOT WORKING! (< 1% padded)")
            return False
        else:
            print(f"⚠️  Masking percentage differs by {abs(pad_percentage - expected_pct):.1f}%")
            return False
    else:
        print(f"\n⚠️  No metadata found")
        if pad_percentage > 5:
            print(f"✅ Some masking detected ({pad_percentage:.1f}%)")
            return True
        else:
            print(f"❌ No significant masking detected")
            return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
    else:
        npz_file = '../data/raw/baseline_1M_raw.npz'
    
    success = check_masking(npz_file)
    sys.exit(0 if success else 1)
