#!/usr/bin/env python3
"""
Comprehensive pre-training verification
"""
import numpy as np
from pathlib import Path

def verify_dataset(path):
    """Verify a dataset is ready for training"""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {path}")
    print(f"{'='*80}")
    
    if not Path(path).exists():
        print(f"❌ File not found!")
        return False
    
    # Load dataset
    data = np.load(path, allow_pickle=False)
    X = data['X']
    y = data['y']
    
    # Check shapes
    print(f"\n📊 Data Shapes:")
    print(f"   X: {X.shape}")
    print(f"   y: {y.shape}")
    
    # Check class balance
    n_pspl = (y == 0).sum()
    n_binary = (y == 1).sum()
    print(f"\n⚖️  Class Balance:")
    print(f"   PSPL:   {n_pspl:,} ({100*n_pspl/len(y):.1f}%)")
    print(f"   Binary: {n_binary:,} ({100*n_binary/len(y):.1f}%)")
    
    # Check shuffling (first vs last chunk)
    chunk_size = min(100000, len(y) // 10)
    first_binary_pct = 100 * (y[:chunk_size] == 1).mean()
    last_binary_pct = 100 * (y[-chunk_size:] == 1).mean()
    
    print(f"\n🔀 Shuffling Check:")
    print(f"   First {chunk_size:,}: {first_binary_pct:.1f}% binary")
    print(f"   Last {chunk_size:,}:  {last_binary_pct:.1f}% binary")
    
    if abs(first_binary_pct - last_binary_pct) < 5:
        print(f"   ✅ Well shuffled")
        shuffled = True
    else:
        print(f"   ❌ NOT well shuffled!")
        shuffled = False
    
    # Check masking
    pad_count = (X == -1).sum()
    pad_pct = 100 * pad_count / X.size
    
    print(f"\n🎭 Masking Check:")
    print(f"   PAD_VALUE (-1) count: {pad_count:,}")
    print(f"   Masking percentage: {pad_pct:.2f}%")
    
    if 15 <= pad_pct <= 25:  # Expect ~20%
        print(f"   ✅ Masking correct (~20% expected)")
        masked = True
    elif pad_pct < 1:
        print(f"   ❌ NO masking applied!")
        masked = False
    else:
        print(f"   ⚠️  Unusual masking percentage")
        masked = False
    
    # Check standardization (sample)
    sample_size = min(1000, len(X))
    sample_means = []
    sample_stds = []
    
    for i in np.random.choice(len(X), sample_size, replace=False):
        valid = (X[i] != -1) & (X[i] != 0)
        if valid.sum() > 10:
            sample_means.append(X[i][valid].mean())
            sample_stds.append(X[i][valid].std())
    
    mean_of_means = np.mean(sample_means)
    mean_of_stds = np.mean(sample_stds)
    
    print(f"\n📏 Standardization Check:")
    print(f"   Mean (sample avg): {mean_of_means:.4f}")
    print(f"   Std (sample avg):  {mean_of_stds:.4f}")
    
    if abs(mean_of_means) < 0.1 and abs(mean_of_stds - 1.0) < 0.2:
        print(f"   ✅ Standardized (mean≈0, std≈1)")
        standardized = True
    else:
        print(f"   ⚠️  May not be standardized")
        standardized = False
    
    # Check for NaN/Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    print(f"\n🔍 Data Quality:")
    print(f"   NaN values: {nan_count}")
    print(f"   Inf values: {inf_count}")
    
    quality_ok = (nan_count == 0) and (inf_count == 0)
    if quality_ok:
        print(f"   ✅ No invalid values")
    else:
        print(f"   ❌ Contains invalid values!")
    
    # Overall verdict
    all_ok = shuffled and masked and standardized and quality_ok
    
    print(f"\n{'='*80}")
    if all_ok:
        print(f"✅ DATASET READY FOR TRAINING!")
    else:
        print(f"❌ DATASET HAS ISSUES - FIX BEFORE TRAINING!")
    print(f"{'='*80}")
    
    return all_ok

# Verify both datasets
distinct_ok = verify_dataset('../data/raw/distinct_100k_final.npz')
baseline_ok = verify_dataset('../data/raw/baseline_1M_final.npz')

print(f"\n{'='*80}")
print(f"FINAL VERDICT")
print(f"{'='*80}")
print(f"Distinct dataset: {'✅ READY' if distinct_ok else '❌ NOT READY'}")
print(f"Baseline dataset: {'✅ READY' if baseline_ok else '❌ NOT READY'}")

if distinct_ok and baseline_ok:
    print(f"\n🚀 ALL SYSTEMS GO! Ready to request GPU access!")
else:
    print(f"\n⚠️  Fix issues before training!")

print(f"{'='*80}")
