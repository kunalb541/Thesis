#!/usr/bin/env python3
"""
Diagnostic script to detect data generation biases and potential cheating.

Checks:
1. Parameter distributions by class (u0, t0, tE for PSPL/Binary)
2. Light curve statistics (mean, std, skewness, kurtosis)
3. Temporal patterns (t0 distribution, correlation with accuracy)
4. Cross-class parameter overlaps

Run: python diagnose_bias.py --data /path/to/test.h5
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def load_data(data_path: str):
    """Load HDF5 data and parameters."""
    with h5py.File(data_path, 'r') as f:
        flux = f['flux'][:]
        labels = f['labels'][:]
        
        # Load parameters for each class
        params = {}
        for class_name in ['flat', 'pspl', 'binary']:
            key = f'params_{class_name}'
            if key in f:
                params[class_name] = f[key][:]
        
        # Get metadata
        metadata = dict(f.attrs)
    
    return flux, labels, params, metadata


def analyze_parameter_distributions(params: dict, labels: np.ndarray):
    """Analyze parameter distributions by class."""
    print("\n" + "="*70)
    print("PARAMETER DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Check if u0 exists in both PSPL and Binary
    for class_name in ['pspl', 'binary']:
        if class_name not in params:
            print(f"  WARNING: {class_name} parameters not found!")
            continue
        
        p = params[class_name]
        fields = p.dtype.names if p.dtype.names else []
        
        print(f"\n{class_name.upper()} parameters (n={len(p)}):")
        print(f"  Available fields: {fields}")
        
        for field in ['u0', 't0', 'tE', 's', 'q', 'rho', 'alpha']:
            if field in fields:
                values = p[field]
                valid = ~np.isnan(values) & ~np.isinf(values)
                if valid.sum() > 0:
                    v = values[valid]
                    print(f"  {field}: min={v.min():.4f}, max={v.max():.4f}, "
                          f"mean={v.mean():.4f}, std={v.std():.4f}")


def check_u0_overlap(params: dict):
    """Check u0 distribution overlap between PSPL and Binary."""
    print("\n" + "="*70)
    print("U0 OVERLAP ANALYSIS (CRITICAL FOR BIAS DETECTION)")
    print("="*70)
    
    if 'pspl' not in params or 'binary' not in params:
        print("  Cannot perform analysis - missing PSPL or Binary parameters")
        return
    
    pspl_u0 = params['pspl']['u0'] if 'u0' in params['pspl'].dtype.names else None
    binary_u0 = params['binary']['u0'] if 'u0' in params['binary'].dtype.names else None
    
    if pspl_u0 is None or binary_u0 is None:
        print("  Cannot perform analysis - u0 not available")
        return
    
    # Compute distributions
    bins = np.linspace(0, 1, 21)
    
    pspl_hist, _ = np.histogram(pspl_u0, bins=bins, density=True)
    binary_hist, _ = np.histogram(binary_u0, bins=bins, density=True)
    
    # KL divergence (measure of distribution difference)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    pspl_hist = pspl_hist + eps
    binary_hist = binary_hist + eps
    pspl_hist /= pspl_hist.sum()
    binary_hist /= binary_hist.sum()
    
    kl_div = np.sum(pspl_hist * np.log(pspl_hist / binary_hist))
    
    print(f"\n  PSPL u0 range: [{pspl_u0.min():.4f}, {pspl_u0.max():.4f}]")
    print(f"  Binary u0 range: [{binary_u0.min():.4f}, {binary_u0.max():.4f}]")
    print(f"\n  KL Divergence (PSPL || Binary): {kl_div:.4f}")
    
    if kl_div > 0.1:
        print("  ⚠️  WARNING: Significant distribution mismatch detected!")
        print("     The model might be learning u0 distribution instead of physics!")
    else:
        print("  ✓ Distributions appear similar (good)")
    
    # Check specific ranges
    print("\n  Events by u0 range:")
    for low, high in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        pspl_count = ((pspl_u0 >= low) & (pspl_u0 < high)).sum()
        binary_count = ((binary_u0 >= low) & (binary_u0 < high)).sum()
        pspl_frac = pspl_count / len(pspl_u0) * 100
        binary_frac = binary_count / len(binary_u0) * 100
        
        bias_indicator = "⚠️" if abs(pspl_frac - binary_frac) > 10 else "✓"
        print(f"    u0 in [{low:.1f}, {high:.1f}): "
              f"PSPL={pspl_count} ({pspl_frac:.1f}%), "
              f"Binary={binary_count} ({binary_frac:.1f}%) {bias_indicator}")
    
    # Create diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.hist(pspl_u0, bins=30, alpha=0.6, label='PSPL', density=True)
    ax.hist(binary_u0, bins=30, alpha=0.6, label='Binary', density=True)
    ax.axvline(0.3, color='red', linestyle='--', label='u0=0.3 threshold')
    ax.set_xlabel('Impact Parameter u0')
    ax.set_ylabel('Density')
    ax.set_title('u0 Distribution by Class')
    ax.legend()
    
    ax = axes[1]
    # Cumulative distribution
    ax.hist(pspl_u0, bins=50, alpha=0.6, label='PSPL', cumulative=True, density=True)
    ax.hist(binary_u0, bins=50, alpha=0.6, label='Binary', cumulative=True, density=True)
    ax.set_xlabel('Impact Parameter u0')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative u0 Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('u0_distribution_check.png', dpi=150)
    plt.close()
    print("\n  Saved: u0_distribution_check.png")


def check_t0_distribution(params: dict, labels: np.ndarray):
    """Check t0 distribution for temporal bias."""
    print("\n" + "="*70)
    print("T0 (PEAK TIME) DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for class_name in ['pspl', 'binary']:
        if class_name not in params:
            continue
        if 't0' not in params[class_name].dtype.names:
            continue
        
        t0 = params[class_name]['t0']
        print(f"\n  {class_name.upper()} t0: "
              f"min={t0.min():.1f}, max={t0.max():.1f}, "
              f"mean={t0.mean():.1f}, std={t0.std():.1f}")


def check_light_curve_statistics(flux: np.ndarray, labels: np.ndarray):
    """Check if light curve statistics differ between classes in suspicious ways."""
    print("\n" + "="*70)
    print("LIGHT CURVE STATISTICS BY CLASS")
    print("="*70)
    
    class_names = ['Flat', 'PSPL', 'Binary']
    
    for label, name in enumerate(class_names):
        mask = labels == label
        class_flux = flux[mask]
        
        # Compute statistics on valid (non-zero) observations
        valid_flux = class_flux[class_flux > 0]
        
        print(f"\n  {name} (n={mask.sum()}):")
        print(f"    Valid observations: {len(valid_flux)}")
        print(f"    Mean: {valid_flux.mean():.4f}")
        print(f"    Std: {valid_flux.std():.4f}")
        print(f"    Skewness: {stats.skew(valid_flux):.4f}")
        print(f"    Kurtosis: {stats.kurtosis(valid_flux):.4f}")
        
        # Check for baseline offset
        # Flat events should be centered at 1.0 (baseline magnification)
        if label == 0:
            baseline_deviation = abs(valid_flux.mean() - 1.0)
            if baseline_deviation > 0.1:
                print(f"    ⚠️  Flat baseline deviates from 1.0 by {baseline_deviation:.3f}")


def check_sequence_length_bias(flux: np.ndarray, labels: np.ndarray):
    """Check if sequence lengths differ by class."""
    print("\n" + "="*70)
    print("SEQUENCE LENGTH ANALYSIS")
    print("="*70)
    
    class_names = ['Flat', 'PSPL', 'Binary']
    
    for label, name in enumerate(class_names):
        mask = labels == label
        class_flux = flux[mask]
        
        # Count valid observations per event
        valid_counts = (class_flux > 0).sum(axis=1)
        
        print(f"\n  {name}: valid obs mean={valid_counts.mean():.1f}, "
              f"std={valid_counts.std():.1f}, "
              f"min={valid_counts.min()}, max={valid_counts.max()}")
    
    # Statistical test
    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            if j <= i:
                continue
            
            mask_i = labels == i
            mask_j = labels == j
            
            counts_i = (flux[mask_i] > 0).sum(axis=1)
            counts_j = (flux[mask_j] > 0).sum(axis=1)
            
            stat, pval = stats.mannwhitneyu(counts_i, counts_j)
            
            if pval < 0.01:
                print(f"\n  ⚠️  {name_i} vs {name_j}: Sequence lengths significantly different (p={pval:.2e})")


def main():
    parser = argparse.ArgumentParser(description="Diagnose data generation biases")
    parser.add_argument('--data', required=True, help="Path to HDF5 test data")
    args = parser.parse_args()
    
    print("="*70)
    print("MICROLENSING DATA BIAS DIAGNOSTIC TOOL")
    print("="*70)
    print(f"Data: {args.data}")
    
    # Load data
    flux, labels, params, metadata = load_data(args.data)
    
    print(f"\nDataset size: {len(labels)}")
    print(f"Class distribution: Flat={sum(labels==0)}, PSPL={sum(labels==1)}, Binary={sum(labels==2)}")
    
    if metadata:
        print(f"\nMetadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
    
    # Run analyses
    analyze_parameter_distributions(params, labels)
    check_u0_overlap(params)
    check_t0_distribution(params, labels)
    check_light_curve_statistics(flux, labels)
    check_sequence_length_bias(flux, labels)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
