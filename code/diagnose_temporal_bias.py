#!/usr/bin/env python3
"""
Temporal Bias Diagnostic Tool
==============================

Checks for temporal bias in microlensing datasets:
1. t0 distribution by class
2. Peak position distribution
3. Temporal features correlation with labels
4. Visual inspection of time-dependent patterns

Author: Kunal Bhatia
Version: 14.1
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_dataset(data_path):
    """Load dataset and extract parameters"""
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    # Load timestamps
    if 'timestamps' in data:
        timestamps = data['timestamps']
    else:
        timestamps = np.linspace(-120, 120, X.shape[1])
    
    # Load parameters
    params = {'flat': [], 'pspl': [], 'binary': []}
    
    if 'params_flat_json' in data:
        params['flat'] = json.loads(str(data['params_flat_json']))
    if 'params_pspl_json' in data:
        params['pspl'] = json.loads(str(data['params_pspl_json']))
    if 'params_binary_json' in data:
        params['binary'] = json.loads(str(data['params_binary_json']))
    
    return X, y, timestamps, params


def extract_peak_times(X, y, timestamps, pad_value=-1.0):
    """Extract empirical peak times from light curves"""
    peak_times = {0: [], 1: [], 2: []}
    
    for i in range(len(X)):
        flux = X[i]
        valid_mask = flux != pad_value
        
        if valid_mask.sum() > 10:
            valid_flux = flux[valid_mask]
            valid_times = timestamps[valid_mask]
            
            # Find peak (maximum flux)
            peak_idx = valid_flux.argmax()
            peak_time = valid_times[peak_idx]
            
            peak_times[int(y[i])].append(peak_time)
    
    return peak_times


def compute_temporal_features(X, timestamps, pad_value=-1.0):
    """Compute temporal features that model might learn"""
    features = {
        'first_observation_time': [],
        'last_observation_time': [],
        'peak_time': [],
        'rising_duration': [],
        'falling_duration': [],
        'asymmetry': []
    }
    
    for i in range(len(X)):
        flux = X[i]
        valid_mask = flux != pad_value
        
        if valid_mask.sum() > 10:
            valid_flux = flux[valid_mask]
            valid_times = timestamps[valid_mask]
            
            # First and last observation
            features['first_observation_time'].append(valid_times[0])
            features['last_observation_time'].append(valid_times[-1])
            
            # Peak time
            peak_idx = valid_flux.argmax()
            peak_time = valid_times[peak_idx]
            features['peak_time'].append(peak_time)
            
            # Rising/falling durations
            rise_time = peak_time - valid_times[0]
            fall_time = valid_times[-1] - peak_time
            features['rising_duration'].append(rise_time)
            features['falling_duration'].append(fall_time)
            features['asymmetry'].append((rise_time - fall_time) / (rise_time + fall_time + 1e-10))
        else:
            features['first_observation_time'].append(np.nan)
            features['last_observation_time'].append(np.nan)
            features['peak_time'].append(np.nan)
            features['rising_duration'].append(np.nan)
            features['falling_duration'].append(np.nan)
            features['asymmetry'].append(np.nan)
    
    return {k: np.array(v) for k, v in features.items()}


def diagnose_temporal_bias(data_path, output_dir='temporal_bias_diagnosis'):
    """Complete temporal bias diagnosis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TEMPORAL BIAS DIAGNOSIS")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    X, y, timestamps, params = load_dataset(data_path)
    print(f"  Events: {len(X)}")
    print(f"  Flat:   {(y==0).sum()}")
    print(f"  PSPL:   {(y==1).sum()}")
    print(f"  Binary: {(y==2).sum()}")
    
    # Check 1: Parameter t0 distributions
    print("\n" + "="*70)
    print("CHECK 1: Parameter t0 Distributions")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PSPL t0
    if params['pspl']:
        t0_pspl = [p['t_0'] for p in params['pspl']]
        axes[0].hist(t0_pspl, bins=30, alpha=0.7, color='darkred', edgecolor='black')
        axes[0].axvline(np.mean(t0_pspl), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(t0_pspl):.1f}')
        axes[0].set_xlabel('t₀ (days)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('PSPL t₀ Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        print(f"\nPSPL t0 statistics:")
        print(f"  Mean: {np.mean(t0_pspl):.2f}")
        print(f"  Std:  {np.std(t0_pspl):.2f}")
        print(f"  Min:  {np.min(t0_pspl):.2f}")
        print(f"  Max:  {np.max(t0_pspl):.2f}")
    
    # Binary t0
    if params['binary']:
        t0_binary = [p['t_0'] for p in params['binary']]
        axes[1].hist(t0_binary, bins=30, alpha=0.7, color='darkblue', edgecolor='black')
        axes[1].axvline(np.mean(t0_binary), color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(t0_binary):.1f}')
        axes[1].set_xlabel('t₀ (days)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Binary t₀ Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        print(f"\nBinary t0 statistics:")
        print(f"  Mean: {np.mean(t0_binary):.2f}")
        print(f"  Std:  {np.std(t0_binary):.2f}")
        print(f"  Min:  {np.min(t0_binary):.2f}")
        print(f"  Max:  {np.max(t0_binary):.2f}")
        
        # Statistical test
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(t0_pspl, t0_binary)
        print(f"\nKolmogorov-Smirnov test (PSPL vs Binary t0):")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {pval:.4f}")
        if pval < 0.05:
            print(f"  ⚠️  SIGNIFICANT DIFFERENCE (p < 0.05)")
        else:
            print(f"  ✅ No significant difference (p >= 0.05)")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'param_t0_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: param_t0_distributions.png")
    plt.close()
    
    # Check 2: Empirical peak times from light curves
    print("\n" + "="*70)
    print("CHECK 2: Empirical Peak Times (from light curves)")
    print("="*70)
    
    peak_times = extract_peak_times(X, y, timestamps)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['gray', 'darkred', 'darkblue']
    labels = ['Flat', 'PSPL', 'Binary']
    
    for class_idx, (color, label) in enumerate(zip(colors, labels)):
        if len(peak_times[class_idx]) > 0:
            ax.hist(peak_times[class_idx], bins=40, alpha=0.6, color=color, 
                   label=label, edgecolor='black')
            
            mean_peak = np.mean(peak_times[class_idx])
            ax.axvline(mean_peak, color=color, linestyle='--', linewidth=2)
            
            print(f"\n{label} empirical peak times:")
            print(f"  Mean: {mean_peak:.2f}")
            print(f"  Std:  {np.std(peak_times[class_idx]):.2f}")
            print(f"  Min:  {np.min(peak_times[class_idx]):.2f}")
            print(f"  Max:  {np.max(peak_times[class_idx]):.2f}")
    
    ax.set_xlabel('Peak Time (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Empirical Peak Time Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'empirical_peak_times.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: empirical_peak_times.png")
    plt.close()
    
    # Check for separation
    if len(peak_times[1]) > 0 and len(peak_times[2]) > 0:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(peak_times[1], peak_times[2])
        print(f"\nKolmogorov-Smirnov test (PSPL vs Binary peaks):")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {pval:.4f}")
        if pval < 0.05:
            print(f"  ⚠️  SIGNIFICANT DIFFERENCE - MODEL CAN CHEAT!")
        else:
            print(f"  ✅ No significant difference")
    
    # Check 3: Temporal features correlation
    print("\n" + "="*70)
    print("CHECK 3: Temporal Features vs. Class Labels")
    print("="*70)
    
    temporal_features = compute_temporal_features(X, timestamps)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    feature_names = list(temporal_features.keys())
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]
        feature_vals = temporal_features[feature_name]
        
        # Remove NaNs
        valid_mask = ~np.isnan(feature_vals)
        feature_vals_clean = feature_vals[valid_mask]
        y_clean = y[valid_mask]
        
        if len(feature_vals_clean) > 0:
            # Plot by class
            for class_idx, (color, label) in enumerate(zip(colors, labels)):
                class_mask = y_clean == class_idx
                if class_mask.sum() > 0:
                    ax.hist(feature_vals_clean[class_mask], bins=30, alpha=0.5, 
                           color=color, label=label, edgecolor='black', density=True)
            
            ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Compute correlation with labels
            from scipy.stats import spearmanr
            corr, pval = spearmanr(feature_vals_clean, y_clean)
            ax.set_title(f'Corr: {corr:.3f} (p={pval:.2e})', fontsize=9)
            
            print(f"\n{feature_name}:")
            print(f"  Correlation with label: {corr:.4f} (p={pval:.2e})")
            if abs(corr) > 0.1 and pval < 0.05:
                print(f"  ⚠️  SIGNIFICANT CORRELATION - POTENTIAL BIAS!")
    
    plt.suptitle('Temporal Features by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_features.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: temporal_features.png")
    plt.close()
    
    # Check 4: Visual inspection - show events from different temporal regions
    print("\n" + "="*70)
    print("CHECK 4: Visual Inspection Across Temporal Regions")
    print("="*70)
    
    # Divide time into 3 regions
    if len(peak_times[1]) > 0:  # PSPL
        pspl_peaks = np.array(peak_times[1])
        pspl_indices = np.where(y == 1)[0]
        
        # Find events in different temporal regions
        early_region = pspl_peaks < -30
        mid_region = (pspl_peaks >= -30) & (pspl_peaks <= 10)
        late_region = pspl_peaks > 10
        
        print(f"\nPSPL events by temporal region:")
        print(f"  Early (t < -30): {early_region.sum()} ({early_region.mean()*100:.1f}%)")
        print(f"  Middle (-30 ≤ t ≤ 10): {mid_region.sum()} ({mid_region.mean()*100:.1f}%)")
        print(f"  Late (t > 10): {late_region.sum()} ({late_region.mean()*100:.1f}%)")
        
        # Plot examples from each region
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        
        regions = [
            ('Early (peak < -30 days)', early_region),
            ('Middle (-30 ≤ peak ≤ 10 days)', mid_region),
            ('Late (peak > 10 days)', late_region)
        ]
        
        for row, (title, mask) in enumerate(regions):
            if mask.sum() > 0:
                # Select up to 3 examples
                region_indices = pspl_indices[mask]
                selected = region_indices[:3]
                
                for col, idx in enumerate(selected):
                    ax = axes[row, col]
                    
                    flux = X[idx]
                    valid_mask = flux != -1.0
                    times = timestamps[valid_mask]
                    fluxes = flux[valid_mask]
                    
                    # Convert to magnitudes
                    baseline = 20.0
                    mags = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
                    
                    ax.scatter(times, mags, c='darkred', s=10, alpha=0.7, edgecolors='black', linewidth=0.3)
                    ax.invert_yaxis()
                    
                    # Mark peak
                    peak_idx = fluxes.argmax()
                    peak_t = times[peak_idx]
                    peak_m = mags[peak_idx]
                    ax.axvline(peak_t, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                    
                    if row == 0:
                        ax.set_title(f'Example {col+1}', fontsize=10, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel('Magnitude', fontsize=10)
                        ax.text(-0.3, 0.5, title, transform=ax.transAxes, rotation=90,
                               va='center', ha='center', fontsize=11, fontweight='bold')
                    if row == 2:
                        ax.set_xlabel('Time (days)', fontsize=10)
                    
                    ax.grid(True, alpha=0.3)
                    ax.text(0.05, 0.95, f'Peak: {peak_t:.0f}d', transform=ax.transAxes,
                           va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.suptitle('PSPL Events Across Temporal Regions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_regions_visual.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: temporal_regions_visual.png")
        plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    print("\n✅ Check parameter t0 distributions")
    print("✅ Check empirical peak times from light curves")
    print("✅ Check temporal feature correlations")
    print("✅ Visual inspection across temporal regions")
    print(f"\nAll diagnostic plots saved to: {output_dir}/")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose temporal bias in microlensing datasets'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset (.npz file)')
    parser.add_argument('--output', type=str, default='temporal_bias_diagnosis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    diagnose_temporal_bias(args.data, args.output)


if __name__ == '__main__':
    main()
