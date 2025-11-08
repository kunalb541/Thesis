#!/usr/bin/env python3
"""
Dataset Validation for Binary Microlensing

Validates that binary events have proper caustic crossings.
Generates comprehensive reports and visualizations.

Author: Kunal Bhatia
Version: 6.0
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
from tqdm import tqdm


def load_dataset(path: str) -> Dict:
    """Load and parse dataset"""
    print(f"Loading dataset: {path}")
    data = np.load(path)
    
    dataset = {
        'X': data['X'],
        'y': data['y'],
        'timestamps': data['timestamps'],
        'perm': data.get('perm', None)
    }
    
    # Parse metadata
    if 'meta_json' in data:
        dataset['meta'] = json.loads(str(data['meta_json']))
    
    # Parse parameters if available
    if 'params_pspl_json' in data:
        dataset['params_pspl'] = json.loads(str(data['params_pspl_json']))
    
    if 'params_binary_json' in data:
        dataset['params_binary'] = json.loads(str(data['params_binary_json']))
    
    # Parse statistics if available
    if 'stats_json' in data:
        dataset['stats'] = json.loads(str(data['stats_json']))
    
    return dataset


def analyze_magnifications(dataset: Dict) -> Dict:
    """Analyze magnification distributions"""
    X = dataset['X']
    y = dataset['y']
    
    # Separate PSPL and Binary
    pspl_data = X[y == 0]
    binary_data = X[y == 1]
    
    # Calculate max magnifications (assuming flux normalization around 1)
    pspl_max_mags = []
    binary_max_mags = []
    
    for flux in pspl_data:
        valid = flux > 0
        if valid.any():
            pspl_max_mags.append(flux[valid].max())
    
    for flux in binary_data:
        valid = flux > 0
        if valid.any():
            binary_max_mags.append(flux[valid].max())
    
    pspl_max_mags = np.array(pspl_max_mags)
    binary_max_mags = np.array(binary_max_mags)
    
    # Statistics
    stats = {
        'pspl': {
            'count': len(pspl_max_mags),
            'mean': pspl_max_mags.mean(),
            'median': np.median(pspl_max_mags),
            'std': pspl_max_mags.std(),
            'min': pspl_max_mags.min(),
            'max': pspl_max_mags.max(),
            'above_20': (pspl_max_mags > 20).sum()
        },
        'binary': {
            'count': len(binary_max_mags),
            'mean': binary_max_mags.mean(),
            'median': np.median(binary_max_mags),
            'std': binary_max_mags.std(),
            'min': binary_max_mags.min(),
            'max': binary_max_mags.max(),
            'above_20': (binary_max_mags > 20).sum()
        }
    }
    
    return stats, pspl_max_mags, binary_max_mags


def validate_binary_caustics(
    dataset: Dict,
    min_magnification: float = 20.0
) -> Tuple[int, int, List[int]]:
    """
    Validate that binary events have caustic crossings.
    
    Returns:
        n_valid: Number of valid binaries
        n_invalid: Number of invalid binaries
        invalid_indices: Indices of invalid binaries
    """
    X = dataset['X']
    y = dataset['y']
    
    binary_indices = np.where(y == 1)[0]
    
    n_valid = 0
    n_invalid = 0
    invalid_indices = []
    
    for idx in binary_indices:
        flux = X[idx]
        valid = flux > 0
        
        if valid.any():
            max_flux = flux[valid].max()
            
            # Check if it has caustic-like spike
            if max_flux >= min_magnification:
                n_valid += 1
            else:
                n_invalid += 1
                invalid_indices.append(idx)
        else:
            n_invalid += 1
            invalid_indices.append(idx)
    
    return n_valid, n_invalid, invalid_indices


def plot_examples(
    dataset: Dict,
    n_examples: int = 12,
    output_dir: Path = None
) -> None:
    """Plot example light curves"""
    
    X = dataset['X']
    y = dataset['y']
    timestamps = dataset['timestamps']
    
    # Select examples
    pspl_indices = np.where(y == 0)[0][:n_examples//2]
    binary_indices = np.where(y == 1)[0][:n_examples//2]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    all_indices = list(pspl_indices) + list(binary_indices)
    all_labels = ['PSPL'] * len(pspl_indices) + ['Binary'] * len(binary_indices)
    
    for ax_idx, (data_idx, label) in enumerate(zip(all_indices, all_labels)):
        if ax_idx >= len(axes):
            break
        
        ax = axes[ax_idx]
        flux = X[data_idx]
        valid = flux > -0.5  # Not padding
        
        if valid.any():
            ax.scatter(timestamps[valid], flux[valid], s=1, alpha=0.7,
                      color='blue' if label == 'PSPL' else 'red')
            
            max_flux = flux[valid].max()
            ax.set_title(f'{label} (max: {max_flux:.1f})', fontsize=10)
        else:
            ax.set_title(f'{label} (invalid)', fontsize=10)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Example Light Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'examples.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_magnification_histogram(
    pspl_mags: np.ndarray,
    binary_mags: np.ndarray,
    output_dir: Path = None
) -> None:
    """Plot magnification distribution histograms"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSPL histogram
    axes[0].hist(pspl_mags, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=20, color='red', linestyle='--', label='Caustic threshold')
    axes[0].set_title('PSPL Max Magnifications')
    axes[0].set_xlabel('Max Magnification')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Binary histogram
    axes[1].hist(binary_mags, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(x=20, color='red', linestyle='--', label='Caustic threshold')
    axes[1].set_title('Binary Max Magnifications')
    axes[1].set_xlabel('Max Magnification')
    axes[1].set_ylabel('Count')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Combined
    axes[2].hist(pspl_mags, bins=50, alpha=0.5, color='blue', label='PSPL', density=True)
    axes[2].hist(binary_mags, bins=50, alpha=0.5, color='red', label='Binary', density=True)
    axes[2].axvline(x=20, color='black', linestyle='--', label='Caustic threshold')
    axes[2].set_title('Comparison (Normalized)')
    axes[2].set_xlabel('Max Magnification')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Magnification Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'magnifications.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_report(
    dataset: Dict,
    stats: Dict,
    n_valid: int,
    n_invalid: int,
    output_path: Path
) -> None:
    """Generate validation report"""
    
    meta = dataset.get('meta', {})
    
    report = {
        'dataset_info': {
            'n_samples': len(dataset['y']),
            'n_pspl': int((dataset['y'] == 0).sum()),
            'n_binary': int((dataset['y'] == 1).sum()),
            'n_points': dataset['X'].shape[1],
            'vbm_available': meta.get('vbm_available', False)
        },
        'magnification_stats': stats,
        'binary_validation': {
            'n_valid': n_valid,
            'n_invalid': n_invalid,
            'valid_fraction': n_valid / (n_valid + n_invalid) if (n_valid + n_invalid) > 0 else 0
        },
        'parameters': meta.get('binary_params', {})
    }
    
    # Add statistics from generation if available
    if 'stats' in dataset:
        report['generation_stats'] = dataset['stats']
    
    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Dataset: {len(dataset['y'])} samples")
    print(f"  PSPL: {report['dataset_info']['n_pspl']}")
    print(f"  Binary: {report['dataset_info']['n_binary']}")
    
    print(f"\nBinary Validation:")
    print(f"  Valid (mag > 20): {n_valid} ({100*n_valid/(n_valid+n_invalid):.1f}%)")
    print(f"  Invalid: {n_invalid}")
    
    print(f"\nMagnification Statistics:")
    print(f"  PSPL mean: {stats['pspl']['mean']:.2f}")
    print(f"  Binary mean: {stats['binary']['mean']:.2f}")
    print(f"  Binary > 20x: {stats['binary']['above_20']} ({100*stats['binary']['above_20']/stats['binary']['count']:.1f}%)")
    
    if report['binary_validation']['valid_fraction'] < 0.8:
        print("\n⚠️  WARNING: Less than 80% of binaries have strong caustics!")
        print("   Consider using 'critical' binary parameters")
    else:
        print("\n✅ Dataset validation PASSED!")


def main():
    parser = argparse.ArgumentParser(description='Validate microlensing dataset')
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--output_dir', default='validation_results', help='Output directory')
    parser.add_argument('--n_examples', type=int, default=12, help='Number of examples to plot')
    parser.add_argument('--min_magnification', type=float, default=20.0, 
                       help='Minimum magnification for valid binary')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.data)
    
    print(f"\nDataset loaded:")
    print(f"  Shape: {dataset['X'].shape}")
    print(f"  Classes: {np.unique(dataset['y'], return_counts=True)}")
    
    # Analyze magnifications
    print("\nAnalyzing magnifications...")
    stats, pspl_mags, binary_mags = analyze_magnifications(dataset)
    
    # Validate binaries
    print("\nValidating binary events...")
    n_valid, n_invalid, invalid_indices = validate_binary_caustics(
        dataset, args.min_magnification
    )
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_examples(dataset, args.n_examples, output_dir)
    plot_magnification_histogram(pspl_mags, binary_mags, output_dir)
    
    # Generate report
    report_path = output_dir / 'validation_report.json'
    generate_report(dataset, stats, n_valid, n_invalid, report_path)
    
    print(f"\n✓ Validation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
