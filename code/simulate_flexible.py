"""
Flexible microlensing event simulation with configurable binary regimes
Supports systematic benchmarking experiments
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_experiments import (
    BINARY_REGIMES, PSPL_BASELINE_MIN, PSPL_BASELINE_MAX,
    PSPL_T0_MIN, PSPL_T0_MAX, PSPL_U0_MIN, PSPL_U0_MAX,
    PSPL_TE_MIN, PSPL_TE_MAX, BINARY_ALPHA_MIN, BINARY_ALPHA_MAX,
    BINARY_TE_MIN, BINARY_TE_MAX, BINARY_T0_MIN, BINARY_T0_MAX,
    TIME_MIN, VBM_REL_TOL, VBM_TOL, PAD_VALUE
)

# Initialize VBMicrolensing
VBM = VBMicrolensing.VBMicrolensing()
VBM.RelTol = VBM_REL_TOL
VBM.Tol = VBM_TOL

def generate_pspl_event(n_observations=1500, 
                       time_min=0,
                       time_max=1000,
                       mag_error_std=0.10, 
                       cadence_mask_prob=0.20):
    """
    Generate PSPL event
    """
    timestamps = np.linspace(time_min, time_max, n_observations)
    baseline = np.random.uniform(PSPL_BASELINE_MIN, PSPL_BASELINE_MAX)
    t0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
    u0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
    tE = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)

    # Paczynski magnification
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))

    # Convert to magnitudes
    magnitudes = baseline - 2.5 * np.log10(magnification)

    # Add photometric errors
    magnitudes += np.random.normal(0, mag_error_std, size=magnitudes.shape)

    # Apply cadence masking
    mask = np.random.rand(n_observations) < cadence_mask_prob
    magnitudes[mask] = np.nan

    # Convert to flux
    flux = 10 ** (-(magnitudes - baseline) / 2.5)

    return timestamps, flux, {'t0': t0, 'u0': u0, 'tE': tE, 'baseline': baseline}

def generate_binary_event(n_points=1500,
                         time_min=0,
                         time_max=1000,
                         binary_regime='mixed_binary',
                         mag_error_std=0.10, 
                         cadence_mask_prob=0.20):
    """
    Generate Binary event using VBMicrolensing with configurable regime
    
    Args:
        n_points: Number of time points
        time_min: Start time
        time_max: End time
        binary_regime: Name of binary regime from BINARY_REGIMES
        mag_error_std: Photometric error
        cadence_mask_prob: Fraction of missing observations
    """
    timestamps = np.linspace(time_min, time_max, n_points)
    
    # Get parameter ranges for this regime
    regime = BINARY_REGIMES[binary_regime]
    
    # Sample binary parameters from regime-specific ranges
    s = np.random.uniform(regime['s_min'], regime['s_max'])
    q = np.random.uniform(regime['q_min'], regime['q_max'])
    rho = np.random.uniform(regime['rho_min'], regime['rho_max'])
    u0 = np.random.uniform(regime['u0_min'], regime['u0_max'])
    
    # These remain the same across regimes
    alpha = np.random.uniform(BINARY_ALPHA_MIN, BINARY_ALPHA_MAX)
    tE = np.random.uniform(BINARY_TE_MIN, BINARY_TE_MAX)
    t0 = np.random.uniform(BINARY_T0_MIN, BINARY_T0_MAX)

    # VBMicrolensing expects log parameters
    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    try:
        # Generate magnification
        magnifications = np.array(VBM.BinaryLightCurve(params, timestamps)[0])
        
        # Add photometric errors
        magnifications += np.random.normal(0, mag_error_std, size=magnifications.shape)
        
        # Apply cadence masking
        mask = np.random.rand(n_points) < cadence_mask_prob
        magnifications[mask] = np.nan
        
        return timestamps, magnifications, {
            's': s, 'q': q, 'rho': rho, 'alpha': alpha, 
            'tE': tE, 't0': t0, 'u0': u0, 'regime': binary_regime
        }
    except Exception as e:
        # If VBM fails, try again with different parameters
        print(f"VBM error: {e}, retrying...")
        return generate_binary_event(n_points, time_min, time_max, binary_regime, 
                                    mag_error_std, cadence_mask_prob)

def pad_missing_data(data, pad_value=PAD_VALUE):
    """Replace NaN with pad value"""
    return np.nan_to_num(data, nan=pad_value)

def simulate_dataset(n_pspl, n_binary, 
                    binary_regime='mixed_binary',
                    n_points=1500,
                    time_min=0,
                    time_max=1000,
                    mag_error_std=0.10,
                    cadence_mask_prob=0.20,
                    output_file='events.npz'):
    """
    Generate full dataset with specified parameters
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of Binary events
        binary_regime: Which binary parameter regime to use
        n_points: Number of time points per event
        time_min, time_max: Time range
        mag_error_std: Photometric error standard deviation
        cadence_mask_prob: Fraction of missing observations
        output_file: Path to save data
    """
    print("=" * 80)
    print("SIMULATION PARAMETERS")
    print("=" * 80)
    print(f"PSPL events: {n_pspl}")
    print(f"Binary events: {n_binary}")
    print(f"Binary regime: {binary_regime}")
    print(f"  Description: {BINARY_REGIMES[binary_regime]['description']}")
    regime = BINARY_REGIMES[binary_regime]
    print(f"  s range: [{regime['s_min']:.2f}, {regime['s_max']:.2f}]")
    print(f"  q range: [{regime['q_min']:.2f}, {regime['q_max']:.2f}]")
    print(f"  rho range: [{regime['rho_min']:.3f}, {regime['rho_max']:.3f}]")
    print(f"  u0 range: [{regime['u0_min']:.2f}, {regime['u0_max']:.2f}]")
    print(f"Time points: {n_points}")
    print(f"Time range: [{time_min}, {time_max}]")
    print(f"Photometric error: {mag_error_std:.3f} mag")
    print(f"Missing observations: {cadence_mask_prob*100:.1f}%")
    print("=" * 80)
    
    print(f"\nGenerating {n_pspl} PSPL events...")
    pspl_data = []
    pspl_params = []
    
    for i in tqdm(range(n_pspl)):
        t, flux, params = generate_pspl_event(
            n_observations=n_points,
            time_min=time_min,
            time_max=time_max,
            mag_error_std=mag_error_std,
            cadence_mask_prob=cadence_mask_prob
        )
        flux_padded = pad_missing_data(flux)
        pspl_data.append(flux_padded)
        pspl_params.append(params)
    
    print(f"\nGenerating {n_binary} Binary events ({binary_regime})...")
    binary_data = []
    binary_params = []
    
    for i in tqdm(range(n_binary)):
        t, mags, params = generate_binary_event(
            n_points=n_points,
            time_min=time_min,
            time_max=time_max,
            binary_regime=binary_regime,
            mag_error_std=mag_error_std,
            cadence_mask_prob=cadence_mask_prob
        )
        mags_padded = pad_missing_data(mags)
        binary_data.append(mags_padded)
        binary_params.append(params)
    
    # Combine into arrays
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]
    X_binary = np.array(binary_data)[:, :, np.newaxis]
    
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save metadata
    metadata = {
        'n_pspl': n_pspl,
        'n_binary': n_binary,
        'binary_regime': binary_regime,
        'binary_regime_description': BINARY_REGIMES[binary_regime]['description'],
        'n_points': n_points,
        'time_min': time_min,
        'time_max': time_max,
        'mag_error_std': mag_error_std,
        'cadence_mask_prob': cadence_mask_prob,
    }
    
    metadata_file = output_file.replace('.npz', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaving to {output_file}...")
    np.savez_compressed(output_file, 
                       X=X, 
                       y=y,
                       pspl_params=pspl_params,
                       binary_params=binary_params,
                       metadata=metadata)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Total events: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Data saved: {output_file}")
    print(f"Metadata saved: {metadata_file}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate microlensing events with configurable parameters'
    )
    parser.add_argument('--n_pspl', type=int, default=500_000,
                       help='Number of PSPL events')
    parser.add_argument('--n_binary', type=int, default=500_000,
                       help='Number of Binary events')
    parser.add_argument('--binary_regime', type=str, default='mixed_binary',
                       choices=list(BINARY_REGIMES.keys()),
                       help='Binary parameter regime')
    parser.add_argument('--n_points', type=int, default=1500,
                       help='Number of time points')
    parser.add_argument('--time_min', type=float, default=0,
                       help='Start time')
    parser.add_argument('--time_max', type=float, default=1000,
                       help='End time')
    parser.add_argument('--mag_error', type=float, default=0.10,
                       help='Photometric error (mag)')
    parser.add_argument('--cadence', type=float, default=0.20,
                       help='Fraction of missing observations (0-1)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Print available regimes
    print("\nAvailable binary regimes:")
    for name, regime in BINARY_REGIMES.items():
        print(f"  {name}: {regime['description']}")
    print()
    
    simulate_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_regime=args.binary_regime,
        n_points=args.n_points,
        time_min=args.time_min,
        time_max=args.time_max,
        mag_error_std=args.mag_error,
        cadence_mask_prob=args.cadence,
        output_file=args.output
    )
