#!/usr/bin/env python3
"""
Microlensing Event Simulation - Production Version (FIXED)
==================================================
Generates realistic PSPL and binary microlensing light curves.
Uses VBBinaryLensing for accurate binary magnifications.

FIXES APPLIED:
- Added --cadence_mask_prob CLI argument
- Added --mag_error_std CLI argument
- Updated all function signatures to pass these parameters
- Fixed version string to v10.0

Author: Kunal Bhatia
Version: 10.0 - Production Ready (FIXED)
"""

import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import sys

# Import configuration
from config import (
    N_POINTS, TIME_MIN, TIME_MAX,
    PSPL_T0_MIN, PSPL_T0_MAX, PSPL_U0_MIN, PSPL_U0_MAX, PSPL_TE_MIN, PSPL_TE_MAX,
    BASELINE_MIN, BASELINE_MAX,
    CADENCE_MASK_PROB, MAG_ERROR_STD, PAD_VALUE,
    BINARY_PARAM_SETS, VBM_TOLERANCE, MAX_BINARY_ATTEMPTS,
    get_config_summary, validate_config
)

# Try to import VBBinaryLensing
try:
    import VBBinaryLensing
    HAS_VBB = True
    print("✅ VBBinaryLensing loaded successfully")
except ImportError:
    HAS_VBB = False
    print("⚠️  VBBinaryLensing not found, will use approximation")


def pspl_magnification(t, t_E, u_0, t_0):
    """
    Compute PSPL magnification
    
    Args:
        t: Time array
        t_E: Einstein crossing time
        u_0: Impact parameter
        t_0: Time of closest approach
        
    Returns:
        Magnification array
    """
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A


def binary_magnification_vbb(t, t_E, u_0, t_0, s, q, alpha, rho=0.001):
    """
    Compute binary lens magnification using VBBinaryLensing
    
    Args:
        t: Time array
        t_E: Einstein crossing time
        u_0: Impact parameter
        t_0: Time of closest approach
        s: Binary separation (in Einstein radii)
        q: Mass ratio (secondary/primary)
        alpha: Source trajectory angle
        rho: Source size (normalized)
        
    Returns:
        Magnification array
    """
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = VBM_TOLERANCE
    VBB.RelTol = VBM_TOLERANCE
    
    # Source trajectory
    tau = (t - t_0) / t_E
    source_x = u_0 * np.cos(alpha) + tau * np.sin(alpha)
    source_y = u_0 * np.sin(alpha) - tau * np.cos(alpha)
    
    mag = np.zeros_like(t)
    for i, (sx, sy) in enumerate(zip(source_x, source_y)):
        mag[i] = VBB.BinaryMag2(s, q, sx, sy, rho)
    
    return mag


def binary_magnification_approx(t, t_E, u_0, t_0, s, q, alpha, rho=0.001):
    """
    Approximate binary lens magnification (fallback when VBB unavailable)
    
    Uses PSPL with caustic perturbations
    """
    # Base PSPL magnification
    A_pspl = pspl_magnification(t, t_E, u_0, t_0)
    
    # Add caustic-like features
    tau = (t - t_0) / t_E
    source_x = u_0 * np.cos(alpha) + tau * np.sin(alpha)
    source_y = u_0 * np.sin(alpha) - tau * np.cos(alpha)
    
    # Simplified caustic positions
    caustic_x = s * np.array([0.5, -0.5, 0, 0])
    caustic_y = s * np.array([0, 0, 0.5, -0.5])
    
    # Perturbation from caustics
    perturbation = 0
    for cx, cy in zip(caustic_x, caustic_y):
        dist = np.sqrt((source_x - cx)**2 + (source_y - cy)**2)
        caustic_width = 0.05 * np.sqrt(q)
        perturbation += q * np.exp(-dist**2 / (2 * caustic_width**2))
    
    A_binary = A_pspl * (1 + perturbation * 3)
    
    return A_binary


def generate_pspl_params(n_events, seed=None):
    """
    Generate PSPL parameters using config values
    
    Returns:
        list: List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = []
    for _ in range(n_events):
        t_E = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)
        t_0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
        u_0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
        m_source = np.random.uniform(BASELINE_MIN, BASELINE_MAX)
        
        params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t_0,
            'm_source': m_source
        })
    
    return params


def generate_binary_params(n_events, params_set='baseline', seed=None):
    """
    Generate binary lens parameters using config values
    
    Args:
        n_events: Number of events to generate
        params_set: Which binary configuration to use
        seed: Random seed
        
    Returns:
        list: List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    if params_set not in BINARY_PARAM_SETS:
        print(f"⚠️  Unknown params_set '{params_set}', using 'baseline'")
        params_set = 'baseline'
    
    config = BINARY_PARAM_SETS[params_set]
    
    params = []
    
    pbar = tqdm(total=n_events, desc=f"  Generating {params_set} binary params")
    
    for _ in range(n_events):
        # Generate parameters
        t_E = np.random.uniform(config['tE_min'], config['tE_max'])
        t_0 = np.random.uniform(config['t0_min'], config['t0_max'])
        u_0 = np.random.uniform(config['u0_min'], config['u0_max'])
        s = np.random.uniform(config['s_min'], config['s_max'])
        q = np.random.uniform(config['q_min'], config['q_max'])
        alpha = np.random.uniform(config['alpha_min'], config['alpha_max'])
        rho = np.random.uniform(config['rho_min'], config['rho_max'])
        m_source = np.random.uniform(BASELINE_MIN, BASELINE_MAX)
        
        params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t_0,
            's': s,
            'q': q,
            'alpha': alpha,
            'rho': rho,
            'm_source': m_source
        })
        pbar.update(1)
    
    pbar.close()
    
    return params


def generate_light_curve(params, event_type, timestamps):
    """
    Generate a single light curve
    
    Args:
        params: Parameter dictionary
        event_type: 'pspl' or 'binary'
        timestamps: Time array
        
    Returns:
        Flux array
    """
    if event_type == 'pspl':
        A = pspl_magnification(timestamps, params['t_E'], params['u_0'], params['t_0'])
    else:  # binary
        if HAS_VBB:
            A = binary_magnification_vbb(
                timestamps, params['t_E'], params['u_0'], params['t_0'],
                params['s'], params['q'], params['alpha'], params['rho']
            )
        else:
            A = binary_magnification_approx(
                timestamps, params['t_E'], params['u_0'], params['t_0'],
                params['s'], params['q'], params['alpha'], params['rho']
            )
    
    # Convert to flux (magnification is already flux ratio)
    flux = A
    
    return flux


def add_observational_effects(flux, error_mag=MAG_ERROR_STD, 
                              cadence_missing=CADENCE_MASK_PROB, 
                              pad_value=PAD_VALUE):
    """
    Add realistic observational effects
    
    Args:
        flux: Clean flux array
        error_mag: Photometric error
        cadence_missing: Fraction of missing observations
        pad_value: Value for missing data
        
    Returns:
        flux_obs: Observed flux with noise and gaps
    """
    flux_obs = flux.copy()
    
    # Add photometric noise (fractional error in flux)
    noise = np.random.normal(0, error_mag, size=len(flux))
    flux_obs = flux_obs * (1 + noise)
    
    # Add missing observations
    mask = np.random.random(len(flux)) < cadence_missing
    flux_obs[mask] = pad_value
    
    # Ensure flux is positive where not masked
    flux_obs[flux_obs != pad_value] = np.maximum(flux_obs[flux_obs != pad_value], 0.01)
    
    return flux_obs


def generate_single_event(args):
    """
    Wrapper for parallel generation
    
    Args:
        args: (index, params, event_type, timestamps, cadence, error)
        
    Returns:
        (flux, label, params)
    """
    # FIXED: Unpack all 6 arguments including cadence and error
    idx, params, event_type, timestamps, cadence, error = args
    
    # Generate clean light curve
    flux = generate_light_curve(params, event_type, timestamps)
    
    # FIXED: Pass cadence and error to add_observational_effects
    flux_obs = add_observational_effects(flux, error_mag=error, cadence_missing=cadence)
    
    # Label: 0 = PSPL, 1 = Binary
    label = 0 if event_type == 'pspl' else 1
    
    return flux_obs, label, params


def simulate_dataset(n_pspl, n_binary, binary_params='baseline', 
                     num_workers=1, seed=None, save_params=False,
                     cadence_mask_prob=None, mag_error_std=None):
    """
    Generate complete dataset
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of binary events
        binary_params: Binary configuration
        num_workers: Number of parallel workers
        seed: Random seed
        save_params: Whether to save event parameters
        cadence_mask_prob: Override CADENCE_MASK_PROB from config (NEW)
        mag_error_std: Override MAG_ERROR_STD from config (NEW)
        
    Returns:
        X: Light curves (n_events, n_points)
        y: Labels (n_events,)
        timestamps: Time array
        params_dict: Event parameters (if save_params=True)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps
    timestamps = np.linspace(TIME_MIN, TIME_MAX, N_POINTS)
    
    # FIXED: Use overrides if provided, otherwise use config defaults
    cadence = CADENCE_MASK_PROB if cadence_mask_prob is None else cadence_mask_prob
    error = MAG_ERROR_STD if mag_error_std is None else mag_error_std
    
    # Generate parameters
    print(f"\n{'='*70}")
    print(f"GENERATING {n_pspl} PSPL + {n_binary} BINARY EVENTS")
    print(f"{'='*70}")
    print(f"Binary config: {binary_params}")
    print(f"Time window: [{TIME_MIN}, {TIME_MAX}] days")
    print(f"Cadence mask: {cadence*100:.0f}% missing")  # FIXED: Use local variable
    print(f"Photometric error: {error:.3f}")  # FIXED: Use local variable
    
    print("\nGenerating PSPL parameters...")
    params_pspl = generate_pspl_params(n_pspl, seed=seed)
    
    print(f"\nGenerating Binary parameters ({binary_params})...")
    params_binary = generate_binary_params(n_binary, params_set=binary_params, seed=seed+1 if seed else None)
    
    # FIXED: Prepare arguments with cadence and error parameters
    args_list = []
    args_list.extend([(i, p, 'pspl', timestamps, cadence, error) for i, p in enumerate(params_pspl)])
    args_list.extend([(i, p, 'binary', timestamps, cadence, error) for i, p in enumerate(params_binary)])
    
    # Generate events in parallel
    print(f"\nGenerating light curves ({num_workers} workers)...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_event, args_list),
                total=len(args_list),
                desc="  Processing events"
            ))
    else:
        results = [generate_single_event(args) for args in tqdm(args_list, desc="  Processing events")]
    
    # Unpack results
    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    all_params = [r[2] for r in results]
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    all_params = [all_params[i] for i in shuffle_idx]
    
    # Statistics
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total events: {len(X)}")
    print(f"  PSPL: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  Binary: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    
    # Binary statistics
    binary_mag = []
    for i, label in enumerate(y):
        if label == 1:
            flux = X[i]
            valid_flux = flux[flux != PAD_VALUE]
            if len(valid_flux) > 0:
                binary_mag.append(valid_flux.max())
    
    if binary_mag:
        binary_mag = np.array(binary_mag)
        print(f"\nBinary Statistics:")
        print(f"  Max mag mean: {binary_mag.mean():.1f}")
        print(f"  Max mag median: {np.median(binary_mag):.1f}")
        print(f"  Max mag range: [{binary_mag.min():.1f}, {binary_mag.max():.1f}]")
    
    # Prepare return
    params_dict = None
    if save_params:
        params_pspl_list = [all_params[i] for i in range(len(all_params)) if y[i] == 0]
        params_binary_list = [all_params[i] for i in range(len(all_params)) if y[i] == 1]
        params_dict = {
            'pspl': params_pspl_list,
            'binary': params_binary_list
        }
    
    return X, y, timestamps, params_dict


def main():
    parser = argparse.ArgumentParser(
        description='Generate microlensing dataset with complete events'
    )
    parser.add_argument('--n_pspl', type=int, default=10000,
                       help='Number of PSPL events')
    parser.add_argument('--n_binary', type=int, default=10000,
                       help='Number of binary events')
    parser.add_argument('--binary_params', type=str, default='baseline',
                       choices=list(BINARY_PARAM_SETS.keys()),
                       help='Binary parameter configuration')
    parser.add_argument('--output', type=str, default='../data/raw/dataset.npz',
                       help='Output file path')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_params', action='store_true',
                       help='Save event parameters in output file')
    
    # FIXED: Added CLI arguments for experimental control
    parser.add_argument('--cadence_mask_prob', type=float, default=None,
                       help='Override CADENCE_MASK_PROB from config (for experiments)')
    parser.add_argument('--mag_error_std', type=float, default=None,
                       help='Override MAG_ERROR_STD from config (for experiments)')
    
    parser.add_argument('--show_config', action='store_true',
                       help='Show configuration and exit')
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        get_config_summary()
        print()
        validate_config()
        return
    
    print("="*70)
    print("BINARY MICROLENSING SIMULATION v10.0")  # FIXED: Updated version string
    print("="*70)
    print(f"PSPL events: {args.n_pspl}")
    print(f"Binary events: {args.n_binary}")
    print(f"Binary config: {args.binary_params}")
    print(f"Workers: {args.num_workers}")
    print(f"Save params: {args.save_params}")
    
    # Show override information if provided
    if args.cadence_mask_prob is not None:
        print(f"Cadence override: {args.cadence_mask_prob*100:.0f}% missing")
    if args.mag_error_std is not None:
        print(f"Error override: {args.mag_error_std:.3f} mag")
    
    # Validate configuration
    print("\n" + "="*70)
    if not validate_config():
        print("\n⚠️  Configuration has warnings but will proceed...")
    print("="*70)
    
    # FIXED: Generate dataset with override parameters
    X, y, timestamps, params_dict = simulate_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_params=args.binary_params,
        num_workers=args.num_workers,
        seed=args.seed,
        save_params=args.save_params,
        cadence_mask_prob=args.cadence_mask_prob,
        mag_error_std=args.mag_error_std
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'X': X,
        'y': y,
        'timestamps': timestamps
    }
    
    if params_dict:
        # Save parameters as JSON strings
        save_dict['params_pspl_json'] = json.dumps(params_dict['pspl'])
        save_dict['params_binary_json'] = json.dumps(params_dict['binary'])
    
    np.savez(output_path, **save_dict)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()