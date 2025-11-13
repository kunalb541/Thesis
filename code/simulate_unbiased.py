#!/usr/bin/env python3
"""
Temporal-Bias-Free Data Generation
===================================

Generates microlensing datasets with guaranteed no temporal bias:
1. Forces identical t0 distributions for PSPL and Binary
2. Adds random time shifts to all events
3. Validates temporal independence

Version: 14.1 - Temporal Bias Fix v3
Author: Kunal Bhatia
"""

import numpy as np
import sys
sys.path.append('/mnt/user-data/uploads')

# Import from uploaded simulate.py
from simulate import (
    generate_flat_params, generate_light_curve,
    add_observational_effects, generate_single_event
)

from config import (
    N_POINTS, TIME_MIN, TIME_MAX,
    PSPL_TE_MIN, PSPL_TE_MAX, PSPL_U0_MIN, PSPL_U0_MAX,
    BASELINE_MIN, BASELINE_MAX,
    CADENCE_MASK_PROB, MAG_ERROR_STD, PAD_VALUE,
    BINARY_PARAM_SETS
)

import argparse
from pathlib import Path
import json
from tqdm import tqdm
from multiprocessing import Pool


def generate_temporally_unbiased_params(n_pspl, n_binary, binary_params='baseline', seed=None):
    """
    Generate PSPL and Binary parameters with IDENTICAL t0 distributions
    
    Strategy:
    1. Generate shared t0 samples from uniform distribution
    2. Shuffle and assign to PSPL and Binary
    3. This guarantees identical t0 distributions
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of Binary events
        binary_params: Binary configuration
        seed: Random seed
        
    Returns:
        pspl_params, binary_params
    """
    if seed is not None:
        np.random.seed(seed)
    
    print("\n" + "="*70)
    print("TEMPORAL-BIAS-FREE PARAMETER GENERATION")
    print("="*70)
    
    # Step 1: Generate shared t0 pool
    total_events = n_pspl + n_binary
    
    # Define t0 range - wider than before to ensure variety
    T0_MIN = -70.0  # Even earlier peaks possible
    T0_MAX = 50.0   # Even later peaks possible
    
    print(f"\nGenerating shared t0 pool:")
    print(f"  Total events: {total_events}")
    print(f"  t0 range: [{T0_MIN}, {T0_MAX}] days")
    
    # Generate uniform t0 distribution
    shared_t0 = np.random.uniform(T0_MIN, T0_MAX, total_events)
    
    # Shuffle to break any ordering
    np.random.shuffle(shared_t0)
    
    # Split into PSPL and Binary
    t0_pspl = shared_t0[:n_pspl]
    t0_binary = shared_t0[n_pspl:]
    
    print(f"\nPSPL t0 statistics:")
    print(f"  Mean: {np.mean(t0_pspl):.2f}")
    print(f"  Std:  {np.std(t0_pspl):.2f}")
    print(f"  Min:  {np.min(t0_pspl):.2f}")
    print(f"  Max:  {np.max(t0_pspl):.2f}")
    
    print(f"\nBinary t0 statistics:")
    print(f"  Mean: {np.mean(t0_binary):.2f}")
    print(f"  Std:  {np.std(t0_binary):.2f}")
    print(f"  Min:  {np.min(t0_binary):.2f}")
    print(f"  Max:  {np.max(t0_binary):.2f}")
    
    # Verify distributions are identical (KS test)
    from scipy.stats import ks_2samp
    stat, pval = ks_2samp(t0_pspl, t0_binary)
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {pval:.4f}")
    if pval < 0.05:
        print(f"  ⚠️  WARNING: Distributions differ (p < 0.05)")
    else:
        print(f"  ✅ Distributions identical (p >= 0.05)")
    
    # Step 2: Generate PSPL parameters
    print(f"\nGenerating PSPL parameters...")
    pspl_params = []
    
    for i in tqdm(range(n_pspl), desc="  PSPL"):
        t_E = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)
        u_0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
        m_source = np.random.uniform(BASELINE_MIN, BASELINE_MAX)
        
        pspl_params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t0_pspl[i],  # Use pre-assigned t0
            'm_source': m_source,
            'type': 'pspl'
        })
    
    # Step 3: Generate Binary parameters
    print(f"Generating Binary parameters ({binary_params})...")
    
    if binary_params not in BINARY_PARAM_SETS:
        print(f"  ⚠️  Unknown params_set '{binary_params}', using 'baseline'")
        binary_params = 'baseline'
    
    config = BINARY_PARAM_SETS[binary_params]
    binary_params_list = []
    
    for i in tqdm(range(n_binary), desc="  Binary"):
        t_E = np.random.uniform(config['tE_min'], config['tE_max'])
        u_0 = np.random.uniform(config['u0_min'], config['u0_max'])
        s = np.random.uniform(config['s_min'], config['s_max'])
        q = np.random.uniform(config['q_min'], config['q_max'])
        alpha = np.random.uniform(config['alpha_min'], config['alpha_max'])
        rho = np.random.uniform(config['rho_min'], config['rho_max'])
        m_source = np.random.uniform(BASELINE_MIN, BASELINE_MAX)
        
        binary_params_list.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t0_binary[i],  # Use pre-assigned t0
            's': s,
            'q': q,
            'alpha': alpha,
            'rho': rho,
            'm_source': m_source,
            'type': 'binary'
        })
    
    print("\n✅ Parameters generated with identical t0 distributions")
    
    return pspl_params, binary_params_list


def add_random_time_shifts(X, y, params_dict, timestamps, max_shift=20.0, pad_value=-1.0):
    """
    Add random time shifts to break any residual temporal patterns
    
    Strategy:
    - Shift each event's observation window by a random amount
    - Preserves event morphology but changes absolute timing
    - Further decorrelates temporal position from class
    
    Args:
        X: Light curves [N, T]
        y: Labels [N]
        params_dict: Parameter dictionary
        timestamps: Original timestamps
        max_shift: Maximum shift in days (default: 20)
        pad_value: Padding value
        
    Returns:
        X_shifted, params_dict_shifted
    """
    print("\n" + "="*70)
    print("APPLYING RANDOM TIME SHIFTS")
    print("="*70)
    print(f"Max shift: ±{max_shift} days")
    
    X_shifted = X.copy()
    params_shifted = {'flat': [], 'pspl': [], 'binary': []}
    
    for i in tqdm(range(len(X)), desc="  Shifting events"):
        # Random shift
        shift = np.random.uniform(-max_shift, max_shift)
        
        # Shift light curve
        flux = X[i]
        valid_mask = flux != pad_value
        
        if valid_mask.sum() > 0:
            # Create new timeline
            new_timestamps = timestamps + shift
            
            # Interpolate flux onto original timeline
            valid_times_old = timestamps[valid_mask]
            valid_flux = flux[valid_mask]
            
            # Linear interpolation
            flux_interp = np.interp(
                timestamps,
                valid_times_old + shift,
                valid_flux,
                left=pad_value,
                right=pad_value
            )
            
            X_shifted[i] = flux_interp
        
        # Update parameters
        class_idx = int(y[i])
        if class_idx == 0:
            # Flat - no t0 to shift
            params_shifted['flat'].append(params_dict['flat'][len(params_shifted['flat'])])
        elif class_idx == 1:
            # PSPL
            param = params_dict['pspl'][len(params_shifted['pspl'])].copy()
            param['t_0'] += shift
            params_shifted['pspl'].append(param)
        else:
            # Binary
            param = params_dict['binary'][len(params_shifted['binary'])].copy()
            param['t_0'] += shift
            params_shifted['binary'].append(param)
    
    print("✅ Random time shifts applied")
    
    return X_shifted, params_shifted


def simulate_temporally_unbiased_dataset(
    n_flat, n_pspl, n_binary,
    binary_params='baseline',
    num_workers=1,
    seed=None,
    apply_time_shifts=True,
    max_shift=20.0,
    cadence_mask_prob=None,
    mag_error_std=None
):
    """
    Generate dataset with guaranteed no temporal bias
    
    Args:
        n_flat: Number of flat events
        n_pspl: Number of PSPL events
        n_binary: Number of binary events
        binary_params: Binary configuration
        num_workers: Parallel workers
        seed: Random seed
        apply_time_shifts: Whether to apply random time shifts
        max_shift: Maximum time shift
        cadence_mask_prob: Override cadence
        mag_error_std: Override error
        
    Returns:
        X, y, timestamps, params_dict
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps
    timestamps = np.linspace(TIME_MIN, TIME_MAX, N_POINTS)
    
    # Use overrides if provided
    cadence = CADENCE_MASK_PROB if cadence_mask_prob is None else cadence_mask_prob
    error = MAG_ERROR_STD if mag_error_std is None else mag_error_std
    
    print("="*70)
    print("TEMPORAL-BIAS-FREE DATASET GENERATION")
    print("="*70)
    print(f"Dataset composition:")
    print(f"  Flat:   {n_flat}")
    print(f"  PSPL:   {n_pspl}")
    print(f"  Binary: {n_binary}")
    print(f"  Total:  {n_flat + n_pspl + n_binary}")
    print(f"Binary topology: {binary_params}")
    print(f"Time window: [{TIME_MIN}, {TIME_MAX}] days")
    print(f"Apply time shifts: {apply_time_shifts}")
    
    # Generate Flat parameters (no temporal bias possible)
    print("\nGenerating Flat parameters...")
    params_flat = generate_flat_params(n_flat, seed=seed)
    
    # Generate PSPL and Binary with identical t0 distributions
    pspl_params, binary_params_list = generate_temporally_unbiased_params(
        n_pspl, n_binary,
        binary_params=binary_params,
        seed=seed+1 if seed else None
    )
    
    # Prepare arguments for parallel generation
    args_list = []
    args_list.extend([(i, p, 'flat', timestamps, cadence, error) 
                      for i, p in enumerate(params_flat)])
    args_list.extend([(i, p, 'pspl', timestamps, cadence, error) 
                      for i, p in enumerate(pspl_params)])
    args_list.extend([(i, p, 'binary', timestamps, cadence, error) 
                      for i, p in enumerate(binary_params_list)])
    
    # Generate events
    print(f"\nGenerating light curves ({num_workers} workers)...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_event, args_list),
                total=len(args_list),
                desc="  Processing"
            ))
    else:
        results = [generate_single_event(args) for args in tqdm(args_list, desc="  Processing")]
    
    # Unpack results
    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    all_params = [r[2] for r in results]
    
    # Organize parameters by class
    params_dict = {
        'flat': [all_params[i] for i in range(len(all_params)) if y[i] == 0],
        'pspl': [all_params[i] for i in range(len(all_params)) if y[i] == 1],
        'binary': [all_params[i] for i in range(len(all_params)) if y[i] == 2]
    }
    
    # Apply random time shifts
    if apply_time_shifts:
        X, params_dict = add_random_time_shifts(
            X, y, params_dict, timestamps,
            max_shift=max_shift,
            pad_value=PAD_VALUE
        )
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Re-organize params after shuffle
    params_final = {'flat': [], 'pspl': [], 'binary': []}
    flat_idx, pspl_idx, binary_idx = 0, 0, 0
    
    for i in shuffle_idx:
        orig_class = y[list(shuffle_idx).index(i)]
        if orig_class == 0:
            params_final['flat'].append(params_dict['flat'][flat_idx])
            flat_idx += 1
        elif orig_class == 1:
            params_final['pspl'].append(params_dict['pspl'][pspl_idx])
            pspl_idx += 1
        else:
            params_final['binary'].append(params_dict['binary'][binary_idx])
            binary_idx += 1
    
    # Final statistics
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total events: {len(X)}")
    print(f"  Flat:   {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  PSPL:   {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Binary: {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")
    
    # Verify final t0 distributions
    if len(params_final['pspl']) > 0 and len(params_final['binary']) > 0:
        t0_pspl_final = [p['t_0'] for p in params_final['pspl']]
        t0_binary_final = [p['t_0'] for p in params_final['binary']]
        
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(t0_pspl_final, t0_binary_final)
        
        print(f"\n Final t0 distribution test (after shuffling):")
        print(f"  PSPL t0:   mean={np.mean(t0_pspl_final):.1f}, std={np.std(t0_pspl_final):.1f}")
        print(f"  Binary t0: mean={np.mean(t0_binary_final):.1f}, std={np.std(t0_binary_final):.1f}")
        print(f"  KS test: stat={stat:.4f}, p={pval:.4f}")
        
        if pval >= 0.05:
            print(f"  ✅ TEMPORAL BIAS ELIMINATED (p >= 0.05)")
        else:
            print(f"  ⚠️  WARNING: Distributions still differ (p < 0.05)")
    
    return X, y, timestamps, params_final


def main():
    parser = argparse.ArgumentParser(
        description='Generate temporally-unbiased microlensing dataset'
    )
    parser.add_argument('--n_flat', type=int, default=10000)
    parser.add_argument('--n_pspl', type=int, default=10000)
    parser.add_argument('--n_binary', type=int, default=10000)
    parser.add_argument('--binary_params', type=str, default='baseline',
                       choices=list(BINARY_PARAM_SETS.keys()))
    parser.add_argument('--output', type=str, 
                       default='../data/raw/dataset_unbiased.npz')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_time_shifts', action='store_true',
                       help='Disable random time shifts')
    parser.add_argument('--max_shift', type=float, default=20.0,
                       help='Maximum time shift in days')
    parser.add_argument('--cadence_mask_prob', type=float, default=None)
    parser.add_argument('--mag_error_std', type=float, default=None)
    
    args = parser.parse_args()
    
    # Generate dataset
    X, y, timestamps, params_dict = simulate_temporally_unbiased_dataset(
        n_flat=args.n_flat,
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_params=args.binary_params,
        num_workers=args.num_workers,
        seed=args.seed,
        apply_time_shifts=not args.no_time_shifts,
        max_shift=args.max_shift,
        cadence_mask_prob=args.cadence_mask_prob,
        mag_error_std=args.mag_error_std
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'X': X,
        'y': y,
        'timestamps': timestamps,
        'n_classes': 3,
        'class_names': ['Flat', 'PSPL', 'Binary'],
        'params_flat_json': json.dumps(params_dict['flat']),
        'params_pspl_json': json.dumps(params_dict['pspl']),
        'params_binary_json': json.dumps(params_dict['binary']),
        'temporal_bias_fix': 'v3',
        'time_shifts_applied': not args.no_time_shifts,
        'max_shift': args.max_shift if not args.no_time_shifts else 0.0
    }
    
    np.savez(output_path, **save_dict)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"Dataset saved to: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Temporal bias fix: v3")
    print(f"Time shifts: {'Applied' if not args.no_time_shifts else 'Not applied'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
