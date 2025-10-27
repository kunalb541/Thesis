"""
Parallelized microlensing event simulation
Uses multiprocessing to speed up generation significantly
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import os

# Import config
from config import *

def init_worker():
    """Initialize VBMicrolensing for each worker process"""
    global VBM
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.RelTol = VBM_REL_TOL
    VBM.Tol = VBM_TOL

def generate_pspl_event_parallel(args):
    """
    Generate single PSPL event (worker function)
    Args: (event_id, n_observations, mag_error_std, cadence_mask_prob)
    """
    event_id, n_observations, mag_error_std, cadence_mask_prob = args
    
    # Set per-event random seed for reproducibility
    np.random.seed(42 + event_id)
    
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_observations)
    baseline = np.random.uniform(PSPL_BASELINE_MIN, PSPL_BASELINE_MAX)
    t0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
    u0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
    tE = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)

    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    magnitudes = baseline - 2.5 * np.log10(magnification)
    magnitudes += np.random.normal(0, mag_error_std, size=magnitudes.shape)

    mask = np.random.rand(n_observations) < cadence_mask_prob
    magnitudes[mask] = np.nan

    flux = 10 ** (-(magnitudes - baseline) / 2.5)
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)

    return flux_padded, {'t0': t0, 'u0': u0, 'tE': tE, 'baseline': baseline}

def generate_binary_event_parallel(args):
    """
    Generate single Binary event (worker function)
    Args: (event_id, n_points, mag_error_std, cadence_mask_prob, binary_params_dict)
    """
    event_id, n_points, mag_error_std, cadence_mask_prob, binary_params_dict = args
    
    # Set per-event random seed
    np.random.seed(42 + 1000000 + event_id)
    
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)
    
    # Sample parameters
    s = np.random.uniform(binary_params_dict['s_min'], binary_params_dict['s_max'])
    q = np.random.uniform(binary_params_dict['q_min'], binary_params_dict['q_max'])
    rho = np.random.uniform(binary_params_dict['rho_min'], binary_params_dict['rho_max'])
    alpha = np.random.uniform(binary_params_dict['alpha_min'], binary_params_dict['alpha_max'])
    tE = np.random.uniform(binary_params_dict['tE_min'], binary_params_dict['tE_max'])
    t0 = np.random.uniform(binary_params_dict['t0_min'], binary_params_dict['t0_max'])
    u0 = np.random.uniform(binary_params_dict['u0_min'], binary_params_dict['u0_max'])

    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    try:
        # VBM is initialized per worker
        magnifications = np.array(VBM.BinaryLightCurve(params, timestamps)[0])
        magnifications += np.random.normal(0, mag_error_std, size=magnifications.shape)
        
        mask = np.random.rand(n_points) < cadence_mask_prob
        magnifications[mask] = np.nan
        
        mags_padded = np.nan_to_num(magnifications, nan=PAD_VALUE)
        
        return mags_padded, {
            's': s, 'q': q, 'rho': rho, 'alpha': alpha,
            'tE': tE, 't0': t0, 'u0': u0
        }
    except Exception as e:
        # If VBM fails, return NaN - will be filtered out
        return None, None

def simulate_dataset_parallel(n_pspl, n_binary, output_file, 
                              cadence_mask_prob=CADENCE_MASK_PROB,
                              mag_error_std=MAG_ERROR_STD, 
                              binary_params='standard',
                              n_workers=None):
    """
    Generate full dataset using multiprocessing
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of Binary events
        output_file: Where to save
        cadence_mask_prob: Fraction of missing observations
        mag_error_std: Photometric error (magnitudes)
        binary_params: 'standard', 'easy', or 'hard'
        n_workers: Number of CPU workers (None = auto-detect)
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    print("=" * 80)
    print("PARALLEL SIMULATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  PSPL events: {n_pspl}")
    print(f"  Binary events: {n_binary}")
    print(f"  Cadence: {(1-cadence_mask_prob)*100:.0f}% coverage")
    print(f"  Photometric error: {mag_error_std:.3f} mag")
    print(f"  Binary difficulty: {binary_params}")
    print(f"  CPU workers: {n_workers}")
    print("=" * 80)
    print()
    
    # Get binary parameter set
    param_set = BINARY_PARAM_SETS.get(binary_params, BINARY_PARAMS_STANDARD)
    
    # Generate PSPL events in parallel
    print(f"Generating {n_pspl} PSPL events with {n_workers} workers...")
    
    pspl_args = [(i, N_POINTS, mag_error_std, cadence_mask_prob) 
                 for i in range(n_pspl)]
    
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        pspl_results = list(tqdm(
            pool.imap(generate_pspl_event_parallel, pspl_args),
            total=n_pspl,
            desc="PSPL events"
        ))
    
    pspl_data = [r[0] for r in pspl_results]
    pspl_params = [r[1] for r in pspl_results]
    
    # Generate Binary events in parallel
    print(f"\nGenerating {n_binary} Binary events ({binary_params}) with {n_workers} workers...")
    
    binary_args = [(i, N_POINTS, mag_error_std, cadence_mask_prob, param_set)
                   for i in range(n_binary)]
    
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        binary_results = list(tqdm(
            pool.imap(generate_binary_event_parallel, binary_args),
            total=n_binary,
            desc="Binary events"
        ))
    
    # Filter out failed generations
    binary_data = []
    binary_params_list = []
    failed_count = 0
    
    for data, params in binary_results:
        if data is not None:
            binary_data.append(data)
            binary_params_list.append(params)
        else:
            failed_count += 1
    
    if failed_count > 0:
        print(f"\nWarning: {failed_count} binary events failed to generate (will regenerate)")
        
        # Regenerate failed events
        while len(binary_data) < n_binary:
            remaining = n_binary - len(binary_data)
            print(f"Regenerating {remaining} failed events...")
            
            binary_args = [(n_binary + i, N_POINTS, mag_error_std, cadence_mask_prob, param_set)
                          for i in range(remaining)]
            
            with Pool(processes=n_workers, initializer=init_worker) as pool:
                extra_results = list(tqdm(
                    pool.imap(generate_binary_event_parallel, binary_args),
                    total=remaining,
                    desc="Retry Binary"
                ))
            
            for data, params in extra_results:
                if data is not None:
                    binary_data.append(data)
                    binary_params_list.append(params)
                if len(binary_data) >= n_binary:
                    break
    
    # Combine datasets
    print("\nCombining and shuffling...")
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]
    X_binary = np.array(binary_data[:n_binary])[:, :, np.newaxis]
    
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    np.savez_compressed(output_file, 
                       X=X, 
                       y=y,
                       pspl_params=pspl_params,
                       binary_params=binary_params_list)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)
    print(f"Total events: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Output: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pspl', type=int, default=N_PSPL)
    parser.add_argument('--n_binary', type=int, default=N_BINARY)
    parser.add_argument('--output', type=str, default=DATA_DIR + '/events_1M.npz')
    parser.add_argument('--cadence', type=float, default=CADENCE_MASK_PROB,
                       help='Fraction of missing observations (0-1)')
    parser.add_argument('--error', type=float, default=MAG_ERROR_STD,
                       help='Photometric error in magnitudes')
    parser.add_argument('--binary_difficulty', type=str, default='standard',
                       choices=['standard', 'easy', 'hard'],
                       help='Binary event difficulty')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of CPU workers (default: auto-detect)')
    args = parser.parse_args()
    
    simulate_dataset_parallel(
        args.n_pspl, args.n_binary, args.output,
        cadence_mask_prob=args.cadence,
        mag_error_std=args.error,
        binary_params=args.binary_difficulty,
        n_workers=args.n_workers
    )
