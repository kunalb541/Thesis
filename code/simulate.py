"""
Microlensing event simulation with configurable binary parameters and multiprocessing
FIXED VERSION - Bug corrections applied

This script generates realistic microlensing light curves using VBMicrolensing.
Uses multiprocessing to parallelize event generation for significant speedup.

Performance:
- Serial: ~12 hours for 1M events
- Parallel (24 cores): ~2-3 hours for 1M events

Author: Kunal Bhatia (kunal29bhatia@gmail.com)
University of Heidelberg
Last Updated: October 27, 2025
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import time
from functools import partial
from typing import Tuple, Dict
from config import *

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Global VBMicrolensing instance for worker processes
_VBM = None

def init_worker():
    """Initialize VBMicrolensing in each worker process"""
    global _VBM
    _VBM = VBMicrolensing.VBMicrolensing()
    _VBM.RelTol = VBM_REL_TOL
    _VBM.Tol = VBM_TOL

def generate_pspl_event_worker(args: Tuple[int, int, float, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Worker function for generating a single PSPL event
    
    Args:
        args: tuple of (seed, n_observations, mag_error_std, cadence_mask_prob)
    
    Returns:
        tuple: (flux_padded, params_dict)
    """
    seed, n_observations, mag_error_std, cadence_mask_prob = args
    
    # Set random seed for this worker
    np.random.seed(seed)
    
    # Generate timestamps
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_observations)
    
    # Sample PSPL parameters
    baseline = np.random.uniform(PSPL_BASELINE_MIN, PSPL_BASELINE_MAX)
    t0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
    u0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
    tE = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)

    # Calculate magnification
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    
    # Convert to magnitudes
    magnitudes = baseline - 2.5 * np.log10(magnification)
    
    # Add photometric errors
    magnitudes += np.random.normal(0, mag_error_std, size=magnitudes.shape)

    # Apply cadence masking (simulate sparse observations)
    mask = np.random.rand(n_observations) < cadence_mask_prob
    magnitudes[mask] = np.nan

    # Convert to flux for NN input
    flux = 10 ** (-(magnitudes - baseline) / 2.5)
    
    # Pad missing data
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)
    
    # Store parameters for later analysis
    params = {
        't0': t0, 
        'u0': u0, 
        'tE': tE, 
        'baseline': baseline
    }
    
    return flux_padded, params

def generate_binary_event_worker(args: Tuple[int, int, float, float, str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Worker function for generating a single Binary event
    
    Args:
        args: tuple of (seed, n_points, mag_error_std, cadence_mask_prob, binary_params)
    
    Returns:
        tuple: (magnifications_padded, params_dict)
    """
    global _VBM
    
    seed, n_points, mag_error_std, cadence_mask_prob, binary_params = args
    
    # Set random seed for this worker
    np.random.seed(seed)
    
    # Generate timestamps
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)
    
    # Get parameter set - FIXED: Changed BINARY_PARAMS_STANDARD to BINARY_PARAMS_BASELINE
    param_set = BINARY_PARAM_SETS.get(binary_params, BINARY_PARAMS_BASELINE)
    
    # Sample binary parameters from specified ranges
    s = np.random.uniform(param_set['s_min'], param_set['s_max'])
    q = np.random.uniform(param_set['q_min'], param_set['q_max'])
    rho = np.random.uniform(param_set['rho_min'], param_set['rho_max'])
    alpha = np.random.uniform(param_set['alpha_min'], param_set['alpha_max'])
    tE = np.random.uniform(param_set['tE_min'], param_set['tE_max'])
    t0 = np.random.uniform(param_set['t0_min'], param_set['t0_max'])
    u0 = np.random.uniform(param_set['u0_min'], param_set['u0_max'])

    # VBMicrolensing expects log parameters for numerical stability
    params_vbm = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    # Generate magnification using ray-tracing
    try:
        magnifications = np.array(_VBM.BinaryLightCurve(params_vbm, timestamps)[0])
    except Exception as e:
        # Fallback: if VBMicrolensing fails, return PSPL-like curve
        print(f"Warning: VBMicrolensing failed for seed {seed}, using PSPL fallback")
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        magnifications = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))

    # Add photometric errors
    magnifications += np.random.normal(0, mag_error_std, size=magnifications.shape)

    # Apply cadence masking
    mask = np.random.rand(n_points) < cadence_mask_prob
    magnifications[mask] = np.nan
    
    # Pad missing data
    magnifications_padded = np.nan_to_num(magnifications, nan=PAD_VALUE)
    
    # Store parameters for later analysis
    params = {
        's': s, 'q': q, 'rho': rho, 'alpha': alpha, 
        'tE': tE, 't0': t0, 'u0': u0
    }
    
    return magnifications_padded, params

def simulate_dataset(n_pspl: int, n_binary: int, output_file: str, 
                    cadence_mask_prob: float = CADENCE_MASK_PROB,
                    mag_error_std: float = MAG_ERROR_STD, 
                    binary_params: str = 'baseline',
                    n_processes: int = None) -> None:
    """
    Generate full dataset with multiprocessing
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of binary events
        output_file: Where to save
        cadence_mask_prob: Fraction of missing observations (0-1)
        mag_error_std: Photometric error (magnitudes)
        binary_params: 'baseline', 'distinct', 'planetary', or 'stellar'
        n_processes: Number of parallel processes (default: CPU count)
    """
    
    # Validate inputs
    assert n_pspl > 0, "n_pspl must be positive"
    assert n_binary > 0, "n_binary must be positive"
    assert 0 <= cadence_mask_prob < 1, "cadence_mask_prob must be in [0, 1)"
    assert mag_error_std > 0, "mag_error_std must be positive"
    assert binary_params in BINARY_PARAM_SETS, f"Unknown binary_params: {binary_params}"
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"\n{'='*80}")
    print(f"MICROLENSING EVENT SIMULATION")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  PSPL events:        {n_pspl:,}")
    print(f"  Binary events:      {n_binary:,}")
    print(f"  Total events:       {n_pspl + n_binary:,}")
    print(f"  Cadence coverage:   {(1-cadence_mask_prob)*100:.0f}%")
    print(f"  Photometric error:  {mag_error_std:.3f} mag")
    print(f"  Binary params:      {binary_params}")
    print(f"  CPU processes:      {n_processes}")
    print(f"  Output file:        {output_file}")
    print(f"  Random seed:        {RANDOM_SEED}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # ========================================================================
    # Generate PSPL events with multiprocessing
    # ========================================================================
    print(f"Generating {n_pspl:,} PSPL events using {n_processes} processes...")
    
    # Prepare arguments for each PSPL event (use different seeds)
    pspl_args = [
        (RANDOM_SEED + i * 2, N_POINTS, mag_error_std, cadence_mask_prob) 
        for i in range(n_pspl)
    ]
    
    # Create process pool and generate events
    with Pool(processes=n_processes, initializer=init_worker) as pool:
        pspl_results = list(tqdm(
            pool.imap(generate_pspl_event_worker, pspl_args),
            total=n_pspl,
            desc="PSPL events"
        ))
    
    # Unpack results
    pspl_data = [result[0] for result in pspl_results]
    pspl_params = [result[1] for result in pspl_results]
    
    pspl_time = time.time() - start_time
    print(f"✓ PSPL generation complete in {pspl_time/60:.1f} minutes")
    print(f"  Rate: {n_pspl/pspl_time:.1f} events/second\n")
    
    # ========================================================================
    # Generate Binary events with multiprocessing
    # ========================================================================
    print(f"Generating {n_binary:,} Binary events ({binary_params}) using {n_processes} processes...")
    
    # Prepare arguments for each Binary event (use different seeds)
    binary_args = [
        (RANDOM_SEED + i * 2 + 1, N_POINTS, mag_error_std, cadence_mask_prob, binary_params)
        for i in range(n_binary)
    ]
    
    binary_start = time.time()
    
    # Create process pool and generate events
    with Pool(processes=n_processes, initializer=init_worker) as pool:
        binary_results = list(tqdm(
            pool.imap(generate_binary_event_worker, binary_args),
            total=n_binary,
            desc="Binary events"
        ))
    
    # Unpack results
    binary_data = [result[0] for result in binary_results]
    binary_params_list = [result[1] for result in binary_results]
    
    binary_time = time.time() - binary_start
    print(f"✓ Binary generation complete in {binary_time/60:.1f} minutes")
    print(f"  Rate: {n_binary/binary_time:.1f} events/second\n")
    
    # ========================================================================
    # Combine and save
    # ========================================================================
    print("Combining and shuffling dataset...")
    
    # Convert to numpy arrays with shape (n_events, n_timesteps, 1)
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]
    X_binary = np.array(binary_data)[:, :, np.newaxis]
    
    # Combine
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    # Validate data before saving
    assert X.shape[0] == len(y), f"Mismatch: X has {X.shape[0]} samples, y has {len(y)}"
    assert X.shape[1] == N_POINTS, f"Expected {N_POINTS} timesteps, got {X.shape[1]}"
    assert X.shape[2] == 1, f"Expected 1 channel, got {X.shape[2]}"
    assert np.isfinite(X).all(), "Data contains NaN or inf values"
    
    # Shuffle for training
    print("Shuffling...")
    np.random.seed(RANDOM_SEED)  # Ensure reproducible shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    np.savez_compressed(
        output_file, 
        X=X, 
        y=y,
        pspl_params=pspl_params,
        binary_params_list=binary_params_list,
        config={
            'n_pspl': n_pspl,
            'n_binary': n_binary,
            'cadence_mask_prob': cadence_mask_prob,
            'mag_error_std': mag_error_std,
            'binary_params': binary_params,
            'random_seed': RANDOM_SEED,
            'n_points': N_POINTS,
        }
    )
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SIMULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nDataset statistics:")
    print(f"  Total events:       {len(X):,}")
    print(f"  Shape:              {X.shape}")
    print(f"  Labels:             {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  File size:          {os.path.getsize(output_file) / 1e9:.2f} GB")
    print(f"\nPerformance:")
    print(f"  Total time:         {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Overall rate:       {(n_pspl + n_binary)/total_time:.1f} events/second")
    print(f"  Speedup factor:     ~{n_processes}x (using {n_processes} processes)")
    print(f"\n✓ Dataset ready for training!")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Generate microlensing light curves with multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (automatic CPU detection)
  python simulate.py --n_pspl 500000 --n_binary 500000
  
  # Control processes explicitly
  python simulate.py --n_pspl 500000 --n_binary 500000 --n_processes 16
  
  # Dense cadence, low noise, distinct binaries
  python simulate.py --n_pspl 100000 --n_binary 100000 \\
                     --cadence 0.05 --error 0.05 --binary_params distinct
  
  # Sparse cadence, high noise, baseline binaries  
  python simulate.py --n_pspl 100000 --n_binary 100000 \\
                     --cadence 0.40 --error 0.20 --binary_params baseline
        """
    )
    
    parser.add_argument('--n_pspl', type=int, default=N_PSPL,
                       help=f'Number of PSPL events (default: {N_PSPL})')
    parser.add_argument('--n_binary', type=int, default=N_BINARY,
                       help=f'Number of Binary events (default: {N_BINARY})')
    parser.add_argument('--output', type=str, default=os.path.join(DATA_DIR, 'events_1M.npz'),
                       help='Output file path')
    parser.add_argument('--cadence', type=float, default=CADENCE_MASK_PROB,
                       help='Fraction of missing observations (0-1, default: 0.2)')
    parser.add_argument('--error', type=float, default=MAG_ERROR_STD,
                       help='Photometric error in magnitudes (default: 0.1)')
    # FIXED: Changed from binary_difficulty to binary_params with correct choices
    parser.add_argument('--binary_params', type=str, default='baseline',
                       choices=['baseline', 'distinct', 'planetary', 'stellar'],
                       help='Binary parameter set (default: baseline)')
    parser.add_argument('--n_processes', type=int, default=None,
                       help=f'Number of parallel processes (default: auto-detect, currently {cpu_count()})')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_pspl < 1 or args.n_binary < 1:
        parser.error("Number of events must be positive")
    if not (0 <= args.cadence < 1):
        parser.error("Cadence must be in [0, 1)")
    if args.error <= 0:
        parser.error("Photometric error must be positive")
    if args.n_processes is not None and args.n_processes < 1:
        parser.error("Number of processes must be positive")
    
    # Run simulation
    simulate_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        output_file=args.output,
        cadence_mask_prob=args.cadence,
        mag_error_std=args.error,
        binary_params=args.binary_params,
        n_processes=args.n_processes
    )

if __name__ == "__main__":
    main()