"""
Parallelized flexible microlensing event simulation
Uses multiprocessing to speed up generation significantly
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_experiments import (
    BINARY_REGIMES, PSPL_BASELINE_MIN, PSPL_BASELINE_MAX,
    PSPL_T0_MIN, PSPL_T0_MAX, PSPL_U0_MIN, PSPL_U0_MAX,
    PSPL_TE_MIN, PSPL_TE_MAX, BINARY_ALPHA_MIN, BINARY_ALPHA_MAX,
    BINARY_TE_MIN, BINARY_TE_MAX, BINARY_T0_MIN, BINARY_T0_MAX,
    VBM_REL_TOL, VBM_TOL, PAD_VALUE
)

def init_worker():
    """Initialize VBMicrolensing for each worker process"""
    global VBM
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.RelTol = VBM_REL_TOL
    VBM.Tol = VBM_TOL

def generate_pspl_worker(args):
    """Generate single PSPL event (worker function)"""
    event_id, n_observations, time_min, time_max, mag_error_std, cadence_mask_prob = args
    
    np.random.seed(42 + event_id)
    
    timestamps = np.linspace(time_min, time_max, n_observations)
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

def generate_binary_worker(args):
    """Generate single Binary event (worker function)"""
    event_id, n_points, time_min, time_max, regime_params, mag_error_std, cadence_mask_prob = args
    
    np.random.seed(42 + 1000000 + event_id)
    
    timestamps = np.linspace(time_min, time_max, n_points)
    
    # Sample parameters from regime-specific ranges
    s = np.random.uniform(regime_params['s_min'], regime_params['s_max'])
    q = np.random.uniform(regime_params['q_min'], regime_params['q_max'])
    rho = np.random.uniform(regime_params['rho_min'], regime_params['rho_max'])
    u0 = np.random.uniform(regime_params['u0_min'], regime_params['u0_max'])
    
    alpha = np.random.uniform(BINARY_ALPHA_MIN, BINARY_ALPHA_MAX)
    tE = np.random.uniform(BINARY_TE_MIN, BINARY_TE_MAX)
    t0 = np.random.uniform(BINARY_T0_MIN, BINARY_T0_MAX)

    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    try:
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
        return None, None

def simulate_dataset_parallel(n_pspl, n_binary, 
                              binary_regime='mixed_binary',
                              n_points=1500,
                              time_min=0,
                              time_max=1000,
                              mag_error_std=0.10,
                              cadence_mask_prob=0.20,
                              output_file='events.npz',
                              n_workers=None):
    """
    Generate full dataset using multiprocessing
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 32)  # Cap at 32 to avoid overload
    
    print("=" * 80)
    print("PARALLEL SIMULATION")
    print("=" * 80)
    print(f"PSPL events: {n_pspl}")
    print(f"Binary events: {n_binary}")
    print(f"Binary regime: {binary_regime}")
    regime = BINARY_REGIMES[binary_regime]
    print(f"  Description: {regime['description']}")
    print(f"  s range: [{regime['s_min']:.2f}, {regime['s_max']:.2f}]")
    print(f"  q range: [{regime['q_min']:.2f}, {regime['q_max']:.2f}]")
    print(f"  u0 range: [{regime['u0_min']:.2f}, {regime['u0_max']:.2f}]")
    print(f"Time points: {n_points}")
    print(f"Photometric error: {mag_error_std:.3f} mag")
    print(f"Missing observations: {cadence_mask_prob*100:.1f}%")
    print(f"CPU workers: {n_workers}")
    print("=" * 80)
    
    # Generate PSPL events in parallel
    print(f"\nGenerating {n_pspl} PSPL events with {n_workers} workers...")
    
    pspl_args = [(i, n_points, time_min, time_max, mag_error_std, cadence_mask_prob) 
                 for i in range(n_pspl)]
    
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        pspl_results = list(tqdm(
            pool.imap(generate_pspl_worker, pspl_args),
            total=n_pspl,
            desc="PSPL"
        ))
    
    pspl_data = [r[0] for r in pspl_results]
    pspl_params = [r[1] for r in pspl_results]
    
    # Generate Binary events in parallel
    print(f"\nGenerating {n_binary} Binary events ({binary_regime}) with {n_workers} workers...")
    
    regime_params = BINARY_REGIMES[binary_regime]
    binary_args = [(i, n_points, time_min, time_max, regime_params, mag_error_std, cadence_mask_prob)
                   for i in range(n_binary)]
    
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        binary_results = list(tqdm(
            pool.imap(generate_binary_worker, binary_args),
            total=n_binary,
            desc="Binary"
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
        print(f"\nWarning: {failed_count} binary events failed to generate")
        # Regenerate failed events
        while len(binary_data) < n_binary:
            remaining = n_binary - len(binary_data)
            print(f"Regenerating {remaining} failed events...")
            
            extra_args = [(n_binary + i, n_points, time_min, time_max, regime_params, 
                          mag_error_std, cadence_mask_prob)
                         for i in range(remaining)]
            
            with Pool(processes=n_workers, initializer=init_worker) as pool:
                extra_results = list(tqdm(
                    pool.imap(generate_binary_worker, extra_args),
                    total=remaining,
                    desc="Retry"
                ))
            
            for data, params in extra_results:
                if data is not None:
                    binary_data.append(data)
                    binary_params_list.append(params)
                if len(binary_data) >= n_binary:
                    break
    
    # Combine and shuffle
    print("\nCombining and shuffling...")
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]
    X_binary = np.array(binary_data[:n_binary])[:, :, np.newaxis]
    
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save metadata
    metadata = {
        'n_pspl': n_pspl,
        'n_binary': n_binary,
        'binary_regime': binary_regime,
        'binary_regime_description': regime['description'],
        'n_points': n_points,
        'time_min': time_min,
        'time_max': time_max,
        'mag_error_std': mag_error_std,
        'cadence_mask_prob': cadence_mask_prob,
        'n_workers': n_workers
    }
    
    metadata_file = output_file.replace('.npz', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaving to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    np.savez_compressed(output_file, 
                       X=X, 
                       y=y,
                       pspl_params=pspl_params,
                       binary_params=binary_params_list,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pspl', type=int, default=100000)
    parser.add_argument('--n_binary', type=int, default=100000)
    parser.add_argument('--binary_regime', type=str, default='mixed_binary',
                       choices=list(BINARY_REGIMES.keys()))
    parser.add_argument('--mag_error', type=float, default=0.10)
    parser.add_argument('--cadence', type=float, default=0.20)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of CPU workers (default: auto-detect)')
    args = parser.parse_args()
    
    simulate_dataset_parallel(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_regime=args.binary_regime,
        mag_error_std=args.mag_error,
        cadence_mask_prob=args.cadence,
        output_file=args.output,
        n_workers=args.n_workers
    )
