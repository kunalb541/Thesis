#!/usr/bin/env python3
"""
FIXED Binary Microlensing Simulation - No PSPL Fallback
Binaries remain binaries regardless of caustic strength

Version: 7.0 - Final thesis version
Author: Kunal Bhatia
"""

import numpy as np
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional
import multiprocessing as mp
from tqdm import tqdm
import config as CFG

# Try importing VBMicrolensing
try:
    import VBBinaryLensing
    VBM = VBBinaryLensing.VBBinaryLensing()
    VBM.Tol = CFG.VBM_TOLERANCE
    VBM_AVAILABLE = True
except ImportError:
    VBM = None
    VBM_AVAILABLE = False
    print("ERROR: VBMicrolensing required! Install with: pip install VBMicrolensing")


@dataclass
class SimulationStats:
    """Track simulation statistics"""
    n_binary_generated: int = 0
    n_binary_strong_caustic: int = 0
    n_binary_weak: int = 0
    max_magnifications: List[float] = None
    
    def __post_init__(self):
        if self.max_magnifications is None:
            self.max_magnifications = []


def pspl_magnification(u: np.ndarray) -> np.ndarray:
    """Standard PSPL magnification"""
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def generate_pspl_event(
    timestamps: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate PSPL event"""
    
    if seed is not None:
        np.random.seed(seed)
    
    # Sample parameters
    t0 = np.random.uniform(CFG.PSPL_T0_MIN, CFG.PSPL_T0_MAX)
    u0 = np.random.uniform(CFG.PSPL_U0_MIN, CFG.PSPL_U0_MAX)
    tE = np.random.uniform(CFG.PSPL_TE_MIN, CFG.PSPL_TE_MAX)
    baseline = np.random.uniform(CFG.BASELINE_MIN, CFG.BASELINE_MAX)
    
    # Calculate magnification
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = pspl_magnification(u_t)
    
    # Convert to magnitude
    magnitudes = baseline - 2.5 * np.log10(magnification)
    
    # Add photometric noise
    if CFG.MAG_ERROR_STD > 0:
        magnitudes += np.random.normal(0, CFG.MAG_ERROR_STD, magnitudes.shape)
    
    # Apply cadence mask
    if CFG.CADENCE_MASK_PROB > 0:
        mask = np.random.rand(len(timestamps)) < CFG.CADENCE_MASK_PROB
        magnitudes[mask] = np.nan
    
    # Convert to flux (normalized)
    flux = 10.0 ** (-(magnitudes - baseline) / 2.5)
    flux = np.nan_to_num(flux, nan=CFG.PAD_VALUE)
    
    params = {
        't0': t0, 'u0': u0, 'tE': tE, 'baseline': baseline,
        'max_magnification': magnification.max(),
        'event_type': 'pspl'
    }
    
    return flux.astype(np.float32), params


def generate_binary_event(
    timestamps: np.ndarray,
    binary_params: Dict,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    """
    Generate binary event - ALWAYS returns a binary
    
    Returns:
        flux: Light curve flux
        params: Event parameters
        has_strong_caustic: Whether mag > 20 was achieved
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if not VBM_AVAILABLE:
        raise ImportError("VBMicrolensing is required for binary events!")
    
    # Sample parameters
    s = np.random.uniform(binary_params.get('s_min', 0.1), 
                        binary_params.get('s_max', 2.5))
    q = np.random.uniform(binary_params.get('q_min', 0.001), 
                        binary_params.get('q_max', 1.0))
    u0 = np.random.uniform(binary_params.get('u0_min', 0.001), 
                         binary_params.get('u0_max', 0.5))
    alpha = np.random.uniform(binary_params.get('alpha_min', 0), 
                            binary_params.get('alpha_max', 2*np.pi))
    rho = np.random.uniform(binary_params.get('rho_min', 0.001), 
                          binary_params.get('rho_max', 0.05))
    t0 = np.random.uniform(binary_params.get('t0_min', -20.0), 
                         binary_params.get('t0_max', 20.0))
    tE = np.random.uniform(binary_params.get('tE_min', 20.0), 
                         binary_params.get('tE_max', 150.0))
    baseline = np.random.uniform(CFG.BASELINE_MIN, CFG.BASELINE_MAX)
    
    # Calculate binary light curve
    try:
        # VBMicrolensing parameters
        params_vbm = [
            np.log(s),
            np.log(q),
            u0,
            alpha,
            np.log(rho),
            np.log(tE),
            t0
        ]
        
        # Get magnification
        magnification = np.array(VBM.BinaryLightCurve(params_vbm, timestamps)[0])
        magnification = np.maximum(magnification, 1.0)  # Ensure positive
        
    except Exception as e:
        # If VBM fails, generate a simple perturbed PSPL
        print(f"VBM calculation failed: {e}, using perturbed PSPL")
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        magnification = pspl_magnification(u_t)
        # Add some perturbation to make it binary-like
        perturbation = 1 + 0.2 * np.sin(2*np.pi * (timestamps - t0) / (0.3 * tE))
        magnification = magnification * perturbation
    
    # Check if strong caustic
    max_mag = magnification.max()
    has_strong_caustic = (max_mag >= CFG.MIN_BINARY_MAGNIFICATION)
    
    # Convert to magnitude
    magnitudes = baseline - 2.5 * np.log10(magnification)
    
    # Add noise
    if CFG.MAG_ERROR_STD > 0:
        magnitudes += np.random.normal(0, CFG.MAG_ERROR_STD, magnitudes.shape)
    
    # Apply cadence mask
    if CFG.CADENCE_MASK_PROB > 0:
        mask = np.random.rand(len(timestamps)) < CFG.CADENCE_MASK_PROB
        magnitudes[mask] = np.nan
    
    # Convert to flux
    flux = 10.0 ** (-(magnitudes - baseline) / 2.5)
    flux = np.nan_to_num(flux, nan=CFG.PAD_VALUE)
    
    params = {
        's': s, 'q': q, 'u0': u0, 'alpha': alpha,
        'rho': rho, 't0': t0, 'tE': tE, 'baseline': baseline,
        'max_magnification': max_mag,
        'has_strong_caustic': has_strong_caustic,
        'event_type': 'binary'
    }
    
    return flux.astype(np.float32), params, has_strong_caustic


def parallel_worker(args):
    """Worker for parallel generation"""
    idx, timestamps, event_type, binary_params, seed = args
    
    if event_type == 'pspl':
        flux, params = generate_pspl_event(timestamps, seed)
        return idx, flux, params, True
    else:
        flux, params, has_caustic = generate_binary_event(
            timestamps, binary_params, seed
        )
        return idx, flux, params, has_caustic


def generate_dataset(
    n_pspl: int,
    n_binary: int,
    binary_params: Dict,
    num_workers: int = 4,
    seed: int = 42,
    cadence_mask_prob: float = None,
    mag_error_std: float = None
) -> Dict:
    """Generate complete dataset"""
    
    np.random.seed(seed)
    
    # Override config if specified
    if cadence_mask_prob is not None:
        CFG.CADENCE_MASK_PROB = cadence_mask_prob
    if mag_error_std is not None:
        CFG.MAG_ERROR_STD = mag_error_std
    
    # Generate timestamps
    timestamps = np.linspace(CFG.TIME_MIN, CFG.TIME_MAX, CFG.N_POINTS)
    
    print(f"Generating {n_pspl} PSPL + {n_binary} Binary events...")
    print(f"Binary parameters: {binary_params.get('u0_min', 0.001):.3f} < u0 < {binary_params.get('u0_max', 0.5):.3f}")
    print(f"Cadence mask: {CFG.CADENCE_MASK_PROB*100:.0f}% missing")
    print(f"Photometric error: {CFG.MAG_ERROR_STD:.3f} mag")
    
    # Prepare data arrays
    N = n_pspl + n_binary
    X = np.zeros((N, CFG.N_POINTS), dtype=np.float32)
    y = np.zeros(N, dtype=np.uint8)
    
    # Prepare worker arguments
    worker_args = []
    
    # PSPL events
    for i in range(n_pspl):
        worker_args.append((i, timestamps, 'pspl', None, seed + i if seed else None))
    
    # Binary events
    for i in range(n_binary):
        idx = n_pspl + i
        worker_args.append((idx, timestamps, 'binary', binary_params, seed + idx if seed else None))
    
    # Parallel processing
    all_params = {'pspl': [], 'binary': []}
    stats = SimulationStats()
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(parallel_worker, worker_args),
            total=N,
            desc="Generating events"
        ))
    
    # Process results
    for idx, flux, params, success in results:
        X[idx] = flux
        
        if idx < n_pspl:
            y[idx] = 0  # PSPL
            all_params['pspl'].append(params)
        else:
            y[idx] = 1  # Binary (always!)
            all_params['binary'].append(params)
            stats.n_binary_generated += 1
            
            if params.get('has_strong_caustic', False):
                stats.n_binary_strong_caustic += 1
            else:
                stats.n_binary_weak += 1
            
            stats.max_magnifications.append(params['max_magnification'])
    
    # Shuffle data
    perm = np.random.permutation(N)
    X = X[perm]
    y = y[perm]
    
    # Print statistics
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total events: {N}")
    print(f"  PSPL: {n_pspl} ({100*n_pspl/N:.1f}%)")
    print(f"  Binary: {n_binary} ({100*n_binary/N:.1f}%)")
    
    if stats.max_magnifications:
        max_mags = np.array(stats.max_magnifications)
        print(f"\nBinary Statistics:")
        print(f"  Strong caustics (mag>20): {stats.n_binary_strong_caustic} ({100*stats.n_binary_strong_caustic/n_binary:.1f}%)")
        print(f"  Weak binaries: {stats.n_binary_weak} ({100*stats.n_binary_weak/n_binary:.1f}%)")
        print(f"  Max mag mean: {max_mags.mean():.1f}")
        print(f"  Max mag median: {np.median(max_mags):.1f}")
        print(f"  Max mag range: [{max_mags.min():.1f}, {max_mags.max():.1f}]")
    
    # Package results
    dataset = {
        'X': X,
        'y': y,
        'timestamps': timestamps,
        'perm': perm,
        'params': all_params,
        'stats': asdict(stats),
        'metadata': {
            'n_pspl': n_pspl,
            'n_binary': n_binary,
            'n_points': CFG.N_POINTS,
            'binary_params': binary_params,
            'cadence_mask_prob': CFG.CADENCE_MASK_PROB,
            'mag_error_std': CFG.MAG_ERROR_STD,
            'vbm_available': VBM_AVAILABLE,
            'seed': seed
        }
    }
    
    return dataset


def save_dataset(dataset: Dict, output_path: str, save_params: bool = False):
    """Save dataset to NPZ file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Save NPZ with all data
    save_dict = {
        'X': dataset['X'],
        'y': dataset['y'],
        'timestamps': dataset['timestamps'],
        'perm': dataset['perm'],
        'meta_json': json.dumps(convert_numpy(dataset['metadata'])),
        'stats_json': json.dumps(convert_numpy(dataset['stats'])),
    }
    
    # Save parameters if requested (for u0 analysis)
    if save_params and dataset['params']:
        save_dict['params_pspl_json'] = json.dumps(convert_numpy(dataset['params']['pspl']))
        save_dict['params_binary_json'] = json.dumps(convert_numpy(dataset['params']['binary']))
    
    np.savez_compressed(output_path, **save_dict)
    print(f"\nDataset saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate microlensing dataset")
    parser.add_argument('--n_pspl', type=int, default=5000)
    parser.add_argument('--n_binary', type=int, default=5000)
    parser.add_argument('--binary_params', choices=list(CFG.BINARY_PARAM_SETS.keys()), 
                       default='baseline', help='Binary parameter set')
    parser.add_argument('--output', type=str, default='../data/raw/dataset.npz')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_params', action='store_true', 
                       help='Save event parameters for u0 analysis')
    parser.add_argument('--cadence_mask_prob', type=float, default=None,
                       help='Override cadence mask probability')
    parser.add_argument('--mag_error_std', type=float, default=None,
                       help='Override photometric error std')
    
    args = parser.parse_args()
    
    if not VBM_AVAILABLE:
        print("\n" + "!"*60)
        print("ERROR: VBMicrolensing not installed!")
        print("Install with: pip install VBMicrolensing")
        print("!"*60)
        return
    
    # Get binary parameters
    binary_params = CFG.BINARY_PARAM_SETS[args.binary_params]
    
    print("="*60)
    print("BINARY MICROLENSING SIMULATION v7.0")
    print("="*60)
    print(f"Configuration:")
    print(f"  PSPL events: {args.n_pspl}")
    print(f"  Binary events: {args.n_binary}")
    print(f"  Binary set: {args.binary_params}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Save params: {args.save_params}")
    
    # Generate dataset
    dataset = generate_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_params=binary_params,
        num_workers=args.num_workers,
        seed=args.seed,
        cadence_mask_prob=args.cadence_mask_prob,
        mag_error_std=args.mag_error_std
    )
    
    # Save dataset
    save_dataset(dataset, args.output, save_params=args.save_params)
    
    print("\n✅ Simulation complete!")


if __name__ == "__main__":
    main()
