#!/usr/bin/env python3
"""
Fixed Binary Microlensing Simulation with Guaranteed Caustic Crossings

Version: 6.2 - Fixed JSON serialization for NumPy types
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
    VBM.Tol = CFG.VBM_TOLERANCE  # Set tolerance
    VBM_AVAILABLE = True
except ImportError:
    VBM = None
    VBM_AVAILABLE = False
    print("WARNING: VBMicrolensing not available. Binary events will be simulated as PSPL!")


@dataclass
class SimulationStats:
    """Track simulation statistics"""
    n_binary_attempts: int = 0
    n_binary_success: int = 0
    n_binary_failed: int = 0
    n_binary_fallback: int = 0
    max_magnifications: List[float] = None
    caustic_fractions: List[float] = None
    
    def __post_init__(self):
        if self.max_magnifications is None:
            self.max_magnifications = []
        if self.caustic_fractions is None:
            self.caustic_fractions = []


def pspl_magnification(u: np.ndarray) -> np.ndarray:
    """Standard PSPL magnification"""
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def generate_pspl_event(
    timestamps: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate PSPL event with proper parameters"""
    
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
    seed: Optional[int] = None,
    max_attempts: int = 10
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    """
    Generate binary event with GUARANTEED caustic crossing
    
    Returns:
        flux: Light curve flux
        params: Event parameters
        success: Whether caustic crossing was achieved
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if not VBM_AVAILABLE:
        # Fall back to PSPL but warn and mark
        flux, params = generate_pspl_event(timestamps, seed)
        params['is_fallback'] = True
        params['actual_class'] = 'pspl'
        params['event_type'] = 'binary_fallback'
        return flux, params, False
    
    # Try multiple times to get a good caustic crossing
    for attempt in range(max_attempts):
        # Sample parameters (using provided ranges)
        s = np.random.uniform(binary_params.get('s_min', 0.7), 
                            binary_params.get('s_max', 1.5))
        q = np.random.uniform(binary_params.get('q_min', 0.01), 
                            binary_params.get('q_max', 0.5))
        u0 = np.random.uniform(binary_params.get('u0_min', 0.001), 
                             binary_params.get('u0_max', 0.05))
        alpha = np.random.uniform(binary_params.get('alpha_min', 0), 
                                binary_params.get('alpha_max', np.pi))
        rho = np.random.uniform(binary_params.get('rho_min', 0.001), 
                              binary_params.get('rho_max', 0.01))
        t0 = np.random.uniform(binary_params.get('t0_min', -20.0), 
                             binary_params.get('t0_max', 20.0))
        tE = np.random.uniform(binary_params.get('tE_min', 30.0), 
                             binary_params.get('tE_max', 100.0))
        baseline = np.random.uniform(CFG.BASELINE_MIN, CFG.BASELINE_MAX)
        
        # Calculate binary light curve
        try:
            # VBMicrolensing expects log parameters
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
            
            # Check for caustic crossing
            max_mag = magnification.max()
            
            if max_mag >= CFG.MIN_BINARY_MAGNIFICATION:
                # SUCCESS! Strong caustic crossing detected
                
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
                
                # Find caustic points
                caustic_mask = magnification > (0.5 * max_mag)
                n_caustic = caustic_mask.sum()
                
                params = {
                    's': s, 'q': q, 'u0': u0, 'alpha': alpha,
                    'rho': rho, 't0': t0, 'tE': tE, 'baseline': baseline,
                    'max_magnification': max_mag,
                    'n_caustic_points': int(n_caustic),  # Convert to Python int
                    'attempt': attempt + 1,
                    'event_type': 'binary',
                    'is_fallback': False,
                    'actual_class': 'binary'
                }
                
                return flux.astype(np.float32), params, True
                
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Binary generation failed after {max_attempts} attempts: {e}")
    
    # All attempts failed - fall back to PSPL
    print(f"WARNING: Could not generate caustic crossing, using PSPL fallback")
    flux, params = generate_pspl_event(timestamps, seed)
    params['is_fallback'] = True
    params['actual_class'] = 'pspl'
    params['event_type'] = 'binary_fallback'
    params['binary_attempts'] = max_attempts
    return flux, params, False


def parallel_worker(args):
    """Worker for parallel generation"""
    idx, timestamps, event_type, binary_params, seed = args
    
    if event_type == 'pspl':
        flux, params = generate_pspl_event(timestamps, seed)
        success = True
    else:
        flux, params, success = generate_binary_event(
            timestamps, binary_params, seed, CFG.MAX_BINARY_ATTEMPTS
        )
    
    return idx, flux, params, success


def generate_dataset(
    n_pspl: int,
    n_binary: int,
    binary_params: Dict,
    num_workers: int = 4,
    seed: int = 42,
    filter_fallbacks: bool = True
) -> Dict:
    """Generate complete dataset with validation"""
    
    np.random.seed(seed)
    
    # Generate timestamps
    timestamps = np.linspace(CFG.TIME_MIN, CFG.TIME_MAX, CFG.N_POINTS)
    
    print(f"Generating {n_pspl} PSPL + {n_binary} Binary events...")
    print(f"Binary parameters: {binary_params.get('u0_min', 0.001):.3f} < u0 < {binary_params.get('u0_max', 0.05):.3f}")
    
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
    all_params = {'pspl': [], 'binary': [], 'binary_fallback': []}
    stats = SimulationStats()
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(parallel_worker, worker_args),
            total=N,
            desc="Generating events"
        ))
    
    # Process results
    fallback_indices = []
    for idx, flux, params, success in results:
        X[idx] = flux
        
        if idx < n_pspl:
            y[idx] = 0  # PSPL
            all_params['pspl'].append(params)
        else:
            if params.get('is_fallback', False):
                # Binary that fell back to PSPL
                fallback_indices.append(idx)
                all_params['binary_fallback'].append(params)
                stats.n_binary_fallback += 1
                y[idx] = 0 if filter_fallbacks else 1  # Mark as PSPL if filtering
            else:
                y[idx] = 1  # Binary
                all_params['binary'].append(params)
            
            if success and 'max_magnification' in params:
                stats.max_magnifications.append(params['max_magnification'])
                stats.n_binary_success += 1
            else:
                stats.n_binary_failed += 1
    
    # Filter out fallbacks if requested
    if filter_fallbacks and len(fallback_indices) > 0:
        print(f"\nFiltering {len(fallback_indices)} fallback events...")
        keep_mask = np.ones(N, dtype=bool)
        keep_mask[fallback_indices] = False
        X = X[keep_mask]
        y = y[keep_mask]
        N = len(y)
    
    # Shuffle data
    perm = np.random.permutation(N)
    X = X[perm]
    y = y[perm]
    
    # Validation statistics
    print("\n" + "="*60)
    print("GENERATION COMPLETE - VALIDATION")
    print("="*60)
    
    if VBM_AVAILABLE and len(stats.max_magnifications) > 0:
        max_mags = np.array(stats.max_magnifications)
        print(f"Binary Events Generated: {stats.n_binary_success}/{n_binary}")
        print(f"Binary Generation Failures: {stats.n_binary_failed}")
        print(f"Binary Fallbacks (PSPL): {stats.n_binary_fallback}")
        print(f"Max Magnification Statistics:")
        print(f"  Mean: {max_mags.mean():.1f}")
        print(f"  Median: {np.median(max_mags):.1f}")
        print(f"  Min: {max_mags.min():.1f}")
        print(f"  Max: {max_mags.max():.1f}")
        print(f"  > 20x: {(max_mags > 20).sum()} ({100*(max_mags > 20).mean():.1f}%)")
        
        if (max_mags > 20).mean() < 0.8:
            print("\n⚠️  WARNING: Less than 80% of binaries have strong caustics!")
            print("   Consider using 'critical' binary parameters for better detection")
    else:
        print("⚠️  VBMicrolensing not available - binaries simulated as PSPL")
    
    # Check class balance
    pspl_count = (y == 0).sum()
    binary_count = (y == 1).sum()
    print(f"\nClass Distribution:")
    print(f"  PSPL: {pspl_count} ({100*pspl_count/N:.1f}%)")
    print(f"  Binary: {binary_count} ({100*binary_count/N:.1f}%)")
    
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
            'vbm_available': VBM_AVAILABLE,
            'seed': seed,
            'filter_fallbacks': filter_fallbacks,
            'n_fallbacks': stats.n_binary_fallback,
            'config': {
                'MIN_BINARY_MAGNIFICATION': CFG.MIN_BINARY_MAGNIFICATION,
                'MAG_ERROR_STD': CFG.MAG_ERROR_STD,
                'CADENCE_MASK_PROB': CFG.CADENCE_MASK_PROB,
            }
        }
    }
    
    return dataset


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_dataset(dataset: Dict, output_path: str):
    """Save dataset to NPZ file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ with all data
    save_dict = {
        'X': dataset['X'],
        'y': dataset['y'],
        'timestamps': dataset['timestamps'],
        'perm': dataset['perm'],
        'meta_json': json.dumps(convert_numpy_types(dataset['metadata'])),
        'stats_json': json.dumps(convert_numpy_types(dataset['stats'])),
    }
    
    # Save parameters if requested
    if dataset['params']:
        save_dict['params_pspl_json'] = json.dumps(convert_numpy_types(dataset['params']['pspl']))
        save_dict['params_binary_json'] = json.dumps(convert_numpy_types(dataset['params']['binary']))
        if dataset['params']['binary_fallback']:
            save_dict['params_binary_fallback_json'] = json.dumps(convert_numpy_types(dataset['params']['binary_fallback']))
    
    np.savez_compressed(output_path, **save_dict)
    print(f"\nDataset saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate microlensing dataset with caustic validation")
    parser.add_argument('--n_pspl', type=int, default=5000)
    parser.add_argument('--n_binary', type=int, default=5000)
    parser.add_argument('--binary_params', choices=list(CFG.BINARY_PARAM_SETS.keys()), 
                       default='critical', help='Binary parameter set')
    parser.add_argument('--output', type=str, default='../data/raw/dataset.npz')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_params', action='store_true', help='Save event parameters')
    parser.add_argument('--keep_fallbacks', action='store_true', help='Keep fallback events as binaries')
    
    args = parser.parse_args()
    
    # Get binary parameters
    binary_params = CFG.BINARY_PARAM_SETS[args.binary_params]
    
    print("="*60)
    print("BINARY MICROLENSING SIMULATION v6.2")
    print("="*60)
    print(f"Configuration:")
    print(f"  PSPL events: {args.n_pspl}")
    print(f"  Binary events: {args.n_binary}")
    print(f"  Binary set: {args.binary_params}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Filter fallbacks: {not args.keep_fallbacks}")
    print(f"  VBMicrolensing: {'Available' if VBM_AVAILABLE else 'NOT AVAILABLE'}")
    
    if not VBM_AVAILABLE:
        print("\n" + "!"*60)
        print("CRITICAL WARNING: VBMicrolensing not installed!")
        print("Binary events will be simulated as PSPL!")
        print("Install with: pip install VBMicrolensing")
        print("!"*60 + "\n")
    
    # Generate dataset
    dataset = generate_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_params=binary_params,
        num_workers=args.num_workers,
        seed=args.seed,
        filter_fallbacks=not args.keep_fallbacks
    )
    
    # Save dataset
    save_dataset(dataset, args.output)
    
    print("\n✅ Simulation complete!")


if __name__ == "__main__":
    main()
