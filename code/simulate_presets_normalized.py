#!/usr/bin/env python3
"""
FINAL FIXED VERSION: Proper baseline normalization for both PSPL and Binary
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count

try:
    import VBBinaryLensing as vb
    VBM_AVAILABLE = True
except ImportError:
    VBM_AVAILABLE = False

try:
    import config as CFG
except ImportError:
    class CFG:
        TIME_MIN = -100
        TIME_MAX = 100
        N_POINTS = 1500

from preset_sampler import sample_distinct_binary_params


def normalize_light_curve(flux):
    """
    Normalize light curve so baseline (unmagnified) flux = 1.0
    
    Uses 10th percentile as baseline estimate to be robust to noise
    This ensures both PSPL and Binary events have same baseline distribution
    """
    valid = flux > 0
    if valid.sum() < 10:
        return flux
    
    # Use low percentile as baseline (avoids being affected by peaks)
    baseline = np.percentile(flux[valid], 10)
    
    if baseline > 0.01:  # Sanity check
        flux = flux / baseline
    
    return flux


def generate_pspl_light_curve(u0, tE, t0, times, mag_error_std, rng, mask=None):
    """Generate PSPL light curve with proper normalization"""
    u = np.sqrt(u0**2 + ((times - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    flux = A
    
    # CRITICAL: Normalize baseline to 1.0
    flux = normalize_light_curve(flux)
    
    # Add noise in magnitude space
    mag = -2.5 * np.log10(np.maximum(flux, 1e-10))
    mag_noise = rng.normal(0, mag_error_std, len(mag))
    mag_with_noise = mag + mag_noise
    flux = 10 ** (mag_with_noise / -2.5)
    
    if mask is not None:
        flux[mask] = -1.0
    
    return flux


def generate_binary_with_preset_worker(args):
    """Worker function for parallel binary generation with normalization"""
    idx, params, times, mag_error_std, seed, vbb_tol = args
    
    vbm = vb.VBBinaryLensing()
    vbm.Tol = vbb_tol
    rng = np.random.RandomState(seed + idx)
    
    vb_params = [
        np.log10(params['s']),
        np.log10(params['q']),
        params['u0'],
        params['alpha'],
        np.log10(params['rho']),
        np.log10(params['tE']),
        params['t0']
    ]
    
    try:
        result = vbm.BinaryLightCurve(vb_params, times)
        magnification = np.array(result[0])
        flux = magnification
        
        # CRITICAL: Normalize baseline to 1.0 (same as PSPL!)
        flux = normalize_light_curve(flux)
        
        # Add photometric noise
        mag = -2.5 * np.log10(np.maximum(flux, 1e-10))
        mag_noise = rng.normal(0, mag_error_std, len(mag))
        mag_with_noise = mag + mag_noise
        flux = 10 ** (mag_with_noise / -2.5)
        
        return (idx, flux, params)
        
    except Exception as e:
        return (idx, None, None)


def generate_pspl_worker(args):
    """Worker function for parallel PSPL generation"""
    idx, times, mag_error_std, seed = args
    rng = np.random.RandomState(seed + idx)
    
    u0 = rng.uniform(0.001, 1.0)
    tE = rng.uniform(10.0, 200.0)
    t0 = rng.uniform(times.min() + 20, times.max() - 20)
    
    flux = generate_pspl_light_curve(u0, tE, t0, times, mag_error_std, rng, mask=None)
    
    return (idx, flux)


def main():
    parser = argparse.ArgumentParser(description='Generate distinct dataset with proper normalization')
    parser.add_argument('--n_pspl', type=int, default=100000)
    parser.add_argument('--n_binary', type=int, default=100000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cadence_mask_prob', type=float, default=0.0)
    parser.add_argument('--mag_error_std', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    
    if not VBM_AVAILABLE:
        print("ERROR: VBBinaryLensing required!")
        return 1
    
    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    
    print("="*80)
    print("DISTINCT DATASET WITH PROPER BASELINE NORMALIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PSPL events:     {args.n_pspl:,}")
    print(f"  Binary events:   {args.n_binary:,}")
    print(f"  Normalization:   Baseline = 1.0 for both classes")
    print(f"  Photometric err: {args.mag_error_std:.3f} mag")
    print(f"  Workers:         {n_workers}")
    
    times = np.linspace(CFG.TIME_MIN, CFG.TIME_MAX, CFG.N_POINTS)
    
    # Generate binary events
    print(f"\n{'='*80}")
    print("GENERATING BINARY EVENTS")
    print(f"{'='*80}")
    
    binary_params_list = sample_distinct_binary_params(args.n_binary, seed=args.seed)
    worker_args = [
        (i, params, times, args.mag_error_std, args.seed, 1e-5)
        for i, params in enumerate(binary_params_list)
    ]
    
    X_binary = [None] * args.n_binary
    params_binary = [None] * args.n_binary
    failed_count = 0
    
    with Pool(n_workers) as pool:
        for idx, flux, params in tqdm(
            pool.imap_unordered(generate_binary_with_preset_worker, worker_args),
            total=args.n_binary,
            desc="Binary events"
        ):
            if flux is not None:
                X_binary[idx] = flux
                params_binary[idx] = params
            else:
                failed_count += 1
    
    X_binary = [x for x in X_binary if x is not None]
    params_binary = [p for p in params_binary if p is not None]
    X_binary = np.array(X_binary, dtype=np.float32)
    
    print(f"✓ Generated {len(X_binary):,} binary events ({failed_count} failures)")
    
    # Generate PSPL events
    print(f"\n{'='*80}")
    print("GENERATING PSPL EVENTS")
    print(f"{'='*80}")
    
    pspl_args = [(i, times, args.mag_error_std, args.seed) for i in range(args.n_pspl)]
    X_pspl = [None] * args.n_pspl
    
    with Pool(n_workers) as pool:
        for idx, flux in tqdm(
            pool.imap_unordered(generate_pspl_worker, pspl_args),
            total=args.n_pspl,
            desc="PSPL events"
        ):
            X_pspl[idx] = flux
    
    X_pspl = np.array(X_pspl, dtype=np.float32)
    print(f"✓ Generated {len(X_pspl):,} PSPL events")
    
    # Verify normalization
    print(f"\n{'='*80}")
    print("NORMALIZATION VERIFICATION")
    print(f"{'='*80}")
    
    binary_baselines = np.array([x[x>0].min() for x in X_binary[:1000] if (x>0).sum()>0])
    pspl_baselines = np.array([x[x>0].min() for x in X_pspl[:1000] if (x>0).sum()>0])
    
    print(f"Binary baseline: {binary_baselines.mean():.3f} ± {binary_baselines.std():.3f}")
    print(f"PSPL baseline:   {pspl_baselines.mean():.3f} ± {pspl_baselines.std():.3f}")
    print(f"Difference:      {abs(binary_baselines.mean() - pspl_baselines.mean()):.3f}")
    
    if abs(binary_baselines.mean() - pspl_baselines.mean()) < 0.05:
        print("✅ Normalization successful - baselines are similar!")
    else:
        print("⚠️  WARNING: Baselines still different!")
    
    # Combine
    X = np.vstack([X_pspl, X_binary])
    y = np.concatenate([
        np.zeros(len(X_pspl), dtype=np.int32),
        np.ones(len(X_binary), dtype=np.int32)
    ])
    
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    meta = {
        'n_pspl': len(X_pspl),
        'n_binary': len(X_binary),
        'presets_used': True,
        'baseline_normalized': True,
        'mag_error_std': args.mag_error_std,
        'seed': args.seed
    }
    
    np.savez(
        output_path,
        X=X,
        y=y,
        timestamps=times,
        params_binary_json=json.dumps(params_binary),
        meta_json=json.dumps(meta)
    )
    
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"\n{'='*80}")
    print("✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())
