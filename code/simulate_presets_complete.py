#!/usr/bin/env python3
"""
Complete standalone script for generating distinct topology dataset with presets
Multiprocessing-enabled (both binary and PSPL generation).
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import multiprocessing as mp
import os

try:
    import VBBinaryLensing as vb
    VBM_AVAILABLE = True
except ImportError:
    VBM_AVAILABLE = False
    print("Warning: VBBinaryLensing not available")

# Import config or define inline
try:
    import config as CFG
except ImportError:
    # Define inline if config not available
    class CFG:
        TIME_MIN = -100
        TIME_MAX = 100
        N_POINTS = 1500

from preset_sampler import sample_distinct_binary_params


def generate_pspl_light_curve(u0, tE, t0, times, mag_error_std, rng, mask=None):
    """
    Generate a PSPL (Point Source Point Lens) light curve
    """
    u = np.sqrt(u0**2 + ((times - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    flux = A
    mag_noise = rng.normal(0, mag_error_std, len(flux))
    flux = flux * 10 ** (mag_noise / -2.5)
    if mask is not None:
        flux[mask] = -1.0  # Mark as missing
    return flux


def generate_pspl_sample(i, rng, times, mag_error_std, mask):
    """Generate one PSPL event with random parameters"""
    u0 = rng.uniform(0.001, 1.0)
    tE = rng.uniform(10.0, 200.0)
    t0 = rng.uniform(times.min() + 20, times.max() - 20)
    return generate_pspl_light_curve(u0, tE, t0, times, mag_error_std, rng, mask)


def generate_binary_with_preset(params, times, vbm, rng, mag_error_std, mask=None):
    """Generate binary light curve from preset parameters using VBBinaryLensing"""
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
        mag = vbm.BinaryLightCurve(vb_params, times)[0]
        mag = np.array(mag)
        flux = 10 ** ((mag - mag.min()) / -2.5)
        mag_noise = rng.normal(0, mag_error_std, len(flux))
        flux = flux * 10 ** (mag_noise / -2.5)
        if mask is not None:
            flux[mask] = -1.0
        return flux, params
    except Exception as e:
        # Per-event failures are handled upstream
        return None, None


# ----------------------- Multiprocessing worker functions -----------------------

def _make_mask(rng, n_points, prob):
    if prob <= 0:
        return None
    return rng.rand(n_points) < prob


def _pspl_worker(args):
    """
    Worker for PSPL events.
    Returns: (index, flux_array or None)
    """
    i, times, mag_error_std, cadence_prob, seed = args
    rng = np.random.RandomState(seed + i)
    mask = _make_mask(rng, len(times), cadence_prob)
    try:
        flux = generate_pspl_sample(i, rng, times, mag_error_std, mask)
        return i, flux.astype(np.float32, copy=False)
    except Exception:
        return i, None


def _binary_worker(args):
    """
    Worker for Binary events using VBBinaryLensing.
    Returns: (index, flux_array or None, params or None)
    """
    i, params, times, mag_error_std, cadence_prob, seed, tol = args
    rng = np.random.RandomState(seed + i)
    mask = _make_mask(rng, len(times), cadence_prob)

    # Create a VBB instance in each process to avoid pickling issues
    try:
        vbm = vb.VBBinaryLensing()
        vbm.Tol = tol
    except Exception:
        return i, None, None

    flux, p = generate_binary_with_preset(params, times, vbm, rng, mag_error_std, mask)
    if flux is None:
        return i, None, None
    return i, flux.astype(np.float32, copy=False), p


# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate distinct topology dataset with battle-tested presets (multiprocessing)')
    parser.add_argument('--n_pspl', type=int, default=100000, help='Number of PSPL events')
    parser.add_argument('--n_binary', type=int, default=100000, help='Number of binary events')
    parser.add_argument('--output', type=str, required=True, help='Output .npz file path')
    parser.add_argument('--cadence_mask_prob', type=float, default=0.0, help='Probability of missing each observation')
    parser.add_argument('--mag_error_std', type=float, default=0.05, help='Photometric error (magnitudes)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 0), help='Number of worker processes')
    parser.add_argument('--vbb_tol', type=float, default=1e-5, help='VBBinaryLensing tolerance per worker')
    args = parser.parse_args()

    if not VBM_AVAILABLE:
        print("ERROR: VBBinaryLensing is required for binary light curve generation!")
        print("Install with: pip install VBBinaryLensing")
        return 1

    print("="*80)
    print("GENERATING DISTINCT TOPOLOGY DATASET WITH BATTLE-TESTED PRESETS (MP)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PSPL events:     {args.n_pspl:,}")
    print(f"  Binary events:   {args.n_binary:,}")
    print(f"  Cadence masking: {args.cadence_mask_prob*100:.1f}%")
    print(f"  Photometric err: {args.mag_error_std:.3f} mag")
    print(f"  Random seed:     {args.seed}")
    print(f"  Workers:         {args.workers}")
    print(f"  VBB Tol:         {args.vbb_tol:g}")

    # Initialize (master)
    rng_master = np.random.RandomState(args.seed)
    times = np.linspace(CFG.TIME_MIN, CFG.TIME_MAX, CFG.N_POINTS).astype(np.float64, copy=False)

    # -------------------- Generate binary events (parallel) --------------------
    print(f"\n{'='*80}")
    print(f"GENERATING BINARY EVENTS FROM PRESETS (parallel)")
    print(f"{'='*80}")

    binary_params_list = sample_distinct_binary_params(args.n_binary, seed=args.seed)

    X_binary_list = [None] * args.n_binary
    params_binary_list = [None] * args.n_binary
    failed_count = 0

    # Reasonable chunk size for imap_unordered
    def _chunk_size(n, w):
        return max(1, n // (w * 10 if w > 0 else 10))

    b_chunksize = _chunk_size(len(binary_params_list), args.workers)

    with mp.get_context("spawn").Pool(processes=args.workers, maxtasksperchild=64) as pool:
        jobs = (
            (i, p, times, args.mag_error_std, args.cadence_mask_prob, args.seed + 10_000_000, args.vbb_tol)
            for i, p in enumerate(binary_params_list)
        )
        for i, flux, p in tqdm(
            pool.imap_unordered(_binary_worker, jobs, chunksize=b_chunksize),
            total=args.n_binary,
            desc="Binary events",
        ):
            if flux is not None:
                X_binary_list[i] = flux
                params_binary_list[i] = p
            else:
                failed_count += 1

    # Compact to only successful ones
    X_binary = np.array([x for x in X_binary_list if x is not None], dtype=np.float32)
    params_binary = [p for p in params_binary_list if p is not None]

    print(f"✓ Generated {len(X_binary):,} binary events ({failed_count} failures)")

    # -------------------- Generate PSPL events (parallel) --------------------
    print(f"\n{'='*80}")
    print(f"GENERATING PSPL EVENTS (parallel)")
    print(f"{'='*80}")

    X_pspl_list = [None] * args.n_pspl
    p_chunksize = _chunk_size(args.n_pspl, args.workers)

    with mp.get_context("spawn").Pool(processes=args.workers, maxtasksperchild=256) as pool:
        jobs = (
            (i, times, args.mag_error_std, args.cadence_mask_prob, args.seed + 20_000_000)
            for i in range(args.n_pspl)
        )
        for i, flux in tqdm(
            pool.imap_unordered(_pspl_worker, jobs, chunksize=p_chunksize),
            total=args.n_pspl,
            desc="PSPL events",
        ):
            X_pspl_list[i] = flux

    X_pspl = np.array(X_pspl_list, dtype=np.float32)
    print(f"✓ Generated {len(X_pspl):,} PSPL events")

    # -------------------- Combine and shuffle --------------------
    print(f"\n{'='*80}")
    print(f"COMBINING AND SHUFFLING")
    print(f"{'='*80}")

    X = np.vstack([X_pspl, X_binary])
    y = np.concatenate([
        np.zeros(len(X_pspl), dtype=np.int32),
        np.ones(len(X_binary), dtype=np.int32)
    ])

    # Shuffle (reproducible with master RNG)
    perm = rng_master.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Total events: {len(X):,}")
    print(f"  PSPL: {(y==0).sum():,}")
    print(f"  Binary: {(y==1).sum():,}")

    # -------------------- Statistics --------------------
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")

    if len(X_binary) > 0:
        binary_max = X_binary.max(axis=1)
        print(f"\nPeak flux:")
        print(f"  Binary: mean={binary_max.mean():.2f}, median={np.median(binary_max):.2f}, max={binary_max.max():.2f}")
    else:
        print("\nPeak flux:\n  Binary: n/a (no successful binary events)")

    if len(X_pspl) > 0:
        pspl_max = X_pspl.max(axis=1)
        if len(X_binary) > 0:
            print(f"  PSPL:   mean={pspl_max.mean():.2f}, median={np.median(pspl_max):.2f}, max={pspl_max.max():.2f}")
            print(f"  Ratio:  {binary_max.mean() / pspl_max.mean():.2f}×")
        else:
            print(f"  PSPL:   mean={pspl_max.mean():.2f}, median={np.median(pspl_max):.2f}, max={pspl_max.max():.2f}")
    else:
        print("  PSPL:   n/a")

    if args.cadence_mask_prob > 0:
        pad_count = int((X == -1).sum())
        print(f"\nCadence masking:")
        print(f"  Masked observations: {pad_count:,} ({100*pad_count/X.size:.1f}%)")

    # -------------------- Save --------------------
    print(f"\n{'='*80}")
    print(f"SAVING DATASET")
    print(f"{'='*80}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        'n_pspl': int(len(X_pspl)),
        'n_binary': int(len(X_binary)),
        'presets_used': True,
        'vbmicrolensing_available': VBM_AVAILABLE,
        'cadence_mask_prob': float(args.cadence_mask_prob),
        'mag_error_std': float(args.mag_error_std),
        'seed': int(args.seed),
        'workers': int(args.workers),
        'vbb_tol': float(args.vbb_tol),
        'time_min': float(CFG.TIME_MIN),
        'time_max': float(CFG.TIME_MAX),
        'n_points': int(CFG.N_POINTS),
    }

    np.savez(
        output_path,
        X=X,
        y=y,
        timestamps=times,
        params_binary_json=json.dumps(params_binary),
        meta_json=json.dumps(meta)
    )

    print(f"✓ Dataset saved to: {output_path}")
    try:
        print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    except Exception:
        pass

    print(f"\n{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()  # safe on Windows/conda
    exit(main())
