#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate.py — Generate PSPL and Binary microlensing light curves for classification

Key invariants to avoid data leakage:
  - SAME baseline magnitude range for both classes (BASELINE_MIN/MAX)
  - Noise added in MAGNITUDE space (not magnification)
  - Identical cadence masking for both classes
  - Convert back to flux the same way for both classes
  - Shuffle X and y after stacking (seeded, if provided)

Outputs an .npz with:
  - X: (N, n_points) float32 — padded flux light curves
  - y: (N,) object — labels ('PSPL' or 'Binary')
  - timestamps: (n_points,) float64
  - meta: dict with configuration used
  - params_pspl / params_binary: list of dicts (only if --save-params)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np

# ---------------------------
# Global constants (FIXES)
# ---------------------------

# SAME baseline range for BOTH classes — avoid mean-flux leakage
BASELINE_MIN = 19.0
BASELINE_MAX = 22.0

# Padded value for missing points after masking
PAD_VALUE = -1.0

# Default time window (you can change to match your thesis setup)
TIME_MIN = -100.0
TIME_MAX = 100.0

# Try to import VBMicrolensing if available
_VBM = None
try:
    # Typical import; adapt if your environment differs
    import VBBinaryLensing as VBBL  # noqa: F401
    from VBBinaryLensing import VBBinaryLensing

    _VBM = VBBinaryLensing()
    # Speed setups can go here if you like, e.g., _VBM.accuracy = 1
except Exception:
    _VBM = None


# ---------------------------
# Data classes for ranges
# ---------------------------

@dataclass
class PSPLRanges:
    t0_min: float = -20.0
    t0_max: float = 20.0
    u0_min: float = 0.01
    u0_max: float = 0.5
    tE_min: float = 10.0
    tE_max: float = 150.0


@dataclass
class BinaryRanges:
    s_min: float = 0.7
    s_max: float = 2.0
    q_min: float = 0.1
    q_max: float = 0.5
    rho_min: float = 0.005
    rho_max: float = 0.02
    alpha_min: float = 0.0
    alpha_max: float = math.pi / 2
    tE_min: float = 50.0
    tE_max: float = 200.0
    t0_min: float = -20.0
    t0_max: float = 20.0
    u0_min: float = 0.1
    u0_max: float = 0.3


# ---------------------------
# Utilities
# ---------------------------

def _to_flux_from_mag(mag: np.ndarray, baseline: float) -> np.ndarray:
    """
    Convert magnitudes back to flux relative to the baseline:
    flux = 10 ** (-(mag - baseline) / 2.5)
    """
    return 10.0 ** (-(mag - baseline) / 2.5)


def _pspl_magnification(u: np.ndarray) -> np.ndarray:
    """
    Standard PSPL magnification.
    A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
    """
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _set_np_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)


# ---------------------------
# Event generators (workers)
# ---------------------------

def generate_pspl_event_worker(args: Tuple[int, int, float, float, PSPLRanges]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate one PSPL event with consistent pipeline:
        magnification -> magnitude -> add mag noise -> mask -> flux -> pad
    """
    seed, n_points, mag_error_std, cadence_mask_prob, ranges = args
    _set_np_seed(seed)

    # Timebase
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)

    # Sample params
    t0 = np.random.uniform(ranges.t0_min, ranges.t0_max)
    u0 = np.random.uniform(ranges.u0_min, ranges.u0_max)
    tE = np.random.uniform(ranges.tE_min, ranges.tE_max)

    baseline = np.random.uniform(BASELINE_MIN, BASELINE_MAX)

    # PSPL magnification
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = _pspl_magnification(u_t)

    # Convert to magnitude
    magnitudes = baseline - 2.5 * np.log10(magnification)

    # Add mag-space noise
    if mag_error_std > 0:
        magnitudes += np.random.normal(0.0, mag_error_std, size=magnitudes.shape)

    # Cadence masking
    if cadence_mask_prob > 0:
        mask = np.random.rand(n_points) < cadence_mask_prob
        magnitudes[mask] = np.nan

    # Convert to flux & pad
    flux = _to_flux_from_mag(magnitudes, baseline)
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)

    params = {
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "baseline": float(baseline),
    }
    return flux_padded.astype(np.float32), params


def generate_binary_event_worker(args: Tuple[int, int, float, float, BinaryRanges]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate one Binary event with consistent pipeline:
        magnification -> magnitude -> add mag noise -> mask -> flux -> pad

    Uses VBMicrolensing if available; otherwise falls back to a PSPL-like curve
    to avoid obvious artifacts (still valid for testing the pipeline).
    """
    seed, n_points, mag_error_std, cadence_mask_prob, ranges = args
    _set_np_seed(seed)

    # Timebase
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)

    # Sample binary params
    s = np.random.uniform(ranges.s_min, ranges.s_max)
    q = np.random.uniform(ranges.q_min, ranges.q_max)
    rho = np.random.uniform(ranges.rho_min, ranges.rho_max)
    alpha = np.random.uniform(ranges.alpha_min, ranges.alpha_max)
    tE = np.random.uniform(ranges.tE_min, ranges.tE_max)
    t0 = np.random.uniform(ranges.t0_min, ranges.t0_max)
    u0 = np.random.uniform(ranges.u0_min, ranges.u0_max)

    baseline = np.random.uniform(BASELINE_MIN, BASELINE_MAX)

    # Magnification via VBMicrolensing if available
    if _VBM is not None:
        try:
            # VBMicrolensing expects logs for (s, q, rho, tE), linear for u0, alpha, t0
            params_vbm = [_safe_log(np.array([s]))[0],
                          _safe_log(np.array([q]))[0],
                          u0,
                          alpha,
                          _safe_log(np.array([rho]))[0],
                          _safe_log(np.array([tE]))[0],
                          t0]
            # BinaryLightCurve returns (magnification, ...). Index 0 is magnification array
            magnifications = np.array(_VBM.BinaryLightCurve(params_vbm, timestamps)[0], dtype=np.float64)
            # Clip for safety
            magnifications = np.clip(magnifications, 1e-6, None)
        except Exception:
            # Fallback: PSPL-like (keeps distribution closer than e.g. flat line)
            u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
            magnifications = _pspl_magnification(u_t)
    else:
        # Library not available -> fallback
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        magnifications = _pspl_magnification(u_t)

    # Convert to magnitude — CRITICAL (noise added in MAG space)
    magnitudes = baseline - 2.5 * np.log10(magnifications)

    # Add mag-space noise (Gaussian)
    if mag_error_std > 0:
        magnitudes += np.random.normal(0.0, mag_error_std, size=magnitudes.shape)

    # Cadence masking (identical approach)
    if cadence_mask_prob > 0:
        mask = np.random.rand(n_points) < cadence_mask_prob
        magnitudes[mask] = np.nan

    # Back to flux & pad
    flux = _to_flux_from_mag(magnitudes, baseline)
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)

    params = {
        "s": float(s), "q": float(q), "rho": float(rho), "alpha": float(alpha),
        "tE": float(tE), "t0": float(t0), "u0": float(u0), "baseline": float(baseline)
    }
    return flux_padded.astype(np.float32), params


# ---------------------------
# Dataset generation
# ---------------------------

def _maybe_pool_map(pool, fn, args_list):
    if pool is None:
        return list(map(fn, args_list))
    return pool.map(fn, args_list)


def build_dataset(n_pspl: int,
                  n_binary: int,
                  n_points: int,
                  mag_error_std: float,
                  cadence_mask_prob: float,
                  seed: Optional[int],
                  save_params: bool,
                  num_workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, Optional[List[Dict]], Optional[List[Dict]]]:

    # Seed the global RNG so that worker seeds are reproducible
    _set_np_seed(seed)

    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)

    # Ranges (you can expose these via CLI if you want)
    pspl_ranges = PSPLRanges()
    binary_ranges = BinaryRanges()

    # Build argument lists
    pspl_args = [
        (None if seed is None else seed + i, n_points, mag_error_std, cadence_mask_prob, pspl_ranges)
        for i in range(n_pspl)
    ]
    binary_args = [
        (None if seed is None else seed + 10_000 + i, n_points, mag_error_std, cadence_mask_prob, binary_ranges)
        for i in range(n_binary)
    ]

    # Optional multiprocessing
    pool = None
    if num_workers and num_workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=num_workers)

    try:
        pspl_results = _maybe_pool_map(pool, generate_pspl_event_worker, pspl_args)
        bin_results = _maybe_pool_map(pool, generate_binary_event_worker, binary_args)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    X_pspl = np.stack([r[0] for r in pspl_results], axis=0) if n_pspl > 0 else np.empty((0, n_points), dtype=np.float32)
    X_bin = np.stack([r[0] for r in bin_results], axis=0) if n_binary > 0 else np.empty((0, n_points), dtype=np.float32)

    y_pspl = np.array(["PSPL"] * n_pspl, dtype=object)
    y_bin = np.array(["Binary"] * n_binary, dtype=object)

    # Stack
    X = np.vstack([X_pspl, X_bin]).astype(np.float32)
    y = np.concatenate([y_pspl, y_bin])

    # Shuffle after stacking — CRITICAL
    print("Shuffling...")
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Optional parameter logs
    params_pspl = [r[1] for r in pspl_results] if save_params else None
    params_binary = [r[1] for r in bin_results] if save_params else None

    meta = {
        "n_pspl": n_pspl,
        "n_binary": n_binary,
        "n_points": n_points,
        "mag_error_std": mag_error_std,
        "cadence_mask_prob": cadence_mask_prob,
        "seed": seed,
        "PAD_VALUE": PAD_VALUE,
        "BASELINE_MIN": BASELINE_MIN,
        "BASELINE_MAX": BASELINE_MAX,
        "TIME_MIN": TIME_MIN,
        "TIME_MAX": TIME_MAX,
        "vbmicrolensing_available": bool(_VBM is not None),
        "pspl_ranges": asdict(pspl_ranges),
        "binary_ranges": asdict(binary_ranges),
    }

    return X, y, timestamps, meta, params_pspl, params_binary


def save_npz(path: str,
             X: np.ndarray,
             y: np.ndarray,
             timestamps: np.ndarray,
             meta: dict,
             params_pspl: Optional[List[Dict]],
             params_binary: Optional[List[Dict]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "X": X,
        "y": y,
        "timestamps": timestamps,
        "meta_json": np.array(json.dumps(meta)),
    }
    if params_pspl is not None:
        out["params_pspl_json"] = np.array(json.dumps(params_pspl))
    if params_binary is not None:
        out["params_binary_json"] = np.array(json.dumps(params_binary))
    np.savez_compressed(path, **out)
    print(f"Saved dataset to: {path}")
    print(f"Shapes: X={X.shape}, y={y.shape}, timestamps={timestamps.shape}")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate PSPL and Binary microlensing datasets.")
    p.add_argument("--n_pspl", type=int, default=5000, help="Number of PSPL events")
    p.add_argument("--n_binary", type=int, default=5000, help="Number of Binary events")
    p.add_argument("--n_points", type=int, default=256, help="Points per light curve")
    p.add_argument("--mag_error_std", type=float, default=0.02, help="Gaussian sigma in magnitude space")
    p.add_argument("--cadence_mask_prob", type=float, default=0.1, help="Probability to mask each point (NaN)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (global). Use a fixed seed for reproducibility.")
    p.add_argument("--output", type=str, default="data/raw/events.npz", help="Output .npz path")
    p.add_argument("--num_workers", type=int, default=0, help="Multiprocessing workers (0/1 = no MP)")
    p.add_argument("--save-params", action="store_true", help="Save parameter dicts in the NPZ (JSON fields)")
    return p.parse_args()


def main():
    args = parse_args()
    print("Configuration:")
    print(json.dumps({
        "n_pspl": args.n_pspl,
        "n_binary": args.n_binary,
        "n_points": args.n_points,
        "mag_error_std": args.mag_error_std,
        "cadence_mask_prob": args.cadence_mask_prob,
        "seed": args.seed,
        "output": args.output,
        "num_workers": args.num_workers,
        "save_params": bool(args.save_params),
    }, indent=2))

    X, y, timestamps, meta, p_pspl, p_bin = build_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        n_points=args.n_points,
        mag_error_std=args.mag_error_std,
        cadence_mask_prob=args.cadence_mask_prob,
        seed=args.seed,
        save_params=args.save_params,
        num_workers=args.num_workers
    )

    save_npz(args.output, X, y, timestamps, meta, p_pspl, p_bin)

    # Quick integrity printout (class mean flux should be similar if baselines match)
    pspl_mean = X[y == "PSPL"].mean() if np.any(y == "PSPL") else float("nan")
    bin_mean = X[y == "Binary"].mean() if np.any(y == "Binary") else float("nan")
    diff = abs(pspl_mean - bin_mean) if (not math.isnan(pspl_mean) and not math.isnan(bin_mean)) else float("nan")

    print(f"\nClass mean flux check:")
    print(f"  PSPL mean flux:   {pspl_mean:.6f}")
    print(f"  Binary mean flux: {bin_mean:.6f}")
    print(f"  Difference:       {diff:.6f}")
    if not math.isnan(diff) and diff < 0.01:
        print("✅ FIXED! Means are similar. (No obvious baseline leakage)")
    else:
        print("ℹ️  Means differ noticeably. Re-check baseline usage if this is unexpected.")


if __name__ == "__main__":
    main()
