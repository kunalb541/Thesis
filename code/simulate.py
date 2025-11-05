#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate.py — FAST unified generator (PSPL + Binary) for classification

FIXED: Masking now works correctly with USE_SHARED_MASK = False
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm

try:
    import config as CFG
except Exception:
    class _Fallback:
        PAD_VALUE = -1
        TIME_MIN = -100
        TIME_MAX = 100
        NORMALIZE_PER_EVENT = True
        USE_SHARED_MASK = False
        MASK_POOL_SIZE = 256
        BINARY_PARAM_SETS = {'baseline': {
            's_min': 0.7, 's_max': 2.0, 'q_min': 0.1, 'q_max': 0.5, 'rho_min': 0.005, 'rho_max': 0.02,
            'alpha_min': 0.0, 'alpha_max': math.pi / 2, 'tE_min': 50.0, 'tE_max': 200.0,
            't0_min': -20.0, 't0_max': 20.0, 'u0_min': 0.1, 'u0_max': 0.3
        }}
    CFG = _Fallback()

PAD_VALUE = float(CFG.PAD_VALUE)
TIME_MIN = float(CFG.TIME_MIN)
TIME_MAX = float(CFG.TIME_MAX)
NORMALIZE_PER_EVENT = bool(getattr(CFG, "NORMALIZE_PER_EVENT", True))
USE_SHARED_MASK = bool(getattr(CFG, "USE_SHARED_MASK", False))
MASK_POOL_SIZE = int(getattr(CFG, "MASK_POOL_SIZE", 256))

BASELINE_MIN = 19.0
BASELINE_MAX = 22.0

_VBM = None
try:
    import VBBinaryLensing as VBBL
    from VBBinaryLensing import VBBinaryLensing
    _VBM = VBBinaryLensing()
except Exception:
    _VBM = None


@dataclass
class PSPLRanges:
    t0_min: float = getattr(CFG, "PSPL_T0_MIN", -20.0)
    t0_max: float = getattr(CFG, "PSPL_T0_MAX", 20.0)
    u0_min: float = getattr(CFG, "PSPL_U0_MIN", 0.01)
    u0_max: float = getattr(CFG, "PSPL_U0_MAX", 0.5)
    tE_min: float = getattr(CFG, "PSPL_TE_MIN", 10.0)
    tE_max: float = getattr(CFG, "PSPL_TE_MAX", 150.0)


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
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _to_flux_from_mag(mag: np.ndarray, baseline: float) -> np.ndarray:
    return 10.0 ** (-(mag - baseline) / 2.5)


def _pspl_magnification(u: np.ndarray) -> np.ndarray:
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _set_np_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)


def _maybe_norm_event(flux: np.ndarray) -> np.ndarray:
    if not NORMALIZE_PER_EVENT:
        return flux
    finite = np.isfinite(flux) & (flux != PAD_VALUE)
    if finite.any():
        m = np.median(flux[finite])
        if m > 0:
            flux = flux / m
    return flux


# FIX: Updated worker functions to accept cadence_mask_prob and generate masks
def generate_pspl_event_worker(args: Tuple[int, np.ndarray, float, float, Optional[np.ndarray], PSPLRanges]) -> Tuple[np.ndarray, Dict[str, float]]:
    seed, timestamps, mag_error_std, cadence_mask_prob, mask, ranges = args
    _set_np_seed(seed)
    
    t0 = np.random.uniform(ranges.t0_min, ranges.t0_max)
    u0 = np.random.uniform(ranges.u0_min, ranges.u0_max)
    tE = np.random.uniform(ranges.tE_min, ranges.tE_max)
    baseline = np.random.uniform(BASELINE_MIN, BASELINE_MAX)

    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = _pspl_magnification(u_t)
    magnitudes = baseline - 2.5 * np.log10(magnification)

    if mag_error_std > 0:
        magnitudes += np.random.normal(0.0, mag_error_std, size=magnitudes.shape)

    # FIX: Generate mask if not provided
    if mask is None and cadence_mask_prob > 0:
        mask = np.random.rand(len(timestamps)) < cadence_mask_prob
    
    if mask is not None:
        magnitudes[mask] = np.nan
    
    flux = _to_flux_from_mag(magnitudes, baseline)
    flux = _maybe_norm_event(flux)
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)
    params = {"t0": float(t0), "u0": float(u0), "tE": float(tE), "baseline": float(baseline)}
    return flux_padded.astype(np.float32), params


def generate_binary_event_worker(args: Tuple[int, np.ndarray, float, float, Optional[np.ndarray], BinaryRanges]) -> Tuple[np.ndarray, Dict[str, float]]:
    seed, timestamps, mag_error_std, cadence_mask_prob, mask, ranges = args
    _set_np_seed(seed)
    
    s = np.random.uniform(ranges.s_min, ranges.s_max)
    q = np.random.uniform(ranges.q_min, ranges.q_max)
    rho = np.random.uniform(ranges.rho_min, ranges.rho_max)
    alpha = np.random.uniform(ranges.alpha_min, ranges.alpha_max)
    tE = np.random.uniform(ranges.tE_min, ranges.tE_max)
    t0 = np.random.uniform(ranges.t0_min, ranges.t0_max)
    u0 = np.random.uniform(ranges.u0_min, ranges.u0_max)
    baseline = np.random.uniform(BASELINE_MIN, BASELINE_MAX)

    magnifications = None
    if _VBM is not None:
        try:
            params_vbm = [_safe_log(np.array([s]))[0],
                          _safe_log(np.array([q]))[0],
                          u0, alpha,
                          _safe_log(np.array([rho]))[0],
                          _safe_log(np.array([tE]))[0],
                          t0]
            magnifications = np.array(_VBM.BinaryLightCurve(params_vbm, timestamps)[0], dtype=np.float64)
            magnifications = np.clip(magnifications, 1e-6, None)
        except Exception:
            u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
            magnifications = _pspl_magnification(u_t)
    else:
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        magnifications = _pspl_magnification(u_t)

    magnitudes = baseline - 2.5 * np.log10(magnifications)

    if mag_error_std > 0:
        magnitudes += np.random.normal(0.0, mag_error_std, size=magnitudes.shape)

    # FIX: Generate mask if not provided
    if mask is None and cadence_mask_prob > 0:
        mask = np.random.rand(len(timestamps)) < cadence_mask_prob
    
    if mask is not None:
        magnitudes[mask] = np.nan
    
    flux = _to_flux_from_mag(magnitudes, baseline)
    flux = _maybe_norm_event(flux)
    flux_padded = np.nan_to_num(flux, nan=PAD_VALUE)

    params = {"s": float(s), "q": float(q), "rho": float(rho), "alpha": float(alpha),
              "tE": float(tE), "t0": float(t0), "u0": float(u0), "baseline": float(baseline)}
    return flux_padded.astype(np.float32), params


def _parallel_map_unordered(fn, args_list, num_workers: int):
    if num_workers is None or num_workers <= 1:
        for a in tqdm(args_list, desc=f"Generating {fn.__name__.split('_')[1]}"):
            yield fn(a)
        return
    import multiprocessing as mp
    n = len(args_list)
    chunksize = max(1, n // (num_workers * 8))
    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(fn, args_list, chunksize=chunksize), total=n, desc=f"Generating {fn.__name__.split('_')[1]} (Parallel)"):
            yield res


def build_dataset(n_pspl: int,
                  n_binary: int,
                  n_points: int,
                  mag_error_std: float,
                  cadence_mask_prob: float,
                  seed: Optional[int],
                  save_params: bool,
                  num_workers: int,
                  binary_ranges: BinaryRanges) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict,
                                             Optional[List[Dict]], Optional[List[Dict]], np.ndarray]:

    _set_np_seed(seed)
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points, dtype=np.float64)

    pspl_ranges = PSPLRanges()

    N = n_pspl + n_binary
    X = np.empty((N, n_points), dtype=np.float32)
    y = np.empty((N,), dtype=np.uint8)

    params_pspl: Optional[List[Dict]] = [] if save_params else None
    params_binary: Optional[List[Dict]] = [] if save_params else None

    shared_masks: Optional[List[np.ndarray]] = None
    if USE_SHARED_MASK:
        rng = np.random.RandomState(seed if seed is not None else None)
        shared_masks = []
        for _ in range(MASK_POOL_SIZE):
            shared_masks.append(rng.rand(n_points) < cadence_mask_prob)

    def pick_mask(i):
        if not USE_SHARED_MASK or shared_masks is None or MASK_POOL_SIZE == 0:
            return None
        return shared_masks[i % MASK_POOL_SIZE]

    # FIX: Pass cadence_mask_prob to workers
    pspl_args = [
        (None if seed is None else seed + i, timestamps, mag_error_std, cadence_mask_prob, pick_mask(i), pspl_ranges)
        for i in range(n_pspl)
    ]
    binary_args = [
        (None if seed is None else seed + 10_000 + i, timestamps, mag_error_std, cadence_mask_prob, pick_mask(i), binary_ranges)
        for i in range(n_binary)
    ]

    idx = 0
    for flux, params in _parallel_map_unordered(generate_pspl_event_worker, pspl_args, num_workers):
        X[idx, :] = flux
        y[idx] = 0
        if save_params and params_pspl is not None:
            params_pspl.append(params)
        idx += 1

    for flux, params in _parallel_map_unordered(generate_binary_event_worker, binary_args, num_workers):
        X[idx, :] = flux
        y[idx] = 1
        if save_params and params_binary is not None:
            params_binary.append(params)
        idx += 1

    print("Creating permutation (deferred shuffle)...")
    if seed is not None:
        np.random.seed(seed)
    perm = np.random.permutation(N).astype(np.int64)

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
        "label_map": {0: "PSPL", 1: "Binary"},
        "shuffle_perm_available": True,
        "perm_seed": seed,
        "normalize_per_event": NORMALIZE_PER_EVENT,
        "use_shared_mask": USE_SHARED_MASK,
        "mask_pool_size": MASK_POOL_SIZE,
    }

    return X, y, timestamps, meta, params_pspl, params_binary, perm


def save_npz(path: str,
             X: np.ndarray,
             y: np.ndarray,
             timestamps: np.ndarray,
             meta: dict,
             params_pspl: Optional[List[Dict]],
             params_binary: Optional[List[Dict]],
             perm: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "X": X,
        "y": y,
        "timestamps": timestamps,
        "meta_json": np.array(json.dumps(meta)),
        "perm": perm,
    }
    if params_pspl is not None:
        out["params_pspl_json"] = np.array(json.dumps(params_pspl))
    if params_binary is not None:
        out["params_binary_json"] = np.array(json.dumps(params_binary))
    np.savez(path, **out)
    print(f"Saved dataset to: {path}")
    print(f"Shapes: X={X.shape}, y={y.shape}, timestamps={timestamps.shape}, perm={perm.shape}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate PSPL and Binary microlensing datasets (fast).")
    p.add_argument("--n_pspl", type=int, default=5000)
    p.add_argument("--n_binary", type=int, default=5000)
    p.add_argument("--n_points", type=int, default=256)
    p.add_argument("--mag_error_std", type=float, default=0.02)
    p.add_argument("--cadence_mask_prob", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="data/raw/events_fast.npz")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save-params", action="store_true")
    choices = list(CFG.BINARY_PARAM_SETS.keys())
    p.add_argument("--binary_params", type=str, default="baseline", choices=choices, 
                   help=f"Select binary parameter set from config: {choices}")
    return p.parse_args()


def main():
    args = parse_args()
    
    binary_ranges_data = CFG.BINARY_PARAM_SETS.get(args.binary_params, CFG.BINARY_PARAM_SETS['baseline'])
    binary_ranges = BinaryRanges(**binary_ranges_data)
    
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
        "TIME_MIN": TIME_MIN,
        "TIME_MAX": TIME_MAX,
        "PAD_VALUE": PAD_VALUE,
        "NORMALIZE_PER_EVENT": NORMALIZE_PER_EVENT,
        "USE_SHARED_MASK": USE_SHARED_MASK,
        "binary_params_set": args.binary_params,
    }, indent=2))

    X, y, timestamps, meta, p_pspl, p_bin, perm = build_dataset(
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        n_points=args.n_points,
        mag_error_std=args.mag_error_std,
        cadence_mask_prob=args.cadence_mask_prob,
        seed=args.seed,
        save_params=args.save_params,
        num_workers=args.num_workers,
        binary_ranges=binary_ranges
    )

    save_npz(args.output, X, y, timestamps, meta, p_pspl, p_bin, perm)

    pspl_mean = X[y == 0].mean() if np.any(y == 0) else float("nan")
    bin_mean = X[y == 1].mean() if np.any(y == 1) else float("nan")
    diff = abs(pspl_mean - bin_mean) if (not math.isnan(pspl_mean) and not math.isnan(bin_mean)) else float("nan")

    print(f"\nClass mean flux check (pre-shuffle):")
    print(f"  PSPL mean flux:   {pspl_mean:.6f}")
    print(f"  Binary mean flux: {bin_mean:.6f}")
    print(f"  Difference:       {diff:.6f}")
    if not math.isnan(diff) and diff < 0.01:
        print("✅ Means are similar (likely normalized or matched priors).")
    else:
        print("ℹ️ Means differ; expected if priors/physics differ.")


if __name__ == "__main__":
    main()

