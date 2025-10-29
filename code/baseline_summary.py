#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_summary.py — Summarize an NPZ dataset (shape, class balance, padding rate, class means).
Usage:
  python baseline_summary.py --data data/raw/events_baseline_1M.npz --out results/analysis/summary.json
"""
from __future__ import annotations

import argparse, os, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--pad_value", type=float, default=-1.0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, t = d["X"], d["y"], d["timestamps"]
    N, L = X.shape

    # apply perm for reporting (doesn't matter, but keep consistent)
    if "perm" in d.files:
        perm = d["perm"]
        y = y[perm]

    # class labels -> uint8
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([0 if (str(v).lower().startswith("pspl")) else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8, copy=False)

    pad_mask = (X == args.pad_value)
    pad_rate = float(pad_mask.sum() / X.size)

    m0 = float(X[y == 0].mean()) if np.any(y == 0) else float("nan")
    m1 = float(X[y == 1].mean()) if np.any(y == 1) else float("nan")
    diff = abs(m0 - m1) if (np.isfinite(m0) and np.isfinite(m1)) else float("nan")

    summary = {
        "path": os.path.abspath(args.data),
        "N": int(N), "L": int(L),
        "time_min": float(t.min()), "time_max": float(t.max()),
        "class_counts": {"PSPL": int((y == 0).sum()), "Binary": int((y == 1).sum())},
        "pad_value": float(args.pad_value),
        "pad_rate": pad_rate,
        "class_mean_flux": {"PSPL": m0, "Binary": m1, "diff": diff},
        "has_perm": bool("perm" in d.files),
        "has_params": bool("params_pspl_json" in d.files or "params_binary_json" in d.files),
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {os.path.abspath(args.out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
