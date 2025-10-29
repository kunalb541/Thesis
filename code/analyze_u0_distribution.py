#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_u0_distribution.py — Inspect u0 distributions if params are saved in the NPZ.
- Works only if simulate.py was run with --save-params (params_pspl_json / params_binary_json)
- Produces histogram PNGs and prints summary stats.
Usage:
  python analyze_u0_distribution.py --data data/raw/events_baseline_1M.npz --out results/analysis/u0
"""
from __future__ import annotations

import argparse, os, json
import numpy as np

def load_params(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    pspl, binary = None, None
    if "params_pspl_json" in d.files:
        pspl = json.loads(d["params_pspl_json"].item())
    if "params_binary_json" in d.files:
        binary = json.loads(d["params_binary_json"].item())
    return pspl, binary

def extract_u0(params_list, key="u0"):
    if not params_list:
        return None
    vals = []
    for p in params_list:
        if key in p:
            vals.append(float(p[key]))
    return np.array(vals, dtype=float) if len(vals) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to .npz produced by simulate.py with --save-params")
    ap.add_argument("--out", required=False, default=None, help="Directory to save plots (optional)")
    args = ap.parse_args()

    pspl_params, bin_params = load_params(args.data)
    if pspl_params is None and bin_params is None:
        print("No params found in NPZ. Re-run simulate.py with --save-params to enable u0 analysis.")
        return 0

    u0_pspl = extract_u0(pspl_params) if pspl_params is not None else None
    u0_bin  = extract_u0(bin_params) if bin_params is not None else None

    def summarize(name, arr):
        if arr is None:
            print(f"{name}: (none)")
            return
        print(f"{name}: N={arr.size} min={arr.min():.4f} p50={np.median(arr):.4f} mean={arr.mean():.4f} max={arr.max():.4f}")

    summarize("PSPL u0", u0_pspl)
    summarize("Binary u0", u0_bin)

    if args.out:
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        if u0_pspl is not None:
            plt.figure()
            plt.hist(u0_pspl, bins=80, alpha=0.9)
            plt.xlabel("u0"); plt.ylabel("count"); plt.title("PSPL u0")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, "u0_pspl.png"), dpi=200); plt.close()
        if u0_bin is not None:
            plt.figure()
            plt.hist(u0_bin, bins=80, alpha=0.9)
            plt.xlabel("u0"); plt.ylabel("count"); plt.title("Binary u0")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, "u0_binary.png"), dpi=200); plt.close()
        if u0_pspl is not None and u0_bin is not None:
            plt.figure()
            # overlay with same bins
            bins = np.linspace(min(u0_pspl.min(), u0_bin.min()), max(u0_pspl.max(), u0_bin.max()), 80)
            plt.hist(u0_pspl, bins=bins, alpha=0.5, label="PSPL", density=True)
            plt.hist(u0_bin, bins=bins, alpha=0.5, label="Binary", density=True)
            plt.xlabel("u0"); plt.ylabel("density"); plt.title("u0 comparison"); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(args.out, "u0_overlay.png"), dpi=200); plt.close()
        print(f"Plots saved to: {os.path.abspath(args.out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
