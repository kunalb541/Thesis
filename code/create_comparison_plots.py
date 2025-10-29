#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_comparison_plots.py — Visual comparisons between PSPL and Binary samples.
- Plots per-class median and IQR envelopes
- Plots a few random examples from each class
Usage:
  python create_comparison_plots.py --data data/raw/events_baseline_1M.npz --out results/plots
"""
from __future__ import annotations

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def compute_envelope(X: np.ndarray, mask_val: float):
    finite = np.isfinite(X) & (X != mask_val)
    # Replace masked with nan for quantiles
    Xq = X.copy().astype(np.float64)
    Xq[~finite] = np.nan
    median = np.nanmedian(Xq, axis=0)
    q25 = np.nanpercentile(Xq, 25, axis=0)
    q75 = np.nanpercentile(Xq, 75, axis=0)
    return median, q25, q75

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad_value", type=float, default=-1.0)
    ap.add_argument("--n_examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, t = d["X"], d["y"], d["timestamps"]
    if "perm" in d.files:
        perm = d["perm"]
        X, y = X[perm], y[perm]
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([0 if (str(v).lower().startswith("pspl")) else 1 for v in y], dtype=np.uint8)

    os.makedirs(args.out, exist_ok=True)

    X0 = X[y == 0]
    X1 = X[y == 1]

    # Envelope plots
    for cls, Xc, name in [(0, X0, "PSPL"), (1, X1, "Binary")]:
        med, q25, q75 = compute_envelope(Xc, args.pad_value)
        plt.figure(figsize=(10,4))
        plt.plot(t, med, label=f"{name} median")
        plt.fill_between(t, q25, q75, alpha=0.3, label="IQR")
        plt.xlabel("Time"); plt.ylabel("Flux"); plt.title(f"{name}: median ± IQR")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"{name.lower()}_envelope.png"), dpi=200); plt.close()

    # Random examples
    rng = np.random.default_rng(args.seed)
    def plot_examples(Xc, name):
        n = min(args.n_examples, Xc.shape[0])
        idx = rng.choice(Xc.shape[0], size=n, replace=False)
        plt.figure(figsize=(12, 2*n))
        for i, j in enumerate(idx, 1):
            x = Xc[j]
            m = np.isfinite(x) & (x != args.pad_value)
            ax = plt.subplot(n, 1, i)
            ax.plot(t[m], x[m], lw=0.8)
            ax.set_title(f"{name} example {i}")
            ax.set_xlabel("Time"); ax.set_ylabel("Flux")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"{name.lower()}_examples.png"), dpi=200); plt.close()

    plot_examples(X0, "PSPL")
    plot_examples(X1, "Binary")
    print(f"Plots written to {os.path.abspath(args.out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
