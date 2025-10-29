"""
Utility functions for microlensing classification
- GPU detection for NVIDIA (CUDA) and AMD (ROCm)
- Dataset loading with saved permutation support
- Normalization, padding helpers, shared mask generation
- Lightweight plotting helpers

Author: Kunal Bhatia
"""

from __future__ import annotations

import os
import json
from typing import Tuple, List, Optional

import numpy as np

# Try to import config for consistent settings (PAD_VALUE, plotting style, etc.)
try:
    import config as CFG
except Exception:
    class _Fallback:
        PAD_VALUE = -1
        PLOT_STYLE = None
    CFG = _Fallback()

# ---------------------------------------------------------------------------
# GPU utilities (Torch optional)
# ---------------------------------------------------------------------------

def detect_gpu_backend() -> str:
    """
    Detect the available GPU backend.
    Returns:
        'nvidia' if CUDA is available
        'amd'    if ROCm/HIP is available
        'cpu'    otherwise
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu"

    import torch
    if torch.cuda.is_available():
        # NVIDIA CUDA is available
        # (Some ROCm builds may also report cuda available; prefer explicit HIP check next)
        if hasattr(torch.version, "hip") and torch.version.hip:
            return "amd"
        return "nvidia"
    # CUDA not available; check for HIP explicitly
    if hasattr(torch.version, "hip") and torch.version.hip:
        return "amd"
    return "cpu"


def check_gpu_availability() -> int:
    """
    Returns number of visible GPUs (0 if none).
    """
    try:
        import torch
    except Exception:
        return 0

    if detect_gpu_backend() in ("nvidia", "amd") and torch.cuda.is_available():
        try:
            return torch.cuda.device_count()
        except Exception:
            return 0
    return 0


def get_device_name() -> str:
    """
    Human-friendly device string.
    """
    try:
        import torch
    except Exception:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    count = torch.cuda.device_count()
    names = []
    for i in range(count):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append(f"GPU{i}")
    return ", ".join(names) if names else "gpu"


def setup_gpu_environment(verbose: bool = True) -> str:
    """
    Set a few environment toggles for stability/perf.
    Returns backend: 'nvidia' | 'amd' | 'cpu'
    """
    backend = detect_gpu_backend()

    # Determinism toggles (leave to training script for strict reproducibility)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    if verbose:
        print(f"[utils] Detected backend: {backend}")
        if backend != "cpu":
            try:
                import torch
                print(f"[utils] GPUs visible: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gmem = getattr(props, "total_memory", 0) / 1e9
                    print(f"  - GPU {i}: {props.name} | {gmem:.2f} GB")
            except Exception:
                pass
    return backend


def print_gpu_summary() -> None:
    """
    Pretty-print a one-shot GPU summary.
    """
    backend = setup_gpu_environment(verbose=False)
    try:
        import torch
    except Exception:
        print("GPU: CPU only (PyTorch not installed).")
        return

    if not torch.cuda.is_available():
        print("GPU: CPU only (no CUDA/HIP device).")
        return

    n = torch.cuda.device_count()
    print(f"GPU backend: {backend} | {n} device(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = getattr(props, "total_memory", 0) / 1e9
        print(f"  [{i}] {props.name} | {total_gb:.2f} GB")

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_npz_dataset(npz_path: str, apply_perm: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load dataset produced by simulate.py.
    - Applies saved permutation if present
    - Maps string labels to uint8 if necessary

    Returns:
        X: (N, L) float32
        y: (N,) uint8 (0=PSPL, 1=Binary)
        timestamps: (L,) float64
        meta: dict
    """
    d = np.load(npz_path, allow_pickle=False)
    X = d["X"]
    y = d["y"]
    if apply_perm and "perm" in d.files:
        perm = d["perm"]
        X = X[perm]
        y = y[perm]

    # Normalize label dtype
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([0 if (str(v).lower().startswith("pspl")) else 1 for v in y], dtype=np.uint8)
    else:
        y = y.astype(np.uint8, copy=False)

    timestamps = d["timestamps"]
    meta = {}
    if "meta_json" in d.files:
        try:
            meta = json.loads(d["meta_json"].item())
        except Exception:
            meta = {}
    return X, y, timestamps, meta


def apply_pad_to_zero(X: np.ndarray, pad_value: Optional[float] = None) -> np.ndarray:
    """
    Replace PAD_VALUE entries with 0.0 (neutral for convs).
    Returns a new array (does not modify in-place).
    """
    if pad_value is None:
        pad_value = getattr(CFG, "PAD_VALUE", -1)
    X = X.copy()
    X[X == pad_value] = 0.0
    return X


def normalize_per_event(X: np.ndarray, pad_value: Optional[float] = None) -> np.ndarray:
    """
    Divide each row by its median ignoring PAD_VALUE entries.
    Returns a new array.
    """
    if pad_value is None:
        pad_value = getattr(CFG, "PAD_VALUE", -1)
    X = X.copy()
    finite = np.isfinite(X) & (X != pad_value)
    # Compute per-row median safely
    med = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        mask = finite[i]
        if mask.any():
            med[i] = np.nanmedian(X[i, mask])
        else:
            med[i] = 1.0
    med[med <= 0] = 1.0
    X[finite] = X[finite] / med[:, None][finite]
    return X


def generate_shared_masks(n_points: int, prob: float, pool_size: int, seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Create a pool of boolean masks (True = missing) with given probability.
    """
    rng = np.random.RandomState(seed if seed is not None else None)
    masks = [(rng.rand(n_points) < prob) for _ in range(max(0, int(pool_size)))]
    return masks

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _maybe_set_style():
    style = getattr(CFG, "PLOT_STYLE", None)
    if style:
        try:
            import matplotlib.pyplot as plt  # noqa
            plt.style.use(style)
        except Exception:
            pass

def plot_lightcurve(t: np.ndarray, f: np.ndarray, title: str = "", save_path: Optional[str] = None):
    """
    Quick plot of a single light curve.
    Treat PAD_VALUE as missing.
    """
    _maybe_set_style()
    import matplotlib.pyplot as plt

    pad_value = getattr(CFG, "PAD_VALUE", -1)
    mask = np.isfinite(f) & (f != pad_value)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(t[mask], f[mask], lw=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux (arb.)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=getattr(CFG, "DPI", 150))
        plt.close(fig)
    else:
        plt.show()

# ---------------------------------------------------------------------------
# Simple sanity checks
# ---------------------------------------------------------------------------

def class_mean_flux(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Return mean flux per class (pre-shuffle values). Useful for quick diagnostics.
    """
    m0 = X[y == 0].mean() if np.any(y == 0) else float("nan")
    m1 = X[y == 1].mean() if np.any(y == 1) else float("nan")
    diff = abs(m0 - m1) if (np.isfinite(m0) and np.isfinite(m1)) else float("nan")
    return float(m0), float(m1), float(diff)


if __name__ == "__main__":
    # On-demand self-test
    print("[utils] Backend:", detect_gpu_backend())
    print("[utils] GPUs:", check_gpu_availability())
    print("[utils] Device(s):", get_device_name())