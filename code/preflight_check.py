#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight_check.py — Quick environment and config sanity checks
- Verifies Python packages, GPU visibility, and config consistency
- Checks path existence (data/raw, models, results, logs)
- Confirms simulator/evaluator entrypoints are executable
"""

from __future__ import annotations

import sys
import os
import importlib
from pathlib import Path

REQUIRED_PY = ["numpy", "torch", "sklearn", "matplotlib"]
OPTIONAL_PY = ["VBBinaryLensing"]

def section(title: str):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def check_imports(pkgs):
    ok = True
    for name in pkgs:
        try:
            importlib.import_module(name)
            print(f"[ok] import {name}")
        except Exception as e:
            print(f"[!!] missing {name}: {e}")
            ok = False
    return ok

def check_gpu():
    try:
        import torch
    except Exception as e:
        print(f"[!!] torch not importable: {e}")
        return False
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[ok] CUDA available: {n} device(s)")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}")
        return True
    else:
        # ROCm can still be available via torch.version.hip
        if hasattr(torch.version, "hip") and torch.version.hip:
            print("[ok] ROCm/HIP build detected")
            return True
        print("[..] No GPU detected (CPU-only)")
        return True  # allow CPU

def check_paths(base_dir: Path):
    ok = True
    needed = [
        base_dir/"data"/"raw",
        base_dir/"models",
        base_dir/"results",
    ]
    for p in needed:
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"[mk] created {p}")
            except Exception as e:
                print(f"[!!] cannot create {p}: {e}")
                ok = False
        else:
            print(f"[ok] {p}")
    return ok

def check_config():
    try:
        import config as CFG
    except Exception as e:
        print(f"[!!] cannot import config: {e}")
        return False
    ok = True
    # PAD_VALUE should be -1 for consistency with simulate/train/eval
    pad = getattr(CFG, "PAD_VALUE", None)
    if pad != -1:
        print(f"[!!] PAD_VALUE expected -1, found {pad}")
        ok = False
    else:
        print("[ok] PAD_VALUE = -1")
    # Time window consistency
    tmin = getattr(CFG, "TIME_MIN", None)
    tmax = getattr(CFG, "TIME_MAX", None)
    if tmin is None or tmax is None:
        print("[!!] TIME_MIN/TIME_MAX missing")
        ok = False
    else:
        print(f"[ok] TIME_MIN={tmin} TIME_MAX={tmax}")
    # Early detection checkpoints sanity
    edc = getattr(CFG, "EARLY_DETECTION_CHECKPOINTS", [])
    if edc:
        bad = [x for x in edc if not (0 < x <= 1)]
        if bad:
            print(f"[!!] invalid EARLY_DETECTION_CHECKPOINTS: {bad}")
            ok = False
        else:
            print(f"[ok] EARLY_DETECTION_CHECKPOINTS: {edc}")
    return ok

def check_scripts(base_dir: Path):
    ok = True
    for name in ["simulate.py", "train.py", "evaluate.py", "utils.py"]:
        p = base_dir / name
        if not p.exists():
            print(f"[!!] missing {p}")
            ok = False
            continue
        if os.access(p, os.X_OK):
            print(f"[ok] {name} executable")
        else:
            try:
                os.chmod(p, 0o755)
                print(f"[fix] chmod +x {name}")
            except Exception as e:
                print(f"[..] could not chmod +x {name}: {e}")
    return ok

def main():
    base_dir = Path(__file__).resolve().parent
    section("Python packages")
    ok_pkgs = check_imports(REQUIRED_PY)
    check_imports(OPTIONAL_PY)  # optional

    section("GPU")
    ok_gpu = check_gpu()

    section("Project paths")
    ok_paths = check_paths(base_dir)

    section("Config")
    ok_cfg = check_config()

    section("Scripts")
    ok_scripts = check_scripts(base_dir)

    all_ok = ok_pkgs and ok_gpu and ok_paths and ok_cfg and ok_scripts
    print("\n" + "="*70)
    if all_ok:
        print("Preflight: OK ✅")
        return 0
    else:
        print("Preflight: issues found ❌")
        print("Hints:")
        print("  - pip install -r requirements.txt")
        print("  - verify simulate/train/evaluate use PAD_VALUE = -1 consistently")
        print("  - check GPU driver + CUDA/ROCm runtime if needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())