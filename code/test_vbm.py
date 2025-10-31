#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vbm.py - VBMicrolensing Validation Test

Validates that VBMicrolensing is:
1. Installed correctly
2. Producing binary signatures distinct from PSPL
3. Working across realistic parameter ranges

Run this BEFORE starting data generation to ensure physics is correct.

Author: Kunal Bhatia
Date: October 2025
"""

import numpy as np

print("="*80)
print("VBMicrolensing Validation Test")
print("="*80)

# Test VBM import
try:
    import VBBinaryLensing as VBBL
    vbm = VBBL.VBBinaryLensing()
    print("✓ VBMicrolensing imported successfully")
except ImportError as e:
    print(f"❌ VBMicrolensing not installed: {e}")
    print("\nInstall with: pip install VBMicrolensing")
    exit(1)

timestamps = np.linspace(-50, 50, 100)

# Test 1: Obvious binary with strong caustic
print("\n" + "="*80)
print("Test 1: Strong Binary Caustic (s=1.0, q=0.1, u0=0.05)")
print("="*80)

s, q, u0, alpha, rho, tE, t0 = 1.0, 0.1, 0.05, 0.5, 0.001, 50.0, 0.0
params_binary = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]

try:
    mags_binary = np.array(vbm.BinaryLightCurve(params_binary, timestamps)[0])
    print(f"  Max magnification: {mags_binary.max():.2f}")
    print(f"  Mean magnification: {mags_binary.mean():.2f}")
    print(f"  Min magnification: {mags_binary.min():.2f}")
    
    if mags_binary.max() > 10:
        print("  ✅ Strong caustic spike detected!")
    else:
        print("  ⚠️ Warning: Caustic spike weaker than expected")
except Exception as e:
    print(f"  ❌ Binary light curve failed: {e}")
    exit(1)

# Test 2: PSPL for comparison
print("\n" + "="*80)
print("Test 2: PSPL Event (u0=0.05)")
print("="*80)

def pspl_magnification(u):
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
mags_pspl = pspl_magnification(u_t)

print(f"  Max magnification: {mags_pspl.max():.2f}")
print(f"  Mean magnification: {mags_pspl.mean():.2f}")
print(f"  Min magnification: {mags_pspl.min():.2f}")

# Test 3: Compare binary vs PSPL
print("\n" + "="*80)
print("Test 3: Binary vs PSPL Comparison")
print("="*80)

ratio = mags_binary.max() / mags_pspl.max()
print(f"  Binary/PSPL peak ratio: {ratio:.2f}")

if ratio > 2.0:
    print("  ✅ Binary events are clearly distinguishable from PSPL!")
    print(f"  Binary peak is {ratio:.1f}× higher due to caustic crossing")
elif ratio > 1.5:
    print(f"  ⚠️ Moderate distinction (ratio = {ratio:.2f})")
    print("  This is acceptable but not ideal for obvious binaries")
else:
    print("  ❌ WARNING: Binary and PSPL look too similar!")
    print("  Check VBMicrolensing parameters or installation")

# Test 4: Check parameter ranges
print("\n" + "="*80)
print("Test 4: Parameter Range Validation")
print("="*80)

test_cases = [
    ("Planetary (q=0.001)", 1.2, 0.001, 0.1),
    ("Brown Dwarf (q=0.05)", 1.0, 0.05, 0.1),
    ("Stellar Binary (q=0.5)", 0.8, 0.5, 0.15),
]

for name, s, q, u0 in test_cases:
    params = [np.log(s), np.log(q), u0, 0.5, np.log(0.001), np.log(50.0), 0.0]
    try:
        mags = np.array(vbm.BinaryLightCurve(params, timestamps)[0])
        print(f"  {name:<25} max mag = {mags.max():>6.2f}  ✓")
    except Exception as e:
        print(f"  {name:<25} ❌ FAILED: {e}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\nIf all tests passed, VBMicrolensing is working correctly.")
print("You can now proceed with data generation.")
print("="*80)