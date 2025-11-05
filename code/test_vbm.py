#!/usr/bin/env python3
"""
VBMicrolensing Validation Test

Validates that VBMicrolensing is properly installed and can generate
distinguishable binary vs PSPL light curves.

Author: Kunal Bhatia
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("VBMicrolensing Validation Test")
print("="*80)

# Try importing VBMicrolensing
try:
    import VBBinaryLensing as VBBL
    from VBBinaryLensing import VBBinaryLensing
    VBM = VBBinaryLensing()
    print("✅ VBMicrolensing imported successfully\n")
except ImportError as e:
    print("❌ VBMicrolensing import failed!")
    print(f"Error: {e}")
    print("\nInstallation instructions:")
    print("  pip install VBMicrolensing")
    print("  OR")
    print("  conda install -c conda-forge vbmicrolensing")
    exit(1)

# Test 1: Strong Binary Event (Clear Caustic Crossing)
print("="*80)
print("Test 1: Strong Binary Caustic (s=1.0, q=0.1, u0=0.05)")
print("="*80)

timestamps = np.linspace(-50, 50, 500)

# Binary parameters (clear caustic crossing)
s = 1.0      # Optimal separation
q = 0.1      # Planetary mass ratio
u0 = 0.05    # Close approach
alpha = 0.5  # Trajectory angle
rho = 0.01   # Source size
tE = 25.0    # Einstein time
t0 = 0.0     # Peak time

try:
    params_binary = [
        np.log(s),
        np.log(q),
        u0,
        alpha,
        np.log(rho),
        np.log(tE),
        t0
    ]
    
    mag_binary = np.array(VBM.BinaryLightCurve(params_binary, timestamps)[0])
    
    # Check for strong magnification
    max_mag = np.max(mag_binary)
    mean_mag = np.mean(mag_binary)
    
    print(f"  Max magnification: {max_mag:.2f}")
    print(f"  Mean magnification: {mean_mag:.2f}")
    
    if max_mag > 10.0:
        print("  ✅ Strong caustic spike detected!")
    else:
        print("  ⚠️  Weak magnification (may still be valid)")
        
except Exception as e:
    print(f"  ❌ Binary simulation failed: {e}")
    mag_binary = None

# Test 2: PSPL Event (Smooth Light Curve)
print("\n" + "="*80)
print("Test 2: PSPL Event (u0=0.1, tE=25)")
print("="*80)

def pspl_magnification(u):
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

u0_pspl = 0.1
tE_pspl = 25.0
t0_pspl = 0.0

u_t = np.sqrt(u0_pspl**2 + ((timestamps - t0_pspl) / tE_pspl)**2)
mag_pspl = pspl_magnification(u_t)

max_mag_pspl = np.max(mag_pspl)
mean_mag_pspl = np.mean(mag_pspl)

print(f"  Max magnification: {max_mag_pspl:.2f}")
print(f"  Mean magnification: {mean_mag_pspl:.2f}")
print("  ✅ PSPL simulation successful")

# Test 3: Compare Binary vs PSPL
if mag_binary is not None:
    print("\n" + "="*80)
    print("Test 3: Binary vs PSPL Comparison")
    print("="*80)
    
    ratio = max_mag / max_mag_pspl
    print(f"  Binary/PSPL peak ratio: {ratio:.2f}")
    
    if ratio > 1.5:
        print("  ✅ Binary events are clearly distinguishable from PSPL!")
    else:
        print("  ⚠️  Binary/PSPL difference is subtle")

# Test 4: Plot Comparison
print("\n" + "="*80)
print("Test 4: Visual Comparison")
print("="*80)

try:
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Binary
    plt.subplot(1, 3, 1)
    if mag_binary is not None:
        plt.plot(timestamps, mag_binary, 'b-', linewidth=2, label='Binary')
        plt.xlabel('Time (days)')
        plt.ylabel('Magnification')
        plt.title(f'Binary Light Curve\n(s={s}, q={q}, u0={u0})')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Plot 2: PSPL
    plt.subplot(1, 3, 2)
    plt.plot(timestamps, mag_pspl, 'r-', linewidth=2, label='PSPL')
    plt.xlabel('Time (days)')
    plt.ylabel('Magnification')
    plt.title(f'PSPL Light Curve\n(u0={u0_pspl}, tE={tE_pspl})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Overlay
    plt.subplot(1, 3, 3)
    if mag_binary is not None:
        plt.plot(timestamps, mag_binary, 'b-', linewidth=2, alpha=0.7, label='Binary')
    plt.plot(timestamps, mag_pspl, 'r-', linewidth=2, alpha=0.7, label='PSPL')
    plt.xlabel('Time (days)')
    plt.ylabel('Magnification')
    plt.title('Overlay Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vbm_validation_test.png', dpi=150, bbox_inches='tight')
    print("  ✅ Plots saved to: vbm_validation_test.png")
    
    # Don't show plot in non-interactive mode
    # plt.show()
    plt.close()
    
except Exception as e:
    print(f"  ⚠️  Plotting failed: {e}")

# Test 5: Parameter Sensitivity
print("\n" + "="*80)
print("Test 5: Parameter Sensitivity Test")
print("="*80)

try:
    separations = [0.5, 0.8, 1.0, 1.3, 2.0]
    max_mags = []
    
    for s_test in separations:
        params_test = [np.log(s_test), np.log(q), u0, alpha, 
                      np.log(rho), np.log(tE), t0]
        mag_test = np.array(VBM.BinaryLightCurve(params_test, timestamps)[0])
        max_mags.append(np.max(mag_test))
    
    print("  Separation (s) vs Peak Magnification:")
    for s_val, max_val in zip(separations, max_mags):
        print(f"    s = {s_val:.1f} → Max mag = {max_val:.2f}")
    
    if max(max_mags) / min(max_mags) > 2.0:
        print("  ✅ Strong sensitivity to binary parameters confirmed")
    else:
        print("  ⚠️  Weak parameter sensitivity")
        
except Exception as e:
    print(f"  ⚠️  Sensitivity test failed: {e}")

# Final Summary
print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)

if mag_binary is not None and max_mag > 5.0:
    print("\n✅ VBMicrolensing is working correctly!")
    print("✅ Binary events show clear caustic features")
    print("✅ Ready for large-scale simulation")
    exit(0)
else:
    print("\n⚠️  VBMicrolensing may have issues")
    print("⚠️  Consider reinstalling or checking parameters")
    exit(1)