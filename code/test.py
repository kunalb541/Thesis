# test_vbm.py
import numpy as np
import VBBinaryLensing as VBBL

vbm = VBBL.VBBinaryLensing()
timestamps = np.linspace(-50, 50, 100)

# Parameters for obvious binary
s, q, u0, alpha, rho, tE, t0 = 1.0, 0.1, 0.05, 0.5, 0.001, 50.0, 0.0
params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]

try:
    mags = np.array(vbm.BinaryLightCurve(params, timestamps)[0])
    print(f"Max magnification: {mags.max():.2f}")
    print(f"Mean magnification: {mags.mean():.2f}")
    
    if mags.max() > 10:
        print("✅ VBMicrolensing working - clear caustic spike!")
    else:
        print("⚠️ No caustic spike - check parameters")
        
except Exception as e:
    print(f"❌ VBMicrolensing failed: {e}")
