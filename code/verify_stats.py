import h5py
import numpy as np

print("Loading full dataset...")
with h5py.File('../data/raw/train_1M_distinct.h5', 'r') as f:
    flux = f['flux'][:]
    delta_t = f['delta_t'][:]
    
print("Computing statistics on ALL non-zero values...")
flux_valid = flux[flux != 0]
dt_valid = delta_t[delta_t != 0]

print("\nCORRECT normalization statistics:")
print(f"flux_mean = {np.mean(flux_valid):.6f}")
print(f"flux_std = {np.std(flux_valid):.6f}")
print(f"delta_t_mean = {np.mean(dt_valid):.6f}")
print(f"delta_t_std = {np.std(dt_valid):.6f}")

print("\nThese should be:")
print("  flux_mean ≈ 19.83")
print("  flux_std ≈ 4.86")
print("  delta_t_mean ≈ 0.083")
print("  delta_t_std ≈ 0.027")
