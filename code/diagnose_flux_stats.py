import h5py
import numpy as np
from pathlib import Path

# -------------------------------
# Robust path handling
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_1M_distinct.h5"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"HDF5 file not found at:\\n{DATA_PATH}")

print(f"Loaded file:\\n{DATA_PATH}\\n")

# -------------------------------
# Load data
# -------------------------------
with h5py.File(DATA_PATH, "r") as f:
    flux = f["flux"][:]          # (N_events, T)
    delta_t = f["delta_t"][:]    # (N_events, T)
    labels = f["labels"][:]      # 0=flat, 1=PSPL, 2=binary

# -------------------------------
# GLOBAL statistics
# -------------------------------
flux_all = flux.reshape(-1)
dt_all = delta_t.reshape(-1)

print("OVERALL STATISTICS:")
print(f"Flux    - Mean: {flux_all.mean():.4f}, Std: {flux_all.std():.4f}")
print(f"Delta_t - Mean: {dt_all.mean():.6f}, Std: {dt_all.std():.6f}\\n")

# -------------------------------
# Per-class diagnostics
# -------------------------------
CLASS_NAMES = {
    0: "Flat",
    1: "PSPL",
    2: "Binary"
}

for cls, name in CLASS_NAMES.items():
    mask = labels == cls
    class_flux = flux[mask].reshape(-1)

    print(f"{name} class:")
    print(
        f"  Flux - Mean: {class_flux.mean():.4f}, "
        f"Std: {class_flux.std():.4f}"
    )
    print()

# -------------------------------
# Sanity warning
# -------------------------------
if flux_all.std() < 0.5:
    print("DIAGNOSIS:")
    print("❌ STD TOO SMALL!")
    print("Likely causes:")
    print("  - Normalization applied twice")
    print("  - Std computed per-event instead of globally")
    print("  - Flux stored in magnitudes but treated as flux")
else:
    print("DIAGNOSIS:")
    print("✅ Flux statistics look physically reasonable")
