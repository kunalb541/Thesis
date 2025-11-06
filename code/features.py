#!/usr/bin/env python3
"""
Physics-Informed Feature Extraction for Microlensing

Extracts features that highlight binary caustic signatures:
- Magnification derivatives (detect sharp spikes)
- Curvature (identify caustic crossings)
- Peak properties (binary vs PSPL distinction)
- High-frequency content (caustic = high frequency)

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0 - Multi-channel physics features
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional


def extract_physics_features(
    flux: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    pad_value: float = -1.0,
    smooth_sigma: float = 2.0
) -> np.ndarray:
    """
    Extract multi-channel physics-informed features from light curve.
    
    Creates 4-channel representation:
    1. Original flux (baseline)
    2. First derivative (rate of change)
    3. Second derivative (acceleration)
    4. Curvature (caustic sensitivity)
    
    Args:
        flux: Original flux array [T]
        timestamps: Time array [T] (optional, for non-uniform sampling)
        pad_value: Sentinel for padding
        smooth_sigma: Gaussian smoothing sigma for derivatives
        
    Returns:
        features: Multi-channel features [4, T]
    """
    # Identify non-padded regions
    non_pad_mask = (flux != pad_value)
    
    # Initialize channels
    n_timesteps = len(flux)
    features = np.zeros((4, n_timesteps), dtype=np.float32)
    
    # Channel 0: Original flux
    features[0] = flux
    
    if not non_pad_mask.any():
        # All padding - return zeros
        return features
    
    # Extract non-padded flux for processing
    flux_valid = flux[non_pad_mask]
    
    # Apply light smoothing to reduce noise before derivatives
    flux_smooth = gaussian_filter1d(flux_valid, sigma=smooth_sigma)
    
    # Channel 1: First derivative (magnification rate)
    if timestamps is not None:
        time_valid = timestamps[non_pad_mask]
        dt = np.gradient(time_valid)
        dmag_dt = np.gradient(flux_smooth) / dt
    else:
        dmag_dt = np.gradient(flux_smooth)
    
    features[1, non_pad_mask] = dmag_dt
    features[1, ~non_pad_mask] = pad_value
    
    # Channel 2: Second derivative (acceleration)
    if timestamps is not None:
        d2mag_dt2 = np.gradient(dmag_dt) / dt
    else:
        d2mag_dt2 = np.gradient(dmag_dt)
    
    features[2, non_pad_mask] = d2mag_dt2
    features[2, ~non_pad_mask] = pad_value
    
    # Channel 3: Curvature (caustic crossing indicator)
    # κ = |f''| / (1 + f'^2)^(3/2)
    numerator = np.abs(d2mag_dt2)
    denominator = (1 + dmag_dt**2)**1.5
    curvature = numerator / np.maximum(denominator, 1e-8)
    
    features[3, non_pad_mask] = curvature
    features[3, ~non_pad_mask] = pad_value
    
    return features


def extract_batch_features(
    X: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    pad_value: float = -1.0
) -> np.ndarray:
    """
    Extract physics features for entire dataset.
    
    Args:
        X: Data array [N, 1, T] or [N, T]
        timestamps: Time array [T]
        pad_value: Padding sentinel
        
    Returns:
        X_features: Multi-channel features [N, 4, T]
    """
    # Handle 2D input
    if X.ndim == 2:
        X = X[:, None, :]
    
    N, C, T = X.shape
    assert C == 1, "Input must be single-channel for feature extraction"
    
    print(f"Extracting physics features for {N} light curves...")
    
    # Preallocate output
    X_features = np.zeros((N, 4, T), dtype=np.float32)
    
    # Process each light curve
    for i in range(N):
        flux = X[i, 0]
        features = extract_physics_features(flux, timestamps, pad_value)
        X_features[i] = features
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{N}")
    
    print(f"✓ Feature extraction complete: {X_features.shape}")
    
    return X_features


def extract_statistical_features(
    flux: np.ndarray,
    pad_value: float = -1.0
) -> np.ndarray:
    """
    Extract statistical summary features (for comparison/ablation).
    
    Features:
    - Mean, median, std, min, max
    - Peak count, peak prominence
    - Asymmetry index
    - High-frequency power (FFT)
    
    Args:
        flux: Flux array [T]
        pad_value: Padding sentinel
        
    Returns:
        features: Statistical features [F]
    """
    non_pad_mask = (flux != pad_value)
    
    if not non_pad_mask.any():
        return np.zeros(10, dtype=np.float32)
    
    flux_valid = flux[non_pad_mask]
    features = []
    
    # Basic statistics
    features.append(np.mean(flux_valid))
    features.append(np.median(flux_valid))
    features.append(np.std(flux_valid))
    features.append(np.min(flux_valid))
    features.append(np.max(flux_valid))
    
    # Peak properties
    peaks, properties = signal.find_peaks(flux_valid, prominence=0.1)
    features.append(len(peaks))  # Peak count
    features.append(np.mean(properties['prominences']) if len(peaks) > 0 else 0)
    
    # Asymmetry index
    median_idx = len(flux_valid) // 2
    left_sum = np.sum(flux_valid[:median_idx])
    right_sum = np.sum(flux_valid[median_idx:])
    asymmetry = (right_sum - left_sum) / (right_sum + left_sum + 1e-8)
    features.append(asymmetry)
    
    # High-frequency content
    fft = np.fft.fft(flux_valid)
    high_freq_power = np.sum(np.abs(fft[len(fft)//4:]))
    features.append(high_freq_power)
    
    # Range
    features.append(np.max(flux_valid) - np.min(flux_valid))
    
    return np.array(features, dtype=np.float32)


def visualize_features(
    flux: np.ndarray,
    timestamps: np.ndarray,
    pad_value: float = -1.0,
    output_path: str = "feature_visualization.png"
):
    """
    Visualize physics features for a single light curve.
    
    Args:
        flux: Flux array [T]
        timestamps: Time array [T]
        pad_value: Padding sentinel
        output_path: Save path for plot
    """
    import matplotlib.pyplot as plt
    
    # Extract features
    features = extract_physics_features(flux, timestamps, pad_value)
    
    non_pad_mask = (flux != pad_value)
    time_valid = timestamps[non_pad_mask]
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    feature_names = [
        "Original Flux",
        "First Derivative (dF/dt)",
        "Second Derivative (d²F/dt²)",
        "Curvature"
    ]
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        feature_valid = features[i, non_pad_mask]
        ax.plot(time_valid, feature_valid, linewidth=1.5)
        ax.set_ylabel(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Highlight high curvature regions (potential caustics)
        if i == 3:
            high_curvature = feature_valid > np.percentile(feature_valid, 95)
            if high_curvature.any():
                ax.scatter(
                    time_valid[high_curvature],
                    feature_valid[high_curvature],
                    color='red', s=30, alpha=0.7,
                    label='High curvature (caustic?)'
                )
                ax.legend()
    
    axes[-1].set_xlabel("Time (days)", fontsize=11)
    plt.suptitle("Physics-Informed Feature Extraction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature visualization saved to: {output_path}")


if __name__ == "__main__":
    print("="*80)
    print("PHYSICS FEATURES MODULE SELF-TEST")
    print("="*80)
    
    # Create synthetic binary-like curve with sharp spike
    np.random.seed(42)
    T = 1500
    timestamps = np.linspace(0, 1000, T)
    
    # Base PSPL
    t0, u0, tE = 500, 0.1, 50
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    flux = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    
    # Add sharp caustic spike
    spike_idx = np.argmin(np.abs(timestamps - 480))
    flux[spike_idx-5:spike_idx+5] *= 2.0
    
    # Add padding
    flux[-100:] = -1.0
    
    # Extract features
    print("\nExtracting features for single light curve...")
    features = extract_physics_features(flux, timestamps)
    
    print(f"✓ Features shape: {features.shape}")
    print(f"  Channel 0 (flux): range [{features[0].min():.3f}, {features[0].max():.3f}]")
    print(f"  Channel 1 (dF/dt): range [{features[1, features[1] != -1.0].min():.3f}, "
          f"{features[1, features[1] != -1.0].max():.3f}]")
    print(f"  Channel 2 (d²F/dt²): range [{features[2, features[2] != -1.0].min():.3f}, "
          f"{features[2, features[2] != -1.0].max():.3f}]")
    print(f"  Channel 3 (curvature): range [{features[3, features[3] != -1.0].min():.3f}, "
          f"{features[3, features[3] != -1.0].max():.3f}]")
    
    # Test batch processing
    print("\nTesting batch feature extraction...")
    X_batch = np.random.randn(100, 1, T).astype(np.float32)
    X_features = extract_batch_features(X_batch, timestamps)
    print(f"✓ Batch features shape: {X_features.shape}")
    
    # Test statistical features
    print("\nExtracting statistical features...")
    stats = extract_statistical_features(flux)
    print(f"✓ Statistical features: {stats.shape}")
    print(f"  Features: {stats}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_features(flux, timestamps)
    
    print("\n✓ All tests passed!")
    print("="*80)
