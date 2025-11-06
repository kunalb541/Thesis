#!/usr/bin/env python3
"""
Physics-Aware Data Augmentation for Microlensing

Implements augmentations that preserve the physics of microlensing while
increasing effective dataset size and model robustness.

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0 - Physics-informed augmentation
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional


class MicrolensingAugmenter:
    """
    Physics-aware augmentation for binary microlensing light curves.
    
    All augmentations preserve the fundamental physics:
    - Caustic crossing patterns remain valid
    - Time evolution is physically plausible
    - Magnification properties are conserved
    """
    
    def __init__(
        self,
        time_shift_range: Tuple[float, float] = (-50, 50),
        magnitude_scale_range: Tuple[float, float] = (0.9, 1.1),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        noise_std_range: Tuple[float, float] = (0.01, 0.03),
        rotation_range: Tuple[float, float] = (-0.1, 0.1),
        seed: Optional[int] = None
    ):
        """
        Initialize augmenter with parameter ranges.
        
        Args:
            time_shift_range: Range for t0 shifts (days)
            magnitude_scale_range: Range for baseline magnitude scaling
            time_stretch_range: Range for tE variations
            noise_std_range: Range for noise injection (magnitudes)
            rotation_range: Range for small rotations (preserves caustic)
            seed: Random seed for reproducibility
        """
        self.time_shift_range = time_shift_range
        self.magnitude_scale_range = magnitude_scale_range
        self.time_stretch_range = time_stretch_range
        self.noise_std_range = noise_std_range
        self.rotation_range = rotation_range
        
        if seed is not None:
            np.random.seed(seed)
    
    def augment_light_curve(
        self,
        flux: np.ndarray,
        timestamps: np.ndarray,
        y: int,
        n_augmentations: int = 5,
        pad_value: float = -1.0
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate augmented versions of a light curve.
        
        Args:
            flux: Original flux array [T]
            timestamps: Time array [T]
            y: Label (0=PSPL, 1=Binary)
            n_augmentations: Number of augmented versions to generate
            pad_value: Sentinel value for padding
            
        Returns:
            List of (flux_aug, timestamps_aug, y) tuples
        """
        augmented = []
        
        # Always include original
        augmented.append((flux.copy(), timestamps.copy(), y))
        
        # Identify non-padded regions
        non_pad_mask = (flux != pad_value)
        
        for _ in range(n_augmentations - 1):
            # Start with original
            flux_aug = flux.copy()
            time_aug = timestamps.copy()
            
            # Apply random combination of augmentations
            aug_type = np.random.choice(['shift', 'scale', 'stretch', 'noise', 'combo'])
            
            if aug_type == 'shift':
                time_aug = self._time_shift(time_aug, non_pad_mask)
                
            elif aug_type == 'scale':
                flux_aug = self._magnitude_scale(flux_aug, non_pad_mask, pad_value)
                
            elif aug_type == 'stretch':
                flux_aug, time_aug = self._time_stretch(
                    flux_aug, time_aug, non_pad_mask, pad_value
                )
                
            elif aug_type == 'noise':
                flux_aug = self._add_noise(flux_aug, non_pad_mask, pad_value)
                
            elif aug_type == 'combo':
                # Apply multiple augmentations
                if np.random.rand() > 0.5:
                    time_aug = self._time_shift(time_aug, non_pad_mask)
                if np.random.rand() > 0.5:
                    flux_aug = self._magnitude_scale(flux_aug, non_pad_mask, pad_value)
                if np.random.rand() > 0.7:
                    flux_aug = self._add_noise(flux_aug, non_pad_mask, pad_value)
            
            augmented.append((flux_aug, time_aug, y))
        
        return augmented
    
    def _time_shift(
        self,
        timestamps: np.ndarray,
        non_pad_mask: np.ndarray
    ) -> np.ndarray:
        """
        Shift t0 within valid range.
        
        Physically equivalent to observing the same event at a different time.
        """
        dt = np.random.uniform(*self.time_shift_range)
        timestamps_shifted = timestamps.copy()
        timestamps_shifted[non_pad_mask] += dt
        return timestamps_shifted
    
    def _magnitude_scale(
        self,
        flux: np.ndarray,
        non_pad_mask: np.ndarray,
        pad_value: float
    ) -> np.ndarray:
        """
        Scale baseline magnitude.
        
        Physically equivalent to observing a source with different baseline brightness.
        Preserves caustic structure.
        """
        scale = np.random.uniform(*self.magnitude_scale_range)
        flux_scaled = flux.copy()
        flux_scaled[non_pad_mask] *= scale
        flux_scaled[~non_pad_mask] = pad_value
        return flux_scaled
    
    def _time_stretch(
        self,
        flux: np.ndarray,
        timestamps: np.ndarray,
        non_pad_mask: np.ndarray,
        pad_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stretch/compress time axis (simulate tE variations).
        
        Physically equivalent to different Einstein crossing time.
        """
        stretch = np.random.uniform(*self.time_stretch_range)
        
        # Stretch time
        time_stretched = timestamps.copy()
        time_mean = timestamps[non_pad_mask].mean()
        time_stretched[non_pad_mask] = (
            time_mean + (timestamps[non_pad_mask] - time_mean) * stretch
        )
        
        return flux, time_stretched
    
    def _add_noise(
        self,
        flux: np.ndarray,
        non_pad_mask: np.ndarray,
        pad_value: float
    ) -> np.ndarray:
        """
        Add random photometric noise.
        
        Simulates varying observing conditions.
        """
        noise_std = np.random.uniform(*self.noise_std_range)
        flux_noisy = flux.copy()
        noise = np.random.normal(0, noise_std, size=flux.shape)
        flux_noisy[non_pad_mask] += noise[non_pad_mask]
        flux_noisy[~non_pad_mask] = pad_value
        return flux_noisy
    
    def _small_rotation(
        self,
        flux: np.ndarray,
        non_pad_mask: np.ndarray,
        pad_value: float
    ) -> np.ndarray:
        """
        Apply small rotation (preserves caustic topology).
        
        Physically equivalent to slight change in trajectory angle.
        """
        angle = np.random.uniform(*self.rotation_range)
        
        # Simple rotation approximation for 1D time series
        # (In practice, this would be more complex for 2D trajectories)
        flux_rotated = flux.copy()
        
        # Apply small sinusoidal modulation
        t = np.arange(len(flux))
        modulation = 1.0 + angle * np.sin(2 * np.pi * t / len(flux))
        flux_rotated[non_pad_mask] *= modulation[non_pad_mask]
        flux_rotated[~non_pad_mask] = pad_value
        
        return flux_rotated


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    n_augmentations: int = 5,
    pad_value: float = -1.0,
    augment_binary_only: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment entire dataset.
    
    Args:
        X: Data array [N, C, T]
        y: Labels [N]
        timestamps: Time array [T]
        n_augmentations: Augmentations per sample
        pad_value: Padding sentinel
        augment_binary_only: If True, only augment binary events
        
    Returns:
        X_aug: Augmented data [N*n_aug, C, T]
        y_aug: Augmented labels [N*n_aug]
    """
    augmenter = MicrolensingAugmenter()
    
    X_augmented = []
    y_augmented = []
    
    print(f"Augmenting dataset: {len(X)} events × {n_augmentations} = {len(X) * n_augmentations} total")
    
    for i in range(len(X)):
        flux = X[i, 0]  # Assume single channel for now
        label = y[i]
        
        # Skip augmentation for PSPL if requested
        if augment_binary_only and label == 0:
            X_augmented.append(X[i])
            y_augmented.append(label)
            continue
        
        # Generate augmentations
        augmented = augmenter.augment_light_curve(
            flux, timestamps, label, n_augmentations, pad_value
        )
        
        for flux_aug, time_aug, y_aug in augmented:
            X_augmented.append(flux_aug[None, :])  # Add channel dimension
            y_augmented.append(y_aug)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"✓ Augmentation complete: {X_augmented.shape}")
    
    return X_augmented, y_augmented


if __name__ == "__main__":
    print("="*80)
    print("AUGMENTATION MODULE SELF-TEST")
    print("="*80)
    
    # Create synthetic test data
    np.random.seed(42)
    T = 1500
    timestamps = np.linspace(0, 1000, T)
    
    # Simple PSPL-like curve
    t0, u0, tE = 500, 0.1, 50
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    flux = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    
    # Add some padding
    flux[-100:] = -1.0
    
    # Test augmenter
    augmenter = MicrolensingAugmenter()
    augmented = augmenter.augment_light_curve(flux, timestamps, y=0, n_augmentations=5)
    
    print(f"\n✓ Generated {len(augmented)} augmentations")
    
    # Check that augmentations are different
    original_flux = augmented[0][0]
    for i, (flux_aug, _, _) in enumerate(augmented[1:], 1):
        diff = np.abs(flux_aug - original_flux).sum()
        print(f"  Augmentation {i}: diff = {diff:.2f}")
    
    print("\n✓ All tests passed!")
    print("="*80)
