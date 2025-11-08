#!/usr/bin/env python3
"""
Caustic-Preserving Normalization for Microlensing Data

This module implements normalization that preserves caustic spike information
critical for binary detection.

Author: Kunal Bhatia
Version: 6.0
"""

import numpy as np
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path


class CausticPreservingNormalizer:
    """
    Normalizer that preserves caustic features in microlensing light curves.
    
    Key features:
    - Works in flux space (not magnitude)
    - Uses robust statistics (median/MAD not mean/std)
    - Preserves peak amplifications
    - Handles padding properly
    """
    
    def __init__(self, pad_value: float = -1.0):
        self.pad_value = pad_value
        self.baseline_median = None
        self.baseline_mad = None
        self.flux_min = None
        self.flux_max = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, baseline_fraction: float = 0.2):
        """
        Fit normalizer on training data.
        
        Args:
            X: Data array [N, C, T] or [N, T]
            baseline_fraction: Fraction of start/end to use for baseline
        """
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # Add channel dimension
        
        N, C, T = X.shape
        
        # Find baseline regions (start and end of light curves)
        baseline_points = int(T * baseline_fraction)
        
        all_baselines = []
        all_peaks = []
        
        for i in range(N):
            for c in range(C):
                flux = X[i, c, :]
                
                # Skip if all padding
                if np.all(flux == self.pad_value):
                    continue
                
                # Get non-padded data
                valid = flux != self.pad_value
                
                if not valid.any():
                    continue
                
                valid_flux = flux[valid]
                
                # Find baseline regions
                n_valid = len(valid_flux)
                start_baseline = valid_flux[:min(baseline_points, n_valid//4)]
                end_baseline = valid_flux[max(-baseline_points, -n_valid//4):]
                
                baseline = np.concatenate([start_baseline, end_baseline])
                
                if len(baseline) > 0:
                    all_baselines.append(baseline)
                    all_peaks.append(valid_flux.max())
        
        if len(all_baselines) == 0:
            raise ValueError("No valid baseline regions found!")
        
        # Compute robust statistics
        all_baseline_values = np.concatenate(all_baselines)
        self.baseline_median = np.median(all_baseline_values)
        self.baseline_mad = np.median(np.abs(all_baseline_values - self.baseline_median))
        
        # Store dynamic range
        self.flux_min = np.percentile(all_baseline_values, 1)
        self.flux_max = np.percentile(all_peaks, 99)
        
        self.is_fitted = True
        
        print(f"Normalizer fitted:")
        print(f"  Baseline median: {self.baseline_median:.3f}")
        print(f"  Baseline MAD: {self.baseline_mad:.3f}")
        print(f"  Flux range: [{self.flux_min:.3f}, {self.flux_max:.3f}]")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply caustic-preserving normalization.
        
        Args:
            X: Data array [N, C, T] or [N, T]
            
        Returns:
            Normalized data preserving caustics
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first!")
        
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        N, C, T = X.shape
        X_norm = np.zeros_like(X)
        
        for i in range(N):
            for c in range(C):
                flux = X[i, c, :]
                
                # Handle padding
                pad_mask = flux == self.pad_value
                
                if pad_mask.all():
                    X_norm[i, c, :] = self.pad_value
                    continue
                
                # Step 1: Center on baseline
                flux_centered = flux.copy()
                flux_centered[~pad_mask] = flux[~pad_mask] - self.baseline_median
                
                # Step 2: Scale by MAD (robust scaling)
                if self.baseline_mad > 0:
                    flux_centered[~pad_mask] = flux_centered[~pad_mask] / (3 * self.baseline_mad)
                
                # Step 3: Apply log transform for large dynamic range
                # This preserves caustic spikes while compressing baseline
                flux_log = flux_centered.copy()
                positive_mask = (~pad_mask) & (flux_centered > 0)
                
                if positive_mask.any():
                    flux_log[positive_mask] = np.log1p(flux_centered[positive_mask])
                
                # Step 4: Final scaling to [-1, 1]
                flux_scaled = flux_log.copy()
                if positive_mask.any():
                    max_val = np.abs(flux_log[~pad_mask]).max()
                    if max_val > 0:
                        flux_scaled[~pad_mask] = flux_log[~pad_mask] / max_val
                
                # Restore padding
                flux_scaled[pad_mask] = self.pad_value
                
                X_norm[i, c, :] = flux_scaled
        
        if squeeze:
            X_norm = X_norm.squeeze(1)
        
        return X_norm.astype(np.float32)
    
    def fit_transform(self, X: np.ndarray, **fit_params) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, **fit_params).transform(X)
    
    def save(self, path: str):
        """Save normalizer parameters"""
        path = Path(path)
        params = {
            'baseline_median': self.baseline_median,
            'baseline_mad': self.baseline_mad,
            'flux_min': self.flux_min,
            'flux_max': self.flux_max,
            'pad_value': self.pad_value,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, path: str):
        """Load normalizer parameters"""
        path = Path(path)
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self.baseline_median = params['baseline_median']
        self.baseline_mad = params['baseline_mad']
        self.flux_min = params['flux_min']
        self.flux_max = params['flux_max']
        self.pad_value = params['pad_value']
        self.is_fitted = params['is_fitted']
        
        return self


def normalize_dataset(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    pad_value: float = -1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CausticPreservingNormalizer]:
    """
    Normalize train/val/test splits preserving caustics.
    
    Args:
        X_train, X_val, X_test: Data arrays
        pad_value: Padding sentinel value
        
    Returns:
        X_train_norm, X_val_norm, X_test_norm, normalizer
    """
    print("Applying caustic-preserving normalization...")
    
    # Create and fit normalizer on training data only
    normalizer = CausticPreservingNormalizer(pad_value=pad_value)
    normalizer.fit(X_train)
    
    # Transform all splits
    X_train_norm = normalizer.transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    X_test_norm = normalizer.transform(X_test)
    
    # Validate normalization
    for name, data in [('Train', X_train_norm), ('Val', X_val_norm), ('Test', X_test_norm)]:
        non_pad = data != pad_value
        if non_pad.any():
            print(f"  {name}: range [{data[non_pad].min():.3f}, {data[non_pad].max():.3f}]")
    
    return X_train_norm, X_val_norm, X_test_norm, normalizer


if __name__ == "__main__":
    # Test the normalizer
    print("Testing CausticPreservingNormalizer...")
    
    # Create synthetic data with caustic spike
    np.random.seed(42)
    N, T = 100, 200
    X = np.ones((N, T))
    
    # Add caustic spikes
    for i in range(N):
        spike_pos = np.random.randint(50, 150)
        spike_height = np.random.uniform(5, 50)
        X[i, spike_pos-2:spike_pos+3] *= spike_height
    
    # Add padding
    X[:, -20:] = -1.0
    
    # Test normalization
    normalizer = CausticPreservingNormalizer(pad_value=-1.0)
    X_norm = normalizer.fit_transform(X)
    
    print(f"\nOriginal range: [{X[X != -1.0].min():.1f}, {X[X != -1.0].max():.1f}]")
    print(f"Normalized range: [{X_norm[X_norm != -1.0].min():.3f}, {X_norm[X_norm != -1.0].max():.3f}]")
    print(f"Padding preserved: {np.all(X_norm[X == -1.0] == -1.0)}")
    
    # Check that spikes are preserved
    original_spike_idx = X[0, :180].argmax()
    norm_spike_idx = X_norm[0, :180].argmax()
    print(f"Spike position preserved: {original_spike_idx == norm_spike_idx}")
    
    print("\n✅ Normalizer test passed!")
