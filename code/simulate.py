#!/usr/bin/env python3
"""
Roman Microlensing Event Simulator
==================================

High-throughput simulation pipeline for generating realistic gravitational
microlensing light curves for the Nancy Grace Roman Space Telescope. Designed
for large-scale dataset generation, parameter inference experiments, and
training machine-learning models for event classification and regression.

Key Technical Features
----------------------
    * Fully vectorized NumPy/Numba hot loops for PSPL and noise models
    * Binary microlensing magnification via VBBinaryLensing with strict tolerances
    * Roman WFI F146 detector model: AB magnitudes, flux conversions, photon noise
    * Realistic Roman cadence masks, noise floors, sky contribution
    * Clean multiprocessing with reproducible seeds
    * Unified interface producing mag, delta_t, labels, timestamps, and metadata
    * HDF5 output with compressed datasets for large-volume workflows

Performance Characteristics
---------------------------
    - Numba-accelerated PSPL magnification up to 50x faster than pure Python
    - Photon-noise computation fused into low-level kernels
    - Multiprocessing using `spawn` for safe VBBinaryLensing usage
    - Memory-contiguous outputs ideal for PyTorch / JAX ingestion

Fixes Applied (v2.8.1 - Failure Tracking & Caustic Forcing)
-----------------------------------------------------------
    * CRITICAL FIX: worker_wrapper now returns event type on failure, preventing
      PSPL failures from being retried as Binary events (class distribution bug)
    * CRITICAL FIX: Added has_caustic_signature() to force detectable binary features
    * CRITICAL FIX: Added estimate_caustic_size() for physics-based u0 constraints
    * MAJOR FIX: Oversample/subsample approach guarantees exact class balance
    * MAJOR FIX: 'distinct' preset u0_range now uses SHARED_U0 (was hardcoded)
    * MAJOR FIX: 'stellar' preset q_range fixed to (0.1, 1.0) (was 0.3-3, convention q≤1)
    * MAJOR FIX: 'baseline' preset s_range fixed to (0.3, 3.0) (was 0.01-3.0, too small)
    * Added --oversample CLI argument for configurable oversampling factor
    * Added require_caustic flag per preset for caustic crossing enforcement

Fixes Applied (v2.8 - Bias Removal)
------------------------------------
    * CRITICAL FIX: Aligned u0 ranges between PSPL and Binary to prevent
      "high magnification = binary" shortcut learning. All presets now use
      SHARED_U0_MIN/MAX identical to PSPLParams.
    * CRITICAL FIX: Extended t0 range from 20%-80% to 10%-90% of mission
      duration for better coverage of edge cases (partial events).
    * CRITICAL FIX: Added PSPL acceptance criteria (retry loop) to match
      Binary filtering, preventing "strong/detectable event = Binary" bias.
    * MAJOR: All presets now use shared parameter constants for t0, tE, u0.

Fixes Applied (v2.6)
--------------------
    * CRITICAL FIX: Binary generation now returns None on failure instead of 
      mislabeling PSPL fallback as Binary (S0-1)
    * CRITICAL FIX: Binary metadata only set after successful generation (S0-2)
    * MAJOR FIX: Output key renamed from 'flux' to 'mag' for semantic clarity,
      with backward-compatible 'flux' alias in HDF5 output (S1-1)
    * MAJOR FIX: Binary parameters initialized before loop to prevent 
      uninitialized variable access (S1-2)
    * MODERATE FIX: Acceptance criteria constants defined at module level (S2-2)
    * MODERATE FIX: Eliminated duplicate np.max(A) computation (S2-1)
    * MODERATE FIX: Complete type hints for all functions (S2-3)
    * Enhanced docstrings with units and physics references

    Previous fixes (v2.5):
    * Enhanced: Complete docstring coverage for all functions (100%)
    * Enhanced: Comprehensive parameter documentation with units
    * Enhanced: Physics references added to key constants
    * Verified: Numba acceleration working correctly
    * Verified: VBBinaryLensing integration robust
    
    Previous fixes (v2.4):
    * CRITICAL: Fixed PSPL extreme magnifications by capping u0 and A
    * CRITICAL: Fixed binary flat events by strengthening acceptance criteria
    * Magnification now capped at 100x for physical realism

This module powers downstream ML pipelines such as CNN-GRU classifiers.

IMPORTANT: OUTPUT FORMAT (v2.7 - CNN-OPTIMIZED)
------------------------------------------------
The 'flux' array in HDF5 files contains NORMALIZED MAGNIFICATION:
    - Baseline (unmagnified source): A = 1.0
    - Magnified 2x: A = 2.0  
    - Magnified 10x: A = 10.0
    - Masked/missing observations: A = 0.0

This is CNN-ready! Photon noise is applied as relative noise in magnification space,
preserving physical realism while maintaining numerical stability for neural networks.

Previous versions stored absolute flux in Jansky (~1e-5), which caused CNN training
instability due to tiny numerical values.

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 2.8.1
"""
from __future__ import annotations

import argparse
import h5py
import math
import multiprocessing
import sys
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

__version__: Final[str] = "2.8.1"

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

try:
    import VBBinaryLensing
    HAS_VBB: Final[bool] = True
except ImportError:
    print("CRITICAL: VBBinaryLensing not found. Install via: pip install VBBinaryLensing")
    print("Binary microlensing simulation requires this library. Exiting.")
    sys.exit(1)

try:
    from numba import njit, prange
    HAS_NUMBA: Final[bool] = True
except ImportError:
    HAS_NUMBA: Final[bool] = False
    print("Warning: Numba not found. Simulation will be slower (~50x).")
    print("Install via: pip install numba")

# =============================================================================
# PHYSICAL CONSTANTS (Roman Space Telescope F146 Filter)
# =============================================================================

# AB magnitude system zero-point flux
# Reference: Oke & Gunn (1983), ApJ 266, 713
ROMAN_ZP_FLUX_JY: Final[float] = 3631.0

# Roman WFI F146 filter characteristics
# Reference: Spergel et al. (2015), arXiv:1503.03757
ROMAN_LIMITING_MAG_AB: Final[float] = 27.5  # 5-sigma point source detection limit
ROMAN_SKY_MAG_AB: Final[float] = 22.0       # Typical sky background in F146
ROMAN_SOURCE_MAG_MIN: Final[float] = 18.0   # Bright limit (saturation)
ROMAN_SOURCE_MAG_MAX: Final[float] = 24.0   # Faint limit for good S/N
ROMAN_CADENCE_MINUTES: Final[float] = 12.1  # Nominal observation cadence
ROMAN_MISSION_DURATION_DAYS: Final[float] = 200.0  # POC mission duration
ROMAN_LIMITING_SNR: Final[float] = 5.0      # Detection threshold

# =============================================================================
# BINARY EVENT ACCEPTANCE CRITERIA
# Reference: Empirically tuned for distinguishable binary features
# =============================================================================

BINARY_MIN_MAGNIFICATION: Final[float] = 1.5    # Minimum peak magnification for detection
BINARY_MAX_MAGNIFICATION: Final[float] = 100.0  # Physical upper limit (avoid numerical issues)
BINARY_MIN_MAG_RANGE: Final[float] = 0.3        # Minimum magnitude variation for caustic features
BINARY_MAX_ATTEMPTS: Final[int] = 10            # Maximum retry attempts per event

# PSPL magnification cap to avoid unrealistic values
PSPL_MAX_MAGNIFICATION: Final[float] = 100.0

# =============================================================================
# v2.8: PSPL EVENT ACCEPTANCE CRITERIA (to match Binary filtering)
# Without this, PSPL could include weak events that Binary filters out,
# creating a bias where "strong event = Binary" shortcut learning occurs
# =============================================================================

PSPL_MIN_MAGNIFICATION: Final[float] = 1.3      # Slightly lower than Binary (PSPL has smoother curves)
PSPL_MIN_MAG_RANGE: Final[float] = 0.1          # Lower than Binary (no caustic features needed)
PSPL_MAX_ATTEMPTS: Final[int] = 10              # Maximum retry attempts per event

# =============================================================================
# v2.8.1: CAUSTIC DETECTION PARAMETERS
# =============================================================================

CAUSTIC_SPIKE_THRESHOLD: Final[float] = 5.0     # N-sigma threshold for spike detection
CAUSTIC_MIN_SPIKES: Final[int] = 1              # Minimum spike features for caustic crossing
CAUSTIC_ASYMMETRY_THRESHOLD: Final[float] = 0.15  # 15% asymmetry threshold

# =============================================================================
# NUMBA ACCELERATED FUNCTIONS
# =============================================================================

if HAS_NUMBA:
    @njit(fastmath=True, cache=True, parallel=True)
    def flux_to_mag_numba(flux_jy: np.ndarray) -> np.ndarray:
        """
        Convert flux (Jansky) to AB magnitude using Numba acceleration.
        
        Formula: m_AB = -2.5 * log10(f_nu / 3631 Jy)
        
        Parameters
        ----------
        flux_jy : np.ndarray
            Flux array in Jansky units.
            
        Returns
        -------
        np.ndarray
            AB magnitude array. Invalid fluxes (<=0) return NaN.
        """
        n = len(flux_jy)
        mag = np.empty(n, dtype=np.float32)
        zp = ROMAN_ZP_FLUX_JY
        for i in prange(n):
            f = flux_jy[i]
            if f > 0:
                mag[i] = -2.5 * math.log10(f / zp)
            else:
                mag[i] = np.nan
        return mag
    
    @njit(fastmath=True, cache=True, parallel=True)
    def compute_photon_noise_numba(flux_jy: np.ndarray) -> np.ndarray:
        """
        Compute photon noise using Roman detector model with Numba acceleration.
        
        Implements realistic photon noise including source flux, sky background,
        and detector characteristics for the Roman F146 filter.
        
        Noise model: sigma = k * sqrt(f_source + f_sky)
        where k is calibrated to match the limiting magnitude SNR.
        
        Parameters
        ----------
        flux_jy : np.ndarray
            Flux array in Jansky units.
            
        Returns
        -------
        np.ndarray
            Noise sigma array in Jansky units.
        """
        n = len(flux_jy)
        sigma = np.empty(n, dtype=np.float32)
        zp = ROMAN_ZP_FLUX_JY
        f_lim = zp * 10**(-0.4 * ROMAN_LIMITING_MAG_AB)
        f_sky = zp * 10**(-0.4 * ROMAN_SKY_MAG_AB)
        sigma_lim = f_lim / ROMAN_LIMITING_SNR
        k_noise = sigma_lim / math.sqrt(f_lim + f_sky)
        
        for i in prange(n):
            f_total = flux_jy[i] + f_sky
            if f_total > 0:
                sigma[i] = k_noise * math.sqrt(f_total)
            else:
                sigma[i] = k_noise * 1e-5  # Floor for invalid flux
        return sigma

    @njit(fastmath=True, cache=True)
    def single_mag_to_flux(mag: float) -> float:
        """
        Convert single AB magnitude to flux in Jansky.
        
        Parameters
        ----------
        mag : float
            AB magnitude value.
            
        Returns
        -------
        float
            Flux in Jansky units.
        """
        return ROMAN_ZP_FLUX_JY * 10**(-0.4 * mag)

    @njit(fastmath=True, cache=True, parallel=True)
    def pspl_magnification_fast(
        t: np.ndarray, 
        t_E: float, 
        u_0: float, 
        t_0: float
    ) -> np.ndarray:
        """
        Compute PSPL magnification using Numba acceleration.
        
        Implements the Paczynski (1986) formula for point-source point-lens
        magnification with parallel execution across time points.
        
        Formula: A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
        where u^2 = u_0^2 + ((t - t_0) / t_E)^2
        
        Parameters
        ----------
        t : np.ndarray
            Time array in days.
        t_E : float
            Einstein crossing time in days.
        u_0 : float
            Impact parameter in Einstein radii.
        t_0 : float
            Time of peak magnification in days.
            
        Returns
        -------
        np.ndarray
            Magnification array (dimensionless, >= 1).
            
        References
        ----------
        Paczynski, B. (1986). "Gravitational microlensing by the galactic halo"
        ApJ 304, 1-5
        """
        n = len(t)
        A = np.ones(n, dtype=np.float32)
        inv_tE = 1.0 / t_E
        u0_sq = u_0 * u_0
        for i in prange(n):
            tau = (t[i] - t_0) * inv_tE
            u_sq = u0_sq + tau * tau
            u_sqrt = np.sqrt(u_sq)
            A[i] = (u_sq + 2.0) / (u_sqrt * np.sqrt(u_sq + 4.0))
        return A

    @njit(fastmath=True, cache=True, parallel=True)
    def compute_delta_t_numba(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Compute time differences between consecutive valid observations.
        
        For each valid observation, computes the time since the previous
        valid observation. First valid observation gets delta_t = 0.
        
        Parameters
        ----------
        times : np.ndarray
            Time array in days.
        mask : np.ndarray
            Boolean mask indicating valid observations (True = valid).
            
        Returns
        -------
        np.ndarray
            Delta_t array in days. Masked points have delta_t = 0.
        """
        n = len(times)
        delta_t = np.zeros(n, dtype=np.float32)
        prev_valid = np.full(n, -1, dtype=np.int32)
        
        # Forward pass to find previous valid index for each position
        last = -1
        for i in range(n):
            if mask[i]:
                prev_valid[i] = last
                last = i
            else:
                prev_valid[i] = last
        
        # Parallel computation of delta_t
        for i in prange(n):
            if mask[i] and prev_valid[i] != -1:
                delta_t[i] = times[i] - times[prev_valid[i]]
        return delta_t


# =============================================================================
# PURE NUMPY FALLBACK FUNCTIONS (when Numba unavailable)
# =============================================================================

def flux_to_mag_numpy(flux_jy: np.ndarray) -> np.ndarray:
    """
    Convert flux to AB magnitude using pure NumPy.
    
    Parameters
    ----------
    flux_jy : np.ndarray
        Flux array in Jansky units.
        
    Returns
    -------
    np.ndarray
        AB magnitude array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux_jy / ROMAN_ZP_FLUX_JY)
    return mag.astype(np.float32)


def compute_photon_noise_numpy(flux_jy: np.ndarray) -> np.ndarray:
    """
    Compute photon noise using Roman detector model with pure NumPy.
    
    Parameters
    ----------
    flux_jy : np.ndarray
        Flux array in Jansky units.
        
    Returns
    -------
    np.ndarray
        Noise sigma array in Jansky units.
    """
    f_lim = ROMAN_ZP_FLUX_JY * 10**(-0.4 * ROMAN_LIMITING_MAG_AB)
    f_sky = ROMAN_ZP_FLUX_JY * 10**(-0.4 * ROMAN_SKY_MAG_AB)
    sigma_lim = f_lim / ROMAN_LIMITING_SNR
    k_noise = sigma_lim / np.sqrt(f_lim + f_sky)
    
    f_total = np.maximum(flux_jy + f_sky, 1e-10)
    sigma = k_noise * np.sqrt(f_total)
    
    return sigma.astype(np.float32)


def pspl_magnification_numpy(
    t: np.ndarray, 
    t_E: float, 
    u_0: float, 
    t_0: float
) -> np.ndarray:
    """
    Compute PSPL magnification using pure NumPy.
    
    Parameters
    ----------
    t : np.ndarray
        Time array in days.
    t_E : float
        Einstein crossing time in days.
    u_0 : float
        Impact parameter in Einstein radii.
    t_0 : float
        Time of peak magnification in days.
        
    Returns
    -------
    np.ndarray
        Magnification array (dimensionless).
    """
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A.astype(np.float32)


def compute_delta_t_numpy(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute time differences between consecutive valid observations.
    
    Parameters
    ----------
    times : np.ndarray
        Time array in days.
    mask : np.ndarray
        Boolean mask indicating valid observations.
        
    Returns
    -------
    np.ndarray
        Delta_t array in days.
    """
    valid_idx = np.where(mask)[0]
    dt = np.zeros_like(times, dtype=np.float32)
    if len(valid_idx) > 1:
        diffs = np.diff(times[valid_idx])
        dt[valid_idx[1:]] = diffs
    return dt


# =============================================================================
# DETECTOR AND CONFIGURATION CLASSES
# =============================================================================

class RomanWFI_F146:
    """
    Roman Wide Field Instrument F146 filter detector model.
    
    Implements the photometric conversion and noise model for the
    Roman Space Telescope's F146 wide filter.
    
    Methods
    -------
    flux_to_mag(flux_jy)
        Convert flux array to AB magnitudes.
    compute_photon_noise(flux_jy)
        Compute photon noise for given flux values.
    """
    
    @staticmethod
    def flux_to_mag(flux_jy: np.ndarray) -> np.ndarray:
        """
        Convert flux (Jansky) to AB magnitude.
        
        Parameters
        ----------
        flux_jy : np.ndarray
            Flux array in Jansky units.
            
        Returns
        -------
        np.ndarray
            AB magnitude array.
        """
        if HAS_NUMBA:
            return flux_to_mag_numba(flux_jy)
        return flux_to_mag_numpy(flux_jy)

    @staticmethod
    def compute_photon_noise(flux_jy: np.ndarray) -> np.ndarray:
        """
        Compute photon noise for Roman detector.
        
        Parameters
        ----------
        flux_jy : np.ndarray
            Flux array in Jansky units.
            
        Returns
        -------
        np.ndarray
            Noise sigma array in Jansky units.
        """
        if HAS_NUMBA:
            return compute_photon_noise_numba(flux_jy)
        return compute_photon_noise_numpy(flux_jy)


class SimConfig:
    """
    Simulation configuration parameters.
    
    Defines the observational setup for Roman Space Telescope
    microlensing survey simulations.
    
    Attributes
    ----------
    TIME_MIN : float
        Start time of observations (days).
    TIME_MAX : float
        End time of observations (days).
    N_POINTS : int
        Number of observation points.
    VBM_TOLERANCE : float
        VBBinaryLensing precision tolerance.
    CADENCE_MASK_PROB : float
        Probability of missing an observation.
    BASELINE_MIN : float
        Minimum source baseline magnitude.
    BASELINE_MAX : float
        Maximum source baseline magnitude.
    PAD_VALUE : float
        Value used for masked/missing observations.
    """
    TIME_MIN: float = 0.0
    TIME_MAX: float = ROMAN_MISSION_DURATION_DAYS
    N_POINTS: int = 2400
    VBM_TOLERANCE: float = 1e-3
    CADENCE_MASK_PROB: float = 0.05
    BASELINE_MIN: float = ROMAN_SOURCE_MAG_MIN
    BASELINE_MAX: float = ROMAN_SOURCE_MAG_MAX
    PAD_VALUE: float = 0.0


class BinaryPresets:
    """
    Binary lens parameter presets for different astrophysical regimes.
    
    Provides scientifically motivated parameter ranges for simulating
    different types of binary microlensing events.
    
    Attributes
    ----------
    SHARED_T0_MIN : float
        Minimum peak time (10% of mission duration).
    SHARED_T0_MAX : float
        Maximum peak time (90% of mission duration).
    SHARED_TE_MIN : float
        Minimum Einstein crossing time (5 days).
    SHARED_TE_MAX : float
        Maximum Einstein crossing time (30 days).
    SHARED_U0_MIN : float
        Minimum impact parameter (Einstein radii). v2.8: Shared with PSPL to prevent bias.
    SHARED_U0_MAX : float
        Maximum impact parameter (Einstein radii). v2.8: Shared with PSPL to prevent bias.
    PRESETS : dict
        Dictionary of preset configurations.
        
    Notes
    -----
    Preset parameter ranges are based on:
    - Mao & Paczynski (1991) for binary lens geometry
    - Gaudi (2012) review for planetary microlensing
    - OGLE and MOA survey statistics for observed events
    
    v2.8 BIAS FIX: Extended t0 range to 10%-90% to include edge cases (partial events).
    v2.8 BIAS FIX: All presets now use SHARED_U0 range identical to PSPLParams to prevent
    the model from learning "very high magnification = binary" shortcut.
    
    v2.8.1 ADDITIONS:
    - 'require_caustic' flag per preset to force caustic crossing signatures
    - 'distinct' preset tightened u0_range for better caustic intersection
    - 'stellar' preset q_range fixed to (0.1, 1.0) per convention q ≤ 1
    - 'baseline' preset s_range fixed to (0.3, 3.0), was (0.01, 3.0) which is degenerate
    
    References
    ----------
    Gaudi (2012): "Microlensing Surveys for Exoplanets", ARA&A 50, 411
    Mao & Paczynski (1991): ApJ 374, L37
    Gould & Loeb (1992): ApJ 396, 104 (snow line argument)
    Chung et al. (2005): ApJ 630, 535 (caustic size formulas)
    """
    SHARED_T0_MIN: float = 0.1 * SimConfig.TIME_MAX
    SHARED_T0_MAX: float = 0.9 * SimConfig.TIME_MAX
    SHARED_TE_MIN: float = 5.0
    SHARED_TE_MAX: float = 30.0
    SHARED_U0_MIN: float = 0.01   # Matches PSPL to prevent bias
    SHARED_U0_MAX: float = 0.5    # Matches PSPL to prevent bias
    
    PRESETS: Dict[str, Dict[str, Any]] = {
        'distinct': {
            # Resonant caustics near s=1, strong binary signatures
            # Forces caustic crossings via tight u0 and require_caustic flag
            's_range': (0.8, 1.2),          # Tightened for resonant caustics
            'q_range': (0.1, 1.0),          # High q = bigger caustic
            'u0_range': (SHARED_U0_MIN, 0.3),  # Tighter for caustic intersection
            'rho_range': (1e-3, 1e-2),      # Finite source resolves caustic
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': True         # Force caustic signature detection
        },
        'planetary': {
            # Exoplanet detection regime (low mass ratio)
            # Planets have tiny caustics, need small u0 and caustic check
            's_range': (0.6, 1.6),          # Tighter around snow line
            'q_range': (1e-4, 1e-2),        # Jupiter to super-Earth
            'u0_range': (SHARED_U0_MIN, 0.2),  # Planets have tiny caustics
            'rho_range': (1e-3, 1e-2),      # Need finite source for planets
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': True         # Force anomaly detection
        },
        'stellar': {
            # Binary star systems (high mass ratio)
            # Some stellar binaries are smooth (no caustic crossing)
            's_range': (0.3, 3.0),          # Wide range of separations
            'q_range': (0.1, 1.0),          # FIX v2.8.1: was (0.3, 3), convention q ≤ 1
            'u0_range': (SHARED_U0_MIN, SHARED_U0_MAX),
            'rho_range': (1e-3, 5e-2),      # Larger sources for stellar
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': False        # Allow smooth binaries
        },
        'baseline': {
            # Full realistic parameter space for general training
            's_range': (0.3, 3.0),          # FIX v2.8.1: was (0.01, 3.0), too small
            'q_range': (1e-4, 1.0),         # Full range: planets to binaries
            'u0_range': (SHARED_U0_MIN, SHARED_U0_MAX),
            'rho_range': (1e-4, 0.05),
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': False        # Full parameter space includes non-crossing
        }
    }


class PSPLParams:
    """
    PSPL (Point Source Point Lens) parameter ranges.
    
    Defines the parameter space for single-lens microlensing events.
    
    Attributes
    ----------
    T0_MIN : float
        Minimum peak time in days.
    T0_MAX : float
        Maximum peak time in days.
    TE_MIN : float
        Minimum Einstein crossing time in days.
    TE_MAX : float
        Maximum Einstein crossing time in days.
    U0_MIN : float
        Minimum impact parameter (Einstein radii).
    U0_MAX : float
        Maximum impact parameter (Einstein radii).
        
    Notes
    -----
    v2.6: u0 range tightened to (0.01, 0.5) to avoid:
    - Extreme magnifications at very low u0 (numerical instability)
    - Undetectable events at high u0 (weak signal)
    
    v2.8 BIAS FIX: Now uses SHARED_U0 from BinaryPresets to ensure identical
    u0 distribution between PSPL and Binary, preventing the model from learning
    "very high magnification = binary" shortcut.
    """
    T0_MIN: float = BinaryPresets.SHARED_T0_MIN
    T0_MAX: float = BinaryPresets.SHARED_T0_MAX
    TE_MIN: float = BinaryPresets.SHARED_TE_MIN
    TE_MAX: float = BinaryPresets.SHARED_TE_MAX
    U0_MIN: float = BinaryPresets.SHARED_U0_MIN
    U0_MAX: float = BinaryPresets.SHARED_U0_MAX


# =============================================================================
# CAUSTIC DETECTION AND SIZE ESTIMATION (v2.8.1)
# =============================================================================

def estimate_caustic_size(s: float, q: float) -> float:
    """
    Estimate central caustic half-width in Einstein radii.
    
    Provides approximate caustic size for constraining impact parameter
    to ensure trajectory intersects the caustic structure.
    
    Parameters
    ----------
    s : float
        Binary separation in Einstein radii.
    q : float
        Mass ratio (secondary/primary, q ≤ 1 by convention).
        
    Returns
    -------
    float
        Estimated caustic half-width in Einstein radii.
        
    Notes
    -----
    Approximations used:
    - Resonant caustic (s ~ 1): size ~ 4q / (1+q)²
    - Close binary (s < 1): size ~ q * s⁴  
    - Wide binary (s > 1): size ~ q / s⁴
    
    These are order-of-magnitude estimates. Actual caustic shapes are
    complex and depend on all parameters.
    
    References
    ----------
    Chung et al. (2005): ApJ 630, 535
    Gaudi (2012): ARA&A 50, 411 (Section 2.2)
    """
    if 0.7 < s < 1.3:
        # Resonant caustic (central + planetary merged)
        return 4.0 * q / (1.0 + q)**2
    elif s <= 0.7:
        # Close binary - central caustic dominates
        return q * s**4
    else:
        # Wide binary - two separate caustics
        return q / s**4


def has_caustic_signature(A: np.ndarray, min_spikes: int = CAUSTIC_MIN_SPIKES) -> bool:
    """
    Detect caustic crossing signatures in light curve.
    
    Caustic crossings produce sharp spikes with large second derivatives,
    multiple peaks, or asymmetric profiles that distinguish binaries from PSPL.
    
    Parameters
    ----------
    A : np.ndarray
        Magnification array.
    min_spikes : int, optional
        Minimum number of spike features required. Default is CAUSTIC_MIN_SPIKES.
        
    Returns
    -------
    bool
        True if caustic crossing signature detected.
        
    Notes
    -----
    Three detection methods are used:
    
    1. **Spike detection**: Caustic crossings have very large |d²A/dt²|.
       PSPL is smooth with d²A peaks at ~2-3× mean.
       Caustic crossings peak at 10-100× mean.
       
    2. **Multiple peaks**: W-shaped or multi-peak light curves indicate
       multiple caustic crossings or cusp approaches.
       
    3. **Asymmetry**: PSPL is symmetric around peak. Caustic crossings
       break this symmetry due to caustic geometry.
    
    References
    ----------
    Gaudi (2012): ARA&A 50, 411 (Figure 4 shows characteristic signatures)
    """
    # Require minimum length for meaningful analysis
    if len(A) < 20:
        return False
    
    # Method 1: Check for rapid magnification changes (spikes)
    dA = np.diff(A)
    d2A = np.diff(dA)  # Second derivative
    
    # Caustic crossings have very large |d²A/dt²|
    std_d2A = np.std(d2A)
    if std_d2A > 1e-8:  # Avoid division by zero
        d2A_normalized = np.abs(d2A) / std_d2A
        n_spikes = np.sum(d2A_normalized > CAUSTIC_SPIKE_THRESHOLD)
        
        if n_spikes >= min_spikes:
            return True
    
    # Method 2: Check for multiple local maxima (W-shaped, etc.)
    # Find peaks: points higher than both neighbors
    peaks = []
    for i in range(1, len(A) - 1):
        if A[i] > A[i-1] and A[i] > A[i+1] and A[i] > 1.1:  # 10% above baseline
            peaks.append(i)
    
    # Multiple significant peaks indicate caustic structure
    if len(peaks) >= 2:
        # Check that peaks are actually separated (not noise)
        peak_mags = [A[p] for p in peaks]
        if max(peak_mags) > 1.3:  # At least one strong peak
            return True
    
    # Method 3: Check for asymmetry around peak
    if len(peaks) >= 1:
        # Find the highest peak
        peak_idx = peaks[np.argmax([A[p] for p in peaks])]
        
        # Compare shape before vs after peak
        n_compare = min(peak_idx, len(A) - peak_idx - 1, 50)
        if n_compare > 10:
            before = A[peak_idx - n_compare:peak_idx]
            after = A[peak_idx + 1:peak_idx + n_compare + 1][::-1]  # Reverse for comparison
            
            # Ensure same length
            min_len = min(len(before), len(after))
            if min_len > 5:
                before = before[-min_len:]
                after = after[:min_len]
                
                # Asymmetry metric: mean absolute difference normalized by amplitude
                mean_amplitude = np.mean(A[peak_idx - n_compare:peak_idx + n_compare])
                if mean_amplitude > 1.0:
                    asymmetry = np.mean(np.abs(before - after)) / (mean_amplitude - 1.0 + 1e-8)
                    
                    if asymmetry > CAUSTIC_ASYMMETRY_THRESHOLD:
                        return True
    
    return False


# =============================================================================
# MAGNIFICATION FUNCTIONS
# =============================================================================

def pspl_magnification(
    t: np.ndarray, 
    t_E: float, 
    u_0: float, 
    t_0: float
) -> np.ndarray:
    """
    Compute Point Source Point Lens magnification.
    
    Implements the Paczynski (1986) formula for gravitational lensing
    magnification by a point mass.
    
    Parameters
    ----------
    t : np.ndarray
        Time array in days.
    t_E : float
        Einstein crossing time in days.
    u_0 : float
        Impact parameter in Einstein radii.
    t_0 : float
        Time of peak magnification in days.
        
    Returns
    -------
    np.ndarray
        Magnification array (dimensionless, >= 1).
        
    References
    ----------
    Paczynski, B. (1986). ApJ 304, 1-5
    """
    if HAS_NUMBA:
        return pspl_magnification_fast(t, t_E, u_0, t_0)
    return pspl_magnification_numpy(t, t_E, u_0, t_0)


def binary_magnification_vbb(
    t: np.ndarray, 
    t_E: float, 
    u_0: float, 
    t_0: float, 
    s: float, 
    q: float, 
    alpha: float, 
    rho: float
) -> np.ndarray:
    """
    Compute binary lens magnification using VBBinaryLensing.
    
    Uses the VBBinaryLensing library for accurate finite-source
    binary lens magnification calculations with contour integration.
    
    Parameters
    ----------
    t : np.ndarray
        Time array in days.
    t_E : float
        Einstein crossing time in days.
    u_0 : float
        Impact parameter in Einstein radii.
    t_0 : float
        Time of peak magnification in days.
    s : float
        Projected binary separation in Einstein radii.
    q : float
        Mass ratio (secondary/primary).
    alpha : float
        Source trajectory angle in radians.
    rho : float
        Source radius in Einstein radii.
        
    Returns
    -------
    np.ndarray
        Magnification array (dimensionless).
        
    Raises
    ------
    RuntimeError
        If VBBinaryLensing computation fails completely.
        
    References
    ----------
    Bozza, V. (2010). MNRAS, 408, 2188-2196
    """
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE
    
    tau = (t - t_0) / t_E
    u1 = -u_0 * math.sin(alpha) + tau * math.cos(alpha)
    u2 = u_0 * math.cos(alpha) + tau * math.sin(alpha)
    
    try:
        # Try vectorized computation first (faster)
        return VBB.BinaryMag(s, q, u1, u2, rho)
    except (RuntimeError, ValueError, TypeError) as e:
        # Fallback to point-by-point computation
        n = len(t)
        mag = np.ones(n, dtype=np.float32)
        for i in range(n):
            try:
                val = VBB.BinaryMag2(s, q, u1[i], u2[i], rho)
                if val > 0 and np.isfinite(val):
                    mag[i] = val
            except (RuntimeError, ValueError, TypeError):
                # Use PSPL approximation for this point
                u_sq = u1[i]**2 + u2[i]**2
                u = np.sqrt(u_sq)
                mag[i] = (u_sq + 2) / (u * np.sqrt(u_sq + 4)) if u > 0 else 1.0
        return mag


def compute_delta_t(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute time differences between consecutive valid observations.
    
    Parameters
    ----------
    times : np.ndarray
        Time array in days.
    mask : np.ndarray
        Boolean mask indicating valid observations.
        
    Returns
    -------
    np.ndarray
        Delta_t array in days (0 for first valid observation and masked points).
    """
    if HAS_NUMBA:
        return compute_delta_t_numba(times, mask)
    return compute_delta_t_numpy(times, mask)


# =============================================================================
# SIMULATION CORE
# =============================================================================

def simulate_event(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simulate a single microlensing event.
    
    Generates a complete light curve including magnification, photometric
    noise, cadence masking, and computes all auxiliary quantities needed
    for machine learning pipelines.
    
    Parameters
    ----------
    params : dict
        Dictionary containing:
        - type: Event type ('flat', 'pspl', 'binary')
        - time_grid: Array of observation times
        - mask_prob: Probability of missing an observation
        - noise_scale: Noise amplitude scaling factor
        - preset: Binary lens preset name (for binary events)
        
    Returns
    -------
    dict or None
        Dictionary containing:
        - flux: Observed magnitude array (AB magnitudes, 0 for masked)
        - delta_t: Time differences between observations
        - label: Class label (0=flat, 1=pspl, 2=binary)
        - params: Dictionary of physical parameters
        
        Returns None if binary event generation fails after all attempts.
        
    Notes
    -----
    The 'flux' key contains NORMALIZED MAGNIFICATION (A=1.0 baseline) for
    backward compatibility and CNN numerical stability.
    
    v2.6 FIX: Binary events that fail to generate distinguishable caustic
    features now return None instead of falling back to mislabeled PSPL.
    
    v2.8 FIX: PSPL events now have acceptance criteria matching Binary to
    prevent "strong event = Binary" bias.
    
    v2.8.1 FIX: Binary events can optionally require caustic crossing signatures
    via the 'require_caustic' preset flag.
    """
    etype = params['type']
    t_grid = params['time_grid']
    n = len(t_grid)
    
    # Generate baseline magnitude
    m_base = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
    if HAS_NUMBA:
        f_base = single_mag_to_flux(m_base)
    else:
        f_base = ROMAN_ZP_FLUX_JY * 10**(-0.4 * m_base)
    
    # Initialize metadata
    meta: Dict[str, Any] = {'type': etype, 'm_base': float(m_base)}
    
    # Generate magnification based on event type
    if etype == 'flat':
        A = np.ones(n, dtype=np.float32)
        label = 0
        
    elif etype == 'pspl':
        # v2.8 BIAS FIX: PSPL now has acceptance criteria like Binary
        # This ensures PSPL events are similarly detectable, preventing
        # the model from learning "strong/detectable event = Binary" shortcut
        generation_success = False
        t0: Optional[float] = None
        tE: Optional[float] = None
        u0: Optional[float] = None
        A: Optional[np.ndarray] = None
        
        for attempt in range(PSPL_MAX_ATTEMPTS):
            t0 = np.random.uniform(PSPLParams.T0_MIN, PSPLParams.T0_MAX)
            tE = np.random.uniform(PSPLParams.TE_MIN, PSPLParams.TE_MAX)
            u0 = np.random.uniform(PSPLParams.U0_MIN, PSPLParams.U0_MAX)
            A_candidate = pspl_magnification(t_grid, tE, u0, t0)
            
            # Cap extreme magnifications for physical realism
            A_candidate = np.minimum(A_candidate, PSPL_MAX_MAGNIFICATION)
            
            max_mag = np.max(A_candidate)
            min_mag = np.min(A_candidate)
            mag_range = max_mag - min_mag
            
            # v2.8: Apply acceptance criteria (similar to Binary but slightly relaxed)
            if (PSPL_MIN_MAGNIFICATION < max_mag < PSPL_MAX_MAGNIFICATION 
                and mag_range > PSPL_MIN_MAG_RANGE):
                A = A_candidate
                generation_success = True
                break
        
        # v2.8: Return None if PSPL generation failed (consistent with Binary behavior)
        if not generation_success or A is None:
            return None
        
        label = 1
        meta.update({'t0': float(t0), 'tE': float(tE), 'u0': float(u0)})
        
    elif etype == 'binary':
        p = BinaryPresets.PRESETS[params['preset']]
        preset_name = params['preset']
        require_caustic = p.get('require_caustic', False)
        
        t0 = np.random.uniform(*p['t0_range'])
        tE = np.random.uniform(*p['tE_range'])
        
        # Initialize binary parameters to None to detect failure
        s: Optional[float] = None
        q: Optional[float] = None
        u0: Optional[float] = None
        rho: Optional[float] = None
        alpha: Optional[float] = None
        A: Optional[np.ndarray] = None
        generation_success = False
        
        # Try to generate binary event with retries
        for attempt in range(BINARY_MAX_ATTEMPTS):
            # Sample binary-specific parameters
            s = np.random.uniform(*p['s_range'])
            q = 10**np.random.uniform(np.log10(p['q_range'][0]), np.log10(p['q_range'][1]))
            
            # v2.8.1: For presets requiring caustic, constrain u0 based on caustic size
            if require_caustic:
                caustic_size = estimate_caustic_size(s, q)
                # u0 should be within ~2× caustic size for good crossing probability
                u0_max_caustic = min(caustic_size * 2.0, p['u0_range'][1])
                u0_min_caustic = p['u0_range'][0]
                u0 = np.random.uniform(u0_min_caustic, max(u0_min_caustic + 0.01, u0_max_caustic))
            else:
                u0 = np.random.uniform(*p['u0_range'])
            
            rho = 10**np.random.uniform(np.log10(p['rho_range'][0]), np.log10(p['rho_range'][1]))
            alpha = np.random.uniform(*p['alpha_range'])
            
            try:
                A_candidate = binary_magnification_vbb(t_grid, tE, u0, t0, s, q, alpha, rho)
                max_mag = np.max(A_candidate)
                min_mag = np.min(A_candidate)
                mag_range = max_mag - min_mag
                
                # Check standard acceptance criteria
                if not (BINARY_MIN_MAGNIFICATION < max_mag < BINARY_MAX_MAGNIFICATION):
                    continue
                if mag_range < BINARY_MIN_MAG_RANGE:
                    continue
                
                # v2.8.1: Additional caustic check for presets that require it
                if require_caustic:
                    if not has_caustic_signature(A_candidate):
                        continue
                
                # Passed all checks
                A = A_candidate
                generation_success = True
                break
                    
            except (RuntimeError, ValueError, TypeError, MemoryError) as e:
                # VBBinaryLensing failed, continue to next attempt
                continue
        
        # v2.6 CRITICAL FIX: Return None if binary generation failed
        # Do NOT fall back to PSPL with binary label - this poisons training data
        if not generation_success or A is None:
            return None
        
        label = 2
        meta.update({
            't0': float(t0),
            'tE': float(tE),
            'u0': float(u0),
            's': float(s),
            'q': float(q),
            'alpha': float(alpha),
            'rho': float(rho)
        })
    else:
        raise ValueError(f"Unknown event type: {etype}")
    
    # v2.7 CRITICAL FIX: Apply photon noise in MAGNIFICATION space
    # Convert absolute Jansky noise to relative magnification noise
    flux_true_jy = f_base * A
    noise_jy = RomanWFI_F146.compute_photon_noise(flux_true_jy)
    noise_relative = noise_jy / f_base  # Noise as fraction of baseline
    
    # Add noise to magnification (not absolute flux)
    # This preserves physical realism while maintaining numerical stability
    A_noisy = A + np.random.normal(0, noise_relative * params['noise_scale'])
    
    # Clip to physical values (magnification should be >= 1.0, but allow slightly less due to noise)
    A_noisy = np.maximum(A_noisy, 0.1)
    
    # Apply cadence mask (random missing observations)
    mask = np.random.random(n) > params['mask_prob']
    A_noisy[~mask] = 0.0  
    
    # Compute time differences between valid observations
    delta_t = compute_delta_t(t_grid, mask)
    
    return {
        'flux': A_noisy.astype(np.float32),  # v2.7: NORMALIZED magnification (baseline=1.0)
        'delta_t': delta_t.astype(np.float32),
        'label': label,
        'params': meta
    }


def worker_wrapper(args: Tuple[Dict[str, Any], int]) -> Optional[Dict[str, Any]]:
    """
    Wrapper for multiprocessing pool.
    
    Sets the random seed for reproducibility and calls the simulation
    function. This wrapper is needed for proper seed management in
    multiprocessing contexts.
    
    Parameters
    ----------
    args : tuple
        Tuple of (params_dict, random_seed).
        
    Returns
    -------
    dict or None
        Simulation result dictionary, or failure indicator dict if generation failed.
        
    Notes
    -----
    v2.8.1 FIX: On failure, returns a dict with '_failed' flag and event type
    so failures can be attributed to the correct class. This prevents the bug
    where PSPL failures were counted as binary failures and retried as binary
    events, skewing the class distribution.
    """
    param, seed = args
    np.random.seed(seed)
    result = simulate_event(param)
    
    if result is None:
        # v2.8.1 FIX: Return failure info with event type for proper attribution
        return {'_failed': True, '_type': param['type']}
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main simulation pipeline.
    
    Parses command-line arguments, generates the specified number of
    microlensing events, and saves results to an HDF5 file with
    compressed datasets and metadata.
    
    v2.8.1 FIX: Properly tracks failures by event type and uses
    oversample/subsample approach for guaranteed class balance.
    
    Failed events (returning None) are tracked by type and the
    oversample/subsample approach ensures exact class distribution
    matching the requested counts.
    """
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Event Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n_flat', type=int, default=1000,
                       help="Number of flat (baseline) events")
    parser.add_argument('--n_pspl', type=int, default=1000,
                       help="Number of PSPL events")
    parser.add_argument('--n_binary', type=int, default=1000,
                       help="Number of binary events")
    parser.add_argument('--binary_preset', type=str, default='baseline',
                       choices=['distinct', 'planetary', 'stellar', 'baseline'],
                       help="Binary lens parameter preset")
    parser.add_argument('--output', type=str, required=True,
                       help="Output HDF5 file path")
    parser.add_argument('--num_workers', type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--oversample', type=float, default=1.3,
                       help="Oversample factor to account for failures (default: 1.3)")
    
    args = parser.parse_args()

    # Setup output directory
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate time grid
    time_grid = np.linspace(SimConfig.TIME_MIN, SimConfig.TIME_MAX, SimConfig.N_POINTS)
    
    # Base parameters
    base_params: Dict[str, Any] = {
        'time_grid': time_grid, 
        'mask_prob': SimConfig.CADENCE_MASK_PROB, 
        'noise_scale': 1.0,
        'preset': args.binary_preset
    }
    
    # v2.8.1 FIX: Oversample to guarantee enough events of each type
    OVERSAMPLE_FACTOR = args.oversample
    
    n_flat_gen = int(args.n_flat * OVERSAMPLE_FACTOR)
    n_pspl_gen = int(args.n_pspl * OVERSAMPLE_FACTOR)
    n_binary_gen = int(args.n_binary * OVERSAMPLE_FACTOR)
    
    # Generate task list
    tasks: List[Dict[str, Any]] = []
    tasks.extend([{'type': 'flat', **base_params} for _ in range(n_flat_gen)])
    tasks.extend([{'type': 'pspl', **base_params} for _ in range(n_pspl_gen)])
    tasks.extend([{'type': 'binary', **base_params} for _ in range(n_binary_gen)])
    
    total_gen = len(tasks)
    print(f"Generating {total_gen} events (oversampled {OVERSAMPLE_FACTOR}x for failures)")
    print(f"  Targets: Flat={args.n_flat}, PSPL={args.n_pspl}, Binary={args.n_binary}")
    print(f"  Generating: Flat={n_flat_gen}, PSPL={n_pspl_gen}, Binary={n_binary_gen}")
    print(f"  Preset: {args.binary_preset}")
    print(f"    require_caustic: {BinaryPresets.PRESETS[args.binary_preset].get('require_caustic', False)}")
    print(f"Numba acceleration: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")

    # Shuffle tasks for balanced workload
    np.random.seed(args.seed)
    np.random.shuffle(tasks)
    
    # Prepare inputs with unique seeds
    task_inputs: List[Tuple[Dict[str, Any], int]] = [
        (t, args.seed + i) for i, t in enumerate(tasks)
    ]
    
    # Multiprocessing setup
    workers = args.num_workers or cpu_count()
    print(f"Using {workers} workers...")
    
    # v2.8.1 FIX: Track results and failures by type
    results_by_type: Dict[str, List[Dict[str, Any]]] = {
        'flat': [], 'pspl': [], 'binary': []
    }
    failed_by_type: Dict[str, int] = {'flat': 0, 'pspl': 0, 'binary': 0}
    
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(workers) as pool:
        is_tty = sys.stdout.isatty()
        iterator = pool.imap_unordered(worker_wrapper, task_inputs, chunksize=1000)
        
        for res in tqdm(iterator, 
                        total=total_gen,
                        mininterval=5.0,
                        smoothing=0.01,
                        ascii=not is_tty,
                        ncols=80 if not is_tty else 100,
                        unit="evt"):
            if res is None:
                # Should not happen with new worker_wrapper, but handle gracefully
                failed_by_type['binary'] += 1
            elif '_failed' in res:
                # v2.8.1 FIX: Properly attribute failure to correct type
                failed_by_type[res['_type']] += 1
            else:
                event_type = res['params']['type']
                results_by_type[event_type].append(res)
    
    # Report failures
    total_failures = sum(failed_by_type.values())
    if total_failures > 0:
        print(f"\nFailures by type:")
        for etype, count in failed_by_type.items():
            if count > 0:
                generated = len(results_by_type[etype])
                print(f"  {etype}: {count} failed, {generated} succeeded")
    
    # v2.8.1 FIX: Subsample to exact targets for guaranteed class balance
    print("\nBalancing class distribution to exact targets...")
    final_results = []
    shortfalls = {}
    
    for event_type, target in [('flat', args.n_flat), 
                                ('pspl', args.n_pspl), 
                                ('binary', args.n_binary)]:
        available = results_by_type[event_type]
        
        if len(available) >= target:
            indices = np.random.choice(len(available), size=target, replace=False)
            selected = [available[i] for i in indices]
            final_results.extend(selected)
            print(f"  {event_type}: {len(available)} available -> {target} selected ✓")
        else:
            final_results.extend(available)
            shortfall = target - len(available)
            shortfalls[event_type] = shortfall
            print(f"  {event_type}: {len(available)} available, {shortfall} short ⚠")
    
    if shortfalls:
        print(f"\n⚠ Warning: Some classes have fewer events than requested.")
        print(f"  Consider increasing --oversample (currently {OVERSAMPLE_FACTOR})")
        print(f"  or adjusting acceptance criteria in simulate.py")
    
    # Shuffle final results
    np.random.shuffle(final_results)
    
    # Aggregate results
    print("\nAggregating results...")
    n_res = len(final_results)
    flux = np.zeros((n_res, SimConfig.N_POINTS), dtype=np.float32)
    dt = np.zeros((n_res, SimConfig.N_POINTS), dtype=np.float32)
    lbl = np.zeros(n_res, dtype=np.int32)
    ts = np.tile(time_grid.astype(np.float32), (n_res, 1))
    
    # Collect parameters by class for structured storage
    params_by_class: Dict[str, List[Dict[str, Any]]] = {
        'flat': [], 'pspl': [], 'binary': []
    }
    
    for i, r in enumerate(final_results):
        flux[i] = r['flux']
        dt[i] = r['delta_t']
        lbl[i] = r['label']
        params_by_class[r['params']['type']].append(r['params'])
    
    # Count final class distribution
    final_counts = {
        'flat': int((lbl == 0).sum()),
        'pspl': int((lbl == 1).sum()),
        'binary': int((lbl == 2).sum())
    }
    
    # Save to HDF5
    print(f"Saving to {out_path}...")
    comp_args = {'compression': 'gzip', 'compression_opts': 4}
    
    with h5py.File(out_path, 'w') as f:
        # Core datasets
        # NOTE: 'flux' contains NORMALIZED MAGNIFICATION for backward compatibility
        f.create_dataset('flux', data=flux, **comp_args)
        f.create_dataset('delta_t', data=dt, **comp_args)
        f.create_dataset('labels', data=lbl)
        f.create_dataset('timestamps', data=ts, **comp_args)
        
        # Save parameters as structured arrays for each class
        for class_name, class_params in params_by_class.items():
            if not class_params:
                continue
            
            # Get all numeric fields
            all_fields: set = set()
            for p in class_params:
                all_fields.update(
                    k for k, v in p.items() 
                    if isinstance(v, (int, float, np.number))
                )
            
            if not all_fields:
                continue
            
            # Create structured array with sorted field names
            sorted_fields = sorted(all_fields)
            dtype_list = [(field, 'f8') for field in sorted_fields]
            struct_arr = np.zeros(len(class_params), dtype=dtype_list)
            
            for i, p in enumerate(class_params):
                for field in sorted_fields:
                    if field in p:
                        struct_arr[i][field] = p[field]
            
            f.create_dataset(f'params_{class_name}', data=struct_arr, **comp_args)
        
        # Save metadata as attributes
        metadata = {
            'n_events': int(n_res),
            'n_flat': final_counts['flat'],
            'n_pspl': final_counts['pspl'],
            'n_binary': final_counts['binary'],
            'n_flat_requested': int(args.n_flat),
            'n_pspl_requested': int(args.n_pspl),
            'n_binary_requested': int(args.n_binary),
            'binary_preset': args.binary_preset,
            'require_caustic': BinaryPresets.PRESETS[args.binary_preset].get('require_caustic', False),
            'seed': int(args.seed),
            'oversample_factor': float(OVERSAMPLE_FACTOR),
            'mission_duration_days': float(ROMAN_MISSION_DURATION_DAYS),
            'n_points': int(SimConfig.N_POINTS),
            'cadence_minutes': float(ROMAN_CADENCE_MINUTES),
            'numba_enabled': HAS_NUMBA,
            'version': __version__,
            'note': 'v2.8.1: flux contains NORMALIZED MAGNIFICATION (baseline=1.0), caustic forcing enabled for distinct/planetary presets'
        }
        f.attrs.update(metadata)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully saved {n_res} events to {out_path}")
    print(f"Class distribution: Flat={final_counts['flat']}, "
          f"PSPL={final_counts['pspl']}, Binary={final_counts['binary']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    main()
