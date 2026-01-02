from __future__ import annotations
import argparse
import h5py
import math
import multiprocessing
import sys
import threading
import time
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
from typing import Any, Dict, Final, Iterator, Optional, Tuple
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

__version__: Final[str] = "7.3.6"

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

try:
    import VBBinaryLensing
    HAS_VBB: Final[bool] = True
except ImportError:
    print("ERROR: VBBinaryLensing not found. Install via: pip install VBBinaryLensing")
    sys.exit(1)

try:
    from numba import njit, prange
    HAS_NUMBA: Final[bool] = True
except ImportError:
    HAS_NUMBA: Final[bool] = False
    print("WARNING: Numba not found. Install for 50x speedup: pip install numba")

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

ROMAN_ZP_FLUX_JY: Final[float] = 3631.0
ROMAN_LIMITING_MAG_AB: Final[float] = 27.5
ROMAN_SKY_MAG_AB: Final[float] = 22.0
ROMAN_SOURCE_MAG_MIN: Final[float] = 18.0
ROMAN_SOURCE_MAG_MAX: Final[float] = 24.0
ROMAN_CADENCE_MINUTES: Final[float] = 15.0
ROMAN_SEASON_DURATION_DAYS: Final[float] = 72.0
ROMAN_SEASON_N_POINTS: Final[int] = int(ROMAN_SEASON_DURATION_DAYS * 24 * 60 / ROMAN_CADENCE_MINUTES)
ROMAN_LIMITING_SNR: Final[float] = 5.0
ROMAN_CADENCE_DAYS: Final[float] = ROMAN_CADENCE_MINUTES / (24.0 * 60.0)

# =============================================================================
# EVENT ACCEPTANCE CRITERIA
# =============================================================================

BINARY_MIN_MAGNIFICATION: Final[float] = 1.5
BINARY_MAX_MAGNIFICATION: Final[float] = 100.0
BINARY_MIN_MAG_RANGE: Final[float] = 0.3
BINARY_MAX_ATTEMPTS: Final[int] = 10

PSPL_MAX_MAGNIFICATION: Final[float] = 100.0
PSPL_MIN_MAGNIFICATION: Final[float] = 1.3
PSPL_MIN_MAG_RANGE: Final[float] = 0.1
PSPL_MAX_ATTEMPTS: Final[int] = 10

CAUSTIC_SPIKE_THRESHOLD: Final[float] = 5.0
CAUSTIC_MIN_SPIKES: Final[int] = 1
CAUSTIC_ASYMMETRY_THRESHOLD: Final[float] = 0.15
CAUSTIC_MIN_ANALYSIS_LENGTH: Final[int] = 20
CAUSTIC_PEAK_THRESHOLD: Final[float] = 1.1
CAUSTIC_STRONG_PEAK_THRESHOLD: Final[float] = 1.3
CAUSTIC_MIN_COMPARISON_POINTS: Final[int] = 5
CAUSTIC_MAX_COMPARISON_WINDOW: Final[int] = 50

# =============================================================================
# OTHER CONSTANTS
# =============================================================================

NOISE_FLOOR_FACTOR: Final[float] = 1e-5
MIN_MAGNIFICATION_CLIP: Final[float] = 0.5
DEFAULT_OVERSAMPLE_FACTOR: Final[float] = 1.5
MIN_U0_OFFSET: Final[float] = 0.01
CAUSTIC_U0_MULTIPLIER: Final[float] = 2.0

MP_CHUNK_SIZE: Final[int] = 25
MAX_TASKS_PER_CHILD: Final[int] = 200
TQDM_MIN_INTERVAL: Final[float] = 5.0
TQDM_SMOOTHING: Final[float] = 0.01
TQDM_NCOLS_TTY: Final[int] = 100
TQDM_NCOLS_NON_TTY: Final[int] = 80

# =============================================================================
# GLOBAL WORKER STATE
# =============================================================================

_WORK_TIME_GRID: Optional[np.ndarray] = None
_WORK_MASK_PROB: Optional[float] = None
_WORK_PRESET: Optional[str] = None


def _init_worker(time_grid: np.ndarray, mask_prob: float, preset: str) -> None:
    """Initialize worker process with shared data."""
    global _WORK_TIME_GRID, _WORK_MASK_PROB, _WORK_PRESET
    _WORK_TIME_GRID = time_grid
    _WORK_MASK_PROB = mask_prob
    _WORK_PRESET = preset


# =============================================================================
# NUMBA ACCELERATED FUNCTIONS
# =============================================================================

if HAS_NUMBA:
    @njit(fastmath=True, cache=True, parallel=True)
    def compute_photon_noise_numba(flux_jy: np.ndarray) -> np.ndarray:
        """Compute photon noise using Roman detector model."""
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
                sigma[i] = k_noise * NOISE_FLOOR_FACTOR
        return sigma

    @njit(fastmath=True, cache=True)
    def single_mag_to_flux(mag: float) -> float:
        """Convert single AB magnitude to flux in Jansky."""
        return ROMAN_ZP_FLUX_JY * 10**(-0.4 * mag)

    @njit(fastmath=True, cache=True, parallel=True)
    def pspl_magnification_fast(
        t: np.ndarray,
        t_E: float,
        u_0: float,
        t_0: float
    ) -> np.ndarray:
        """Compute PSPL magnification with Numba acceleration."""
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

    @njit(fastmath=True, cache=True)
    def compute_delta_t_numba(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute time differences between consecutive valid observations."""
        n = len(times)
        delta_t = np.zeros(n, dtype=np.float32)
        prev_valid = np.full(n, -1, dtype=np.int32)

        last = -1
        for i in range(n):
            if mask[i]:
                prev_valid[i] = last
                last = i
            else:
                prev_valid[i] = last

        for i in range(n):
            if mask[i] and prev_valid[i] != -1:
                delta_t[i] = times[i] - times[prev_valid[i]]
        return delta_t


# =============================================================================
# PURE NUMPY FALLBACK FUNCTIONS
# =============================================================================

def compute_photon_noise_numpy(flux_jy: np.ndarray) -> np.ndarray:
    """Compute photon noise using Roman detector model with pure NumPy."""
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
    """Compute PSPL magnification using pure NumPy."""
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A.astype(np.float32)


def compute_delta_t_numpy(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute time differences between consecutive valid observations."""
    valid_idx = np.where(mask)[0]
    dt = np.zeros_like(times, dtype=np.float32)
    if len(valid_idx) > 1:
        diffs = np.diff(times[valid_idx])
        dt[valid_idx[1:]] = diffs
    return dt


# =============================================================================
# DETECTOR AND CONFIGURATION
# =============================================================================

class RomanWFI_F146:
    """Roman WFI F146 filter detector model."""

    @staticmethod
    def compute_photon_noise(flux_jy: np.ndarray) -> np.ndarray:
        """Compute photon noise for Roman detector."""
        if HAS_NUMBA:
            return compute_photon_noise_numba(flux_jy)
        return compute_photon_noise_numpy(flux_jy)


class SimConfig:
    """Simulation configuration parameters."""
    TIME_MIN: float = 0.0
    TIME_MAX: float = ROMAN_SEASON_DURATION_DAYS
    N_POINTS: int = ROMAN_SEASON_N_POINTS
    VBM_TOLERANCE: float = 1e-3
    CADENCE_MASK_PROB: float = 0.05
    BASELINE_MIN: float = ROMAN_SOURCE_MAG_MIN
    BASELINE_MAX: float = ROMAN_SOURCE_MAG_MAX
    PAD_VALUE: float = 0.0


def generate_time_grid() -> np.ndarray:
    """Generate observation time grid with exact 15-minute cadence."""
    return SimConfig.TIME_MIN + np.arange(SimConfig.N_POINTS) * ROMAN_CADENCE_DAYS


class BinaryPresets:
    """Binary lens parameter presets."""
    
    SHARED_T0_MIN: float = 0.25 * SimConfig.TIME_MAX
    SHARED_T0_MAX: float = 0.75 * SimConfig.TIME_MAX
    SHARED_TE_MIN: float = 3.0
    SHARED_TE_MAX: float = 18.0
    SHARED_U0_MIN: float = 0.001
    SHARED_U0_MAX: float = 1.0

    PRESETS: Dict[str, Dict[str, Any]] = {
        'distinct': {
            's_range': (0.8, 1.2),
            'q_range': (0.1, 1.0),
            'u0_range': (SHARED_U0_MIN, 0.3),
            'rho_range': (1e-3, 1e-2),
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': True
        },
        'general': {
            's_range': (0.3, 3.0),
            'q_range': (1e-4, 1.0),
            'u0_range': (SHARED_U0_MIN, SHARED_U0_MAX),
            'rho_range': (1e-4, 0.05),
            'alpha_range': (0, 2*math.pi),
            't0_range': (SHARED_T0_MIN, SHARED_T0_MAX),
            'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX),
            'require_caustic': False
        }
    }


class PSPLParams:
    """PSPL parameter ranges."""
    T0_MIN: float = BinaryPresets.SHARED_T0_MIN
    T0_MAX: float = BinaryPresets.SHARED_T0_MAX
    TE_MIN: float = BinaryPresets.SHARED_TE_MIN
    TE_MAX: float = BinaryPresets.SHARED_TE_MAX
    U0_MIN: float = BinaryPresets.SHARED_U0_MIN
    U0_MAX: float = BinaryPresets.SHARED_U0_MAX


# =============================================================================
# CAUSTIC DETECTION
# =============================================================================

def estimate_caustic_size(s: float, q: float) -> float:
    """Estimate central caustic half-width in Einstein radii."""
    if 0.7 < s < 1.3:
        return 4.0 * q / (1.0 + q)**2
    elif s <= 0.7:
        return q * s**4
    else:
        return q / s**4


def has_caustic_signature(A: np.ndarray, min_spikes: int = CAUSTIC_MIN_SPIKES) -> bool:
    """Detect caustic crossing signatures in light curve."""
    if len(A) < CAUSTIC_MIN_ANALYSIS_LENGTH:
        return False

    dA = np.diff(A)
    d2A = np.diff(dA)
    std_d2A = np.std(d2A)
    if std_d2A > 1e-8:
        d2A_normalized = np.abs(d2A) / std_d2A
        n_spikes = np.sum(d2A_normalized > CAUSTIC_SPIKE_THRESHOLD)
        if n_spikes >= min_spikes:
            return True

    peaks = []
    for i in range(1, len(A) - 1):
        if A[i] > A[i-1] and A[i] > A[i+1] and A[i] > CAUSTIC_PEAK_THRESHOLD:
            peaks.append(i)

    if len(peaks) >= 2:
        peak_mags = [A[p] for p in peaks]
        if max(peak_mags) > CAUSTIC_STRONG_PEAK_THRESHOLD:
            return True

    if len(peaks) >= 1:
        peak_idx = peaks[np.argmax([A[p] for p in peaks])]
        n_compare = min(peak_idx, len(A) - peak_idx - 1, CAUSTIC_MAX_COMPARISON_WINDOW)
        if n_compare > CAUSTIC_MIN_COMPARISON_POINTS * 2:
            before = A[peak_idx - n_compare:peak_idx]
            after = A[peak_idx + 1:peak_idx + n_compare + 1][::-1]

            min_len = min(len(before), len(after))
            if min_len > CAUSTIC_MIN_COMPARISON_POINTS:
                before = before[-min_len:]
                after = after[:min_len]

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
    """Compute Point Source Point Lens magnification."""
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
    """Compute binary lens magnification using VBBinaryLensing."""
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE

    tau = (t - t_0) / t_E
    u1 = -u_0 * math.sin(alpha) + tau * math.cos(alpha)
    u2 = u_0 * math.cos(alpha) + tau * math.sin(alpha)

    try:
        result = VBB.BinaryMag(s, q, u1, u2, rho)
        return np.asarray(result, dtype=np.float32)
    except (RuntimeError, ValueError, TypeError):
        n = len(t)
        mag = np.ones(n, dtype=np.float32)
        for i in range(n):
            try:
                val = VBB.BinaryMag2(s, q, u1[i], u2[i], rho)
                if val > 0 and np.isfinite(val):
                    mag[i] = val
            except (RuntimeError, ValueError, TypeError):
                u_sq = u1[i]**2 + u2[i]**2
                u = np.sqrt(u_sq)
                mag[i] = (u_sq + 2) / (u * np.sqrt(u_sq + 4)) if u > 0 else 1.0
        return mag


def compute_delta_t(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute time differences between consecutive valid observations."""
    if HAS_NUMBA:
        return compute_delta_t_numba(times, mask)
    return compute_delta_t_numpy(times, mask)


# =============================================================================
# SIMULATION CORE
# =============================================================================

def simulate_event(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simulate a single microlensing event."""
    etype = params['type']
    t_grid = params['time_grid']
    n = len(t_grid)

    m_base = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
    if HAS_NUMBA:
        f_base = single_mag_to_flux(m_base)
    else:
        f_base = ROMAN_ZP_FLUX_JY * 10**(-0.4 * m_base)

    meta: Dict[str, Any] = {'type': etype, 'm_base': float(m_base)}

    if etype == 'flat':
        A = np.ones(n, dtype=np.float32)
        label = 0

    elif etype == 'pspl':
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

            A_candidate = np.minimum(A_candidate, PSPL_MAX_MAGNIFICATION)

            max_mag = np.max(A_candidate)
            min_mag = np.min(A_candidate)
            mag_range = max_mag - min_mag

            if (PSPL_MIN_MAGNIFICATION < max_mag < PSPL_MAX_MAGNIFICATION
                    and mag_range > PSPL_MIN_MAG_RANGE):
                A = A_candidate
                generation_success = True
                break

        if not generation_success or A is None:
            return None

        label = 1
        meta.update({'t0': float(t0), 'tE': float(tE), 'u0': float(u0)})

    elif etype == 'binary':
        p = BinaryPresets.PRESETS[params['preset']]
        require_caustic = p.get('require_caustic', False)

        t0 = np.random.uniform(*p['t0_range'])
        tE = np.random.uniform(*p['tE_range'])

        s: Optional[float] = None
        q: Optional[float] = None
        u0: Optional[float] = None
        rho: Optional[float] = None
        alpha: Optional[float] = None
        A: Optional[np.ndarray] = None
        generation_success = False

        for attempt in range(BINARY_MAX_ATTEMPTS):
            s = np.random.uniform(*p['s_range'])
            q = 10**np.random.uniform(np.log10(p['q_range'][0]), np.log10(p['q_range'][1]))

            if require_caustic:
                caustic_size = estimate_caustic_size(s, q)
                u0_max_caustic = min(caustic_size * CAUSTIC_U0_MULTIPLIER, p['u0_range'][1])
                u0_min_caustic = p['u0_range'][0]

                if u0_max_caustic <= u0_min_caustic:
                    continue

                u0 = np.random.uniform(u0_min_caustic, u0_max_caustic)
            else:
                u0 = np.random.uniform(*p['u0_range'])

            rho = 10**np.random.uniform(np.log10(p['rho_range'][0]), np.log10(p['rho_range'][1]))
            alpha = np.random.uniform(*p['alpha_range'])

            try:
                A_candidate = binary_magnification_vbb(t_grid, tE, u0, t0, s, q, alpha, rho)
                max_mag = np.max(A_candidate)
                min_mag = np.min(A_candidate)
                mag_range = max_mag - min_mag

                if not (BINARY_MIN_MAGNIFICATION < max_mag < BINARY_MAX_MAGNIFICATION):
                    continue
                if mag_range < BINARY_MIN_MAG_RANGE:
                    continue

                if require_caustic:
                    if not has_caustic_signature(A_candidate):
                        continue

                A = A_candidate
                generation_success = True
                break

            except (RuntimeError, ValueError, TypeError):
                continue

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

    flux_true_jy = f_base * A
    noise_jy = RomanWFI_F146.compute_photon_noise(flux_true_jy)
    flux_noisy_jy = flux_true_jy + np.random.normal(0, noise_jy)

    A_noisy = np.maximum(flux_noisy_jy / f_base, MIN_MAGNIFICATION_CLIP)

    mask = np.random.random(n) > params['mask_prob']
    A_noisy[~mask] = 0.0

    delta_t = compute_delta_t(t_grid, mask)

    return {
        'flux': A_noisy.astype(np.float32),
        'delta_t': delta_t.astype(np.float32),
        'label': label,
        'params': meta
    }


def worker_wrapper(args: Tuple[str, int]) -> Dict[str, Any]:
    """Worker wrapper for multiprocessing pool (never raises, always returns _type)."""
    etype, seed = args
    np.random.seed(seed)
    
    # Explicit initialization check (no asserts - they're ignorable under -O)
    if _WORK_TIME_GRID is None or _WORK_MASK_PROB is None or _WORK_PRESET is None:
        return {'_failed': True, '_type': etype, '_seed': seed, '_err': 'worker_not_initialized'}
    
    params = {
        'type': etype,
        'time_grid': _WORK_TIME_GRID,
        'mask_prob': _WORK_MASK_PROB,
        'preset': _WORK_PRESET,
        'noise_scale': 1.0,
    }
    
    # Full try/except to catch any uncaught exceptions
    try:
        result = simulate_event(params)
    except Exception as e:
        return {'_failed': True, '_type': etype, '_seed': seed, '_err': repr(e)}

    if result is None:
        return {'_failed': True, '_type': etype, '_seed': seed}

    # Always include _type even on success for robust inflight tracking
    result['_type'] = etype
    result['_seed'] = seed
    return result


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.oversample < 1.0:
        raise ValueError(f"Oversample factor must be >= 1.0, got {args.oversample}")
    if args.n_flat < 0 or args.n_pspl < 0 or args.n_binary < 0:
        raise ValueError("Event counts must be >= 0")
    if args.n_flat + args.n_pspl + args.n_binary == 0:
        raise ValueError("At least one event type must have n > 0")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main simulation pipeline with streaming HDF5 writes."""
    parser = argparse.ArgumentParser(
        description="Roman Microlensing Event Simulator v7.3.6 (Final Bulletproof)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n_flat', type=int, default=1000)
    parser.add_argument('--n_pspl', type=int, default=1000)
    parser.add_argument('--n_binary', type=int, default=1000)
    parser.add_argument('--binary_preset', type=str, default='general',
                        choices=['distinct', 'general'])
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--oversample', type=float, default=DEFAULT_OVERSAMPLE_FACTOR)

    args = parser.parse_args()
    validate_args(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time_grid = generate_time_grid()

    n_total = args.n_flat + args.n_pspl + args.n_binary
    n_points = SimConfig.N_POINTS

    max_tasks = int(np.ceil(n_total * args.oversample))

    print("=" * 60)
    print("Roman Microlensing Simulator v7.3.6 (Final Bulletproof)")
    print("=" * 60)
    print(f"Season: {SimConfig.TIME_MAX:.1f} days, {n_points} obs")
    print(f"Cadence: {ROMAN_CADENCE_MINUTES:.1f} min")
    print(f"Target: Flat={args.n_flat}, PSPL={args.n_pspl}, Binary={args.n_binary}")
    print(f"Total: {n_total:,} events")
    print(f"Preset: {args.binary_preset}")
    print(f"Caustic: {BinaryPresets.PRESETS[args.binary_preset]['require_caustic']}")
    print(f"Oversample: {args.oversample}x (max {max_tasks:,} tasks)")
    print(f"Output: {out_path}")
    print(f"Seed: {args.seed}")
    print(f"Numba: {'ON' if HAS_NUMBA else 'OFF'}")
    print("=" * 60)

    workers = args.num_workers or cpu_count()
    print(f"Using {workers} workers (max {MAX_TASKS_PER_CHILD} tasks/worker)...")

    comp_args = {
        'compression': 'lzf',
        'chunks': (min(1000, n_total), n_points)
    }
    comp_args_1d = {
        'compression': 'lzf',
        'chunks': (min(1000, n_total),)
    }

    dtype_params = np.dtype([
        ('m_base', 'f4'),
        ('t0', 'f4'),
        ('tE', 'f4'),
        ('u0', 'f4'),
        ('s', 'f4'),
        ('q', 'f4'),
        ('alpha', 'f4'),
        ('rho', 'f4'),
    ])

    quotas = {
        'flat': args.n_flat,
        'pspl': args.n_pspl,
        'binary': args.n_binary
    }
    
    write_pos = {
        'flat': 0,
        'pspl': 0,
        'binary': 0
    }
    
    # Track in-flight tasks per event type
    inflight = {
        'flat': 0,
        'pspl': 0,
        'binary': 0
    }
    
    row_offsets = {
        'flat': 0,
        'pspl': args.n_flat,
        'binary': args.n_flat + args.n_pspl
    }

    # Thread-safe state management
    state_lock = threading.Lock()
    done_event = threading.Event()
    submitted_holder = {'n': 0}

    def finalize_task_and_maybe_reserve_row(etype_task: str, is_success: bool) -> Optional[int]:
        """
        Atomically:
          - decrement inflight for the completed task
          - if is_success and quota not full, reserve a row (increment write_pos) and return it
          - set done_event if quotas filled
        
        IMPORTANT: etype_task must be a valid type from inflight dict.
        Malformed results should be handled before calling this function.
        """
        with state_lock:
            # Decrement per-type inflight
            if etype_task in inflight and inflight[etype_task] > 0:
                inflight[etype_task] -= 1

            if not is_success:
                # Check completion anyway
                if all(write_pos[k] >= quotas[k] for k in quotas):
                    done_event.set()
                return None

            # Reserve row only if quota still available
            if etype_task not in quotas or write_pos[etype_task] >= quotas[etype_task]:
                # Check completion anyway
                if all(write_pos[k] >= quotas[k] for k in quotas):
                    done_event.set()
                return None

            row = row_offsets[etype_task] + write_pos[etype_task]
            write_pos[etype_task] += 1

            if all(write_pos[k] >= quotas[k] for k in quotas):
                done_event.set()

            return row

    def task_stream_limited() -> Iterator[Tuple[str, int]]:
        """Generate tasks until quotas full or cap reached (quota + inflight aware)."""
        seed = args.seed
        etypes = ('flat', 'pspl', 'binary')
        rr = 0

        while True:
            with state_lock:
                if done_event.is_set() or submitted_holder['n'] >= max_tasks:
                    return
                
                # Only submit if (already written + already in-flight) still below quota
                needed = [
                    t for t in etypes 
                    if (write_pos[t] + inflight[t]) < quotas[t]
                ]
                
                if needed:
                    etype = needed[rr % len(needed)]
                    rr += 1
                    submitted_holder['n'] += 1
                    inflight[etype] += 1
                    task = (etype, seed)
                    seed += 1
                else:
                    # Check if anything is actually in-flight
                    inflight_total = sum(inflight.values())
                    if inflight_total <= 0:
                        # Nothing in-flight and nothing needed: either done or true stall
                        if all(write_pos[k] >= quotas[k] for k in quotas):
                            done_event.set()
                            return
                        # True stall - shouldn't happen, but fail fast
                        print(f"\nWARNING: Stall detected - no tasks in-flight, quotas not met")
                        print(f"  write_pos: {dict(write_pos)}")
                        print(f"  inflight: {dict(inflight)}")
                        print(f"  quotas: {dict(quotas)}")
                        return
                    task = None

            if task is not None:
                yield task
            else:
                # Everything remaining is currently in-flight; wait for results.
                time.sleep(0.01)

    failed_counts: Dict[str, int] = {}
    error_counts: Dict[str, int] = {}  # Track exception types
    accepted_counts = {'flat': 0, 'pspl': 0, 'binary': 0}
    
    mbase_min = float("inf")
    mbase_max = float("-inf")

    ctx = multiprocessing.get_context('spawn')

    print("\nGenerating and streaming to HDF5...")
    
    with h5py.File(out_path, 'w') as f:
        ds_flux = f.create_dataset('flux', shape=(n_total, n_points), dtype='f4', **comp_args)
        ds_dt = f.create_dataset('delta_t', shape=(n_total, n_points), dtype='f4', **comp_args)
        ds_labels = f.create_dataset('labels', shape=(n_total,), dtype='i4')
        ds_mbase = f.create_dataset('m_base', shape=(n_total,), dtype='f4', **comp_args_1d)
        ds_params = f.create_dataset('params', shape=(n_total,), dtype=dtype_params, **comp_args_1d)
        
        f.create_dataset('time_grid', data=time_grid.astype(np.float32), compression='lzf')

        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(time_grid, SimConfig.CADENCE_MASK_PROB, args.binary_preset),
            maxtasksperchild=MAX_TASKS_PER_CHILD,
        ) as pool:
            is_tty = sys.stdout.isatty()
            
            iterator = pool.imap_unordered(
                worker_wrapper,
                task_stream_limited(),
                chunksize=MP_CHUNK_SIZE
            )

            pbar = tqdm(
                total=n_total,
                mininterval=TQDM_MIN_INTERVAL,
                smoothing=TQDM_SMOOTHING,
                ascii=not is_tty,
                ncols=TQDM_NCOLS_NON_TTY if not is_tty else TQDM_NCOLS_TTY,
                unit="evt",
                desc="Writing"
            )

            done = False
            for res in iterator:
                # FATAL: Malformed result corrupts inflight accounting
                if (res is None) or (not isinstance(res, dict)):
                    pbar.close()
                    print("\nFATAL: Malformed result from pool; inflight accounting compromised.")
                    with state_lock:
                        print(f"  write_pos: {dict(write_pos)}")
                        print(f"  inflight:  {dict(inflight)}")
                        print(f"  quotas:    {dict(quotas)}")
                    pool.terminate()
                    pool.join()
                    sys.exit(3)

                # _type is source of truth for inflight accounting
                etype_task = res.get('_type', None)
                
                # FATAL: Missing _type corrupts inflight accounting
                if etype_task is None or etype_task not in inflight:
                    pbar.close()
                    print(f"\nFATAL: Result missing valid _type; inflight accounting compromised.")
                    print(f"  Got _type: {etype_task}")
                    print(f"  Result keys: {list(res.keys())}")
                    with state_lock:
                        print(f"  write_pos: {dict(write_pos)}")
                        print(f"  inflight:  {dict(inflight)}")
                    pool.terminate()
                    pool.join()
                    sys.exit(3)

                # Failure case - atomically handle inflight decrement
                if res.get('_failed'):
                    finalize_task_and_maybe_reserve_row(etype_task, is_success=False)
                    failed_counts[etype_task] = failed_counts.get(etype_task, 0) + 1
                    
                    # Track exception types if present
                    if '_err' in res:
                        err_key = res['_err'][:50]  # Truncate long errors
                        error_counts[err_key] = error_counts.get(err_key, 0) + 1
                    
                    with state_lock:
                        pbar.set_postfix({
                            "Ff": failed_counts.get('flat', 0),
                            "Pf": failed_counts.get('pspl', 0),
                            "Bf": failed_counts.get('binary', 0),
                            "Fi": inflight['flat'],
                            "Pi": inflight['pspl'],
                            "Bi": inflight['binary'],
                        })
                    continue

                # Validate params['type'] matches _type (accounting source of truth)
                ptype = res.get('params', {}).get('type', None)
                if ptype != etype_task:
                    finalize_task_and_maybe_reserve_row(etype_task, is_success=False)
                    failed_counts['unknown'] = failed_counts.get('unknown', 0) + 1
                    error_counts['type_mismatch'] = error_counts.get('type_mismatch', 0) + 1
                    continue
                
                # Atomically decrement inflight AND reserve row (using _type as source of truth)
                row = finalize_task_and_maybe_reserve_row(etype_task, is_success=True)
                if row is None:
                    # Quota already full for this type; just discard
                    continue

                ds_flux[row] = res['flux']
                ds_dt[row] = res['delta_t']
                ds_labels[row] = int(res['label'])
                
                mb = float(res['params']['m_base'])
                ds_mbase[row] = mb
                mbase_min = min(mbase_min, mb)
                mbase_max = max(mbase_max, mb)

                p = res['params']
                rowp = np.zeros((), dtype=dtype_params)
                rowp['m_base'] = p['m_base']
                rowp['t0'] = p.get('t0', np.nan)
                rowp['tE'] = p.get('tE', np.nan)
                rowp['u0'] = p.get('u0', np.nan)
                rowp['s'] = p.get('s', np.nan)
                rowp['q'] = p.get('q', np.nan)
                rowp['alpha'] = p.get('alpha', np.nan)
                rowp['rho'] = p.get('rho', np.nan)
                ds_params[row] = rowp

                accepted_counts[etype_task] += 1
                pbar.update(1)

                if done_event.is_set():
                    done = True
                    break

            pbar.close()

            # Explicit cleanup
            if done:
                pool.terminate()
                pool.join()

        # Verify quotas were met
        with state_lock:
            all_full = all(write_pos[k] >= quotas[k] for k in quotas)
        
        if not all_full:
            shortfalls = {k: quotas[k] - write_pos[k] for k in quotas if write_pos[k] < quotas[k]}
            print("\n" + "=" * 60)
            print("ERROR: Quotas not filled within oversample cap")
            print("=" * 60)
            print(f"Shortfalls: {shortfalls}")
            print(f"Max tasks cap: {max_tasks:,}")
            print(f"Tasks submitted: {submitted_holder['n']:,}")
            print(f"Accepted: {accepted_counts}")
            print(f"Failed: {dict(failed_counts)}")
            if error_counts:
                print(f"Exceptions: {dict(error_counts)}")
            print("\nSuggestions:")
            print(f"  1. Increase --oversample (currently {args.oversample})")
            print(f"  2. Relax acceptance criteria in simulate.py")
            print(f"  3. Check for VBBinaryLensing errors in logs")
            print("=" * 60)
            
            f.attrs['incomplete'] = True
            f.attrs['shortfalls'] = str(shortfalls)
            f.attrs['max_tasks_cap'] = max_tasks
            f.attrs['tasks_submitted'] = int(submitted_holder['n'])
            sys.exit(2)

        metadata = {
            'n_events': int(n_total),
            'n_flat': int(args.n_flat),
            'n_pspl': int(args.n_pspl),
            'n_binary': int(args.n_binary),
            'binary_preset': args.binary_preset,
            'require_caustic': BinaryPresets.PRESETS[args.binary_preset]['require_caustic'],
            'seed': int(args.seed),
            'oversample_factor': float(args.oversample),
            'max_tasks_cap': max_tasks,
            'tasks_submitted': int(submitted_holder['n']),
            'season_duration_days': float(ROMAN_SEASON_DURATION_DAYS),
            'n_points': int(n_points),
            'cadence_minutes': float(ROMAN_CADENCE_MINUTES),
            'cadence_days': float(ROMAN_CADENCE_DAYS),
            'time_grid_start': float(time_grid[0]),
            'time_grid_end': float(time_grid[-1]),
            'numba_enabled': HAS_NUMBA,
            'version': __version__,
            'm_base_min': mbase_min,
            'm_base_max': mbase_max,
        }
        f.attrs.update(metadata)

    file_size_gb = out_path.stat().st_size / 1e9

    print(f"\n{'='*60}")
    print(f"SUCCESS: {n_total:,} events -> {out_path}")
    print(f"Accepted: Flat={accepted_counts['flat']:,}, PSPL={accepted_counts['pspl']:,}, Binary={accepted_counts['binary']:,}")
    
    total_failed = sum(failed_counts.values())
    if total_failed > 0:
        print(f"Failed: {dict(failed_counts)}")
        acceptance_rate = n_total / (n_total + total_failed) * 100
        print(f"Acceptance rate: {acceptance_rate:.1f}%")
        if error_counts:
            print(f"Exceptions: {dict(error_counts)}")
    
    print(f"Tasks submitted: {submitted_holder['n']:,}")
    print(f"m_base: [{mbase_min:.2f}, {mbase_max:.2f}] mag")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"{'='*60}")


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
