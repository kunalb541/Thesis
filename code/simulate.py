import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
import math
import warnings
import time
import sys

warnings.filterwarnings("ignore")

# ============================================================================
# CRITICAL DEPENDENCY CHECK
# ============================================================================
try:
    import VBBinaryLensing
    HAS_VBB = True
    print("VBBinaryLensing detected: High-fidelity binary magnification enabled")
except ImportError:
    print("\nCRITICAL ERROR: VBBinaryLensing not available")
    print("This is required for accurate binary microlensing simulation.")
    print("\nInstallation instructions:")
    print("  pip install VBBinaryLensing")
    print("  or")
    print("  conda install -c conda-forge vbbinarylensing")
    print("\nExiting now.")
    sys.exit(1)

# ============================================================================
# OPTIMIZATION IMPORTS
# ============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("Numba JIT detected: Temporal encoding acceleration enabled")
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found - using vectorized NumPy (slower)")
    print("Install for maximum speed: conda install numba")

# ============================================================================
# CONFIGURATION
# ============================================================================
class SimConfig:
    """Core simulation parameters"""
    N_POINTS = 1000
    TIME_MIN = -100.0
    TIME_MAX = 100.0
    
    VBM_TOLERANCE = 1e-3
    MAX_BINARY_ATTEMPTS = 10
    
    CADENCE_MASK_PROB = 0.05
    MAG_ERROR_STD = 0.03
    BASELINE_MIN = 19.0
    BASELINE_MAX = 21.0
    
    PAD_VALUE = 0.0

class PSPLParams:
    """
    PSPL parameters strictly confined to ensure 'Rise and Fall' is visible.
    Prevents model from using cut-off tails to identify PSPLs.
    """
    T0_MIN = -40.0
    T0_MAX = 40.0
    
    TE_MIN = 5.0
    TE_MAX = 70.0
    
    # CRITICAL ANTI-BIAS: Allow extremely small u0 (high magnification)
    # Prevents model from assuming "High Mag = Binary"
    U0_MIN = 0.0001
    U0_MAX = 0.5

class BinaryPresets:
    """Binary lens topology presets for Roman Space Telescope readiness"""
    
    PRESETS = {
        'distinct': {
            'description': 'Resonant Caustics - Guaranteed crossings',
            's_range': (0.90, 1.10),
            'q_range': (0.1, 1.0),
            'u0_range': (0.0001, 0.4),
            'rho_range': (1e-4, 5e-3),
            'alpha_range': (0, 2*math.pi),
            't0_range': (-40.0, 40.0),  # Match PSPL constraints
            'tE_range': (5.0, 70.0),    # Match PSPL constraints
        },
        
        'planetary': {
            'description': 'Exoplanet focus - small mass ratios',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.0001, 0.01),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 80.0),
            'tE_range': (1.0, 80.0),
        },
        
        'stellar': {
            'description': 'Binary stars - equal masses',
            's_range': (0.3, 3.0),
            'q_range': (0.3, 1.0),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 80.0),
            'tE_range': (10.0, 80.0),
        },
        
        'baseline': {
            'description': 'Standard mixed population',
            's_range': (0.1, 3.0),
            'q_range': (0.0001, 1.0),
            'u0_range': (0.001, 1.0),
            'rho_range': (0.001, 0.1),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 80.0),
            'tE_range': (1.0, 80.0),
        }
    }

class ObservationalPresets:
    """Cadence and photometric error presets"""
    
    CADENCE_PRESETS = {
        'cadence_05': {
            'description': 'Space-based high cadence',
            'mask_prob': 0.05,
            'error': 0.05,
            'example': 'Roman Space Telescope'
        },
        'cadence_15': {
            'description': 'Good ground-based',
            'mask_prob': 0.15,
            'error': 0.10,
            'example': 'Excellent survey conditions'
        },
        'cadence_30': {
            'description': 'Typical ground-based',
            'mask_prob': 0.30,
            'error': 0.10,
            'example': 'LSST typical'
        },
        'cadence_50': {
            'description': 'Sparse ground-based',
            'mask_prob': 0.50,
            'error': 0.10,
            'example': 'Weather-limited'
        }
    }
    
    ERROR_PRESETS = {
        'error_003': {
            'description': 'Excellent space photometry',
            'mask_prob': 0.05,
            'error': 0.03,
            'example': 'JWST-quality'
        },
        'error_005': {
            'description': 'Space-based quality',
            'mask_prob': 0.05,
            'error': 0.05,
            'example': 'Roman Space Telescope'
        },
        'error_010': {
            'description': 'High-quality ground',
            'mask_prob': 0.05,
            'error': 0.10,
            'example': 'Professional observatories'
        },
        'error_015': {
            'description': 'Typical ground',
            'mask_prob': 0.05,
            'error': 0.15,
            'example': 'Wide-field surveys'
        }
    }

# ============================================================================
# MAGNIFICATION MODELS
# ============================================================================
def pspl_magnification(t: np.ndarray, t_E: float, u_0: float, t_0: float) -> np.ndarray:
    """
    Point Source Point Lens magnification.
    
    Args:
        t: Array of observation times
        t_E: Einstein crossing time
        u_0: Impact parameter
        t_0: Time of closest approach
    
    Returns:
        Magnification array
    """
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A

def binary_magnification_vbb(t: np.ndarray, t_E: float, u_0: float, t_0: float, 
                             s: float, q: float, alpha: float, rho: float) -> np.ndarray:
    """
    Binary lens magnification using VBBinaryLensing (high-fidelity).
    
    Args:
        t: Array of observation times
        t_E: Einstein crossing time
        u_0: Impact parameter
        t_0: Time of closest approach
        s: Binary separation in Einstein radii
        q: Mass ratio
        alpha: Source trajectory angle
        rho: Normalized source radius
    
    Returns:
        Magnification array with caustic features
    """
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE
    VBB.RelTol = SimConfig.VBM_TOLERANCE
    
    tau = (t - t_0) / t_E
    source_x = u_0 * np.cos(alpha) + tau * np.sin(alpha)
    source_y = u_0 * np.sin(alpha) - tau * np.cos(alpha)
    
    mag = np.zeros_like(t, dtype=np.float64)
    for i, (sx, sy) in enumerate(zip(source_x, source_y)):
        mag[i] = VBB.BinaryMag2(s, q, sx, sy, rho)
    
    return mag

# ============================================================================
# PARAMETER GENERATION
# ============================================================================
def generate_flat_params(n_events: int, seed: int = None) -> list:
    """Generate flat (no event) parameters"""
    if seed is not None:
        np.random.seed(seed)
    
    params = []
    for _ in range(n_events):
        m_source = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
        params.append({'m_source': m_source, 'type': 'flat'})
    
    return params

def generate_pspl_params(n_events: int, seed: int = None) -> list:
    """
    Generate PSPL parameters with temporal invariance.
    Uses log-uniform sampling for u0 to match statistical reality.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = []
    for _ in range(n_events):
        t_E = np.random.uniform(PSPLParams.TE_MIN, PSPLParams.TE_MAX)
        t_0 = np.random.uniform(PSPLParams.T0_MIN, PSPLParams.T0_MAX)
        
        # Log-uniform sampling favors small impact parameters
        u_0 = np.exp(np.random.uniform(np.log(PSPLParams.U0_MIN), np.log(PSPLParams.U0_MAX)))
        m_source = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
        
        params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t_0,
            'm_source': m_source,
            'type': 'pspl'
        })
    
    return params

def generate_binary_params(n_events: int, preset: str = 'baseline', seed: int = None) -> list:
    """Generate binary parameters from topology preset"""
    if seed is not None:
        np.random.seed(seed)
    
    if preset not in BinaryPresets.PRESETS:
        print(f"Warning: Unknown preset '{preset}', using 'baseline'")
        preset = 'baseline'
    
    config = BinaryPresets.PRESETS[preset]
    params = []
    
    for _ in range(n_events):
        t_E = np.random.uniform(config['tE_range'][0], config['tE_range'][1])
        t_0 = np.random.uniform(config['t0_range'][0], config['t0_range'][1])
        
        # Log-uniform for u0 and rho
        u0_min, u0_max = config['u0_range']
        u_0 = np.exp(np.random.uniform(np.log(u0_min), np.log(u0_max)))
        
        s = np.random.uniform(config['s_range'][0], config['s_range'][1])
        q = np.random.uniform(config['q_range'][0], config['q_range'][1])
        alpha = np.random.uniform(config['alpha_range'][0], config['alpha_range'][1])
        
        rho = np.exp(np.random.uniform(np.log(config['rho_range'][0]), np.log(config['rho_range'][1])))
        m_source = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
        
        params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t_0,
            's': s,
            'q': q,
            'alpha': alpha,
            'rho': rho,
            'm_source': m_source,
            'type': 'binary'
        })
    
    return params

# ============================================================================
# LIGHT CURVE GENERATION
# ============================================================================
def generate_light_curve(params: dict, event_type: str, timestamps: np.ndarray) -> np.ndarray:
    """Generate clean magnification light curve"""
    if event_type == 'flat':
        A = np.ones_like(timestamps, dtype=np.float64)
    elif event_type == 'pspl':
        A = pspl_magnification(timestamps, params['t_E'], params['u_0'], params['t_0'])
    else:  # binary
        A = binary_magnification_vbb(
            timestamps, params['t_E'], params['u_0'], params['t_0'],
            params['s'], params['q'], params['alpha'], params['rho']
        )
    
    return A

def add_observational_effects(flux: np.ndarray, error_mag: float, 
                              cadence_missing: float, pad_value: float) -> np.ndarray:
    """
    Add photometric noise and missing observations.
    
    Args:
        flux: Clean flux array
        error_mag: Photometric error standard deviation
        cadence_missing: Probability of missing observation
        pad_value: Value to use for missing data
    
    Returns:
        Observed flux with noise and gaps
    """
    flux_obs = flux.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, error_mag, size=len(flux))
    flux_obs = flux_obs * (1 + noise)
    
    # Apply masking
    mask = np.random.random(len(flux)) < cadence_missing
    flux_obs[mask] = pad_value
    
    # Floor flux to prevent negative values
    flux_obs[flux_obs != pad_value] = np.maximum(flux_obs[flux_obs != pad_value], 0.001)
    
    return flux_obs

def generate_single_event(args: tuple) -> tuple:
    """Worker function for parallel generation"""
    idx, params, event_type, timestamps, cadence, error, pad_value = args
    
    flux = generate_light_curve(params, event_type, timestamps)
    flux_obs = add_observational_effects(flux, error, cadence, pad_value)
    
    label = {'flat': 0, 'pspl': 1, 'binary': 2}[event_type]
    
    return flux_obs, label, params

# ============================================================================
# TEMPORAL ENCODING (CAUSALITY-PRESERVING)
# ============================================================================
if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def calculate_delta_t_per_event(timestamps: np.ndarray, flux_obs: np.ndarray, 
                                    pad_value: float) -> np.ndarray:
        """
        Calculate delta_t using Numba JIT compilation.
        Parallel processing across events for maximum speed.
        
        Critical: Only uses past observations - maintains causality.
        """
        n_events = flux_obs.shape[0]
        n_points = timestamps.shape[0]
        delta_t_array = np.zeros_like(flux_obs)
        
        for i in prange(n_events):
            last_obs_time = timestamps[0]
            
            for j in range(1, n_points):
                if flux_obs[i, j] != pad_value:
                    delta_t_array[i, j] = timestamps[j] - last_obs_time
                    last_obs_time = timestamps[j]
                else:
                    delta_t_array[i, j] = 0.0
        
        return delta_t_array
else:
    def calculate_delta_t_per_event(timestamps: np.ndarray, flux_obs: np.ndarray, 
                                    pad_value: float) -> np.ndarray:
        """
        Vectorized NumPy fallback for delta_t calculation.
        Slower than Numba but maintains causality.
        """
        n_events, n_points = flux_obs.shape
        time_grid = np.tile(timestamps, (n_events, 1))
        is_observed = (flux_obs != pad_value)
        
        observed_times = time_grid.copy()
        observed_times[~is_observed] = np.nan
        
        mask = ~np.isnan(observed_times)
        idx = np.maximum.accumulate(mask * np.arange(n_points), axis=1)
        last_valid_times = np.take_along_axis(observed_times, idx, axis=1)
        
        previous_last_obs = np.roll(last_valid_times, 1, axis=1)
        previous_last_obs[:, 0] = timestamps[0]
        
        delta_t = time_grid - previous_last_obs
        delta_t[~is_observed] = 0.0
        delta_t[:, 0] = 0.0
        
        return delta_t

# ============================================================================
# MAIN SIMULATION
# ============================================================================
def simulate_dataset(
    n_flat: int,
    n_pspl: int,
    n_binary: int,
    binary_preset: str = 'baseline',
    observational_preset: str = None,
    cadence_mask_prob: float = None,
    mag_error_std: float = None,
    num_workers: int = 1,
    seed: int = None,
    save_params: bool = True
) -> tuple:
    """
    Generate complete microlensing dataset for Roman Space Telescope readiness.
    
    Returns:
        (flux, delta_t, labels, timestamps, params_dict)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get observational parameters
    if observational_preset:
        if observational_preset in ObservationalPresets.CADENCE_PRESETS:
            obs_config = ObservationalPresets.CADENCE_PRESETS[observational_preset]
        elif observational_preset in ObservationalPresets.ERROR_PRESETS:
            obs_config = ObservationalPresets.ERROR_PRESETS[observational_preset]
        else:
            print(f"Warning: Unknown preset '{observational_preset}'")
            obs_config = {'mask_prob': SimConfig.CADENCE_MASK_PROB, 'error': SimConfig.MAG_ERROR_STD}
        
        cadence = obs_config['mask_prob']
        error = obs_config['error']
    else:
        cadence = cadence_mask_prob if cadence_mask_prob is not None else SimConfig.CADENCE_MASK_PROB
        error = mag_error_std if mag_error_std is not None else SimConfig.MAG_ERROR_STD
    
    # Generate timestamps
    timestamps = np.linspace(SimConfig.TIME_MIN, SimConfig.TIME_MAX, SimConfig.N_POINTS, dtype=np.float64)
    
    print("=" * 80)
    print("Roman Space Telescope Microlensing Simulation")
    print("=" * 80)
    print(f"Dataset: Flat={n_flat}, PSPL={n_pspl}, Binary={n_binary}")
    print(f"Binary topology: {binary_preset}")
    print(f"Cadence: {cadence*100:.0f}% missing | Error: {error:.3f} mag")
    print(f"Causality enforcement: Strict t0/tE bounds, high-mag PSPLs enabled")
    print(f"Workers: {num_workers}")
    print(f"Accelerator: {'Numba JIT' if HAS_NUMBA else 'NumPy Vectorization'}")
    print("=" * 80)
    
    # Generate parameters
    print("\nGenerating parameters...")
    params_flat = generate_flat_params(n_flat, seed=seed)
    params_pspl = generate_pspl_params(n_pspl, seed=seed+1 if seed else None)
    params_binary = generate_binary_params(n_binary, preset=binary_preset, 
                                           seed=seed+2 if seed else None)
    
    # Prepare arguments
    args_list = []
    args_list.extend([(i, p, 'flat', timestamps, cadence, error, SimConfig.PAD_VALUE) 
                      for i, p in enumerate(params_flat)])
    args_list.extend([(i, p, 'pspl', timestamps, cadence, error, SimConfig.PAD_VALUE) 
                      for i, p in enumerate(params_pspl)])
    args_list.extend([(i, p, 'binary', timestamps, cadence, error, SimConfig.PAD_VALUE) 
                      for i, p in enumerate(params_binary)])
    
    # Generate events
    print("\nGenerating light curves...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_event, args_list),
                total=len(args_list),
                desc="Events"
            ))
    else:
        results = [generate_single_event(args) for args in tqdm(args_list, desc="Events")]
    
    # Unpack results
    flux = np.array([r[0] for r in results], dtype=np.float64)
    labels = np.array([r[1] for r in results], dtype=np.int64)
    all_params = [r[2] for r in results]
    
    # Organize parameters by class
    params_dict = {
        'flat': [all_params[i] for i in range(len(all_params)) if labels[i] == 0],
        'pspl': [all_params[i] for i in range(len(all_params)) if labels[i] == 1],
        'binary': [all_params[i] for i in range(len(all_params)) if labels[i] == 2]
    }
    
    # Calculate causal temporal encoding
    print("\nComputing causal temporal encoding (delta_t)...")
    start_time = time.time()
    
    delta_t_array = calculate_delta_t_per_event(timestamps, flux, float(SimConfig.PAD_VALUE))
    
    elapsed = time.time() - start_time
    print(f"Delta_t calculation completed in {elapsed:.2f} seconds")
    
    # Shuffle dataset
    shuffle_idx = np.random.permutation(len(flux))
    flux = flux[shuffle_idx]
    delta_t_array = delta_t_array[shuffle_idx]
    labels = labels[shuffle_idx]
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Dataset Generation Complete")
    print("=" * 80)
    print(f"Total events: {len(flux):,}")
    print(f"  Flat:   {(labels==0).sum():,} ({(labels==0).mean()*100:.1f}%)")
    print(f"  PSPL:   {(labels==1).sum():,} ({(labels==1).mean()*100:.1f}%)")
    print(f"  Binary: {(labels==2).sum():,} ({(labels==2).mean()*100:.1f}%)")
    print("=" * 80)
    
    return flux, delta_t_array, labels, timestamps, params_dict if save_params else None

# ============================================================================
# OPTIMIZED SAVING UTILITIES
# ============================================================================
def params_to_numpy(params_list):
    """Convert list of dicts to numpy arrays (optimized for speed)"""
    if not params_list:
        return None, None, None
    
    # Separate numeric and string values
    all_numeric = []
    all_strings = []
    numeric_keys = []
    string_keys = []
    
    # Get keys and determine which are numeric
    first_params = params_list[0]
    for key in first_params.keys():
        value = first_params[key]
        if isinstance(value, (int, float, np.floating, np.integer)):
            numeric_keys.append(key)
        else:
            string_keys.append(key)
    
    # Pre-allocate arrays
    n_params = len(params_list)
    numeric_values = np.empty((n_params, len(numeric_keys)), dtype=np.float32)
    string_values = []
    
    # Fill arrays
    for j in range(n_params):
        param_dict = params_list[j]
        # Fill numeric values
        for i, key in enumerate(numeric_keys):
            numeric_values[j, i] = float(param_dict[key])
        # Collect string values
        if string_keys:
            string_row = [str(param_dict[key]) for key in string_keys]
            string_values.append(string_row)
    
    return numeric_values, numeric_keys, (string_values, string_keys) if string_keys else None

def numpy_to_params(numeric_values, numeric_keys, string_data):
    """Convert numpy arrays back to list of dicts"""
    if numeric_values is None:
        return []
    
    params_list = []
    n_params = len(numeric_values)
    
    if string_data:
        string_values, string_keys = string_data
        for j in range(n_params):
            param_dict = {}
            # Add numeric values
            for i, key in enumerate(numeric_keys):
                param_dict[key] = numeric_values[j, i]
            # Add string values
            for i, key in enumerate(string_keys):
                param_dict[key] = string_values[j][i]
            params_list.append(param_dict)
    else:
        for j in range(n_params):
            param_dict = {}
            for i, key in enumerate(numeric_keys):
                param_dict[key] = numeric_values[j, i]
            params_list.append(param_dict)
    
    return params_list

# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate microlensing dataset for Roman Space Telescope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline 1M events (Roman quality)
  python simulate.py --preset baseline_1M
 
  # Topology study
  python simulate.py --preset distinct --n_flat 50000 --n_pspl 50000 --n_binary 50000
 
  # Quick test
  python simulate.py --preset quick_test
        """
    )
    
    parser.add_argument('--n_flat', type=int, default=10000, help="Number of flat events")
    parser.add_argument('--n_pspl', type=int, default=10000, help="Number of PSPL events")
    parser.add_argument('--n_binary', type=int, default=10000, help="Number of binary events")
    
    parser.add_argument('--preset', type=str, choices=[
        'baseline_1M', 'quick_test',
        'distinct', 'planetary', 'stellar', 'baseline',
        'cadence_05', 'cadence_15', 'cadence_30', 'cadence_50',
        'error_003', 'error_005', 'error_010', 'error_015'
    ], help='Predefined experiment preset')
    
    parser.add_argument('--binary_preset', type=str, default='distinct',
                        choices=list(BinaryPresets.PRESETS.keys()),
                        help='Binary topology preset')
    
    parser.add_argument('--cadence_mask_prob', type=float, default=None,
                        help='Fraction of observations to mask')
    parser.add_argument('--mag_error_std', type=float, default=None,
                        help='Photometric error standard deviation')
    
    parser.add_argument('--output', type=str, default='../data/dataset.npz',
                        help='Output file path')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_save_params', action='store_true',
                        help='Skip saving individual event parameters')
    parser.add_argument('--no_compress', action='store_true',
                        help='Disable compression for faster saving (larger files)')
    
    parser.add_argument('--list_presets', action='store_true',
                        help='List all available presets')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\n=== Binary Topology Presets ===")
        for name, config in BinaryPresets.PRESETS.items():
            print(f"  {name:12s} : {config['description']}")
        
        print("\n=== Cadence Presets ===")
        for name, config in ObservationalPresets.CADENCE_PRESETS.items():
            print(f"  {name:12s} : {config['description']} ({config['mask_prob']*100:.0f}% missing)")
        
        print("\n=== Error Presets ===")
        for name, config in ObservationalPresets.ERROR_PRESETS.items():
            print(f"  {name:12s} : {config['description']} (σ={config['error']:.3f})")
        
        print("\n=== Experiment Presets ===")
        print("  baseline_1M  : Full 1M dataset for Roman Space Telescope")
        print("  quick_test   : Small 300-event test dataset")
        return
    
    # Apply preset overrides
    if args.preset:
        if args.preset == 'baseline_1M':
            args.n_flat = 333000
            args.n_pspl = 333000
            args.n_binary = 334000
            args.binary_preset = 'baseline'
            args.cadence_mask_prob = 0.05
            args.mag_error_std = 0.05
            args.output = '../data/baseline_1M.npz'
        
        elif args.preset == 'quick_test':
            args.n_flat = 100
            args.n_pspl = 100
            args.n_binary = 100
            args.binary_preset = 'baseline'
            args.output = '../data/quick_test.npz'
        
        elif args.preset in BinaryPresets.PRESETS:
            args.binary_preset = args.preset
            args.output = f'../data/{args.preset}.npz'
        
        elif args.preset.startswith('cadence_'):
            if args.preset in ObservationalPresets.CADENCE_PRESETS:
                obs = ObservationalPresets.CADENCE_PRESETS[args.preset]
                args.cadence_mask_prob = obs['mask_prob']
                args.mag_error_std = obs['error']
                args.output = f'../data/{args.preset}.npz'
        
        elif args.preset.startswith('error_'):
            if args.preset in ObservationalPresets.ERROR_PRESETS:
                obs = ObservationalPresets.ERROR_PRESETS[args.preset]
                args.cadence_mask_prob = obs['mask_prob']
                args.mag_error_std = obs['error']
                args.output = f'../data/{args.preset}.npz'
    
    # Generate dataset
    flux, delta_t, labels, timestamps, params_dict = simulate_dataset(
        n_flat=args.n_flat,
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_preset=args.binary_preset,
        cadence_mask_prob=args.cadence_mask_prob,
        mag_error_std=args.mag_error_std,
        num_workers=min(args.num_workers, 20),  # Limit workers to avoid overhead
        seed=args.seed,
        save_params=not args.no_save_params
    )
    
    # ====================================================================
    # 🚀 OPTIMIZED SAVING SECTION
    # ====================================================================
    print("\n" + "=" * 80)
    print("Optimized Saving Phase")
    print("=" * 80)
    
    # 1. Reduce precision to float32 (50% smaller files, faster I/O)
    print("Converting to float32 for faster I/O...")
    flux = flux.astype(np.float32)
    delta_t = delta_t.astype(np.float32)
    timestamps = timestamps.astype(np.float32)
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'flux': flux,
        'delta_t': delta_t,
        'labels': labels,
        'timestamps': timestamps,
        'n_classes': 3,
        'class_names': ['Flat', 'PSPL', 'Binary'],
        'binary_preset': args.binary_preset,
        'cadence_mask_prob': args.cadence_mask_prob or SimConfig.CADENCE_MASK_PROB,
        'mag_error_std': args.mag_error_std or SimConfig.MAG_ERROR_STD,
        't0_range': (PSPLParams.T0_MIN, PSPLParams.T0_MAX),
        'tE_range': (PSPLParams.TE_MIN, PSPLParams.TE_MAX),
        'seed': args.seed
    }
    
    if params_dict and not args.no_save_params:
        # 2. Store parameters as numpy arrays with separate handling for strings
        print("Converting parameters to optimized numpy format...")
        
        # Convert each parameter set to numpy arrays
        flat_vals, flat_num_keys, flat_str_data = params_to_numpy(params_dict['flat'])
        pspl_vals, pspl_num_keys, pspl_str_data = params_to_numpy(params_dict['pspl'])
        binary_vals, binary_num_keys, binary_str_data = params_to_numpy(params_dict['binary'])
        
        # Store in save_dict
        if flat_vals is not None:
            save_dict['params_flat_numeric_values'] = flat_vals
            save_dict['params_flat_numeric_keys'] = np.array(flat_num_keys, dtype=object)
            if flat_str_data:
                str_vals, str_keys = flat_str_data
                save_dict['params_flat_string_values'] = np.array(str_vals, dtype=object)
                save_dict['params_flat_string_keys'] = np.array(str_keys, dtype=object)
        
        if pspl_vals is not None:
            save_dict['params_pspl_numeric_values'] = pspl_vals
            save_dict['params_pspl_numeric_keys'] = np.array(pspl_num_keys, dtype=object)
            if pspl_str_data:
                str_vals, str_keys = pspl_str_data
                save_dict['params_pspl_string_values'] = np.array(str_vals, dtype=object)
                save_dict['params_pspl_string_keys'] = np.array(str_keys, dtype=object)
        
        if binary_vals is not None:
            save_dict['params_binary_numeric_values'] = binary_vals
            save_dict['params_binary_numeric_keys'] = np.array(binary_num_keys, dtype=object)
            if binary_str_data:
                str_vals, str_keys = binary_str_data
                save_dict['params_binary_string_values'] = np.array(str_vals, dtype=object)
                save_dict['params_binary_string_keys'] = np.array(str_keys, dtype=object)
    
    # 3. Choose compression based on size and speed preference
    print(f"\nSaving to {output_path}...")
    start_save = time.time()
    
    if args.no_compress:
        np.savez(output_path, **save_dict)  # Fastest saving
        compression_type = "Uncompressed (fastest)"
    else:
        np.savez_compressed(output_path, **save_dict)  # Balanced
        compression_type = "Compressed (balanced)"
    
    save_time = time.time() - start_save
    
    # ====================================================================
    # 🏁 SAVING COMPLETE
    # ====================================================================
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 80)
    print("Dataset Saved Successfully")
    print("=" * 80)
    print(f"File: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Compression: {compression_type}")
    print(f"Total events: {len(flux):,}")

if __name__ == '__main__':
    main()
