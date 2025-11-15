#!/usr/bin/env python3
"""
Microlensing Event Simulation v16.0 
===================================

CRITICAL FIX (v15.1):
- **REMOVED apply_temporal_randomization function** (was creating interpolation artifacts)
- Temporal invariance is now achieved ONLY through wide t_0 sampling (-80 to +60 days)
- This prevents the model from learning "peak at time X = class Y" without introducing
  fictitious linear segments in data gaps

The wide t_0 range is the correct way to enforce temporal invariance - it forces
the model to learn magnification morphology rather than absolute peak timing.

Author: Kunal Bhatia  
Version: 15.1 (Fixed)
"""

import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import math

# ============================================================================
# BUILT-IN CONFIGURATION (No external dependencies)
# ============================================================================

class SimConfig:
    """Built-in simulation parameters"""
    # Temporal
    N_POINTS = 1500
    TIME_MIN = -100
    TIME_MAX = 100
    
    # VBBinaryLensing
    VBM_TOLERANCE = 1e-4
    MAX_BINARY_ATTEMPTS = 10
    
    # Observational - Roman Space Telescope baseline
    CADENCE_MASK_PROB = 0.05  # 5% missing (~15 min sampling)
    MAG_ERROR_STD = 0.05       # Space-based photometry
    BASELINE_MIN = 19.0
    BASELINE_MAX = 22.0
    
    # Data
    PAD_VALUE = -1.0


class PSPLParams:
    """PSPL parameter ranges - TEMPORAL INVARIANT"""
    # CRITICAL: Wide t0 range to prevent temporal shortcuts
    T0_MIN = -80.0   # Can peak well before observation
    T0_MAX = 60.0    # Can peak late in observation
    U0_MIN = 0.001
    U0_MAX = 0.3
    TE_MIN = 20.0
    TE_MAX = 40.0


class BinaryPresets:
    """
    Binary topology presets
    
    Each preset defines a specific science case:
    - distinct: Clear caustics (optimal detection)
    - planetary: Exoplanet search (small q)
    - stellar: Binary stars (large q)
    - baseline: Mixed population (realistic survey conditions)
    """
    
    PRESETS = {
        'distinct': {
            'description': 'Clear caustics - optimal detection',
            's_range': (0.7, 1.5),
            'q_range': (0.01, 0.5),
            'u0_range': (0.001, 0.15),
            'rho_range': (0.001, 0.01),
            'alpha_range': (0, math.pi),
            't0_range': (-80.0, 60.0),  # TEMPORAL INVARIANT
            'tE_range': (30.0, 40.0),
        },
        
        'planetary': {
            'description': 'Exoplanet focus - small mass ratios',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.0001, 0.01),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 60.0),  # TEMPORAL INVARIANT
            'tE_range': (20.0, 40.0),
        },
        
        'stellar': {
            'description': 'Binary stars - equal masses',
            's_range': (0.3, 3.0),
            'q_range': (0.3, 1.0),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 60.0),  # TEMPORAL INVARIANT
            'tE_range': (30.0, 40.0),
        },
        
        'baseline': {
            'description': 'Mixed population - full parameter space',
            's_range': (0.1, 3.0),
            'q_range': (0.0001, 1.0),
            'u0_range': (0.001, 1.0),  # Includes wide u0 (physical limits)
            'rho_range': (0.001, 0.1),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-80.0, 60.0),  # TEMPORAL INVARIANT
            'tE_range': (10.0, 40.0),
        }
    }


class ObservationalPresets:
    """
    Cadence and photometric error presets
    
    Telescope-agnostic naming for comparative studies.
    Default: Roman Space Telescope quality
    
    Cadence studies: How does sampling frequency affect classification?
    Error studies: How does photometric quality affect classification?
    """
    
    CADENCE_PRESETS = {
        'cadence_05': {
            'description': 'High-cadence space-based (~15 min, 5% missing)',
            'mask_prob': 0.05,
            'error': 0.05,
            'example': 'Roman Space Telescope'
        },
        'cadence_15': {
            'description': 'Good ground-based (~1 day, 15% missing)',
            'mask_prob': 0.15,
            'error': 0.10,
            'example': 'Excellent survey conditions'
        },
        'cadence_30': {
            'description': 'Typical ground-based (~3 days, 30% missing)',
            'mask_prob': 0.30,
            'error': 0.10,
            'example': 'LSST typical conditions'
        },
        'cadence_50': {
            'description': 'Sparse ground-based (~5 days, 50% missing)',
            'mask_prob': 0.50,
            'error': 0.10,
            'example': 'Weather-limited survey'
        }
    }
    
    ERROR_PRESETS = {
        'error_003': {
            'description': 'Excellent space-based (0.03 mag)',
            'mask_prob': 0.05,
            'error': 0.03,
            'example': 'JWST-quality photometry'
        },
        'error_005': {
            'description': 'Space-based quality (0.05 mag)',
            'mask_prob': 0.05,
            'error': 0.05,
            'example': 'Roman Space Telescope'
        },
        'error_010': {
            'description': 'High-quality ground (0.10 mag)',
            'mask_prob': 0.05,
            'error': 0.10,
            'example': 'Professional ground-based'
        },
        'error_015': {
            'description': 'Typical ground (0.15 mag)',
            'mask_prob': 0.05,
            'error': 0.15,
            'example': 'Amateur/wide-field surveys'
        }
    }


# ============================================================================
# VBBinaryLensing (Try import)
# ============================================================================

try:
    import VBBinaryLensing
    HAS_VBB = True
except ImportError:
    HAS_VBB = False
    print("⚠️  VBBinaryLensing not found, using approximation")


# ============================================================================
# MAGNIFICATION FUNCTIONS
# ============================================================================

def pspl_magnification(t, t_E, u_0, t_0):
    """Point Source Point Lens magnification"""
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A


def binary_magnification_vbb(t, t_E, u_0, t_0, s, q, alpha, rho=0.001):
    """Binary lens magnification using VBBinaryLensing"""
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE
    VBB.RelTol = SimConfig.VBM_TOLERANCE
    
    tau = (t - t_0) / t_E
    source_x = u_0 * np.cos(alpha) + tau * np.sin(alpha)
    source_y = u_0 * np.sin(alpha) - tau * np.cos(alpha)
    
    mag = np.zeros_like(t)
    for i, (sx, sy) in enumerate(zip(source_x, source_y)):
        mag[i] = VBB.BinaryMag2(s, q, sx, sy, rho)
    
    return mag


def binary_magnification_approx(t, t_E, u_0, t_0, s, q, alpha, rho=0.001):
    """Approximate binary lens magnification (fallback)"""
    A_pspl = pspl_magnification(t, t_E, u_0, t_0)
    
    tau = (t - t_0) / t_E
    source_x = u_0 * np.cos(alpha) + tau * np.sin(alpha)
    source_y = u_0 * np.sin(alpha) - tau * np.cos(alpha)
    
    # Simple caustic perturbation
    caustic_x = s * np.array([0.5, -0.5, 0, 0])
    caustic_y = s * np.array([0, 0, 0.5, -0.5])
    
    perturbation = 0
    for cx, cy in zip(caustic_x, caustic_y):
        dist = np.sqrt((source_x - cx)**2 + (source_y - cy)**2)
        caustic_width = 0.05 * np.sqrt(q)
        perturbation += q * np.exp(-dist**2 / (2 * caustic_width**2))
    
    A_binary = A_pspl * (1 + perturbation * 3)
    return A_binary


# ============================================================================
# PARAMETER GENERATION
# ============================================================================

def generate_flat_params(n_events, seed=None):
    """Generate Flat (no event) parameters"""
    if seed is not None:
        np.random.seed(seed)
    
    params = []
    for _ in range(n_events):
        m_source = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
        params.append({'m_source': m_source, 'type': 'flat'})
    
    return params


def generate_pspl_params(n_events, seed=None):
    """Generate PSPL parameters with temporal invariance"""
    if seed is not None:
        np.random.seed(seed)
    
    params = []
    for _ in range(n_events):
        t_E = np.random.uniform(PSPLParams.TE_MIN, PSPLParams.TE_MAX)
        t_0 = np.random.uniform(PSPLParams.T0_MIN, PSPLParams.T0_MAX)
        u_0 = np.random.uniform(PSPLParams.U0_MIN, PSPLParams.U0_MAX)
        m_source = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
        
        params.append({
            't_E': t_E,
            'u_0': u_0,
            't_0': t_0,
            'm_source': m_source,
            'type': 'pspl'
        })
    
    return params


def generate_binary_params(n_events, preset='baseline', seed=None):
    """Generate Binary parameters from preset"""
    if seed is not None:
        np.random.seed(seed)
    
    if preset not in BinaryPresets.PRESETS:
        print(f"⚠️  Unknown preset '{preset}', using 'baseline'")
        preset = 'baseline'
    
    config = BinaryPresets.PRESETS[preset]
    params = []
    
    for _ in tqdm(range(n_events), desc=f"  Generating {preset} params"):
        t_E = np.random.uniform(config['tE_range'][0], config['tE_range'][1])
        t_0 = np.random.uniform(config['t0_range'][0], config['t0_range'][1])
        u_0 = np.random.uniform(config['u0_range'][0], config['u0_range'][1])
        s = np.random.uniform(config['s_range'][0], config['s_range'][1])
        q = np.random.uniform(config['q_range'][0], config['q_range'][1])
        alpha = np.random.uniform(config['alpha_range'][0], config['alpha_range'][1])
        rho = np.random.uniform(config['rho_range'][0], config['rho_range'][1])
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

def generate_light_curve(params, event_type, timestamps):
    """Generate a single light curve"""
    if event_type == 'flat':
        A = np.ones_like(timestamps)
    elif event_type == 'pspl':
        A = pspl_magnification(timestamps, params['t_E'], params['u_0'], params['t_0'])
    else:  # binary
        if HAS_VBB:
            A = binary_magnification_vbb(
                timestamps, params['t_E'], params['u_0'], params['t_0'],
                params['s'], params['q'], params['alpha'], params['rho']
            )
        else:
            A = binary_magnification_approx(
                timestamps, params['t_E'], params['u_0'], params['t_0'],
                params['s'], params['q'], params['alpha'], params['rho']
            )
    
    return A


def add_observational_effects(flux, error_mag, cadence_missing, pad_value):
    """Add realistic observational effects"""
    flux_obs = flux.copy()
    
    # Photometric noise
    noise = np.random.normal(0, error_mag, size=len(flux))
    flux_obs = flux_obs * (1 + noise)
    
    # Missing observations
    mask = np.random.random(len(flux)) < cadence_missing
    flux_obs[mask] = pad_value
    
    # Ensure positive flux
    flux_obs[flux_obs != pad_value] = np.maximum(flux_obs[flux_obs != pad_value], 0.01)
    
    return flux_obs


def generate_single_event(args):
    """Wrapper for parallel generation"""
    idx, params, event_type, timestamps, cadence, error, pad_value = args
    
    # Generate clean light curve
    flux = generate_light_curve(params, event_type, timestamps)
    
    # Add observational effects
    flux_obs = add_observational_effects(flux, error, cadence, pad_value)
    
    # Label: 0=Flat, 1=PSPL, 2=Binary
    label = {'flat': 0, 'pspl': 1, 'binary': 2}[event_type]
    
    return flux_obs, label, params


# ============================================================================
# MAIN SIMULATION FUNCTION (FIXED - NO TEMPORAL RANDOMIZATION)
# ============================================================================

def simulate_dataset(
    n_flat, n_pspl, n_binary,
    binary_preset='baseline',
    observational_preset=None,
    cadence_mask_prob=None,
    mag_error_std=None,
    num_workers=1,
    seed=None,
    save_params=True
):
    """
    Generate complete 3-class dataset (v15.1 FIXED)
    
    CRITICAL FIX: Removed apply_temporal_randomization post-processing.
    Temporal invariance is achieved through wide t_0 sampling (-80 to +60 days)
    at generation time, which forces the model to learn morphology without
    introducing interpolation artifacts.
    
    Args:
        n_flat, n_pspl, n_binary: Event counts
        binary_preset: Topology preset name
        observational_preset: Cadence/error preset (overrides individual settings)
        cadence_mask_prob: Override cadence
        mag_error_std: Override error
        num_workers: Parallel workers
        seed: Random seed
        save_params: Save event parameters
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
            print(f"⚠️  Unknown observational preset '{observational_preset}'")
            obs_config = {'mask_prob': SimConfig.CADENCE_MASK_PROB, 'error': SimConfig.MAG_ERROR_STD}
        
        cadence = obs_config['mask_prob']
        error = obs_config['error']
    else:
        cadence = cadence_mask_prob if cadence_mask_prob is not None else SimConfig.CADENCE_MASK_PROB
        error = mag_error_std if mag_error_std is not None else SimConfig.MAG_ERROR_STD
    
    # Generate timestamps
    timestamps = np.linspace(SimConfig.TIME_MIN, SimConfig.TIME_MAX, SimConfig.N_POINTS)
    
    # Print configuration
    print("="*70)
    print("MICROLENSING SIMULATION v15.1 (FIXED)")
    print("="*70)
    print(f"Dataset: Flat={n_flat}, PSPL={n_pspl}, Binary={n_binary}")
    print(f"Binary topology: {binary_preset}")
    print(f"Cadence: {cadence*100:.0f}% missing")
    print(f"Error: {error:.3f} mag")
    print(f"Temporal invariance: Via wide t_0 sampling (-80 to +60 days)")
    print(f"  ✅ NO post-hoc interpolation (artifact-free)")
    print(f"Workers: {num_workers}")
    
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
    print(f"\nGenerating light curves...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_event, args_list),
                total=len(args_list),
                desc="  Processing"
            ))
    else:
        results = [generate_single_event(args) for args in tqdm(args_list, desc="  Processing")]
    
    # Unpack
    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    all_params = [r[2] for r in results]
    
    # Organize parameters
    params_dict = {
        'flat': [all_params[i] for i in range(len(all_params)) if y[i] == 0],
        'pspl': [all_params[i] for i in range(len(all_params)) if y[i] == 1],
        'binary': [all_params[i] for i in range(len(all_params)) if y[i] == 2]
    }
    
    # NO TEMPORAL RANDOMIZATION (FIXED)
    # The wide t_0 range already provides temporal invariance
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Statistics
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total: {len(X)}")
    print(f"  Flat:   {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  PSPL:   {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Binary: {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")
    print(f"✅ No interpolation artifacts (clean data)")
    
    return X, y, timestamps, params_dict if save_params else None


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate microlensing dataset v15.1 (FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (1M events, Roman quality - DEFAULT)
  python simulate.py --preset baseline_1M
  
  # Topology studies
  python simulate.py --preset distinct --n_flat 50000 --n_pspl 50000 --n_binary 50000
  python simulate.py --preset planetary --n_flat 50000 --n_pspl 50000 --n_binary 50000
  
  # Cadence studies
  python simulate.py --preset cadence_05 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Roman
  python simulate.py --preset cadence_15 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Good ground
  python simulate.py --preset cadence_30 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Typical ground
  
  # Error studies
  python simulate.py --preset error_003 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Excellent
  python simulate.py --preset error_005 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Roman
  python simulate.py --preset error_010 --n_flat 30000 --n_pspl 30000 --n_binary 30000  # Ground
  
  # Custom
  python simulate.py --n_flat 10000 --n_pspl 10000 --n_binary 10000 \\
      --binary_preset planetary --cadence_mask_prob 0.10 --mag_error_std 0.08
        """
    )
    
    # Event counts
    parser.add_argument('--n_flat', type=int, default=10000)
    parser.add_argument('--n_pspl', type=int, default=10000)
    parser.add_argument('--n_binary', type=int, default=10000)
    
    # Presets
    parser.add_argument('--preset', type=str, choices=[
        'baseline_1M', 'quick_test',
        'distinct', 'planetary', 'stellar', 'baseline',
        'cadence_05', 'cadence_15', 'cadence_30', 'cadence_50',
        'error_003', 'error_005', 'error_010', 'error_015'
    ], help='Use predefined experiment preset')
    
    # Binary topology
    parser.add_argument('--binary_preset', type=str, default='baseline',
                       choices=list(BinaryPresets.PRESETS.keys()))
    
    # Observational
    parser.add_argument('--cadence_mask_prob', type=float, default=None)
    parser.add_argument('--mag_error_std', type=float, default=None)
    
    # System
    parser.add_argument('--output', type=str, default='../data/raw/dataset.npz')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_save_params', action='store_true')
    
    # Info
    parser.add_argument('--list_presets', action='store_true',
                       help='List all available presets')
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        print("\n" + "="*70)
        print("AVAILABLE PRESETS (v15.1 FIXED)")
        print("="*70)
        
        print("\nBinary Topologies:")
        for name, config in BinaryPresets.PRESETS.items():
            print(f"  {name:15s}: {config['description']}")
        
        print("\nCadence Presets:")
        for name, config in ObservationalPresets.CADENCE_PRESETS.items():
            print(f"  {name:15s}: {config['description']}")
            print(f"                  ({config['mask_prob']*100:.0f}% missing, {config['error']:.3f} mag)")
            print(f"                  Example: {config['example']}")
        
        print("\nError Presets:")
        for name, config in ObservationalPresets.ERROR_PRESETS.items():
            print(f"  {name:15s}: {config['description']}")
            print(f"                  ({config['mask_prob']*100:.0f}% missing, {config['error']:.3f} mag)")
            print(f"                  Example: {config['example']}")
        
        print("\nExperiment Presets:")
        print("  baseline_1M    : 1M events, Roman quality (5% missing, 0.05 mag)")
        print("  quick_test     : 300 events, quick validation")
        print("  distinct       : Topology study (clear caustics)")
        print("  planetary      : Topology study (exoplanets)")
        print("  stellar        : Topology study (binary stars)")
        print("  baseline       : Topology study (full parameter space)")
        print()
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
            args.output = '../data/raw/baseline_1M.npz'
        
        elif args.preset == 'quick_test':
            args.n_flat = 100
            args.n_pspl = 100
            args.n_binary = 100
            args.binary_preset = 'baseline'
            args.output = '../data/raw/quick_test.npz'
        
        elif args.preset in BinaryPresets.PRESETS:
            args.binary_preset = args.preset
            args.output = f'../data/raw/{args.preset}.npz'
        
        elif args.preset.startswith('cadence_'):
            obs_name = args.preset
            if obs_name in ObservationalPresets.CADENCE_PRESETS:
                obs = ObservationalPresets.CADENCE_PRESETS[obs_name]
                args.cadence_mask_prob = obs['mask_prob']
                args.mag_error_std = obs['error']
                args.output = f'../data/raw/{args.preset}.npz'
        
        elif args.preset.startswith('error_'):
            obs_name = args.preset
            if obs_name in ObservationalPresets.ERROR_PRESETS:
                obs = ObservationalPresets.ERROR_PRESETS[obs_name]
                args.cadence_mask_prob = obs['mask_prob']
                args.mag_error_std = obs['error']
                args.output = f'../data/raw/{args.preset}.npz'
    
    # Generate dataset (FIXED VERSION - NO TEMPORAL RANDOMIZATION)
    X, y, timestamps, params_dict = simulate_dataset(
        n_flat=args.n_flat,
        n_pspl=args.n_pspl,
        n_binary=args.n_binary,
        binary_preset=args.binary_preset,
        cadence_mask_prob=args.cadence_mask_prob,
        mag_error_std=args.mag_error_std,
        num_workers=args.num_workers,
        seed=args.seed,
        save_params=not args.no_save_params
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'X': X,
        'y': y,
        'timestamps': timestamps,
        'n_classes': 3,
        'class_names': ['Flat', 'PSPL', 'Binary'],
        'version': '15.1',  # FIXED VERSION
        'temporal_randomization': False,  # REMOVED (artifact source)
        'wide_t0_sampling': True,  # This is how we achieve temporal invariance
        't0_range': (PSPLParams.T0_MIN, PSPLParams.T0_MAX),
        'binary_preset': args.binary_preset,
        'cadence_mask_prob': args.cadence_mask_prob or SimConfig.CADENCE_MASK_PROB,
        'mag_error_std': args.mag_error_std or SimConfig.MAG_ERROR_STD
    }
    
    if params_dict:
        save_dict['params_flat_json'] = json.dumps(params_dict['flat'])
        save_dict['params_pspl_json'] = json.dumps(params_dict['pspl'])
        save_dict['params_binary_json'] = json.dumps(params_dict['binary'])
    
    np.savez(output_path, **save_dict)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"Dataset saved: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Version: 15.1 (FIXED - No interpolation artifacts)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
