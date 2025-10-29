"""
Battle-tested binary microlensing parameter presets
Guaranteed to produce distinctive caustic features
"""

import numpy as np
from typing import Tuple, Optional

# Presets from battle-tested configurations
PRESETS = {
    'equal_mass_resonant': {
        's': 1.00, 'q': 0.8, 'u0': 0.03, 'tE': 120, 'rho': 0.003, 'alpha_deg': 30,
        'description': 'Big resonant caustic → multiple sharp peaks/asymmetry'
    },
    'resonant_unequal': {
        's': 0.98, 'q': 0.3, 'u0': 0.05, 'tE': 100, 'rho': 0.005, 'alpha_deg': 45,
        'description': 'Cleaner double horns; less symmetry'
    },
    'wide_binary': {
        's': 1.9, 'q': 0.5, 'u0': 0.15, 'tE': 150, 'rho': 0.01, 'alpha_deg': 10,
        'description': 'Two distinct separated peaks'
    },
    'close_binary': {
        's': 0.65, 'q': 0.4, 'u0': 0.03, 'tE': 140, 'rho': 0.003, 'alpha_deg': 20,
        'description': 'Central caustic triple-peak'
    },
    'fold_caustic': {
        's': 1.05, 'q': 0.2, 'u0': 0.02, 'tE': 90, 'rho': 0.002, 'alpha_deg': 70,
        'description': 'U-trough with spikes at edges'
    },
    'cusp_grazing': {
        's': 0.75, 'q': 0.25, 'u0': 0.06, 'tE': 110, 'rho': 0.004, 'alpha_deg': 35,
        'description': 'Sharp cusp + long asymmetric shoulder'
    },
    'planetary_central': {
        's': 1.10, 'q': 0.005, 'u0': 0.05, 'tE': 70, 'rho': 0.0015, 'alpha_deg': 40,
        'description': 'Short anomaly near PSPL peak'
    },
    'planetary_offpeak': {
        's': 1.8, 'q': 0.002, 'u0': 0.20, 'tE': 80, 'rho': 0.002, 'alpha_deg': 15,
        'description': 'Small bump far from peak'
    },
    'high_speed': {
        's': 1.00, 'q': 0.3, 'u0': 0.03, 'tE': 15, 'rho': 0.002, 'alpha_deg': 25,
        'description': 'Compressed multi-features (fast!)'
    },
    'finite_source': {
        's': 1.02, 'q': 0.5, 'u0': 0.04, 'tE': 120, 'rho': 0.03, 'alpha_deg': 35,
        'description': 'Big ρ softens spikes into broad humps'
    },
    'separated_peaks': {
        's': 0.9, 'q': 0.3, 'u0': 0.2, 'tE': 120, 'rho': 0.01, 'alpha_deg': 25,
        'description': 'Clean distinct peaks with good spacing'
    },
    'ultra_dramatic': {
        's': 1.00, 'q': 0.6, 'u0': 0.01, 'tE': 100, 'rho': 0.001, 'alpha_deg': 30,
        'description': 'Extremely sharp/bright spikes'
    }
}

# Jitter amounts for each parameter
JITTER = {
    's_resonant': 0.03,     # Near s=1.0
    's_wide': 0.2,           # Wide/close binaries
    'q_factor': (0.7, 1.3),  # Multiply by factor
    'q_planetary': (0.7, 1.3),  # For q < 0.01
    'u0_small': 0.01,        # For planetary central
    'u0_large': 0.05,        # For wide binary
    'tE_factor': (0.8, 1.2), # Multiply by factor
    'rho_factor': (0.7, 1.3),
    'alpha_deg': 15          # Degrees
}


def sample_preset_with_jitter(preset_name: str, rng: np.random.RandomState) -> dict:
    """
    Sample parameters from a preset with randomization
    
    Returns dict with keys: s, q, u0, tE, rho, alpha (radians), t0
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    # Base values
    s = preset['s']
    q = preset['q']
    u0 = preset['u0']
    tE = preset['tE']
    rho = preset['rho']
    alpha_deg = preset['alpha_deg']
    
    # Apply jitter
    # Separation
    if 0.9 <= s <= 1.1:  # Resonant
        s += rng.uniform(-JITTER['s_resonant'], JITTER['s_resonant'])
    else:
        s += rng.uniform(-JITTER['s_wide'], JITTER['s_wide'])
    s = np.clip(s, 0.1, 10.0)
    
    # Mass ratio
    if q < 0.01:  # Planetary
        q *= rng.uniform(*JITTER['q_planetary'])
        q = np.clip(q, 0.001, 0.01)
    else:
        q *= rng.uniform(*JITTER['q_factor'])
        q = np.clip(q, 0.01, 1.0)
    
    # Impact parameter
    if u0 < 0.1:
        u0 += rng.uniform(-JITTER['u0_small'], JITTER['u0_small'])
    else:
        u0 += rng.uniform(-JITTER['u0_large'], JITTER['u0_large'])
    u0 = np.clip(u0, 0.001, 1.0)
    
    # Timescale
    tE *= rng.uniform(*JITTER['tE_factor'])
    tE = np.clip(tE, 5.0, 300.0)
    
    # Source size
    rho *= rng.uniform(*JITTER['rho_factor'])
    rho = np.clip(rho, 0.0001, 0.1)
    
    # Trajectory angle
    alpha_deg += rng.uniform(-JITTER['alpha_deg'], JITTER['alpha_deg'])
    alpha = np.deg2rad(alpha_deg % 360)
    
    # Peak time (center of observation window)
    t0 = rng.uniform(-50, 50)  # Days relative to center
    
    return {
        's': s,
        'q': q,
        'u0': u0,
        'tE': tE,
        't0': t0,
        'rho': rho,
        'alpha': alpha,
        'preset': preset_name
    }


def sample_distinct_binary_params(n: int, seed: Optional[int] = None) -> list:
    """
    Sample n binary parameter sets from presets
    
    Focuses on dramatic configurations:
    - 40% resonant (equal-mass, unequal)
    - 30% central caustic (close binary, fold, cusp)
    - 20% ultra-dramatic
    - 10% other interesting
    """
    rng = np.random.RandomState(seed)
    
    # Preset distribution (weights for sampling)
    preset_weights = {
        'equal_mass_resonant': 0.25,
        'resonant_unequal': 0.15,
        'close_binary': 0.15,
        'fold_caustic': 0.10,
        'cusp_grazing': 0.05,
        'ultra_dramatic': 0.20,
        'wide_binary': 0.05,
        'separated_peaks': 0.05
    }
    
    presets = list(preset_weights.keys())
    weights = list(preset_weights.values())
    
    # Sample presets
    chosen_presets = rng.choice(presets, size=n, p=weights)
    
    # Generate parameters
    params = []
    for preset_name in chosen_presets:
        params.append(sample_preset_with_jitter(preset_name, rng))
    
    return params


if __name__ == "__main__":
    # Test sampler
    params = sample_distinct_binary_params(10, seed=42)
    
    print("Sample of 10 binary parameter sets:")
    print("-" * 80)
    for i, p in enumerate(params):
        print(f"{i+1}. Preset: {p['preset']}")
        print(f"   s={p['s']:.3f}, q={p['q']:.4f}, u0={p['u0']:.3f}, "
              f"tE={p['tE']:.1f}d, rho={p['rho']:.4f}")
