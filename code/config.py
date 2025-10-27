"""
Configuration for systematic benchmarking experiments
Goal: Test how different parameters affect binary vs PSPL classification
"""

import os

# ============================================================================
# BENCHMARK EXPERIMENT SUITE
# ============================================================================

EXPERIMENTS = {
    # BASELINE: Wide range covering planetary to stellar binaries
    'baseline': {
        'description': 'Baseline - wide range from planetary to stellar',
        'n_events': 1_000_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },
    
    # DISTINCT: Clear caustic-crossing binaries (most distinguishable)
    'distinct': {
        'description': 'Distinct caustic-crossing events',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'distinct',
    },
    
    # CADENCE EXPERIMENTS
    'cadence_dense': {
        'description': 'Dense observing cadence (LSST-like)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.05,
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },
    
    'cadence_sparse': {
        'description': 'Sparse cadence',
        'n_events': 200_000,
        'cadence_mask_prob': 0.30,
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },
    
    # PHOTOMETRIC ERROR EXPERIMENTS
    'error_low': {
        'description': 'Low photometric errors (space-based)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.05,
        'binary_params': 'baseline',
    },
    
    'error_high': {
        'description': 'High photometric errors',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.20,
        'binary_params': 'baseline',
    },
    
    # PLANETARY vs STELLAR
    'planetary': {
        'description': 'Planetary systems (q << 1)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'planetary',
    },
    
    'stellar': {
        'description': 'Stellar binaries (q ~ 1)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'stellar',
    },
}

# ============================================================================
# BINARY PARAMETERS - COVERING PLANETARY TO STELLAR SYSTEMS
# ============================================================================

# BASELINE: Wide range from planetary to stellar (realistic population)
BINARY_PARAMS_BASELINE = {
    's_min': 0.1, 's_max': 10.0,      # Very wide range: close to very wide binaries
    'q_min': 0.001, 'q_max': 1.0,     # Planetary (0.001) to equal mass (1.0)
    'u0_min': 0.001, 'u0_max': 1.0,   # All impact parameters
    'rho_min': 0.0001, 'rho_max': 0.1,  # All source sizes
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 200,      # Wide range of timescales
    't0_min': 0, 't0_max': 1000,
}

# DISTINCT: Clear caustic-crossing binaries (maximally distinguishable from PSPL)
BINARY_PARAMS_DISTINCT = {
    's_min': 0.8, 's_max': 1.5,       # Wide binary region (s~1, largest caustics)
    'q_min': 0.01, 'q_max': 0.5,      # Asymmetric (more features)
    'u0_min': 0.001, 'u0_max': 0.15,  # Small u0 -> MUST cross caustics
    'rho_min': 0.0001, 'rho_max': 0.01,  # Small source -> sharp features
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 20, 'tE_max': 150,
    't0_min': 0, 't0_max': 1000,
}

# PLANETARY: Planet-hosting systems
BINARY_PARAMS_PLANETARY = {
    's_min': 0.5, 's_max': 3.0,       # Typical planetary separations
    'q_min': 0.0001, 'q_max': 0.01,   # Mass ratio << 1 (Jupiter/Sun ~ 0.001)
    'u0_min': 0.001, 'u0_max': 0.5,   # All impact parameters
    'rho_min': 0.0001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 150,
    't0_min': 0, 't0_max': 1000,
}

# STELLAR: Equal-mass or near-equal binary stars
BINARY_PARAMS_STELLAR = {
    's_min': 0.3, 's_max': 5.0,       # Wide range of stellar separations
    'q_min': 0.3, 'q_max': 1.0,       # Near-equal to equal mass
    'u0_min': 0.001, 'u0_max': 0.8,   # All impact parameters
    'rho_min': 0.001, 'rho_max': 0.1,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 20, 'tE_max': 200,
    't0_min': 0, 't0_max': 1000,
}

BINARY_PARAM_SETS = {
    'baseline': BINARY_PARAMS_BASELINE,
    'distinct': BINARY_PARAMS_DISTINCT,
    'planetary': BINARY_PARAMS_PLANETARY,
    'stellar': BINARY_PARAMS_STELLAR,
}

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

N_EVENTS_TOTAL = 1_000_000
N_PSPL = 500_000
N_BINARY = 500_000
N_POINTS = 1500
TIME_MIN = 0
TIME_MAX = 1000

# PSPL parameters (unchanged)
PSPL_BASELINE_MIN = 19
PSPL_BASELINE_MAX = 22
PSPL_T0_MIN = 0
PSPL_T0_MAX = 1000
PSPL_U0_MIN = 0.01
PSPL_U0_MAX = 1.0
PSPL_TE_MIN = 10
PSPL_TE_MAX = 150

# Observational
MAG_ERROR_STD = 0.1
CADENCE_MASK_PROB = 0.2
PAD_VALUE = 0

# CNN
SEQUENCE_LENGTH = 1500
NUM_CHANNELS = 1

# Training
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Paths (more flexible)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# VBMicrolensing
VBM_REL_TOL = 1e-3
VBM_TOL = 1e-3

# GPU settings (works for both AMD and NVIDIA)
MIXED_PRECISION = True
NUM_GPUS = 4  # Maximum available