"""
Configuration for systematic benchmarking experiments
Goal: Test how different parameters affect binary vs PSPL classification
"""

import os

# ============================================================================
# BENCHMARK EXPERIMENT SUITE
# ============================================================================

EXPERIMENTS = {
    # BASELINE: Standard parameters
    'baseline': {
        'description': 'Baseline configuration',
        'n_events': 1_000_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'standard',
    },
    
    # CADENCE EXPERIMENTS
    'cadence_dense': {
        'description': 'Dense observing cadence (LSST-like)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.05,
        'mag_error_std': 0.10,
        'binary_params': 'standard',
    },
    
    'cadence_sparse': {
        'description': 'Sparse cadence',
        'n_events': 200_000,
        'cadence_mask_prob': 0.30,
        'mag_error_std': 0.10,
        'binary_params': 'standard',
    },
    
    # PHOTOMETRIC ERROR EXPERIMENTS
    'error_low': {
        'description': 'Low photometric errors (space-based)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.05,
        'binary_params': 'standard',
    },
    
    'error_high': {
        'description': 'High photometric errors',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.20,
        'binary_params': 'standard',
    },
    
    # BINARY DIFFICULTY EXPERIMENTS
    'binary_easy': {
        'description': 'Easy binaries (clear caustic crossings)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'easy',
    },
    
    'binary_hard': {
        'description': 'Hard binaries (PSPL-like)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'hard',
    },
    
    # EARLY DETECTION
    'early_500': {
        'description': 'Early detection (33% of event)',
        'n_events': 200_000,
        'n_points': 500,
        'time_max': 333,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'standard',
    },
}

# ============================================================================
# BINARY PARAMETERS - KEY FOR DISTINGUISHABILITY!
# ============================================================================

# STANDARD: Mixed difficulty (for baseline)
BINARY_PARAMS_STANDARD = {
    's_min': 0.3, 's_max': 2.0,
    'q_min': 0.1, 'q_max': 1.0,
    'u0_min': 0.01, 'u0_max': 0.5,
    'rho_min': 0.001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 100,
    't0_min': 0, 't0_max': 1000,
}

# EASY: Clear caustic crossings (most distinguishable!)
BINARY_PARAMS_EASY = {
    's_min': 0.8, 's_max': 1.2,      # Wide binary
    'q_min': 0.1, 'q_max': 0.5,      # Asymmetric
    'u0_min': 0.001, 'u0_max': 0.1,  # SMALL u0 -> crosses caustics!
    'rho_min': 0.0001, 'rho_max': 0.01,  # Sharp features
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 100,
    't0_min': 0, 't0_max': 1000,
}

# HARD: PSPL-like (hardest to distinguish!)
BINARY_PARAMS_HARD = {
    's_min': 0.1, 's_max': 0.3,      # Very close binary
    'q_min': 0.5, 'q_max': 1.0,      # Symmetric
    'u0_min': 0.3, 'u0_max': 0.5,    # LARGE u0 -> misses caustics!
    'rho_min': 0.03, 'rho_max': 0.1,  # Smoothed features
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 100,
    't0_min': 0, 't0_max': 1000,
}

BINARY_PARAM_SETS = {
    'standard': BINARY_PARAMS_STANDARD,
    'easy': BINARY_PARAMS_EASY,
    'hard': BINARY_PARAMS_HARD,
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

# PSPL parameters
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

# Paths
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

# GPU
MIXED_PRECISION = True
NUM_GPUS = 4
