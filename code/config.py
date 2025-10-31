"""
Configuration for systematic benchmarking experiments

This file contains all parameter sets for different experimental conditions.
Modify these to create new experiments without changing core code.

Author: Kunal Bhatia (kunal29bhatia@gmail.com)
University of Heidelberg
"""

import os

# ============================================================================
# PROJECT METADATA
# ============================================================================

PROJECT_NAME = "Microlensing Binary Classification"
AUTHOR = "Kunal Bhatia"
INSTITUTION = "University of Heidelberg"
CONTACT = "kunal29bhatia@gmail.com"

# ============================================================================
# BENCHMARK EXPERIMENT SUITE
# ============================================================================

EXPERIMENTS = {
    # BASELINE: Wide range covering planetary to stellar binaries
    # Represents realistic population for general classification
    'baseline': {
        'description': 'Baseline - wide range from planetary to stellar',
        'n_events': 1_000_000,
        'cadence_mask_prob': 0.20,  # 20% missing observations
        'mag_error_std': 0.10,       # 0.1 mag photometric error
        'binary_params': 'baseline',
    },

    # DISTINCT: Clear caustic-crossing binaries (most distinguishable)
    # Tests best-case performance with guaranteed caustic features
    'distinct': {
        'description': 'Distinct caustic-crossing events',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.10,
        'binary_params': 'distinct',
    },

    # CADENCE EXPERIMENTS
    # Test how observation frequency affects performance

    'cadence_dense': {
        'description': 'Dense observing cadence (LSST-like)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.05,  # 5% missing - excellent coverage
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },

    'cadence_sparse': {
        'description': 'Sparse cadence',
        'n_events': 200_000,
        'cadence_mask_prob': 0.30,  # 30% missing - poor coverage
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },

    'cadence_very_sparse': {
        'description': 'Very sparse cadence',
        'n_events': 200_000,
        'cadence_mask_prob': 0.40,  # 40% missing - very poor coverage
        'mag_error_std': 0.10,
        'binary_params': 'baseline',
    },

    # PHOTOMETRIC ERROR EXPERIMENTS
    # Test how measurement precision affects performance

    'error_low': {
        'description': 'Low photometric errors (space-based quality)',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.05,  # Space-based (Roman, HST)
        'binary_params': 'baseline',
    },

    'error_high': {
        'description': 'High photometric errors',
        'n_events': 200_000,
        'cadence_mask_prob': 0.20,
        'mag_error_std': 0.20,  # Poor ground-based conditions
        'binary_params': 'baseline',
    },

    # BINARY TYPE EXPERIMENTS
    # Test performance across different mass ratio regimes

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
    's_min': 0.1, 's_max': 2.5,      # ✅ Match your notebook
    'q_min': 0.1, 'q_max': 1.0,      # ✅ Match your notebook
    'u0_min': 0.01, 'u0_max': 0.5,   # ✅ Match your notebook
    'rho_min': 0.01, 'rho_max': 0.1, # ✅ Match your notebook
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 100,     # ✅ Match your notebook
    't0_min': 0, 't0_max': 1000,     # ✅ Match your notebook
}

# DISTINCT: Clear caustic-crossing binaries (maximally distinguishable from PSPL)
BINARY_PARAMS_DISTINCT = {
    's_min': 0.8, 's_max': 1.5,      # Optimal for caustics
    'q_min': 0.1, 'q_max': 0.5,
    'u0_min': 0.01, 'u0_max': 0.15,  # Very close approach = caustics
    'rho_min': 0.01, 'rho_max': 0.05, # Smaller = sharper features
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 20, 'tE_max': 150,
    't0_min': 0, 't0_max': 1000,
}

# PLANETARY: Planet-hosting systems
BINARY_PARAMS_PLANETARY = {
    's_min': 0.5, 's_max': 3.0,
    'q_min': 0.0001, 'q_max': 0.01,
    'u0_min': 0.001, 'u0_max': 0.5,
    'rho_min': 0.0001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 10, 'tE_max': 150,
    't0_min': 0, 't0_max': 1000,
}

# STELLAR: Equal-mass or near-equal binary stars
BINARY_PARAMS_STELLAR = {
    's_min': 0.3, 's_max': 5.0,
    'q_min': 0.3, 'q_max': 1.0,
    'u0_min': 0.001, 'u0_max': 0.8,
    'rho_min': 0.001, 'rho_max': 0.1,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'tE_min': 20, 'tE_max': 200,
    't0_min': 0, 't0_max': 1000,
}

# Dictionary mapping parameter set names to configurations
BINARY_PARAM_SETS = {
    'baseline': BINARY_PARAMS_BASELINE,
    'distinct': BINARY_PARAMS_DISTINCT,
    'planetary': BINARY_PARAMS_PLANETARY,
    'stellar': BINARY_PARAMS_STELLAR,
}

# ============================================================================
# DEFAULT SIMULATION PARAMETERS
# ============================================================================

# Dataset size (for baseline)
N_EVENTS_TOTAL = 1_000_000
N_PSPL = 500_000
N_BINARY = 500_000

# Time series parameters
N_POINTS = 100  # ✅ FIXED: Was 1500, now matches training
TIME_MIN = 0    # ✅ FIXED: Match your original notebook (0 to 1000)
TIME_MAX = 1000 # ✅ FIXED: Match your original notebook

# PSPL parameters (for single-lens events)
PSPL_BASELINE_MIN = 19
PSPL_BASELINE_MAX = 22
PSPL_T0_MIN = 0
PSPL_T0_MAX = 1000
PSPL_U0_MIN = 0.01
PSPL_U0_MAX = 1.0
PSPL_TE_MIN = 10
PSPL_TE_MAX = 150

# Observational effects
MAG_ERROR_STD = 0.1
CADENCE_MASK_PROB = 0.2
PAD_VALUE = -1       # unified with simulator

# Normalization: divide each curve by its own median before padding
NORMALIZE_PER_EVENT = False

# Optional fairness: share cadence masks between classes
USE_SHARED_MASK = False  # FIXED: Enable cadence masking
MASK_POOL_SIZE = 256

# ============================================================================
# CNN ARCHITECTURE PARAMETERS
# ============================================================================

SEQUENCE_LENGTH = 100  
NUM_CHANNELS = 1

# Architecture details (for TimeDistributedCNN)
CONV1_FILTERS = 128
CONV2_FILTERS = 64
CONV3_FILTERS = 32
FC1_UNITS = 64
DROPOUT_RATE = 0.3

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Data splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

# Optimization
OPTIMIZER = 'adam'
LR_SCHEDULER = 'plateau'
LR_PATIENCE = 5
LR_FACTOR = 0.5

# Random seed (for reproducibility)
RANDOM_SEED = 42

# ============================================================================
# PATHS
# ============================================================================

# Base directory (automatically detected)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ============================================================================
# VBMICROLENSING PARAMETERS
# ============================================================================

VBM_REL_TOL = 1e-3
VBM_TOL = 1e-3

# ============================================================================
# GPU SETTINGS
# ============================================================================
# These are maximum values - actual usage detected automatically

MIXED_PRECISION = True
NUM_GPUS = 4

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Early detection checkpoints (fractions of event observed)
EARLY_DETECTION_CHECKPOINTS = [0.1, 0.25, 0.33, 0.5, 0.67, 0.83, 1.0]

# Metrics to compute
COMPUTE_ROC = True
COMPUTE_PR = True
COMPUTE_CONFUSION_MATRIX = True

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Plot settings
DPI = 300
FIGSIZE = (10, 6)
PLOT_STYLE = 'seaborn'

# Colors for plots
COLOR_PSPL = 'blue'
COLOR_BINARY = 'red'
COLOR_TRAIN = '#2E86AB'
COLOR_VAL = '#A23B72'

# ============================================================================
# LOGGING
# ============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Console output
VERBOSE = True
PROGRESS_BAR = True

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Whether to save intermediate checkpoints
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQ = 10

# Whether to save best model only or all checkpoints
SAVE_BEST_ONLY = True

# Metrics to monitor for best model
MONITOR_METRIC = 'val_acc'
MONITOR_MODE = 'max'

