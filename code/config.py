"""
Configuration for Microlensing Classification with Transformers

Author: Kunal Bhatia
University of Heidelberg
"""

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Sequence parameters
N_POINTS = 1500
TIME_MIN = 0
TIME_MAX = 1000

# Event parameters (shared between PSPL and Binary)
T0_MIN = 300.0
T0_MAX = 700.0
U0_MIN = 0.01
U0_MAX = 1.0
TE_MIN = 10.0
TE_MAX = 150.0

# Baseline magnitude range
BASELINE_MIN = 19.0
BASELINE_MAX = 22.0

# Observational parameters
MAG_ERROR_STD = 0.10  # photometric error (mag)
CADENCE_MASK_PROB = 0.20  # fraction of missing observations
PAD_VALUE = -1.0  # padding value for missing data

# ============================================================================
# BINARY PARAMETERS
# ============================================================================

# Baseline configuration (mixed population)
BINARY_BASELINE = {
    's_min': 0.1, 's_max': 2.5,
    'q_min': 0.1, 'q_max': 1.0,
    'rho_min': 0.01, 'rho_max': 0.1,
    'alpha_min': 0, 'alpha_max': 3.14159,
}

# Distinct caustic crossings (clear signatures)
BINARY_DISTINCT = {
    's_min': 0.8, 's_max': 1.5,
    'q_min': 0.1, 'q_max': 0.5,
    'rho_min': 0.01, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'u0_max': 0.15,  # close approaches
}

# Planetary systems
BINARY_PLANETARY = {
    's_min': 0.5, 's_max': 3.0,
    'q_min': 0.0001, 'q_max': 0.01,
    'rho_min': 0.0001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'u0_min': 0.001, 'u0_max': 0.5,
}

# Stellar binaries
BINARY_STELLAR = {
    's_min': 0.3, 's_max': 5.0,
    'q_min': 0.3, 'q_max': 1.0,
    'rho_min': 0.001, 'rho_max': 0.1,
    'alpha_min': 0, 'alpha_max': 3.14159,
    'u0_min': 0.001, 'u0_max': 0.8,
    'tE_min': 20.0, 'tE_max': 200.0,
}

BINARY_PARAM_SETS = {
    'baseline': BINARY_BASELINE,
    'distinct': BINARY_DISTINCT,
    'planetary': BINARY_PLANETARY,
    'stellar': BINARY_STELLAR,
}

# ============================================================================
# MODEL PARAMETERS (TRANSFORMER)
# ============================================================================

# Transformer architecture
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DOWNSAMPLE_FACTOR = 3
DROPOUT = 0.3

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 10  # early stopping patience

# Data splits
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Random seed
SEED = 42

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

CONFIDENCE_THRESHOLD = 0.8  # for decision-time analysis
N_SAMPLE_PLOTS = 12  # number of sample plots to generate

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"