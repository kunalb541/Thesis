"""
Lightweight Configuration for Fast Training

Author: Kunal Bhatia
Version: 7.0 - Optimized for speed
"""

import math

# ============================================================================
# LIGHTWEIGHT MODEL PARAMETERS (5x smaller, 10x faster)
# ============================================================================

# Temporal downsampling
DOWNSAMPLE_FACTOR = 5  # 1500 -> 300 points

# Model architecture (reduced from original)
D_MODEL = 128           # Reduced from 256
NHEAD = 4              # Reduced from 8
NUM_LAYERS = 4         # Reduced from 6
DIM_FEEDFORWARD = 256  # Reduced from 1024
DROPOUT = 0.15         # Slightly higher for regularization

# Streaming configuration
MAX_SEQ_LEN = 1500 // DOWNSAMPLE_FACTOR  # 300 points
WINDOW_SIZE = 100      # Reduced from 200
CAUSAL_MASK = True
USE_MULTI_HEAD = True

# ============================================================================
# FASTER TRAINING PARAMETERS
# ============================================================================

# Batch configuration (can use larger batches now!)
BATCH_SIZE = 64              # Increased from 32
GRADIENT_ACCUMULATION = 4    # Reduced from 8 (effective batch = 256)
LEARNING_RATE = 2e-4         # Slightly higher for faster convergence
WEIGHT_DECAY = 1e-5

# Training schedule
EPOCHS = 30                  # Reduced from 100 (faster convergence)
PATIENCE = 10                # Reduced from 15
MIN_DELTA = 0.001

# Loss weights (same as before)
LOSS_WEIGHTS = {
    'classification': 1.0,
    'early_detection': 0.5,
    'caustic_focal': 0.3,
    'temporal_consistency': 0.1,
}

# ============================================================================
# DATA PARAMETERS (unchanged)
# ============================================================================

N_POINTS = 1500
TIME_MIN = -100
TIME_MAX = 100

# PSPL parameters
PSPL_T0_MIN = -20.0
PSPL_T0_MAX = 20.0
PSPL_U0_MIN = 0.01
PSPL_U0_MAX = 1.0
PSPL_TE_MIN = 10.0
PSPL_TE_MAX = 150.0

# Baseline magnitude
BASELINE_MIN = 19.0
BASELINE_MAX = 22.0

# Observational noise
MAG_ERROR_STD = 0.10
CADENCE_MASK_PROB = 0.20

# Padding
PAD_VALUE = -1.0
NORMALIZE_PER_EVENT = False

# ============================================================================
# BINARY PARAMETERS (unchanged)
# ============================================================================

MIN_BINARY_MAGNIFICATION = 20.0
VBM_TOLERANCE = 1e-4
MAX_BINARY_ATTEMPTS = 10

BINARY_CRITICAL = {
    's_min': 0.7, 's_max': 1.5,
    'q_min': 0.01, 'q_max': 0.5,
    'rho_min': 0.001, 'rho_max': 0.01,
    'alpha_min': 0, 'alpha_max': math.pi,
    'u0_min': 0.001, 'u0_max': 0.05,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 30.0, 'tE_max': 100.0,
}

BINARY_BASELINE = {
    's_min': 0.1, 's_max': 2.5,
    'q_min': 0.001, 'q_max': 1.0,
    'rho_min': 0.001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 2 * math.pi,
    'u0_min': 0.001, 'u0_max': 0.3,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 20.0, 'tE_max': 150.0,
}

BINARY_DISTINCT = {
    's_min': 0.8, 's_max': 1.3,
    'q_min': 0.1, 'q_max': 0.5,
    'rho_min': 0.001, 'rho_max': 0.02,
    'alpha_min': 0, 'alpha_max': math.pi,
    'u0_min': 0.001, 'u0_max': 0.1,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 30.0, 'tE_max': 100.0,
}

BINARY_OVERLAPPING = {
    's_min': 0.1, 's_max': 3.0,
    'q_min': 0.0001, 'q_max': 1.0,
    'rho_min': 0.001, 'rho_max': 0.1,
    'alpha_min': 0, 'alpha_max': 2 * math.pi,
    'u0_min': 0.01, 'u0_max': 1.0,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 10.0, 'tE_max': 200.0,
}

BINARY_PLANETARY = {
    's_min': 0.5, 's_max': 2.0,
    'q_min': 0.0001, 'q_max': 0.01,
    'rho_min': 0.0001, 'rho_max': 0.01,
    'alpha_min': 0, 'alpha_max': 2 * math.pi,
    'u0_min': 0.001, 'u0_max': 0.2,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 20.0, 'tE_max': 100.0,
}

BINARY_STELLAR = {
    's_min': 0.3, 's_max': 3.0,
    'q_min': 0.3, 'q_max': 1.0,
    'rho_min': 0.001, 'rho_max': 0.05,
    'alpha_min': 0, 'alpha_max': 2 * math.pi,
    'u0_min': 0.001, 'u0_max': 0.4,
    't0_min': -20.0, 't0_max': 20.0,
    'tE_min': 30.0, 'tE_max': 200.0,
}

BINARY_PARAM_SETS = {
    'critical': BINARY_CRITICAL,
    'baseline': BINARY_BASELINE,
    'distinct': BINARY_DISTINCT,
    'overlapping': BINARY_OVERLAPPING,
    'planetary': BINARY_PLANETARY,
    'stellar': BINARY_STELLAR,
}

# ============================================================================
# DATA SPLITS (unchanged)
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# DISTRIBUTED TRAINING (unchanged)
# ============================================================================

DDP_BACKEND = 'nccl'
FIND_UNUSED_PARAMS = False
BROADCAST_BUFFERS = True
MASTER_PORT = 29500
INIT_METHOD = 'env://'
WORLD_SIZE = None
RANK = None

# ============================================================================
# PATHS (unchanged)
# ============================================================================

DATA_DIR = "../data"
RESULTS_DIR = "../results"
CHECKPOINT_DIR = "../checkpoints"
LOG_DIR = "../logs"

# ============================================================================
# STREAMING INFERENCE (unchanged)
# ============================================================================

BUFFER_SIZE = 200
CONFIDENCE_THRESHOLD = 0.8
CAUSTIC_THRESHOLD = 0.5
MAX_LATENCY_MS = 1.0

ALERT_ON_BINARY = True
ALERT_ON_ANOMALY = True
ALERT_ON_CAUSTIC = True

# ============================================================================
# VALIDATION THRESHOLDS (unchanged)
# ============================================================================

MIN_SAMPLES_PER_CLASS = 1000
MAX_CLASS_IMBALANCE = 10.0
MIN_NON_PAD_RATIO = 0.5
MIN_TRAIN_ACCURACY = 0.6
MIN_VAL_ACCURACY = 0.5
MAX_LOSS = 10.0

# ============================================================================
# RANDOM SEEDS (unchanged)
# ============================================================================

SEED = 42
NUMPY_SEED = 42
TORCH_SEED = 42

# ============================================================================
# DEBUG FLAGS (unchanged)
# ============================================================================

DEBUG_MODE = False
VERBOSE = True
SAVE_INTERMEDIATE = False
PROFILE_PERFORMANCE = False

# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_binary_params(params: dict) -> None:
    """Validate binary parameter ranges"""
    assert params['u0_min'] < params['u0_max']
    assert params['s_min'] < params['s_max']
    assert params['q_min'] < params['q_max']
    assert params['rho_min'] < params['rho_max']
    assert params['t0_min'] < params['t0_max']
    assert params['tE_min'] < params['tE_max']
    assert params['u0_min'] > 0
    assert params['q_min'] > 0
    assert params['rho_min'] > 0
    assert params['tE_min'] > 0

for name, params in BINARY_PARAM_SETS.items():
    validate_binary_params(params)

assert BUFFER_SIZE >= WINDOW_SIZE

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

print("="*60)
print("LIGHTWEIGHT CONFIGURATION LOADED")
print("="*60)
print(f"Model size: {D_MODEL}D × {NUM_LAYERS} layers × {NHEAD} heads")
print(f"Downsampling: {N_POINTS} -> {MAX_SEQ_LEN} points ({DOWNSAMPLE_FACTOR}x)")
print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"Expected speedup: ~10x faster training")
print(f"Expected accuracy: ~90-95% of original performance")
print("="*60)
