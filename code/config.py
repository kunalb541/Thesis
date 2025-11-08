"""
Configuration for Real-Time Binary Microlensing Detection

CRITICAL: This config is optimized for detecting binary caustic crossings.
Low u0 values are ESSENTIAL for binaries to ensure caustic features.

Author: Kunal Bhatia
Date: November 2025
Version: 6.1 - Fixed with parameter validation
"""

import math

# ============================================================================
# CRITICAL DETECTION PARAMETERS
# ============================================================================

# MINIMUM magnification for valid binary detection
MIN_BINARY_MAGNIFICATION = 20.0  # Binaries MUST have caustic spike > 20x

# VBMicrolensing tolerance (critical for convergence)
VBM_TOLERANCE = 1e-4

# Maximum retry attempts for binary generation
MAX_BINARY_ATTEMPTS = 10

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Sequence parameters
N_POINTS = 1500  # Full temporal resolution (no downsampling in data!)
TIME_MIN = -100  # Start before event for baseline
TIME_MAX = 100   # End after event for baseline

# PSPL parameters (Point Source Point Lens)
PSPL_T0_MIN = -20.0
PSPL_T0_MAX = 20.0
PSPL_U0_MIN = 0.01   # Can be wider range for PSPL
PSPL_U0_MAX = 1.0
PSPL_TE_MIN = 10.0
PSPL_TE_MAX = 150.0

# Baseline magnitude range
BASELINE_MIN = 19.0
BASELINE_MAX = 22.0

# Observational noise
MAG_ERROR_STD = 0.10      # Ground-based quality (LSST/OGLE)
CADENCE_MASK_PROB = 0.20  # 20% missing observations

# Padding and normalization
PAD_VALUE = -1.0
NORMALIZE_PER_EVENT = False  # Global normalization for consistency

# ============================================================================
# BINARY PARAMETERS - CRITICAL FOR DETECTION
# ============================================================================

# CRITICAL BINARY CONFIGURATION (force caustic crossings)
BINARY_CRITICAL = {
    's_min': 0.7,        # Optimal for wide caustics
    's_max': 1.5,        
    'q_min': 0.01,       # Minimum for clear perturbation
    'q_max': 0.5,        
    'rho_min': 0.001,    # Small source = sharp features
    'rho_max': 0.01,     
    'alpha_min': 0,
    'alpha_max': math.pi,
    'u0_min': 0.001,     # CRITICAL: Force close approach
    'u0_max': 0.05,      # CRITICAL: Maximum u0 for guaranteed caustics
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 30.0,      # Longer events for clear caustics
    'tE_max': 100.0,
}

# Baseline (realistic mix)
BINARY_BASELINE = {
    's_min': 0.1,
    's_max': 2.5,
    'q_min': 0.001,
    'q_max': 1.0,
    'rho_min': 0.001,
    'rho_max': 0.05,
    'alpha_min': 0,
    'alpha_max': 2 * math.pi,
    'u0_min': 0.001,     # Still prefer close approach
    'u0_max': 0.3,       # But allow some wider
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 20.0,
    'tE_max': 150.0,
}

# Distinct (clear signatures)
BINARY_DISTINCT = {
    's_min': 0.8,
    's_max': 1.3,
    'q_min': 0.1,
    'q_max': 0.5,
    'rho_min': 0.001,
    'rho_max': 0.02,
    'alpha_min': 0,
    'alpha_max': math.pi,
    'u0_min': 0.001,
    'u0_max': 0.1,       # Force close approach
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 30.0,
    'tE_max': 100.0,
}

# Overlapping (includes hard cases)
BINARY_OVERLAPPING = {
    's_min': 0.1,
    's_max': 3.0,
    'q_min': 0.0001,
    'q_max': 1.0,
    'rho_min': 0.001,
    'rho_max': 0.1,
    'alpha_min': 0,
    'alpha_max': 2 * math.pi,
    'u0_min': 0.01,
    'u0_max': 1.0,       # INCLUDES u0 > 0.3 (hard cases)
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 10.0,
    'tE_max': 200.0,
}

# Planetary systems
BINARY_PLANETARY = {
    's_min': 0.5,
    's_max': 2.0,
    'q_min': 0.0001,
    'q_max': 0.01,
    'rho_min': 0.0001,
    'rho_max': 0.01,
    'alpha_min': 0,
    'alpha_max': 2 * math.pi,
    'u0_min': 0.001,
    'u0_max': 0.2,
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 20.0,
    'tE_max': 100.0,
}

# Stellar binaries
BINARY_STELLAR = {
    's_min': 0.3,
    's_max': 3.0,
    'q_min': 0.3,
    'q_max': 1.0,
    'rho_min': 0.001,
    'rho_max': 0.05,
    'alpha_min': 0,
    'alpha_max': 2 * math.pi,
    'u0_min': 0.001,
    'u0_max': 0.4,
    't0_min': -20.0,
    't0_max': 20.0,
    'tE_min': 30.0,
    'tE_max': 200.0,
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
# STREAMING TRANSFORMER PARAMETERS
# ============================================================================

# Model architecture
D_MODEL = 256            # Larger model for complex patterns
NHEAD = 8               # More attention heads
NUM_LAYERS = 6          # Deeper network
DIM_FEEDFORWARD = 1024  # Wider FFN
DROPOUT = 0.2           # Less dropout for larger model

# Streaming configuration
MAX_SEQ_LEN = 1500      # Maximum sequence length
WINDOW_SIZE = 200       # Sliding window for attention
CAUSAL_MASK = True      # Strictly causal attention

# Multi-head outputs
USE_MULTI_HEAD = True   # Binary + Anomaly + Caustic detection

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Batch configuration
BATCH_SIZE = 32              # Per GPU (smaller for larger model)
GRADIENT_ACCUMULATION = 8    # Effective batch = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000

# Training schedule
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 0.001

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Loss weights
LOSS_WEIGHTS = {
    'classification': 1.0,
    'early_detection': 0.5,
    'caustic_focal': 0.3,
    'temporal_consistency': 0.1,
}

# ============================================================================
# STREAMING INFERENCE
# ============================================================================

# Real-time parameters
BUFFER_SIZE = 200           # Circular buffer size
CONFIDENCE_THRESHOLD = 0.8  # Decision threshold
CAUSTIC_THRESHOLD = 0.5     # Caustic detection threshold
MAX_LATENCY_MS = 1.0        # Maximum acceptable latency

# Alert triggers
ALERT_ON_BINARY = True
ALERT_ON_ANOMALY = True
ALERT_ON_CAUSTIC = True

# ============================================================================
# DISTRIBUTED TRAINING
# ============================================================================

# DDP configuration
DDP_BACKEND = 'nccl'        # Use NCCL for GPUs
FIND_UNUSED_PARAMS = False  # All params should be used
BROADCAST_BUFFERS = True    # Sync batch norm stats

# Multi-node settings
MASTER_PORT = 29500
INIT_METHOD = 'env://'      # Use environment variables
WORLD_SIZE = None           # Set dynamically
RANK = None                 # Set dynamically

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = "../data"
RESULTS_DIR = "../results"
CHECKPOINT_DIR = "../checkpoints"
LOG_DIR = "../logs"

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Data quality checks
MIN_SAMPLES_PER_CLASS = 1000
MAX_CLASS_IMBALANCE = 10.0
MIN_NON_PAD_RATIO = 0.5

# Model performance thresholds
MIN_TRAIN_ACCURACY = 0.6
MIN_VAL_ACCURACY = 0.5
MAX_LOSS = 10.0

# ============================================================================
# RANDOM SEEDS
# ============================================================================

SEED = 42
NUMPY_SEED = 42
TORCH_SEED = 42

# ============================================================================
# DEBUG FLAGS
# ============================================================================

DEBUG_MODE = False
VERBOSE = True
SAVE_INTERMEDIATE = False
PROFILE_PERFORMANCE = False

# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_binary_params(params: dict) -> None:
    """Validate binary parameter ranges are physically meaningful"""
    assert params['u0_min'] < params['u0_max'], f"u0_min must be < u0_max"
    assert params['s_min'] < params['s_max'], f"s_min must be < s_max"
    assert params['q_min'] < params['q_max'], f"q_min must be < q_max"
    assert params['rho_min'] < params['rho_max'], f"rho_min must be < rho_max"
    assert params['t0_min'] < params['t0_max'], f"t0_min must be < t0_max"
    assert params['tE_min'] < params['tE_max'], f"tE_min must be < tE_max"
    assert params['u0_min'] > 0, f"u0_min must be > 0"
    assert params['q_min'] > 0, f"q_min must be > 0"
    assert params['rho_min'] > 0, f"rho_min must be > 0"
    assert params['tE_min'] > 0, f"tE_min must be > 0"

# Validate all parameter sets on module load
for name, params in BINARY_PARAM_SETS.items():
    try:
        validate_binary_params(params)
    except AssertionError as e:
        raise ValueError(f"Invalid parameters in {name}: {e}")

# Validate buffer size vs window size
assert BUFFER_SIZE >= WINDOW_SIZE, f"Buffer size ({BUFFER_SIZE}) must be >= window size ({WINDOW_SIZE})"