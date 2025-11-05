"""
Configuration for Microlensing Classification with Transformers

This file centralizes all hyperparameters and configuration settings.
All experiments should reference this file to ensure reproducibility.

Author: Kunal Bhatia
University of Heidelberg
Date: November 2025
Version: 5.6.1 - Enhanced documentation
"""

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Sequence parameters
N_POINTS = 1500  # Number of observation points per light curve
TIME_MIN = 0     # Start time (days)
TIME_MAX = 1000  # End time (days)

# Event parameters (shared between PSPL and Binary)
# These define the ranges for random parameter sampling

# t0: Time of maximum magnification (days)
T0_MIN = 300.0
T0_MAX = 700.0
NORMALIZE_PER_EVENT = False 
# u0: Impact parameter (minimum separation in Einstein radii)
# CRITICAL PARAMETER: Low u0 = close approach = strong signal
U0_MIN = 0.01
U0_MAX = 1.0

# tE: Einstein crossing time (days)
# Typical duration of microlensing event
TE_MIN = 10.0
TE_MAX = 150.0

# Baseline magnitude range (apparent magnitude at infinite separation)
BASELINE_MIN = 19.0  # Bright sources
BASELINE_MAX = 22.0  # Faint sources

# Observational parameters
# These simulate real survey conditions

# MAG_ERROR_STD: Photometric measurement error (magnitudes)
# 0.05 = space-based (Roman)
# 0.10 = ground-based (LSST, OGLE)
# 0.20 = poor conditions
MAG_ERROR_STD = 0.10

# CADENCE_MASK_PROB: Fraction of missing observations
# 0.05 = intensive monitoring
# 0.20 = typical survey (LSST)
# 0.40 = sparse coverage
CADENCE_MASK_PROB = 0.20

# PAD_VALUE: Sentinel value for missing/padded data points
PAD_VALUE = -1.0

# ============================================================================
# BINARY PARAMETERS
# ============================================================================

# Binary lens parameters define the caustic topology
# Different parameter sets explore different physical scenarios

# Baseline configuration (mixed population)
# Represents realistic mixture of planetary and stellar binaries
BINARY_BASELINE = {
    's_min': 0.1,      # Minimum separation (Einstein radii)
    's_max': 2.5,      # Maximum separation
    'q_min': 0.1,      # Minimum mass ratio (companion/primary)
    'q_max': 1.0,      # Maximum mass ratio
    'rho_min': 0.01,   # Minimum source size (Einstein radii)
    'rho_max': 0.1,    # Maximum source size
    'alpha_min': 0,    # Minimum trajectory angle (radians)
    'alpha_max': 3.14159,  # Maximum trajectory angle
}

# Distinct caustic crossings (clear signatures)
# Optimized for maximum distinguishability from PSPL
BINARY_DISTINCT = {
    's_min': 0.8,      # Optimal separation range
    's_max': 1.5,      # Wide, clear caustics
    'q_min': 0.1,      # Moderate to high mass ratios
    'q_max': 0.5,
    'rho_min': 0.01,   # Small sources = sharp features
    'rho_max': 0.05,
    'alpha_min': 0,
    'alpha_max': 3.14159,
    'u0_max': 0.15,    # Force close approaches
}

# Planetary systems
# Low mass-ratio events (exoplanet detection)
BINARY_PLANETARY = {
    's_min': 0.5,      # Typical planet-star separations
    's_max': 3.0,
    'q_min': 0.0001,   # Jupiter/Sun ~ 0.001
    'q_max': 0.01,     # Brown dwarfs
    'rho_min': 0.0001, # Point-like planets
    'rho_max': 0.05,
    'alpha_min': 0,
    'alpha_max': 3.14159,
    'u0_min': 0.001,   # Allow both close and far events
    'u0_max': 0.5,
}

# Stellar binaries
# Equal or comparable-mass systems
BINARY_STELLAR = {
    's_min': 0.3,      # Close to wide binaries
    's_max': 5.0,
    'q_min': 0.3,      # M-dwarf/M-dwarf ~ 0.3-1.0
    'q_max': 1.0,      # Equal mass
    'rho_min': 0.001,
    'rho_max': 0.1,
    'alpha_min': 0,
    'alpha_max': 3.14159,
    'u0_min': 0.001,
    'u0_max': 0.8,     # Include larger impact parameters
    'tE_min': 20.0,    # Typically longer events
    'tE_max': 200.0,
}

# Dictionary for easy access in code
BINARY_PARAM_SETS = {
    'baseline': BINARY_BASELINE,
    'distinct': BINARY_DISTINCT,
    'planetary': BINARY_PLANETARY,
    'stellar': BINARY_STELLAR,
}

# ============================================================================
# MODEL PARAMETERS (TRANSFORMER)
# ============================================================================

# Transformer architecture hyperparameters
# These control model capacity and training behavior

D_MODEL = 64              # Embedding dimension (64 or 128 typical)
NHEAD = 4                 # Number of attention heads (must divide D_MODEL)
NUM_LAYERS = 2            # Number of transformer encoder layers
DIM_FEEDFORWARD = 256     # Hidden dimension in FFN (typically 4× d_model)
DOWNSAMPLE_FACTOR = 3     # Sequence reduction factor (1500 → 500)
DROPOUT = 0.3             # Dropout rate for regularization

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

BATCH_SIZE = 128          # Batch size per GPU (adjust for memory)
LEARNING_RATE = 1e-4      # Adam learning rate
EPOCHS = 50               # Maximum training epochs
PATIENCE = 10             # Early stopping patience

# Data splits (must sum to 1.0)
TRAIN_RATIO = 0.6         # 60% training
VAL_RATIO = 0.2           # 20% validation
TEST_RATIO = 0.2          # 20% test

# Random seed for reproducibility
SEED = 42

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Confidence threshold for decision-time analysis
# Model must be ≥ this confident to make a classification
CONFIDENCE_THRESHOLD = 0.8

# Number of sample plots to generate in evaluation
N_SAMPLE_PLOTS = 12

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

# ============================================================================
# NOTES ON PARAMETER SELECTION
# ============================================================================

"""
Key Parameter Relationships:

1. Impact Parameter (u0) vs Detectability:
   - u0 < 0.15: Binary caustic crossings likely → High accuracy
   - u0 = 0.15-0.30: Marginal detections → Medium accuracy
   - u0 > 0.30: Fundamentally PSPL-like → Low accuracy (physical limit)

2. Separation (s) vs Caustic Size:
   - s ~ 1.0: Maximum caustic area → Easiest to detect
   - s < 0.5 or s > 2.0: Small caustics → Harder to detect

3. Mass Ratio (q) vs Perturbation Strength:
   - q < 0.01: Planetary perturbations (subtle)
   - q = 0.1-0.5: Moderate perturbations
   - q ~ 1.0: Symmetric caustics (different topology)

4. Source Size (rho) vs Feature Sharpness:
   - Small rho: Sharp spikes → Easier to detect
   - Large rho: Smoothed features → Harder to detect

5. Photometric Error vs Accuracy:
   - 0.05 mag (Roman): ~5% accuracy gain over ground-based
   - 0.10 mag (LSST): Baseline performance
   - 0.20 mag: ~5-10% accuracy loss

6. Cadence vs Accuracy:
   - 5% missing: ~5% accuracy gain
   - 20% missing: Baseline
   - 40% missing: ~10% accuracy loss

These relationships guide experimental design and interpretation.
"""
