"""
Configuration for systematic microlensing benchmarking experiments

THESIS GOAL: Test how different parameters affect binary/PSPL classification
- Observing cadence (sparse vs dense sampling)
- Binary topology (easy vs hard to distinguish)
- Photometric errors (survey quality)
- Early detection (partial light curves)
"""

import os

# ============================================================================
# BINARY PARAMETER REGIMES - FOR DISTINGUISHABILITY TESTING
# ============================================================================

# Different binary configurations for benchmarking
# These represent different levels of "distinguishability" from PSPL

BINARY_REGIMES = {
    # EASY: Wide binaries with strong caustic features
    'wide_binary': {
        'description': 'Wide separation binaries - strong caustic crossings, very distinguishable',
        's_min': 1.0,        # Separation (wide binaries)
        's_max': 2.5,
        'q_min': 0.3,        # Mass ratio (not too extreme)
        'q_max': 1.0,
        'rho_min': 0.001,    # Small source (sharp caustic features)
        'rho_max': 0.01,
        'u0_min': 0.01,      # Close approach (strong signal)
        'u0_max': 0.3,
    },
    
    # MODERATE: Intermediate binaries
    'intermediate_binary': {
        'description': 'Intermediate separation - moderate caustic features',
        's_min': 0.5,        # Mix of wide and close
        's_max': 1.5,
        'q_min': 0.2,        # Broader mass ratio range
        'q_max': 1.0,
        'rho_min': 0.01,     # Moderate source size
        'rho_max': 0.05,
        'u0_min': 0.01,
        'u0_max': 0.5,
    },
    
    # HARD: Close binaries, harder to distinguish
    'close_binary': {
        'description': 'Close separation binaries - subtle features, harder to distinguish',
        's_min': 0.1,        # Very close binaries (near resonant)
        's_max': 0.8,
        'q_min': 0.1,        # Extreme mass ratios
        'q_max': 0.5,
        'rho_min': 0.05,     # Large source (smooths caustic features)
        'rho_max': 0.1,
        'u0_min': 0.2,       # Distant approach (weaker signal)
        'u0_max': 0.5,
    },
    
    # CHALLENGING: PSPL-like binaries (very hard!)
    'pspl_like_binary': {
        'description': 'PSPL-like binaries - very subtle features, most challenging',
        's_min': 0.8,        # Near-equal mass at specific separations
        's_max': 1.2,
        'q_min': 0.8,        # Nearly equal masses
        'q_max': 1.0,
        'rho_min': 0.05,     # Large source (smooths features)
        'rho_max': 0.1,
        'u0_min': 0.3,       # Distant approach
        'u0_max': 0.7,
    },
    
    # MIXED: Realistic distribution (your original ranges)
    'mixed_binary': {
        'description': 'Mixed population - realistic distribution of all types',
        's_min': 0.1,
        's_max': 2.5,
        'q_min': 0.1,
        'q_max': 1.0,
        'rho_min': 0.01,
        'rho_max': 0.1,
        'u0_min': 0.01,
        'u0_max': 0.5,
    },
}

# ============================================================================
# CADENCE EXPERIMENTS - OBSERVING STRATEGY TESTING
# ============================================================================

CADENCE_EXPERIMENTS = {
    'lsst_dense': {
        'description': 'LSST deep drilling fields (frequent observations)',
        'cadence_mask_prob': 0.05,  # 5% missing (nearly complete coverage)
        'mag_error_std': 0.05,       # Excellent photometry
        'n_events': 200_000,
    },
    'lsst_regular': {
        'description': 'LSST main survey (regular cadence)',
        'cadence_mask_prob': 0.15,  # 15% missing (good coverage)
        'mag_error_std': 0.08,
        'n_events': 200_000,
    },
    'lsst_sparse': {
        'description': 'LSST wide-fast-deep (sparse)',
        'cadence_mask_prob': 0.30,  # 30% missing (sparse)
        'mag_error_std': 0.10,
        'n_events': 200_000,
    },
    'roman_dense': {
        'description': 'Roman Space Telescope (high cadence)',
        'cadence_mask_prob': 0.03,  # 3% missing (space-based, excellent)
        'mag_error_std': 0.03,       # Space telescope precision
        'n_events': 200_000,
    },
    'ground_sparse': {
        'description': 'Ground-based sparse monitoring',
        'cadence_mask_prob': 0.40,  # 40% missing (poor coverage)
        'mag_error_std': 0.15,       # Ground-based errors
        'n_events': 200_000,
    },
}

# ============================================================================
# PHOTOMETRIC ERROR EXPERIMENTS - SURVEY QUALITY
# ============================================================================

ERROR_EXPERIMENTS = {
    'excellent': {
        'description': 'Space-based or excellent ground-based',
        'mag_error_std': 0.03,
        'n_events': 100_000,
    },
    'good': {
        'description': 'Good ground-based survey',
        'mag_error_std': 0.08,
        'n_events': 100_000,
    },
    'moderate': {
        'description': 'Moderate quality',
        'mag_error_std': 0.12,
        'n_events': 100_000,
    },
    'poor': {
        'description': 'Poor conditions or faint targets',
        'mag_error_std': 0.20,
        'n_events': 100_000,
    },
}

# ============================================================================
# EARLY DETECTION EXPERIMENTS - REAL-TIME CLASSIFICATION
# ============================================================================

EARLY_DETECTION_EXPERIMENTS = {
    'very_early': {
        'description': 'Alert stage - minimal data',
        'n_points': 300,
        'time_max': 200,  # First 20% of event
        'n_events': 100_000,
    },
    'early': {
        'description': 'Early detection - 1/3 of event',
        'n_points': 500,
        'time_max': 333,
        'n_events': 100_000,
    },
    'mid': {
        'description': 'Mid-event - half complete',
        'n_points': 750,
        'time_max': 500,
        'n_events': 100_000,
    },
    'late': {
        'description': 'Late detection - 2/3 complete',
        'n_points': 1000,
        'time_max': 667,
        'n_events': 100_000,
    },
    'full': {
        'description': 'Full light curve',
        'n_points': 1500,
        'time_max': 1000,
        'n_events': 100_000,
    },
}

# ============================================================================
# BASELINE CONFIGURATION (DEFAULT)
# ============================================================================

# Use mixed binary regime and moderate cadence as baseline
DEFAULT_BINARY_REGIME = 'mixed_binary'
DEFAULT_CADENCE = 0.20          # 20% missing (reasonable baseline)
DEFAULT_MAG_ERROR = 0.10        # 0.1 mag error (typical)
DEFAULT_N_POINTS = 1500
DEFAULT_TIME_MAX = 1000

# ============================================================================
# DATASET PARAMETERS
# ============================================================================

N_EVENTS_TOTAL = 1_000_000     # Baseline dataset
N_PSPL = 500_000               # 50/50 split
N_BINARY = 500_000

# ============================================================================
# TIME SERIES PARAMETERS
# ============================================================================

N_POINTS = 1500
TIME_MIN = 0
TIME_MAX = 1000

# ============================================================================
# PSPL PARAMETERS (Standard, well-understood)
# ============================================================================

PSPL_BASELINE_MIN = 19
PSPL_BASELINE_MAX = 22
PSPL_T0_MIN = 0
PSPL_T0_MAX = 1000
PSPL_U0_MIN = 0.01
PSPL_U0_MAX = 1.0
PSPL_TE_MIN = 10
PSPL_TE_MAX = 150

# ============================================================================
# DEFAULT BINARY PARAMETERS (from mixed regime)
# ============================================================================

BINARY_S_MIN = 0.1
BINARY_S_MAX = 2.5
BINARY_Q_MIN = 0.1
BINARY_Q_MAX = 1.0
BINARY_RHO_MIN = 0.01
BINARY_RHO_MAX = 0.1
BINARY_ALPHA_MIN = 0
BINARY_ALPHA_MAX = 3.14159
BINARY_TE_MIN = 10
BINARY_TE_MAX = 100
BINARY_T0_MIN = 0
BINARY_T0_MAX = 1000
BINARY_U0_MIN = 0.01
BINARY_U0_MAX = 0.5

# ============================================================================
# OBSERVATIONAL PARAMETERS
# ============================================================================

MAG_ERROR_STD = 0.1
CADENCE_MASK_PROB = 0.2
PAD_VALUE = 0

# ============================================================================
# CNN PARAMETERS
# ============================================================================

SEQUENCE_LENGTH = 1500
NUM_CHANNELS = 1
CONFIDENCE_THRESHOLD = 0.8

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# VBMICROLENSING SETTINGS
# ============================================================================

VBM_REL_TOL = 1e-3
VBM_TOL = 1e-3

# ============================================================================
# GPU SETTINGS
# ============================================================================

MIXED_PRECISION = True
MULTI_GPU = True
NUM_GPUS = 4

# ============================================================================
# EXPERIMENT MATRIX - ALL COMBINATIONS TO TEST
# ============================================================================

# Full experimental matrix for thesis
EXPERIMENT_MATRIX = [
    # 1. Baseline with different binary regimes
    {'name': 'baseline_wide', 'binary_regime': 'wide_binary', 'cadence': 0.20, 'error': 0.10},
    {'name': 'baseline_intermediate', 'binary_regime': 'intermediate_binary', 'cadence': 0.20, 'error': 0.10},
    {'name': 'baseline_close', 'binary_regime': 'close_binary', 'cadence': 0.20, 'error': 0.10},
    {'name': 'baseline_pspl_like', 'binary_regime': 'pspl_like_binary', 'cadence': 0.20, 'error': 0.10},
    {'name': 'baseline_mixed', 'binary_regime': 'mixed_binary', 'cadence': 0.20, 'error': 0.10},
    
    # 2. Cadence variations (using mixed binary regime)
    {'name': 'cadence_lsst_dense', 'binary_regime': 'mixed_binary', 'cadence': 0.05, 'error': 0.05},
    {'name': 'cadence_lsst_regular', 'binary_regime': 'mixed_binary', 'cadence': 0.15, 'error': 0.08},
    {'name': 'cadence_lsst_sparse', 'binary_regime': 'mixed_binary', 'cadence': 0.30, 'error': 0.10},
    {'name': 'cadence_roman', 'binary_regime': 'mixed_binary', 'cadence': 0.03, 'error': 0.03},
    {'name': 'cadence_ground_sparse', 'binary_regime': 'mixed_binary', 'cadence': 0.40, 'error': 0.15},
    
    # 3. Combined: Hard binaries + sparse cadence (worst case)
    {'name': 'worst_case_close', 'binary_regime': 'close_binary', 'cadence': 0.40, 'error': 0.15},
    {'name': 'worst_case_pspl_like', 'binary_regime': 'pspl_like_binary', 'cadence': 0.40, 'error': 0.15},
    
    # 4. Combined: Easy binaries + dense cadence (best case)
    {'name': 'best_case_wide', 'binary_regime': 'wide_binary', 'cadence': 0.03, 'error': 0.03},
]

def get_binary_params(regime_name):
    """Get binary parameter ranges for a specific regime"""
    return BINARY_REGIMES[regime_name]

def print_experiment_plan():
    """Print summary of all experiments"""
    print("=" * 80)
    print("THESIS BENCHMARKING EXPERIMENT PLAN")
    print("=" * 80)
    print()
    print("BINARY REGIMES:")
    for name, regime in BINARY_REGIMES.items():
        print(f"  {name:20s}: {regime['description']}")
    print()
    print("PLANNED EXPERIMENTS:")
    for i, exp in enumerate(EXPERIMENT_MATRIX, 1):
        print(f"  {i:2d}. {exp['name']:30s} | Binary: {exp['binary_regime']:20s} | "
              f"Missing: {exp['cadence']*100:4.1f}% | Error: {exp['error']:.3f}")
    print()
    print(f"Total experiments: {len(EXPERIMENT_MATRIX)}")
    print("=" * 80)

if __name__ == "__main__":
    print_experiment_plan()
