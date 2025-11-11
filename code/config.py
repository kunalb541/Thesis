"""
Configuration for Real-Time Binary Microlensing Detection
VERSION 11.0 - THREE-CLASS CLASSIFICATION

NEW in v11.0: Added "Flat" class for no-event baseline observations
Classes: 0=Flat, 1=PSPL, 2=Binary

Author: Kunal Bhatia
Date: November 2025
Version: 11.0 - Three-Class Classification
"""

import math
from typing import Dict, Any

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Data generation parameters"""
    # Temporal sampling
    N_POINTS = 1500          # Full temporal resolution
    TIME_MIN = -100          # Days before peak
    TIME_MAX = 100           # Days after peak
    
    # VBBinaryLensing settings
    VBM_TOLERANCE = 1e-4
    MAX_BINARY_ATTEMPTS = 10
    
    # Observational effects
    CADENCE_MASK_PROB = 0.20  # 20% missing (LSST nominal)
    MAG_ERROR_STD = 0.10       # Ground-based photometry
    BASELINE_MIN = 19.0        # Source magnitude range
    BASELINE_MAX = 22.0
    
    # Data handling
    PAD_VALUE = -1.0          # Marker for missing data

# ============================================================================
# EVENT PARAMETERS
# ============================================================================

class FlatConfig:
    """
    Flat (no event) parameters
    Just constant baseline flux
    """
    pass  # No special parameters needed - just baseline magnitude

class PSPLConfig:
    """Point Source Point Lens parameters"""
    T0_MIN = -20.0    # Peak time range
    T0_MAX = 20.0
    U0_MIN = 0.001    # Match binary range
    U0_MAX = 0.3      # Match binary range
    TE_MIN = 20.0     # Match binary range
    TE_MAX = 40.0     # Fits in 200-day window

class BinaryConfig:
    """Binary lens configurations"""
    
    # Three main configurations
    CONFIGS = {
        'critical': {
            'description': 'Forces caustic crossings for clear detection',
            's_range': (0.7, 1.5),
            'q_range': (0.01, 0.5),
            'u0_range': (0.001, 0.05),
            'rho_range': (0.001, 0.01),
            'alpha_range': (0, math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.95
        },
        
        'baseline': {
            'description': 'Realistic mixed population for benchmarking',
            's_range': (0.1, 2.5),
            'q_range': (0.001, 1.0),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.75
        },
        
        'challenging': {
            'description': 'Includes physically undetectable events',
            's_range': (0.1, 3.0),
            'q_range': (0.0001, 1.0),
            'u0_range': (0.01, 1.0),
            'rho_range': (0.001, 0.1),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (10.0, 40.0),
            'expected_accuracy': 0.60
        },
        
        'planetary': {
            'description': 'Planet detection focus',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),
            'u0_range': (0.001, 0.2),
            'rho_range': (0.0001, 0.01),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.75
        },
        
        'stellar': {
            'description': 'Stellar binary focus',
            's_range': (0.3, 3.0),
            'q_range': (0.3, 1.0),
            'u0_range': (0.001, 0.4),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.70
        }
    }
    
    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """Get configuration with proper format for simulate.py"""
        if name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {name}. Choose from: {list(cls.CONFIGS.keys())}")
        
        cfg = cls.CONFIGS[name]
        return {
            's_min': cfg['s_range'][0],
            's_max': cfg['s_range'][1],
            'q_min': cfg['q_range'][0],
            'q_max': cfg['q_range'][1],
            'u0_min': cfg['u0_range'][0],
            'u0_max': cfg['u0_range'][1],
            'rho_min': cfg['rho_range'][0],
            'rho_max': cfg['rho_range'][1],
            'alpha_min': cfg['alpha_range'][0],
            'alpha_max': cfg['alpha_range'][1],
            't0_min': cfg['t0_range'][0],
            't0_max': cfg['t0_range'][1],
            'tE_min': cfg['tE_range'][0],
            'tE_max': cfg['tE_range'][1],
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ModelConfig:
    """Transformer architecture parameters"""
    D_MODEL = 256           # Embedding dimension
    NHEAD = 8              # Attention heads
    NUM_LAYERS = 6         # Transformer layers
    DIM_FEEDFORWARD = 1024 # FFN dimension (4 * d_model)
    DROPOUT = 0.1          # Dropout rate
    MAX_SEQ_LEN = 2000     # Maximum sequence length
    
    # Multi-task learning weights (updated for 3 classes)
    CLASSIFICATION_WEIGHT = 1.0  # Main task: 3-class classification
    ANOMALY_WEIGHT = 0.1         # Auxiliary task
    CAUSTIC_WEIGHT = 0.1         # Auxiliary task

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    MAX_EPOCHS = 50
    
    # Learning rate schedule
    LR_SCHEDULE = 'cosine'
    LR_GAMMA = 0.1
    LR_STEPS = [30, 40]
    
    # Gradient handling
    GRAD_CLIP = 1.0
    USE_AMP = True
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Checkpointing
    SAVE_EVERY = 5
    KEEP_LAST_N = 3

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

class EvaluationConfig:
    """Evaluation and analysis parameters"""
    # u0 dependency analysis
    U0_THRESHOLD = 0.3
    U0_BINS = 10
    
    # Early detection fractions
    EARLY_FRACTIONS = [0.1, 0.25, 0.5, 0.67, 0.833, 1.0]
    
    # Visualization settings
    N_EVOLUTION_PLOTS = 3
    N_EXAMPLE_GRID = 3
    
    # Performance thresholds (updated for 3-class)
    MIN_ACCURACY_WARNING = 0.65  # Lower for 3-class
    TARGET_ACCURACY = 0.75       # Harder with 3 classes

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

class SystemConfig:
    """System and path configuration"""
    DATA_DIR = "../data"
    RESULTS_DIR = "../results"
    
    SEED = 42
    NUMPY_SEED = 42
    TORCH_SEED = 42
    
    DEFAULT_NUM_WORKERS = 4
    DATALOADER_WORKERS = 4
    
    LOG_LEVEL = 'INFO'
    LOG_INTERVAL = 10

# ============================================================================
# EXPERIMENT PRESETS (UPDATED FOR 3-CLASS)
# ============================================================================

class ExperimentPresets:
    """Pre-configured experiment settings"""
    
    EXPERIMENTS = {
        'quick_test': {
            'n_flat': 100,
            'n_pspl': 100,
            'n_binary': 100,
            'binary_params': 'baseline',
            'epochs': 5,
            'description': 'Quick validation test (3-class)'
        },
        
        'baseline_benchmark': {
            'n_flat': 333000,
            'n_pspl': 333000,
            'n_binary': 334000,
            'binary_params': 'baseline',
            'epochs': 50,
            'description': 'Main thesis benchmark (3-class, 1M total)'
        },
        
        'physical_limit': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'challenging',
            'epochs': 50,
            'description': 'Test physical detection limits (3-class)'
        },
        
        'planetary_search': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'planetary',
            'epochs': 50,
            'description': 'Optimize for planet detection (3-class)'
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_configs() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        'simulation': SimulationConfig.__dict__,
        'flat': FlatConfig.__dict__,
        'pspl': PSPLConfig.__dict__,
        'binary': BinaryConfig.CONFIGS,
        'model': ModelConfig.__dict__,
        'training': TrainingConfig.__dict__,
        'evaluation': EvaluationConfig.__dict__,
        'system': SystemConfig.__dict__,
        'experiments': ExperimentPresets.EXPERIMENTS
    }

def print_config_summary():
    """Print configuration summary"""
    print("="*70)
    print("CONFIGURATION SUMMARY - v11.0 (THREE-CLASS CLASSIFICATION)")
    print("="*70)
    
    sim = SimulationConfig
    print(f"\n📊 Data Generation:")
    print(f"  Time window: [{sim.TIME_MIN}, {sim.TIME_MAX}] days")
    print(f"  Points: {sim.N_POINTS}")
    print(f"  Missing: {sim.CADENCE_MASK_PROB*100:.0f}%")
    print(f"  Error: {sim.MAG_ERROR_STD:.2f} mag")
    
    print(f"\n🌟 NEW in v11.0 - THREE CLASSES:")
    print(f"  Class 0: Flat (no event, baseline only)")
    print(f"  Class 1: PSPL (single lens)")
    print(f"  Class 2: Binary (binary lens)")
    
    pspl = PSPLConfig
    print(f"\n⭐ PSPL Parameters:")
    print(f"  u0: [{pspl.U0_MIN}, {pspl.U0_MAX}]")
    print(f"  tE: [{pspl.TE_MIN}, {pspl.TE_MAX}] days")
    
    print(f"\n🧬 Binary Configurations:")
    for name, cfg in BinaryConfig.CONFIGS.items():
        print(f"  {name}: {cfg['description']}")
        print(f"    u0=[{cfg['u0_range'][0]:.3f}, {cfg['u0_range'][1]:.3f}], "
              f"Expected: {cfg['expected_accuracy']*100:.0f}%")
    
    model = ModelConfig
    print(f"\n🤖 Model Architecture:")
    print(f"  Transformer: d={model.D_MODEL}, L={model.NUM_LAYERS}, H={model.NHEAD}")
    print(f"  Output: 3 classes (Flat, PSPL, Binary)")
    
    train = TrainingConfig
    print(f"\n🎯 Training:")
    print(f"  Batch size: {train.BATCH_SIZE} per GPU")
    print(f"  Learning rate: {train.LEARNING_RATE}")
    print(f"  Target accuracy: {EvaluationConfig.TARGET_ACCURACY*100:.0f}% (3-class)")
    
    print("="*70)

def validate_config():
    """Validate configuration consistency"""
    issues = []
    
    sim = SimulationConfig
    pspl = PSPLConfig
    
    time_window = sim.TIME_MAX - sim.TIME_MIN
    max_event_duration = pspl.TE_MAX * 4
    
    if max_event_duration > time_window:
        issues.append(f"⚠️ Event duration ({max_event_duration:.0f}d) > window ({time_window}d)")
    
    model = ModelConfig
    if model.D_MODEL % model.NHEAD != 0:
        issues.append(f"⚠️ d_model ({model.D_MODEL}) must be divisible by nhead ({model.NHEAD})")
    
    train = TrainingConfig
    split_sum = train.TRAIN_RATIO + train.VAL_RATIO + train.TEST_RATIO
    if abs(split_sum - 1.0) > 0.001:
        issues.append(f"⚠️ Data splits sum to {split_sum:.3f}, not 1.0")
    
    binary_baseline = BinaryConfig.CONFIGS['baseline']
    pspl_u0_range = (PSPLConfig.U0_MIN, PSPLConfig.U0_MAX)
    pspl_tE_range = (PSPLConfig.TE_MIN, PSPLConfig.TE_MAX)
    
    if pspl_u0_range != binary_baseline['u0_range']:
        issues.append(f"⚠️ PSPL u0 range {pspl_u0_range} != Binary u0 range {binary_baseline['u0_range']}")
    
    if pspl_tE_range != binary_baseline['tE_range']:
        issues.append(f"⚠️ PSPL tE range {pspl_tE_range} != Binary tE range {binary_baseline['tE_range']}")
    
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Configuration validated successfully!")
        print("✅ Ready for 3-class classification!")
        return True

# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# For existing code that expects old variable names
N_POINTS = SimulationConfig.N_POINTS
TIME_MIN = SimulationConfig.TIME_MIN
TIME_MAX = SimulationConfig.TIME_MAX
PAD_VALUE = SimulationConfig.PAD_VALUE

PSPL_T0_MIN = PSPLConfig.T0_MIN
PSPL_T0_MAX = PSPLConfig.T0_MAX
PSPL_U0_MIN = PSPLConfig.U0_MIN
PSPL_U0_MAX = PSPLConfig.U0_MAX
PSPL_TE_MIN = PSPLConfig.TE_MIN
PSPL_TE_MAX = PSPLConfig.TE_MAX

BASELINE_MIN = SimulationConfig.BASELINE_MIN
BASELINE_MAX = SimulationConfig.BASELINE_MAX
CADENCE_MASK_PROB = SimulationConfig.CADENCE_MASK_PROB
MAG_ERROR_STD = SimulationConfig.MAG_ERROR_STD
VBM_TOLERANCE = SimulationConfig.VBM_TOLERANCE
MAX_BINARY_ATTEMPTS = SimulationConfig.MAX_BINARY_ATTEMPTS

BINARY_PARAM_SETS = {
    name: BinaryConfig.get_config(name)
    for name in BinaryConfig.CONFIGS.keys()
}

# Add aliases for compatibility
BINARY_PARAM_SETS['distinct'] = BINARY_PARAM_SETS['critical']
BINARY_PARAM_SETS['overlapping'] = BINARY_PARAM_SETS['challenging']

def get_config_summary():
    """Wrapper for backwards compatibility"""
    print_config_summary()

if __name__ == "__main__":
    print_config_summary()
    validate_config()