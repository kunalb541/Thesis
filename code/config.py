"""
Configuration for Real-Time Binary Microlensing Detection
IMPROVED VERSION - Cleaner, Consistent, No Redundancy

Author: Kunal Bhatia
Date: November 2025
Version: 10.0 - Production Ready
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

class PSPLConfig:
    """Point Source Point Lens parameters"""
    T0_MIN = -20.0    # Peak time range
    T0_MAX = 20.0
    U0_MIN = 0.01     # Impact parameter range
    U0_MAX = 1.0      # Wide range for PSPL
    TE_MIN = 10.0     # Einstein time range
    TE_MAX = 40.0     # Fits in 200-day window

class BinaryConfig:
    """Binary lens configurations"""
    
    # Three main configurations (removed redundancy)
    CONFIGS = {
        'critical': {
            'description': 'Forces caustic crossings for clear detection',
            's_range': (0.7, 1.5),       # Optimal separation
            'q_range': (0.01, 0.5),      # Clear perturbations
            'u0_range': (0.001, 0.05),   # CRITICAL: Close approach
            'rho_range': (0.001, 0.01),  # Sharp features
            'alpha_range': (0, math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.90
        },
        
        'baseline': {
            'description': 'Realistic mixed population for benchmarking',
            's_range': (0.1, 2.5),       # Wide range
            'q_range': (0.001, 1.0),     # All mass ratios
            'u0_range': (0.001, 0.3),    # Mixed difficulties
            'rho_range': (0.001, 0.05),  
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.72
        },
        
        'challenging': {
            'description': 'Includes physically undetectable events',
            's_range': (0.1, 3.0),       # Very wide range
            'q_range': (0.0001, 1.0),    # Include tiny planets
            'u0_range': (0.01, 1.0),     # INCLUDES u0 > 0.3!
            'rho_range': (0.001, 0.1),   # Some smoothed
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (10.0, 40.0),
            'expected_accuracy': 0.60
        },
        
        # Specialized configs for specific science
        'planetary': {
            'description': 'Planet detection focus',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),   # Planetary mass ratios
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
            'q_range': (0.3, 1.0),       # Comparable masses
            'u0_range': (0.001, 0.4),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-20.0, 20.0),
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.65
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
    # FIXED: Match what's actually used in training
    D_MODEL = 256           # Embedding dimension (was wrongly 64)
    NHEAD = 8              # Attention heads (was wrongly 4)
    NUM_LAYERS = 6         # Transformer layers (was wrongly 3)
    DIM_FEEDFORWARD = 1024 # FFN dimension (4 * d_model)
    DROPOUT = 0.1          # Dropout rate
    MAX_SEQ_LEN = 2000     # Maximum sequence length
    
    # Multi-task learning weights
    BINARY_WEIGHT = 1.0    # Main task
    ANOMALY_WEIGHT = 0.1   # Auxiliary task
    CAUSTIC_WEIGHT = 0.1   # Auxiliary task

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    BATCH_SIZE = 32        # Per GPU
    LEARNING_RATE = 1e-3   # Base learning rate
    WEIGHT_DECAY = 1e-4    
    WARMUP_EPOCHS = 5
    MAX_EPOCHS = 50
    
    # Learning rate schedule
    LR_SCHEDULE = 'cosine'  # 'cosine' or 'step'
    LR_GAMMA = 0.1         # For step schedule
    LR_STEPS = [30, 40]    # For step schedule
    
    # Gradient handling
    GRAD_CLIP = 1.0        # Gradient clipping
    USE_AMP = True         # Mixed precision training
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Checkpointing
    SAVE_EVERY = 5         # Save checkpoint every N epochs
    KEEP_LAST_N = 3       # Keep last N checkpoints

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

class EvaluationConfig:
    """Evaluation and analysis parameters"""
    # u0 dependency analysis
    U0_THRESHOLD = 0.3     # Physical detection limit
    U0_BINS = 10           # Number of bins for analysis
    
    # Chi-square analysis (NEW!)
    CHI2_COMPUTE = True    # Enable chi-square evolution
    CHI2_BINS = 20         # Time bins for chi-square
    
    # Early detection fractions
    EARLY_FRACTIONS = [0.1, 0.25, 0.5, 0.67, 0.833, 1.0]
    
    # Visualization settings
    N_EVOLUTION_PLOTS = 3  # Per event type
    N_EXAMPLE_GRID = 3     # Per classification type
    
    # Performance thresholds
    MIN_ACCURACY_WARNING = 0.6
    TARGET_ACCURACY = 0.7

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

class SystemConfig:
    """System and path configuration"""
    # Paths
    DATA_DIR = "../data"
    RESULTS_DIR = "../results"
    
    # Random seeds for reproducibility
    SEED = 42
    NUMPY_SEED = 42
    TORCH_SEED = 42
    
    # Parallel processing
    DEFAULT_NUM_WORKERS = 4
    DATALOADER_WORKERS = 4
    
    # Logging
    LOG_LEVEL = 'INFO'    # DEBUG, INFO, WARNING, ERROR
    LOG_INTERVAL = 10     # Log every N batches

# ============================================================================
# CHI-SQUARE CONFIGURATION (NEW!)
# ============================================================================

class ChiSquareConfig:
    """Configuration for chi-square goodness-of-fit analysis"""
    
    @staticmethod
    def compute_chi2(observed, model, errors):
        """
        Compute chi-square statistic
        
        Args:
            observed: Observed flux values
            model: Model flux predictions  
            errors: Measurement uncertainties
            
        Returns:
            chi2: Chi-square value
            dof: Degrees of freedom
            reduced_chi2: Reduced chi-square
        """
        import numpy as np
        
        # Only use valid (non-padded) points
        valid_mask = observed != -1.0
        
        if not valid_mask.any():
            return np.inf, 0, np.inf
        
        obs = observed[valid_mask]
        mod = model[valid_mask]
        err = errors[valid_mask] if errors is not None else np.ones_like(obs) * 0.1
        
        # Chi-square calculation
        residuals = (obs - mod) / err
        chi2 = np.sum(residuals ** 2)
        
        # Degrees of freedom
        n_points = len(obs)
        n_params = 3  # Simplified: t0, u0, tE for PSPL
        dof = max(n_points - n_params, 1)
        
        # Reduced chi-square
        reduced_chi2 = chi2 / dof
        
        return chi2, dof, reduced_chi2
    
    # Thresholds for interpretation
    GOOD_FIT_THRESHOLD = 1.5      # Reduced chi2 < 1.5 is good
    ACCEPTABLE_FIT_THRESHOLD = 3.0 # Reduced chi2 < 3.0 is acceptable

# ============================================================================
# EXPERIMENT PRESETS
# ============================================================================

class ExperimentPresets:
    """Pre-configured experiment settings"""
    
    EXPERIMENTS = {
        'quick_test': {
            'n_pspl': 100,
            'n_binary': 100,
            'binary_params': 'baseline',
            'epochs': 5,
            'description': 'Quick validation test'
        },
        
        'baseline_benchmark': {
            'n_pspl': 500000,
            'n_binary': 500000,
            'binary_params': 'baseline',
            'epochs': 50,
            'description': 'Main thesis benchmark'
        },
        
        'physical_limit': {
            'n_pspl': 100000,
            'n_binary': 100000,
            'binary_params': 'challenging',
            'epochs': 50,
            'description': 'Test physical detection limits'
        },
        
        'planetary_search': {
            'n_pspl': 100000,
            'n_binary': 100000,
            'binary_params': 'planetary',
            'epochs': 50,
            'description': 'Optimize for planet detection'
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_configs() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        'simulation': SimulationConfig.__dict__,
        'pspl': PSPLConfig.__dict__,
        'binary': BinaryConfig.CONFIGS,
        'model': ModelConfig.__dict__,
        'training': TrainingConfig.__dict__,
        'evaluation': EvaluationConfig.__dict__,
        'system': SystemConfig.__dict__,
        'experiments': ExperimentPresets.EXPERIMENTS
    }

def validate_config():
    """Validate configuration consistency"""
    issues = []
    
    # Check time window can contain events
    sim = SimulationConfig
    pspl = PSPLConfig
    
    time_window = sim.TIME_MAX - sim.TIME_MIN
    max_event_duration = pspl.TE_MAX * 4  # ~4 tE for full event
    
    if max_event_duration > time_window:
        issues.append(f"⚠️ Event duration ({max_event_duration:.0f}d) > window ({time_window}d)")
    
    # Check model parameters
    model = ModelConfig
    if model.D_MODEL % model.NHEAD != 0:
        issues.append(f"⚠️ d_model ({model.D_MODEL}) must be divisible by nhead ({model.NHEAD})")
    
    # Check data splits
    train = TrainingConfig
    split_sum = train.TRAIN_RATIO + train.VAL_RATIO + train.TEST_RATIO
    if abs(split_sum - 1.0) > 0.001:
        issues.append(f"⚠️ Data splits sum to {split_sum:.3f}, not 1.0")
    
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Configuration validated successfully!")
        return True

def print_config_summary():
    """Print configuration summary"""
    print("="*70)
    print("CONFIGURATION SUMMARY - v11.0")
    print("="*70)
    
    sim = SimulationConfig
    print(f"\n📊 Data Generation:")
    print(f"  Time window: [{sim.TIME_MIN}, {sim.TIME_MAX}] days")
    print(f"  Points: {sim.N_POINTS}")
    print(f"  Missing: {sim.CADENCE_MASK_PROB*100:.0f}%")
    print(f"  Error: {sim.MAG_ERROR_STD:.2f} mag")
    
    print(f"\n🧬 Binary Configurations:")
    for name, cfg in BinaryConfig.CONFIGS.items():
        print(f"  {name}: {cfg['description']}")
        print(f"    u0=[{cfg['u0_range'][0]:.3f}, {cfg['u0_range'][1]:.3f}], "
              f"Expected: {cfg['expected_accuracy']*100:.0f}%")
    
    model = ModelConfig
    print(f"\n🤖 Model Architecture:")
    print(f"  Transformer: d={model.D_MODEL}, L={model.NUM_LAYERS}, H={model.NHEAD}")
    print(f"  Multi-task weights: Binary={model.BINARY_WEIGHT}, "
          f"Anomaly={model.ANOMALY_WEIGHT}, Caustic={model.CAUSTIC_WEIGHT}")
    
    train = TrainingConfig
    print(f"\n🎯 Training:")
    print(f"  Batch size: {train.BATCH_SIZE} per GPU")
    print(f"  Learning rate: {train.LEARNING_RATE}")
    print(f"  Schedule: {train.LR_SCHEDULE}")
    print(f"  Early stopping: patience={train.PATIENCE}")
    
    eval_cfg = EvaluationConfig
    print(f"\n📈 Evaluation:")
    print(f"  u0 threshold: {eval_cfg.U0_THRESHOLD}")
    print(f"  Chi-square analysis: {eval_cfg.CHI2_COMPUTE}")
    print(f"  Target accuracy: {eval_cfg.TARGET_ACCURACY*100:.0f}%")
    
    print("="*70)

# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# For existing code that expects old variable names
N_POINTS = SimulationConfig.N_POINTS
TIME_MIN = SimulationConfig.TIME_MIN
TIME_MAX = SimulationConfig.TIME_MAX
PAD_VALUE = SimulationConfig.PAD_VALUE

# Binary param sets for simulate.py compatibility
BINARY_PARAM_SETS = {
    name: BinaryConfig.get_config(name)
    for name in BinaryConfig.CONFIGS.keys()
}

# Add distinct for compatibility (use critical instead)
BINARY_PARAM_SETS['distinct'] = BINARY_PARAM_SETS['critical']
# Add overlapping for compatibility (use challenging instead)  
BINARY_PARAM_SETS['overlapping'] = BINARY_PARAM_SETS['challenging']

if __name__ == "__main__":
    print_config_summary()
    validate_config()