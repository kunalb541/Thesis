"""
Configuration for Real-Time Binary Microlensing Detection
VERSION 12.0-beta - ARCHITECTURAL FIX (NO CAUSAL TRAINING)

CRITICAL CHANGES in v12.0-beta:
- Smaller model (d_model=128) for faster training
- Wider t0 range to prevent timing artifacts
- Relative positional encoding (architecture fix)
- NO causal truncation (tested and rejected - hurts PSPL performance)

Author: Kunal Bhatia
Date: November 2025
Version: 12.0-beta - Architectural Fix
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
# EVENT PARAMETERS - v12.0-beta FIXES
# ============================================================================

class FlatConfig:
    """
    Flat (no event) parameters
    Just constant baseline flux
    """
    pass  # No special parameters needed - just baseline magnitude

class PSPLConfig:
    """
    Point Source Point Lens parameters
    
    v12.0-beta FIX: WIDER t0 range to prevent timing artifacts
    Now events can peak anywhere from -50 to +50 days
    """
    T0_MIN = -50.0    # CHANGED: Much wider range (was -20)
    T0_MAX = 50.0     # CHANGED: More realistic (was 20)
    U0_MIN = 0.001    
    U0_MAX = 0.3      
    TE_MIN = 20.0     
    TE_MAX = 40.0

class BinaryConfig:
    """Binary lens configurations with wider t0 ranges"""
    
    # Three main configurations
    CONFIGS = {
        'critical': {
            'description': 'Forces caustic crossings for clear detection',
            's_range': (0.7, 1.5),
            'q_range': (0.01, 0.5),
            'u0_range': (0.001, 0.05),
            'rho_range': (0.001, 0.01),
            'alpha_range': (0, math.pi),
            't0_range': (-50.0, 50.0),  # CHANGED: Wider (was -20, 20)
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.85
        },
        
        'baseline': {
            'description': 'Realistic mixed population for benchmarking',
            's_range': (0.1, 2.5),
            'q_range': (0.001, 1.0),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),  # CHANGED: Wider (was -20, 20)
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.70
        },
        
        'challenging': {
            'description': 'Includes physically undetectable events',
            's_range': (0.1, 3.0),
            'q_range': (0.0001, 1.0),
            'u0_range': (0.01, 1.0),
            'rho_range': (0.001, 0.1),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),  # CHANGED: Wider (was -20, 20)
            'tE_range': (10.0, 40.0),
            'expected_accuracy': 0.55
        },
        
        'planetary': {
            'description': 'Planet detection focus',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),
            'u0_range': (0.001, 0.2),
            'rho_range': (0.0001, 0.01),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),  # CHANGED: Wider
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.70
        },
        
        'stellar': {
            'description': 'Stellar binary focus',
            's_range': (0.3, 3.0),
            'q_range': (0.3, 1.0),
            'u0_range': (0.001, 0.4),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),  # CHANGED: Wider
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
# MODEL ARCHITECTURE - v12.0-beta SMALLER MODEL
# ============================================================================

class ModelConfig:
    """
    Transformer architecture parameters
    
    v12.0-beta CHANGES: SMALLER MODEL for faster iteration
    - d_model: 256 → 128 (4x fewer parameters!)
    - nhead: 8 → 4 
    - num_layers: 6 → 4
    - dim_feedforward: 1024 → 512
    
    This gives ~100K parameters instead of ~450K
    Much faster training for debugging!
    """
    D_MODEL = 128              # CHANGED: 256 → 128
    NHEAD = 4                  # CHANGED: 8 → 4
    NUM_LAYERS = 4             # CHANGED: 6 → 4
    DIM_FEEDFORWARD = 512      # CHANGED: 1024 → 512 (4 * d_model)
    DROPOUT = 0.1
    MAX_SEQ_LEN = 2000
    
    # Multi-task learning weights (same as v11)
    CLASSIFICATION_WEIGHT = 1.0
    ANOMALY_WEIGHT = 0.1
    CAUSTIC_WEIGHT = 0.1

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    BATCH_SIZE = 64            # CHANGED: 32 → 64 (smaller model fits more)
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
    CAUSAL_TRAINING = False  # DISABLED - architectural fix is better

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

class EvaluationConfig:
    """Evaluation and analysis parameters"""
    # u0 dependency analysis
    U0_THRESHOLD = 0.3
    U0_BINS = 10
    
    # Early detection fractions
    EARLY_FRACTIONS = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
    
    # Visualization settings
    N_EVOLUTION_PLOTS = 3
    N_EXAMPLE_GRID = 3
    
    # Performance thresholds (realistic for v12.0-beta)
    MIN_ACCURACY_WARNING = 0.60  # Lower due to harder task
    TARGET_ACCURACY = 0.70       # Realistic for 3-class

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
# EXPERIMENT PRESETS (UPDATED FOR v12.0-beta)
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
            'description': 'Quick validation test (v12.0-beta architectural fix)'
        },
        
        'baseline_1M': {
            'n_flat': 333000,
            'n_pspl': 333000,
            'n_binary': 334000,
            'binary_params': 'baseline',
            'epochs': 50,
            'description': 'Main v12.0-beta benchmark (1M total, architectural fix)'
        },
        
        'physical_limit': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'challenging',
            'epochs': 50,
            'description': 'Test physical detection limits (architectural fix)'
        },
        
        'planetary_search': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'planetary',
            'epochs': 50,
            'description': 'Optimize for planet detection (architectural fix)'
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
    print("CONFIGURATION SUMMARY - v12.0-beta (ARCHITECTURAL FIX)")
    print("="*70)
    
    sim = SimulationConfig
    print(f"\n📊 Data Generation:")
    print(f"  Time window: [{sim.TIME_MIN}, {sim.TIME_MAX}] days")
    print(f"  Points: {sim.N_POINTS}")
    print(f"  Missing: {sim.CADENCE_MASK_PROB*100:.0f}%")
    print(f"  Error: {sim.MAG_ERROR_STD:.2f} mag")
    
    print(f"\n🔧 v12.0-beta FIXES:")
    print(f"  ✅ Wider t0 range: [{PSPLConfig.T0_MIN}, {PSPLConfig.T0_MAX}] (was -20, 20)")
    print(f"  ✅ Relative positional encoding (no absolute time)")
    print(f"  ✅ Variable-length training (no padding artifacts)")
    print(f"  ✅ NO causal truncation (preserves PSPL features)")
    print(f"  ✅ Smaller model for faster iteration")
    
    pspl = PSPLConfig
    print(f"\n⭐ PSPL Parameters:")
    print(f"  t0: [{pspl.T0_MIN}, {pspl.T0_MAX}] days (WIDER!)")
    print(f"  u0: [{pspl.U0_MIN}, {pspl.U0_MAX}]")
    print(f"  tE: [{pspl.TE_MIN}, {pspl.TE_MAX}] days")
    
    print(f"\n🧬 Binary Configurations:")
    for name, cfg in BinaryConfig.CONFIGS.items():
        print(f"  {name}: t0=[{cfg['t0_range'][0]:.0f}, {cfg['t0_range'][1]:.0f}] days")
    
    model = ModelConfig
    print(f"\n🤖 Model Architecture (SMALLER v12.0-beta):")
    print(f"  d_model: {model.D_MODEL} (was 256)")
    print(f"  nhead: {model.NHEAD} (was 8)")
    print(f"  num_layers: {model.NUM_LAYERS} (was 6)")
    print(f"  dim_ff: {model.DIM_FEEDFORWARD} (was 1024)")
    print(f"  Est. parameters: ~100K (was ~450K)")
    
    train = TrainingConfig
    print(f"\n🎯 Training:")
    print(f"  Batch size: {train.BATCH_SIZE} per GPU")
    print(f"  Learning rate: {train.LEARNING_RATE}")
    print(f"  Causal training: {train.CAUSAL_TRAINING}")
    if train.CAUSAL_TRAINING:
        print(f"    WARNING: Causal training is enabled!")
        print(f"    This was tested and found to hurt PSPL performance.")
        print(f"    Recommended: Keep disabled (architectural fix is sufficient)")
    else:
        print(f"    ✅ DISABLED (architectural fix is sufficient)")
        print(f"    Full sequences preserve PSPL features")
    
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
    
    # Check if t0 ranges make sense
    if abs(PSPLConfig.T0_MAX - PSPLConfig.T0_MIN) < 50:
        issues.append(f"⚠️ t0 range too narrow: [{PSPLConfig.T0_MIN}, {PSPLConfig.T0_MAX}]")
    
    # Warn if causal training is enabled
    if train.CAUSAL_TRAINING:
        issues.append(f"⚠️ WARNING: Causal training is enabled!")
        issues.append(f"   This was tested and found to degrade PSPL performance (77% → <60%)")
        issues.append(f"   Recommendation: Disable causal training (architectural fix is sufficient)")
    
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Configuration validated successfully!")
        print("✅ Ready for v12.0-beta training (architectural fix)!")
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
    print()
    validate_config()