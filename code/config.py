"""
Configuration for Real-Time Binary Microlensing Classification

Three-Class System: 0=Flat (no event), 1=PSPL, 2=Binary

Key architectural feature: Relative positional encoding prevents
temporal information leakage. Model learns from magnification patterns,
not absolute time positions.

Author: Kunal Bhatia
Institution: University of Heidelberg
Thesis: From Light Curves to Labels - Machine Learning in Microlensing
"""

import math
from typing import Dict, Any

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Data generation parameters for realistic microlensing observations"""
    
    # Temporal sampling
    N_POINTS = 1500          # Full temporal resolution
    TIME_MIN = -100          # Days before peak
    TIME_MAX = 100           # Days after peak
    
    # VBBinaryLensing settings
    VBM_TOLERANCE = 1e-4
    MAX_BINARY_ATTEMPTS = 10
    
    # Observational effects (baseline)
    CADENCE_MASK_PROB = 0.20  # 20% missing (LSST nominal)
    MAG_ERROR_STD = 0.10       # 0.10 mag (ground-based)
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
    Just constant baseline flux with photometric scatter
    """
    pass  # No special parameters - uses baseline magnitude only

class PSPLConfig:
    """
    Point Source Point Lens parameters
    
    Wide t0 range prevents temporal information leakage
    """
    T0_MIN = -50.0    # Wide range to prevent timing artifacts
    T0_MAX = 50.0
    U0_MIN = 0.001    # Close to very far approaches
    U0_MAX = 0.3      # Focus on detectable events
    TE_MIN = 20.0     # Typical timescales
    TE_MAX = 40.0

class BinaryConfig:
    """
    Binary lens configurations across different physical regimes
    
    Configurations span from clear caustic crossings (distinct) to
    near-physical detection limits (challenging).
    """
    
    CONFIGS = {
        'distinct': {
            'description': 'Clear caustic crossings for unambiguous detection',
            's_range': (0.7, 1.5),      # Optimal caustic topology
            'q_range': (0.01, 0.5),     # Moderate to high mass ratios
            'u0_range': (0.001, 0.05),  # Close approaches only
            'rho_range': (0.001, 0.01), # Sharp features
            'alpha_range': (0, math.pi),
            't0_range': (-50.0, 50.0),  # Wide temporal range
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.85
        },
        
        'baseline': {
            'description': 'Realistic mixed population for benchmarking',
            's_range': (0.1, 2.5),      # Wide separation range
            'q_range': (0.001, 1.0),    # All mass ratios
            'u0_range': (0.001, 0.3),   # Mix of close/far
            'rho_range': (0.001, 0.05), # Typical source sizes
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.70
        },
        
        'challenging': {
            'description': 'Near physical detection limit - includes indistinguishable events',
            's_range': (0.1, 3.0),      # Very wide separations
            'q_range': (0.0001, 1.0),   # All mass ratios
            'u0_range': (0.01, 1.0),    # Includes large u0 (PSPL-like)
            'rho_range': (0.001, 0.1),  # Large source sizes (smooth features)
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),
            'tE_range': (10.0, 40.0),
            'expected_accuracy': 0.55   # Lower due to physical ambiguity
        },
        
        'planetary': {
            'description': 'Planet detection focus - low mass ratio systems',
            's_range': (0.5, 2.0),      # Typical planetary separations
            'q_range': (0.0001, 0.01),  # Planetary mass ratios
            'u0_range': (0.001, 0.2),   # Moderate approaches
            'rho_range': (0.0001, 0.01),# Small sources (sharp features)
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.70
        },
        
        'stellar': {
            'description': 'Stellar binary focus - comparable mass systems',
            's_range': (0.3, 3.0),      # Wide range of separations
            'q_range': (0.3, 1.0),      # Near equal mass
            'u0_range': (0.001, 0.4),   # Various impact parameters
            'rho_range': (0.001, 0.05), # Typical sources
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-50.0, 50.0),
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.65
        }
    }
    
    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """Get configuration formatted for simulate.py"""
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
    
    D_MODEL = 128              # Embedding dimension
    NHEAD = 4                  # Attention heads
    NUM_LAYERS = 4             # Transformer blocks
    DIM_FEEDFORWARD = 512      # FFN hidden dimension (4 × d_model)
    DROPOUT = 0.1
    MAX_SEQ_LEN = 2000        # Maximum sequence length
    
    # Multi-task learning weights
    CLASSIFICATION_WEIGHT = 1.0
    FLAT_WEIGHT = 0.5          # Flat detection (high priority)
    PSPL_WEIGHT = 0.5          # PSPL detection (high priority)
    ANOMALY_WEIGHT = 0.2       # General event detection
    CAUSTIC_WEIGHT = 0.2       # Binary-specific features

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """Training hyperparameters"""
    
    # Optimization
    BATCH_SIZE = 64
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
    USE_AMP = True             # Mixed precision training
    
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
    
    # Impact parameter analysis
    U0_THRESHOLD = 0.3         # Physical detection limit
    U0_BINS = 10
    
    # Early detection fractions
    EARLY_FRACTIONS = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
    
    # Visualization settings
    N_EVOLUTION_PLOTS = 3
    N_EXAMPLE_GRID = 3
    
    # Performance thresholds
    MIN_ACCURACY_WARNING = 0.60
    TARGET_ACCURACY = 0.70

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
# EXPERIMENT PRESETS
# ============================================================================

class ExperimentPresets:
    """Pre-configured experimental setups for systematic thesis research"""
    
    EXPERIMENTS = {
        # Quick validation
        'quick_test': {
            'n_flat': 100,
            'n_pspl': 100,
            'n_binary': 100,
            'binary_params': 'baseline',
            'epochs': 5,
            'description': 'Quick validation test'
        },
        
        # Main benchmark
        'baseline_1M': {
            'n_flat': 333000,
            'n_pspl': 333000,
            'n_binary': 334000,
            'binary_params': 'baseline',
            'epochs': 50,
            'description': 'Main benchmark - realistic parameter distribution'
        },
        
        # Binary topology suite
        'distinct': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'distinct',
            'epochs': 50,
            'description': 'Clear caustic crossings'
        },
        
        'planetary': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'planetary',
            'epochs': 50,
            'description': 'Exoplanet detection focus'
        },
        
        'stellar': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'stellar',
            'epochs': 50,
            'description': 'Equal-mass binary systems'
        },
        
        'challenging': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'challenging',
            'epochs': 50,
            'description': 'Near physical detection limit'
        },
        
        # Cadence experiments
        'cadence_05': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'cadence_mask_prob': 0.05,
            'epochs': 50,
            'description': 'Dense cadence (5% missing) - intensive follow-up'
        },
        
        'cadence_20': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'cadence_mask_prob': 0.20,
            'epochs': 50,
            'description': 'LSST nominal cadence (20% missing)'
        },
        
        'cadence_30': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'cadence_mask_prob': 0.30,
            'epochs': 50,
            'description': 'Sparse cadence (30% missing) - poor weather'
        },
        
        'cadence_40': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'cadence_mask_prob': 0.40,
            'epochs': 50,
            'description': 'Very sparse cadence (40% missing) - limited coverage'
        },
        
        # Photometric error experiments
        'error_05': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'mag_error_std': 0.05,
            'epochs': 50,
            'description': 'Low photometric error (0.05 mag) - space-based quality'
        },
        
        'error_10': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'mag_error_std': 0.10,
            'epochs': 50,
            'description': 'Medium photometric error (0.10 mag) - ground-based quality'
        },
        
        'error_20': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'baseline',
            'mag_error_std': 0.20,
            'epochs': 50,
            'description': 'High photometric error (0.20 mag) - poor conditions'
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_configs() -> Dict[str, Any]:
    """Get all configuration as dictionary"""
    return {
        'simulation': vars(SimulationConfig),
        'flat': vars(FlatConfig),
        'pspl': vars(PSPLConfig),
        'binary': BinaryConfig.CONFIGS,
        'model': vars(ModelConfig),
        'training': vars(TrainingConfig),
        'evaluation': vars(EvaluationConfig),
        'system': vars(SystemConfig),
        'experiments': ExperimentPresets.EXPERIMENTS
    }

def print_config_summary():
    """Print human-readable configuration summary"""
    print("="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    sim = SimulationConfig
    print(f"\n📊 Temporal Sampling:")
    print(f"  Window: [{sim.TIME_MIN}, {sim.TIME_MAX}] days")
    print(f"  Points: {sim.N_POINTS}")
    
    print(f"\n🔭 Observational Effects (Baseline):")
    print(f"  Cadence: {sim.CADENCE_MASK_PROB*100:.0f}% missing")
    print(f"  Photometry: {sim.MAG_ERROR_STD:.2f} mag")
    print(f"  Baseline: [{sim.BASELINE_MIN}, {sim.BASELINE_MAX}] mag")
    
    pspl = PSPLConfig
    print(f"\n⭐ PSPL Parameters:")
    print(f"  t0: [{pspl.T0_MIN}, {pspl.T0_MAX}] days (wide to prevent leakage)")
    print(f"  u0: [{pspl.U0_MIN}, {pspl.U0_MAX}]")
    print(f"  tE: [{pspl.TE_MIN}, {pspl.TE_MAX}] days")
    
    print(f"\n🧬 Binary Configurations:")
    for name, cfg in BinaryConfig.CONFIGS.items():
        print(f"  {name:12s}: t0=[{cfg['t0_range'][0]:4.0f}, {cfg['t0_range'][1]:4.0f}] days, "
              f"expected accuracy={cfg['expected_accuracy']*100:.0f}%")
    
    model = ModelConfig
    print(f"\n🤖 Model Architecture:")
    print(f"  d_model: {model.D_MODEL}")
    print(f"  nhead: {model.NHEAD}")
    print(f"  num_layers: {model.NUM_LAYERS}")
    print(f"  dim_feedforward: {model.DIM_FEEDFORWARD}")
    print(f"  Parameters: ~100K")
    
    train = TrainingConfig
    print(f"\n🎯 Training:")
    print(f"  Batch size: {train.BATCH_SIZE} per GPU")
    print(f"  Learning rate: {train.LEARNING_RATE}")
    print(f"  Mixed precision: {train.USE_AMP}")
    print(f"  Max epochs: {train.MAX_EPOCHS}")
    
    print(f"\n📊 Available Experiments: {len(ExperimentPresets.EXPERIMENTS)}")
    print("="*70)

def validate_config():
    """Validate configuration consistency"""
    issues = []
    
    # Check temporal window vs. event duration
    sim = SimulationConfig
    pspl = PSPLConfig
    time_window = sim.TIME_MAX - sim.TIME_MIN
    max_event_duration = pspl.TE_MAX * 4
    
    if max_event_duration > time_window:
        issues.append(f"⚠️  Event duration ({max_event_duration:.0f}d) > window ({time_window}d)")
    
    # Check model configuration
    model = ModelConfig
    if model.D_MODEL % model.NHEAD != 0:
        issues.append(f"⚠️  d_model ({model.D_MODEL}) must be divisible by nhead ({model.NHEAD})")
    
    # Check data splits
    train = TrainingConfig
    split_sum = train.TRAIN_RATIO + train.VAL_RATIO + train.TEST_RATIO
    if abs(split_sum - 1.0) > 0.001:
        issues.append(f"⚠️  Data splits sum to {split_sum:.3f}, not 1.0")
    
    # Check t0 range
    if abs(PSPLConfig.T0_MAX - PSPLConfig.T0_MIN) < 50:
        issues.append(f"⚠️  t0 range too narrow: [{PSPLConfig.T0_MIN}, {PSPLConfig.T0_MAX}]")
    
    # Report
    if issues:
        print("\n⚠️  Configuration Warnings:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Configuration validated successfully!")
        return True

# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# For existing code expecting old variable names
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

def get_config_summary():
    """Wrapper for backwards compatibility"""
    print_config_summary()

if __name__ == "__main__":
    print_config_summary()
    print()
    validate_config()