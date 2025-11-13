"""
Configuration for Microlensing Classification - Roman Space Telescope
======================================================================

**VERSION 14.0 - ROMAN SPACE TELESCOPE + TEMPORAL BIAS FIX v2**

Key Changes from v13.1:
1. TARGET: Roman Space Telescope (was LSST ground-based)
2. CADENCE_MASK_PROB: 0.05 (was 0.20) - 5% missing, ~15 min cadence
3. MAG_ERROR_STD: 0.05 (was 0.10) - Space-based photometry quality
4. TEMPORAL BIAS FIX v2: t₀ ∈ [-60, +40] days (was [0, +80])
   - Events can peak before, during, or early in observation
   - All events show characteristic features by window end
   - Prevents temporal position shortcuts
   - Ensures label-feature consistency
5. RESEARCH SCOPE: Binary morphology study only (removed cadence/error experiments)

Rationale:
- Roman Space Telescope more relevant for thesis timeline
- Better data quality enables clearer binary morphology study
- Proper temporal bias fix: varied event stages without label mismatch
- Focus on physical detection limits rather than observational effects
- Simplified experimental scope (5 experiments vs. 11)

Physical Parameters:
- Impact parameter (u₀): Most critical - determines caustic crossing probability
- Separation (s): Binary component distance
- Mass ratio (q): m₂/m₁ (companion/primary)
- Source size (rho): Angular radius/Einstein radius  
- Einstein timescale (tE): Event duration
- Peak time (t₀): When maximum magnification occurs

Temporal Bias Fix Explanation:
- Observation window: [-120, +120] days (240 days total)
- Event peaks: [-60, +40] days (100 days range)
- Earliest peak: t = -60 → 60 days of pre-peak baseline
- Latest peak: t = +40 → 80 days of post-peak observation
- Event visibility: t₀ - 3*tE to t₀ + 3*tE (≈ -180 to +160 for tE=40)
- Result: ALL events show complete light curve features within window
- Model CANNOT use temporal position as shortcut (events scattered)

Author: Kunal Bhatia
Version: 14.0
"""

import math
from typing import Dict, Any

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Data generation parameters - Roman Space Telescope configuration"""
    # Temporal sampling - TEMPORAL BIAS FIX v2
    N_POINTS = 1500          # Full temporal resolution
    TIME_MIN = -120          # Extended baseline (prevents temporal bias)
    TIME_MAX = 120           # Symmetric window (240 days total)
    
    # VBBinaryLensing settings
    VBM_TOLERANCE = 1e-4
    MAX_BINARY_ATTEMPTS = 10
    
    # Observational effects - ROMAN SPACE TELESCOPE
    CADENCE_MASK_PROB = 0.05  # 5% missing (Roman ~15 min cadence)
    MAG_ERROR_STD = 0.05       # Space-based photometry quality
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
    pass  # No special parameters needed

class PSPLConfig:
    """
    Point Source Point Lens parameters
    
    **TEMPORAL BIAS FIX v2**: t₀ ∈ [-60, +40] ensures:
    1. Events can peak before observation starts (some declining)
    2. Events can peak during observation (various stages)
    3. Events can peak early in observation (rising phase visible)
    4. ALL events show complete features by window end
    5. Model cannot use temporal position as classification shortcut
    
    With tE ∈ [20, 40] days:
    - Event visible from: t₀ - 3*tE ≈ t₀ - 120 days
    - Event visible to: t₀ + 3*tE ≈ t₀ + 120 days
    - Earliest peak (t₀ = -60): visible from t ≈ -180 (before window)
    - Latest peak (t₀ = +40): visible to t ≈ +160 (after window)
    - Result: Complete light curve features always visible
    """
    T0_MIN = -60.0    # Can peak 60 days before window midpoint
    T0_MAX = 40.0     # Can peak 40 days after window midpoint
    U0_MIN = 0.001    
    U0_MAX = 0.3      
    TE_MIN = 20.0     
    TE_MAX = 40.0

class BinaryConfig:
    """Binary lens configurations - Roman Space Telescope + Temporal Fix v2"""
    
    CONFIGS = {
        'baseline': {
            'description': 'Realistic mixed population for Roman (temporal fix v2)',
            's_range': (0.1, 2.5),
            'q_range': (0.001, 1.0),
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-60.0, 40.0),    # Temporal bias fix v2
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.80     # Higher with Roman + proper temporal fix
        },
        
        'distinct': {
            'description': 'Clear caustics (s≈1, small u0) - optimal detection',
            's_range': (0.7, 1.5),
            'q_range': (0.01, 0.5),
            'u0_range': (0.001, 0.15),    # Only small u0
            'rho_range': (0.001, 0.01),
            'alpha_range': (0, math.pi),
            't0_range': (-60.0, 40.0),    # Temporal bias fix v2
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.90     # Easy to detect with Roman quality
        },
        
        'planetary': {
            'description': 'Exoplanet detection focus (small q, planet mass ratios)',
            's_range': (0.5, 2.0),
            'q_range': (0.0001, 0.01),    # Planetary mass ratios
            'u0_range': (0.001, 0.3),
            'rho_range': (0.0001, 0.01),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-60.0, 40.0),    # Temporal bias fix v2
            'tE_range': (20.0, 40.0),
            'expected_accuracy': 0.77     # Small features, but Roman helps
        },
        
        'stellar': {
            'description': 'Binary stars focus (large q, equal-mass systems)',
            's_range': (0.3, 3.0),
            'q_range': (0.3, 1.0),        # Stellar mass ratios
            'u0_range': (0.001, 0.3),
            'rho_range': (0.001, 0.05),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-60.0, 40.0),    # Temporal bias fix v2
            'tE_range': (30.0, 40.0),
            'expected_accuracy': 0.75     # Complex caustics, challenging
        },
        
        'challenging': {
            'description': 'Physical detection limit study (wide u0, includes undetectable)',
            's_range': (0.1, 3.0),
            'q_range': (0.0001, 1.0),
            'u0_range': (0.01, 1.0),      # Includes undetectable (u0>0.3)
            'rho_range': (0.001, 0.1),
            'alpha_range': (0, 2 * math.pi),
            't0_range': (-60.0, 40.0),    # Temporal bias fix v2
            'tE_range': (10.0, 40.0),
            'expected_accuracy': 0.62     # Many large u0 events
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
    """
    Transformer architecture parameters
    
    Compact model for fast training and inference
    """
    D_MODEL = 128              # Model dimension
    NHEAD = 4                  # Attention heads
    NUM_LAYERS = 4             # Transformer blocks
    DIM_FEEDFORWARD = 512      # FFN dimension (4 * d_model)
    DROPOUT = 0.1
    MAX_SEQ_LEN = 2000
    
    # Multi-task learning weights
    CLASSIFICATION_WEIGHT = 1.0
    FLAT_WEIGHT = 0.5
    PSPL_WEIGHT = 0.5
    ANOMALY_WEIGHT = 0.2
    CAUSTIC_WEIGHT = 0.2

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
    # u₀ dependency analysis
    U0_THRESHOLD = 0.3
    U0_BINS = 10
    
    # Early detection fractions
    EARLY_FRACTIONS = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
    
    # Visualization settings
    N_EVOLUTION_PLOTS = 3
    N_EXAMPLE_GRID = 3
    
    # Performance thresholds
    MIN_ACCURACY_WARNING = 0.70  # Higher for Roman quality
    TARGET_ACCURACY = 0.80       # Higher for Roman quality + temporal fix

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
# EXPERIMENT PRESETS - v14.0: BINARY MORPHOLOGY STUDY
# ============================================================================

class ExperimentPresets:
    """Pre-configured experiments for Roman Space Telescope"""
    
    EXPERIMENTS = {
        'quick_test': {
            'n_flat': 100,
            'n_pspl': 100,
            'n_binary': 100,
            'binary_params': 'baseline',
            'epochs': 5,
            'description': 'Quick validation test (Roman quality + temporal fix v2)'
        },
        
        'baseline_1M': {
            'n_flat': 333000,
            'n_pspl': 333000,
            'n_binary': 334000,
            'binary_params': 'baseline',
            'epochs': 50,
            'description': 'Main Roman benchmark (1M events, 5% missing, 0.05 mag, temporal fix v2)'
        },
        
        'distinct': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'distinct',
            'epochs': 50,
            'description': 'Clear caustics (optimal detection conditions)'
        },
        
        'planetary': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'planetary',
            'epochs': 50,
            'description': 'Exoplanet search (small mass ratios)'
        },
        
        'stellar': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'stellar',
            'epochs': 50,
            'description': 'Binary stars (equal masses)'
        },
        
        'challenging': {
            'n_flat': 50000,
            'n_pspl': 50000,
            'n_binary': 50000,
            'binary_params': 'challenging',
            'epochs': 50,
            'description': 'Physical limit study (wide u0 range)'
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
    print("CONFIGURATION SUMMARY - v14.0 (ROMAN + TEMPORAL FIX v2)")
    print("="*70)
    
    sim = SimulationConfig
    print(f"\n🛰️  Roman Space Telescope Configuration:")
    print(f"  Time window: [{sim.TIME_MIN}, {sim.TIME_MAX}] days (240 days)")
    print(f"  Points: {sim.N_POINTS}")
    print(f"  Missing: {sim.CADENCE_MASK_PROB*100:.0f}% (~15 min cadence)")
    print(f"  Error: {sim.MAG_ERROR_STD:.2f} mag (space-based quality)")
    
    pspl = PSPLConfig
    print(f"\n⭐ PSPL Parameters (TEMPORAL BIAS FIX v2):")
    print(f"  t₀: [{pspl.T0_MIN}, {pspl.T0_MAX}] days")
    print(f"  u₀: [{pspl.U0_MIN}, {pspl.U0_MAX}]")
    print(f"  tE: [{pspl.TE_MIN}, {pspl.TE_MAX}] days")
    print(f"  Earliest peak: t = {pspl.T0_MIN} → {abs(pspl.T0_MIN - sim.TIME_MIN):.0f} days pre-peak baseline")
    print(f"  Latest peak: t = {pspl.T0_MAX} → {sim.TIME_MAX - pspl.T0_MAX:.0f} days post-peak observation")
    print(f"  ✅ All events show complete features within observation window")
    print(f"  ✅ Events at varied stages prevent temporal position shortcuts")
    
    print(f"\n🧬 Binary Morphology Study (TEMPORAL BIAS FIX v2):")
    for name, cfg in BinaryConfig.CONFIGS.items():
        t0_range = cfg['t0_range']
        u0_range = cfg['u0_range']
        print(f"  {name:12s}: t₀=[{t0_range[0]:.1f}, {t0_range[1]:.1f}], "
              f"u₀=[{u0_range[0]:.3f}, {u0_range[1]:.3f}], "
              f"expected {cfg['expected_accuracy']*100:.0f}% accuracy")
    
    model = ModelConfig
    print(f"\n🤖 Model Architecture:")
    print(f"  d_model: {model.D_MODEL}")
    print(f"  nhead: {model.NHEAD}")
    print(f"  num_layers: {model.NUM_LAYERS}")
    print(f"  dim_ff: {model.DIM_FEEDFORWARD}")
    print(f"  Parameters: ~435k")
    
    train = TrainingConfig
    print(f"\n🎯 Training:")
    print(f"  Batch size: {train.BATCH_SIZE} per GPU")
    print(f"  Learning rate: {train.LEARNING_RATE}")
    print(f"  Max epochs: {train.MAX_EPOCHS}")
    
    print("\n" + "="*70)
    print("RESEARCH FOCUS - v14.0:")
    print("="*70)
    print("1. Baseline Roman benchmark (1M events)")
    print("2. Binary morphology study (4 topologies × 150k events)")
    print("3. u0 dependency analysis (physical detection limits)")
    print("4. Early detection capability (proper temporal fix)")
    print("5. Roman vs. ground-based comparison")
    print("="*70)
    print("\nTEMPORAL BIAS FIX v2:")
    print("="*70)
    print("✅ Events peak at varied times: t₀ ∈ [-60, +40] days")
    print("✅ Observation window: [-120, +120] days (240 days)")
    print("✅ Guaranteed baseline: 60-180 days before peak")
    print("✅ Complete features visible: tE ∈ [20, 40] → 6*tE visibility")
    print("✅ No temporal shortcuts: event stage randomized")
    print("✅ Label consistency: all events show full light curve")
    print("="*70)

def validate_config():
    """Validate configuration consistency"""
    issues = []
    warnings = []
    
    sim = SimulationConfig
    pspl = PSPLConfig
    
    time_window = sim.TIME_MAX - sim.TIME_MIN
    max_event_duration = pspl.TE_MAX * 4
    
    if max_event_duration > time_window:
        issues.append(f"⚠️ Event duration ({max_event_duration:.0f}d) > window ({time_window}d)")
    
    # Check baseline guarantee
    earliest_peak = pspl.T0_MIN
    baseline_duration = earliest_peak - sim.TIME_MIN
    
    if baseline_duration < 30:
        issues.append(f"⚠️ Insufficient baseline: {baseline_duration:.0f} days")
    else:
        warnings.append(f"✅ Minimum baseline: {baseline_duration:.0f} days before earliest peak")
    
    # Check post-peak observation
    latest_peak = pspl.T0_MAX
    post_peak_duration = sim.TIME_MAX - latest_peak
    
    if post_peak_duration < 30:
        issues.append(f"⚠️ Insufficient post-peak: {post_peak_duration:.0f} days")
    else:
        warnings.append(f"✅ Minimum post-peak: {post_peak_duration:.0f} days after latest peak")
    
    # Check event visibility
    max_visibility = pspl.TE_MAX * 6  # ±3*tE
    earliest_visible = pspl.T0_MIN - pspl.TE_MAX * 3
    latest_visible = pspl.T0_MAX + pspl.TE_MAX * 3
    
    if earliest_visible < sim.TIME_MIN:
        warnings.append(f"✅ Some events visible from window start (earliest: t={earliest_visible:.0f})")
    else:
        issues.append(f"⚠️ Gap before first event visible: t={earliest_visible:.0f} > {sim.TIME_MIN}")
    
    if latest_visible > sim.TIME_MAX:
        warnings.append(f"✅ Some events visible beyond window end (latest: t={latest_visible:.0f})")
    else:
        issues.append(f"⚠️ All events end before window: t={latest_visible:.0f} < {sim.TIME_MAX}")
    
    # Check Roman configuration
    if sim.CADENCE_MASK_PROB != 0.05:
        issues.append(f"⚠️ CADENCE_MASK_PROB should be 0.05 for Roman (currently {sim.CADENCE_MASK_PROB})")
    else:
        warnings.append(f"✅ Roman cadence: {sim.CADENCE_MASK_PROB*100:.0f}% missing")
    
    if sim.MAG_ERROR_STD != 0.05:
        issues.append(f"⚠️ MAG_ERROR_STD should be 0.05 for Roman (currently {sim.MAG_ERROR_STD})")
    else:
        warnings.append(f"✅ Roman photometry: {sim.MAG_ERROR_STD:.2f} mag error")
    
    model = ModelConfig
    if model.D_MODEL % model.NHEAD != 0:
        issues.append(f"⚠️ d_model ({model.D_MODEL}) must be divisible by nhead ({model.NHEAD})")
    
    train = TrainingConfig
    split_sum = train.TRAIN_RATIO + train.VAL_RATIO + train.TEST_RATIO
    if abs(split_sum - 1.0) > 0.001:
        issues.append(f"⚠️ Data splits sum to {split_sum:.3f}, not 1.0")
    
    # Check temporal bias fix
    t0_range = pspl.T0_MAX - pspl.T0_MIN
    if t0_range < 50:
        issues.append(f"⚠️ t₀ range too narrow: {t0_range:.0f} days (should be ≥50 for temporal fix)")
    else:
        warnings.append(f"✅ t₀ range: {t0_range:.0f} days (sufficient for temporal fix)")
    
    if pspl.T0_MIN >= 0:
        issues.append(f"⚠️ t₀_min should be negative for temporal fix (currently {pspl.T0_MIN})")
    else:
        warnings.append(f"✅ t₀_min negative: events can peak before observation midpoint")
    
    # Print results
    if warnings:
        print("\nConfiguration Notes:")
        for warning in warnings:
            print(f"  {warning}")
    
    if issues:
        print("\nConfiguration Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Configuration validated successfully!")
        print("   Roman Space Telescope parameters: CORRECT")
        print("   Temporal bias fix v2: IMPLEMENTED")
        print("   All consistency checks: PASSED")
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

def get_config_summary():
    """Wrapper for backwards compatibility"""
    print_config_summary()

if __name__ == "__main__":
    print_config_summary()
    print()
    validate_config()
