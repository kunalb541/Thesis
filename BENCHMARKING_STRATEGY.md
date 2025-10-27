# Thesis Benchmarking Strategy

## Research Question

**"What are the optimal conditions for classifying microlensing binary events, and how do different survey strategies and binary topologies affect early detection capability?"**

## Key Variables to Benchmark

### 1. Binary Topology (Distinguishability)

Different binary configurations have different caustic features:

#### **Wide Binaries** (Easy to Detect)
- **s**: 1.0-2.5 (wide separation)
- **Features**: Strong, well-separated caustic crossings
- **Expected Performance**: >98% accuracy

#### **Intermediate Binaries** (Moderate)
- **s**: 0.5-1.5 (intermediate separation)
- **Features**: Moderate caustic complexity
- **Expected Performance**: 95-97% accuracy

#### **Close Binaries** (Hard)
- **s**: 0.1-0.8 (close, near-resonant)
- **Features**: Complex, overlapping caustics
- **Expected Performance**: 90-95% accuracy

#### **PSPL-like Binaries** (Very Hard)
- **s**: 0.8-1.2, **q**: 0.8-1.0 (near-equal masses at specific separations)
- **Features**: Subtle deviations from PSPL
- **Expected Performance**: 85-90% accuracy

### 2. Observing Cadence (Survey Strategy)

Different surveys have different sampling:

| Survey | Missing Data | Error | Description |
|--------|-------------|-------|-------------|
| **Roman** | 3% | 0.03 mag | Space telescope, high cadence |
| **LSST Dense** | 5% | 0.05 mag | Deep drilling fields |
| **LSST Regular** | 15% | 0.08 mag | Main survey |
| **LSST Sparse** | 30% | 0.10 mag | Wide-fast-deep |
| **Ground Sparse** | 40% | 0.15 mag | Typical ground-based |

### 3. Photometric Quality

- **Excellent**: 0.03 mag (space-based)
- **Good**: 0.08 mag (good ground-based)
- **Moderate**: 0.12 mag
- **Poor**: 0.20 mag (faint/poor conditions)

### 4. Early Detection (Real-Time Classification)

Test classification at different stages:
- **Very Early**: 300 points (~20% of event)
- **Early**: 500 points (~33% of event)
- **Mid**: 750 points (~50% of event)
- **Late**: 1000 points (~67% of event)
- **Full**: 1500 points (complete light curve)

## Experimental Design

### Phase 1: Baseline (Use Your Existing 1M Dataset)

**Purpose**: Establish performance on mixed population

```bash
# Your current dataset: mixed_binary regime, 20% missing, 0.1 mag error
python train.py --data data/raw/events_1M.npz --output models/baseline.keras
```

**Expected**: ~95% accuracy, ROC AUC > 0.98

### Phase 2: Binary Topology Benchmarking

**Question**: How does binary type affect classification?

Generate 5 datasets (100K each) with different binary regimes:

```bash
# Wide binaries (easiest)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime wide_binary --output data/raw/events_wide.npz

# Intermediate
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime intermediate_binary --output data/raw/events_intermediate.npz

# Close (harder)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime close_binary --output data/raw/events_close.npz

# PSPL-like (hardest)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime pspl_like_binary --output data/raw/events_pspl_like.npz

# Mixed (realistic)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --output data/raw/events_mixed.npz
```

**Expected Result**: Performance decreases from wide → close → PSPL-like

### Phase 3: Cadence Benchmarking

**Question**: How does observing strategy affect classification?

Test 5 cadence scenarios (using mixed_binary regime):

```bash
# Roman (best)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --cadence 0.03 --mag_error 0.03 \
    --output data/raw/events_roman.npz

# LSST Dense
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --cadence 0.05 --mag_error 0.05 \
    --output data/raw/events_lsst_dense.npz

# LSST Regular
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --cadence 0.15 --mag_error 0.08 \
    --output data/raw/events_lsst_regular.npz

# LSST Sparse
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --cadence 0.30 --mag_error 0.10 \
    --output data/raw/events_lsst_sparse.npz

# Ground Sparse (worst)
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime mixed_binary --cadence 0.40 --mag_error 0.15 \
    --output data/raw/events_ground_sparse.npz
```

**Expected Result**: Performance decreases with sparser cadence

### Phase 4: Combined Scenarios

**Question**: What are best-case and worst-case scenarios?

```bash
# Best case: Wide binaries + Roman cadence
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime wide_binary --cadence 0.03 --mag_error 0.03 \
    --output data/raw/events_best_case.npz

# Worst case: PSPL-like binaries + sparse ground-based
python simulate_flexible.py --n_pspl 50000 --n_binary 50000 \
    --binary_regime pspl_like_binary --cadence 0.40 --mag_error 0.15 \
    --output data/raw/events_worst_case.npz
```

### Phase 5: Early Detection

**Question**: How early can we reliably classify?

Use baseline model, but test on truncated light curves:

```python
# In evaluate.py, already implemented!
# Tests classification at 100, 250, 500, 750, 1000, 1250, 1500 points
```

## Automated Execution

### Run All Experiments (Recommended)

```bash
# List all planned experiments
python run_master_experiments.py --list

# Run specific experiments
python run_master_experiments.py --experiments baseline_wide baseline_close cadence_roman

# Run ALL experiments (will take days!)
python run_master_experiments.py --n_pspl 50000 --n_binary 50000 --epochs 30

# Just compare existing results
python run_master_experiments.py --compare_only
```

### Manual Control (For Testing)

```bash
# 1. Generate one dataset
python simulate_flexible.py --n_pspl 10000 --n_binary 10000 \
    --binary_regime wide_binary --output data/raw/test.npz

# 2. Train
python train.py --data data/raw/test.npz --output models/test.keras --epochs 10

# 3. Evaluate
python evaluate.py --model models/test.keras --data data/raw/test.npz \
    --output_dir results/test
```

## Expected Thesis Results

### Table 1: Binary Topology Results

| Binary Type | ROC AUC | Accuracy | Notes |
|-------------|---------|----------|-------|
| Wide | 0.99+ | >98% | Easy to detect |
| Intermediate | 0.97-0.98 | 95-97% | Moderate |
| Close | 0.93-0.96 | 90-95% | Harder |
| PSPL-like | 0.88-0.92 | 85-90% | Very challenging |
| Mixed | 0.95-0.97 | ~95% | Realistic |

### Table 2: Cadence Results

| Survey | ROC AUC | Accuracy | Notes |
|--------|---------|----------|-------|
| Roman | 0.98+ | >97% | Optimal |
| LSST Dense | 0.96-0.98 | 95-97% | Very good |
| LSST Regular | 0.94-0.96 | 93-95% | Good |
| LSST Sparse | 0.90-0.93 | 88-92% | Acceptable |
| Ground Sparse | 0.85-0.89 | 82-87% | Poor |

### Figure 1: Performance vs Cadence
Plot showing ROC AUC decreasing with sparser sampling

### Figure 2: Performance vs Binary Type
Bar chart comparing different binary regimes

### Figure 3: Early Detection Curve
Line plot showing accuracy vs number of observations

### Figure 4: Confusion Matrices
Side-by-side for best/worst case scenarios

## Key Findings for Discussion

1. **Binary topology matters**: Wide binaries are much easier to detect than close/PSPL-like
2. **Cadence is critical**: Dense sampling (LSST/Roman) essential for high accuracy
3. **Early detection is feasible**: Can achieve ~85% accuracy with only 500 points (~1/3 of event)
4. **Survey recommendations**: 
   - LSST deep drilling fields ideal for binary detection
   - Roman's high cadence enables earliest alerts
   - Sparse surveys miss subtle binary features

## Timeline

- **Week 1**: Run Phase 1-2 (binary topology)
- **Week 2**: Run Phase 3 (cadence)
- **Week 3**: Run Phase 4-5 (combined + early detection)
- **Week 4**: Generate figures, write results
- **Week 5**: Discussion and conclusions

## Computational Requirements

Per experiment (100K events, 30 epochs):
- **Time**: 2-4 hours on 4× MI300 GPUs
- **Storage**: ~500 MB per dataset
- **Total for all experiments**: ~15 experiments × 4 hours = 60 hours (2.5 days)

**You have time!** Can run multiple experiments in parallel or sequentially.

## Output for Thesis

You'll have:
- 15+ trained models
- Comprehensive evaluation metrics
- Publication-ready figures
- Clear conclusions about optimal survey strategies
- Quantitative benchmarks for LSST/Roman

This is a **complete, publishable thesis** with systematic benchmarking! 🎓
