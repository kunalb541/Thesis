# Systematic Benchmarking Methodology

## Thesis Goal: What Can We Achieve?

Your thesis aims to **benchmark the best performance** we can get for binary vs PSPL classification by systematically varying:

1. **Observing cadence** (how often we observe)
2. **Photometric precision** (measurement errors)
3. **Binary event difficulty** (how distinguishable they are)
4. **Early detection** (how early we can classify)

## Why This Matters

**For LSST and Roman**: Different surveys have different:
- Cadences (LSST: frequent, ground-based: sparse)
- Photometric precision (space vs ground)
- Science goals (early alerts vs archival)

Your results will show: **"With X cadence and Y photometry, we can achieve Z% accuracy"**

## What Makes Binary Events Distinguishable?

### Key Physical Effects

**Binary lens creates caustics** (curves where magnification → ∞)

When the source crosses caustics → **sharp spikes and wiggles** in light curve

**Most distinguishable when:**
- **Small u0 (0.001-0.1)**: Source crosses caustics → sharp features
- **s ~ 0.8-1.2**: Wide binary → prominent caustic structure
- **Small q (0.1-0.5)**: Asymmetric → non-PSPL shape
- **Small rho (0.0001-0.01)**: Finite source smoothing minimal → sharp peaks

**Least distinguishable when:**
- **Large u0 (0.3-0.5)**: Source misses caustics → looks PSPL-like
- **Extreme s (0.1-0.3 or 2+)**: Very close/wide → weak caustic signatures
- **Large q (~1)**: Symmetric → more PSPL-like
- **Large rho (0.05-0.1)**: Features smoothed → looks smoother

## Benchmark Experiments

### 1. Baseline (Establish Performance)
```python
# Standard mix of easy and hard events
n_events: 1M (500K PSPL + 500K Binary)
cadence: 20% missing (typical ground-based)
error: 0.1 mag (decent photometry)
binary_params: 'standard' (mixed difficulty)
```

**Expected**: 95-98% accuracy, ROC AUC ~0.98

**Purpose**: Reference point for all comparisons

### 2. Cadence Experiments (How Often Must We Observe?)

**Dense (5% missing)** - LSST-like
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --cadence 0.05 --output data/raw/events_cadence_05.npz
```

**Sparse (30% missing)** - Poor coverage
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --cadence 0.30 --output data/raw/events_cadence_30.npz
```

**Expected**: Dense better, but how much? Diminishing returns?

**Thesis insight**: "We need ≥70% coverage to maintain >90% accuracy"

### 3. Photometric Error Experiments (Noise Sensitivity)

**Low error (0.05 mag)** - Space-based quality
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --error 0.05 --output data/raw/events_error_low.npz
```

**High error (0.20 mag)** - Poor photometry
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --error 0.20 --output data/raw/events_error_high.npz
```

**Expected**: Better photometry → better performance, but saturates?

**Thesis insight**: "Roman (0.05 mag) would improve accuracy by X% over ground (0.15 mag)"

### 4. Binary Difficulty Experiments (Easy vs Hard Detection)

**Easy binaries** - Clear caustic crossings
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --binary_difficulty easy --output data/raw/events_binary_easy.npz
```
- Small u0 → crosses caustics
- s~1 → prominent caustics
- Small rho → sharp features

**Hard binaries** - PSPL-like
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --binary_difficulty hard --output data/raw/events_binary_hard.npz
```
- Large u0 → misses caustics
- Extreme s → weak signatures
- Large rho → smoothed

**Expected**: Easy >99%, Hard ~85-90%

**Thesis insight**: "We can reliably detect caustic-crossing events, but ~15% of binaries are PSPL-like and hard to classify"

### 5. Early Detection (Real-Time Classification)

**500 points (33% of event)**
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output data/raw/events_early_500.npz
# Then truncate in evaluate.py
```

**Purpose**: Can we alert early for follow-up observations?

**Expected**: Accuracy degrades with fewer points, but caustic crossings detectable early

**Thesis insight**: "We can classify with >85% accuracy using only 1/3 of the light curve"

## Running the Full Benchmark Suite

### Option 1: Interactive (for testing)
```bash
# Get GPU node
salloc --partition=gpu_mi300 --gres=gpu:4 --time=72:00:00

conda activate microlens
cd ~/thesis-microlens/code

# Generate datasets (takes ~1-2 hours per dataset)
python simulate.py --n_pspl 100000 --n_binary 100000 --cadence 0.05 \
    --output ../data/raw/events_cadence_05.npz

python simulate.py --n_pspl 100000 --n_binary 100000 --cadence 0.30 \
    --output ../data/raw/events_cadence_30.npz

# etc...

# Train each model
python train.py --data ../data/raw/events_cadence_05.npz \
    --output ../models/model_cadence_05.keras --experiment_name cadence_05

# Evaluate
python evaluate.py --model ../models/model_cadence_05.keras \
    --data ../data/raw/events_cadence_05.npz \
    --output_dir ../results/cadence_05
```

### Option 2: Automated Batch Jobs
```bash
# Create SLURM array job (recommended)
sbatch slurm/run_all_experiments.sh
```

## Analysis and Visualization

After running experiments:

```python
cd ~/thesis-microlens/code

# Compare all results
python -c "
from utils import load_experiment_results, compare_cadence_experiments
import matplotlib.pyplot as plt

# Load results
results = load_experiment_results('../results')

# Compare cadences
compare_cadence_experiments('../results', save_path='../results/cadence_comparison.png')

# Create summary table
for r in results:
    print(f'{r[\"experiment\"]}: {r[\"metrics\"][\"roc_auc\"]:.3f}')
"
```

## Expected Thesis Figures

### Figure 1: Baseline Results
- Confusion matrix (PSPL vs Binary)
- ROC curve (AUC ~0.98)
- Sample light curves (correctly classified)

### Figure 2: Cadence Comparison
- Accuracy vs % coverage
- Shows: Dense cadence helps, but diminishing returns

### Figure 3: Photometric Error Sensitivity
- Accuracy vs photometric error
- Shows: Performance degrades with noise, but robust to ~0.15 mag

### Figure 4: Binary Difficulty
- Easy vs Hard binary classification
- Shows: Caustic-crossing events easy, high-u0 events hard

### Figure 5: Early Detection
- Accuracy vs % of event observed
- Shows: Can detect with 30-50% of light curve

### Figure 6: Combined Analysis
- 2D heatmap: cadence vs error
- Shows: Optimal survey strategies

## Thesis Discussion Points

1. **Performance bounds**: "Best case (dense+low error+easy): 99.5%, Worst case (sparse+high error+hard): 82%"

2. **Survey design**: "LSST cadence sufficient for >95% accuracy on standard binaries"

3. **Early classification**: "Real-time alerts feasible with TimeDistributed architecture"

4. **Physical interpretation**: "Caustic crossings detectable, but 15-20% of binaries are intrinsically PSPL-like"

5. **Roman advantage**: "Space-based photometry improves faint-event classification by X%"

## Recommended Experiment Order

**Week 1: Baseline**
- Run baseline (1M events)
- Establish reference performance
- Debug any issues

**Week 2: Cadence sweep**
- Generate 5%, 10%, 20%, 30%, 40% datasets
- Train 5 models
- Create cadence comparison plot

**Week 3: Error sensitivity**
- Generate 0.05, 0.1, 0.15, 0.2 mag datasets
- Train 4 models
- Create error sensitivity plot

**Week 4: Binary difficulty**
- Generate easy, standard, hard datasets
- Analyze which parameters affect detection
- Understand physical limits

**Week 5: Analysis + writing**
- Create all figures
- Write results section
- Relate to LSST/Roman

## Key Metrics to Report

For each experiment:
- **Accuracy**: Overall classification rate
- **ROC AUC**: Discrimination ability
- **Precision/Recall**: Per-class performance
- **Early detection curve**: Accuracy vs observation time
- **Confusion matrix**: Error modes

## Computing Resources

**Per experiment:**
- Data generation: 1-2 GPU-hours
- Training: 2-4 GPU-hours (200K events, 4 GPUs)
- Evaluation: <30 minutes

**Total for full benchmark:**
- ~10 datasets × 3 hours = 30 GPU-hours
- With 4 GPUs, can finish in 1-2 weeks

## Success Criteria

Your thesis successfully benchmarks if you can answer:

1. ✅ What accuracy can we achieve? (baseline)
2. ✅ How does cadence affect performance? (cadence experiments)
3. ✅ How does noise affect performance? (error experiments)
4. ✅ What events are hardest to classify? (binary difficulty)
5. ✅ Can we do early classification? (early detection)
6. ✅ What are optimal survey strategies? (combined analysis)

**You'll have quantitative answers to all of these!** 🎯
