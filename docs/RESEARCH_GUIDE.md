# Research Guide: Real-Time Binary Microlensing Classification

## 🎯 Scientific Questions

This thesis investigates automated binary microlensing detection for next-generation surveys:

1. **Baseline Performance**: What classification accuracy can a modern neural network achieve on Roman Space Telescope quality photometry?

2. **Binary Morphology**: How does binary lens geometry (mass ratio q, projected separation s) affect detectability? Can we map parameter regions where binary classification is reliable?

3. **Physical Detection Limits**: Does the impact parameter u₀ impose a fundamental detectability threshold? At what u₀ do binary events become indistinguishable from single-lens events?

4. **Early Classification**: At what completeness fraction (10%, 25%, 50% of observations) can we achieve reliable classification?

---

## 🛰️ Observational Context

### Why Space-Based Photometry?

**Roman Space Telescope** (our baseline):
- Cadence: ~15 minute sampling
- Photometric precision: 0.05 mag
- Coverage: Continuous bulge monitoring
- **Key advantage**: Resolves short-duration caustic crossings

**Ground-Based Surveys** (OGLE, MOA):
- Cadence: ~1-3 days (weather-dependent)
- Photometric precision: 0.10 mag
- Coverage: Seasonal gaps
- **Challenge**: May miss brief caustic features

**Scientific Impact**: Roman's high cadence and precision should dramatically improve binary detection rates, especially for planetary-mass companions (q < 0.01).

---

## 🧪 Experimental Design

### Dataset Philosophy

We generate synthetic datasets using VBBinaryLensing (Bozza 2010), sampling physically realistic parameter ranges informed by OGLE/MOA discoveries. Each dataset tests a specific region of binary parameter space:

| Parameter | Symbol | Physical Meaning | Typical Range |
|-----------|--------|------------------|---------------|
| Mass ratio | q | m₂/m₁ (companion/primary) | 10⁻⁴ to 1.0 |
| Separation | s | d/θ_E (caustic geometry) | 0.1 to 3.0 |
| Impact parameter | u₀ | Minimum source-lens separation | 0.001 to 1.0 |
| Source size | ρ | Finite source effects | 0.001 to 0.1 |
| Timescale | t_E | Einstein crossing time | 10-40 days |
| Peak time | t₀ | Event maximum | Randomized |

### Four Core Experiments

#### 1. Baseline (1M events)
**Purpose**: Establish space-based performance benchmark

**Configuration**:
```
N = 1,000,000 events (333k Flat / 333k PSPL / 334k Binary)

Binary Parameters:
  s: 0.1 - 3.0      # Full caustic geometry range
  q: 10⁻⁴ - 1.0     # Planets to equal-mass binaries  
  u₀: 0.001 - 1.0   # Test full impact parameter space
  t_E: 10-40 days   # Realistic bulge timescales

Observational:
  Sampling: 5% random gaps (Roman-like)
  Photometry: 0.05 mag Gaussian noise
  Duration: 240 days centered on peak
```

**Scientific Goal**: Does a well-trained classifier achieve the 80%+ accuracy needed for large-scale surveys? What is the per-class (Flat/PSPL/Binary) recall?

#### 2. Distinct Topology (150k events)
**Purpose**: Optimal detection regime

**Configuration**:
```
Focus on clear caustics:
  s: 0.7 - 1.5      # Central caustic + planetary caustic both strong
  q: 0.01 - 0.5     # Moderate mass ratios
  u₀: 0.001 - 0.15  # Close encounters only

Expected: Binary recall 87-92%
```

**Scientific Goal**: In ideal conditions (source crosses both caustics), how accurately can we classify? This sets an upper performance bound.

#### 3. Planetary Topology (150k events)
**Purpose**: Exoplanet microlensing focus

**Configuration**:
```
Small mass ratios:
  s: 0.5 - 2.0      # Planet in lensing zone
  q: 10⁻⁴ - 0.01    # Planetary regime
  u₀: 0.001 - 0.3   # Wider u₀ range

Expected: Binary recall 82-87%
```

**Scientific Goal**: Can Roman cadence detect Earth-to-Jupiter mass planets via brief caustic perturbations? What q threshold enables reliable detection?

#### 4. Stellar Topology (150k events)
**Purpose**: Binary star systems

**Configuration**:
```
Equal-mass systems:
  s: 0.3 - 3.0      # Wide range of caustic structures
  q: 0.3 - 1.0      # Comparable masses
  u₀: 0.001 - 0.3   # Standard range

Expected: Binary recall 78-83%
```

**Scientific Goal**: Do symmetric caustic structures (from equal masses) pose classification challenges? How does wide-binary geometry affect detectability?

---

## 📊 Analysis Framework

### Primary Metrics

1. **Three-Class Accuracy** (Flat / PSPL / Binary)
   - Overall accuracy
   - Per-class precision and recall
   - Confusion matrix analysis

2. **u₀ Dependency** (Binary class only)
   - Bin accuracy by impact parameter
   - Identify detectability threshold
   - Compare to PSPL confusion rate

3. **Early Classification**
   - Accuracy vs. observation completeness (10%, 25%, 50%, 75%, 100%)
   - When can we trigger follow-up observations?

4. **Computational Performance**
   - Inference latency per event
   - Throughput (events/second)
   - Survey-scale feasibility

### Expected Physical Findings

**u₀ Threshold Hypothesis**: We predict a sharp detection threshold at u₀ ≈ 0.3, below which accuracy drops to ~50% (random between PSPL/Binary). 

**Physical Reasoning**: For u₀ > 0.3, the source trajectory doesn't cross caustics. The magnification profile becomes indistinguishable from single-lens (Paczynski) curves, regardless of (s, q). This is a fundamental physical limit, not an algorithm limitation.

**Test**: If we observe u₀ ≈ 0.3 threshold across *all* topologies (distinct, planetary, stellar), this confirms geometric origin rather than training artifacts.

---

## 🚀 Implementation: Transformer Architecture

### Model Overview

We use a transformer encoder (Vaswani+ 2017) adapted for time-series classification:

**Architecture**:
- Input: Flux time series [B, 1500, 1]
- Embedding: 1D → 128D via MLP
- Positional Encoding: Relative positions (observation count, gaps)
- Transformer: 4 layers, 4 heads, d_model=128
- Pooling: Average + Max over sequence
- Output Heads:
  - Classification: 3-way softmax (Flat/PSPL/Binary)
  - Caustic Detection: Binary auxiliary task (sigmoid)
  - Confidence: Self-assessed prediction quality (sigmoid)

**Key Modifications**:
1. **Causal Attention**: At timestep t, can only attend to observations ≤ t (no "cheating" by seeing the future)
2. **Relative Encoding**: No absolute time positions, only observation counts and gaps
3. **Robust Normalization**: Median/MAD instead of mean/std (handles outlier flux spikes)

**Why These Choices?**
- Causal attention → realistic real-time scenario
- Relative encoding → model learns magnification patterns, not peak timing
- Median/MAD → robust to caustic-crossing outliers

### Training Details

**Data Pipeline**:
```python
# Temporal randomization
# - Randomly shift t₀ by ±30 days via interpolation
# - Forces model to learn morphology, not absolute peak timing
# - Prevents overfitting to specific observation windows

# Normalization
# - Median-center and MAD-scale (robust to caustics)
# - Per-event normalization using valid (non-padded) points

# Batching
# - Balanced sampling across classes
# - Pad short sequences to 1500 points with sentinel value
```

**Optimization**:
- Optimizer: AdamW (lr=10⁻³, weight decay=10⁻⁴)
- Scheduler: Cosine annealing with 5-epoch warmup
- Loss: CrossEntropy (classification) + BCE (caustic detection, weight=0.8)
- Batch Size: 64 per GPU × 32 GPUs = 2048 effective
- Epochs: 50 (early stopping patience=15)
- Mixed Precision: FP16 training on AMD MI300

**Hardware**: 8 nodes × 4 AMD MI300X GPUs (32 GPUs total)  
**Time**: ~3-5 hours per 1M-event experiment

---

## 📈 Results Interpretation

### What Constitutes Success?

**Baseline (1M events)**:
- Overall accuracy > 80%
- PSPL recall > 75% (single-lens events)
- Binary recall > 77% (for u₀ < 0.3)
- Inference < 1 ms per event

**Topology Studies**:
- Distinct: Binary recall > 85% (optimal conditions)
- Planetary: Binary recall > 80% (exoplanet regime)
- Stellar: Binary recall > 75% (equal-mass systems)

**u₀ Analysis**:
- Clear threshold at u₀ ≈ 0.3 across all topologies
- Consistent with geometric caustic-crossing requirement
- Validates physical interpretation over ML artifact

**Early Classification**:
- 50% completeness: >70% accuracy (trigger follow-up)
- 25% completeness: >60% accuracy (early alert)
- 10% completeness: ~40% accuracy (too early, random)

### Astrophysical Implications

If we achieve these targets:

1. **Survey Automation**: Roman can automatically classify 10,000+ events/sec during peak bulge season

2. **Exoplanet Detection**: Reliable planetary-mass companion detection down to q ~ 10⁻⁴ with Roman cadence

3. **Follow-up Prioritization**: Early classification at 50% completeness enables targeted high-resolution follow-up (Keck, VLT) 2-3 weeks before peak

4. **Population Statistics**: Large-scale binary/planet occurrence rates from automated classification of 10⁶+ events over mission lifetime

---

## 🔬 Workflow: Step-by-Step

### Phase 1: Quick Validation (30 minutes)

```bash
cd code

# Generate 300 test events
python simulate.py --preset quick_test

# Train 5 epochs (single GPU)
python train.py \
    --data ../data/raw/quick_test.npz \
    --experiment_name quick_test \
    --epochs 5 \
    --batch_size 32

# Evaluate
python evaluate.py \
    --experiment_name quick_test \
    --data ../data/raw/quick_test.npz

# Expected: ~60-70% accuracy (tiny dataset, just checking it runs)
```

### Phase 2: Baseline Experiment (5 hours, 32 GPUs)

```bash
cd code

# 1. Generate 1M events
python simulate.py --preset baseline_1M
# Output: data/raw/baseline_1M.npz (~4 GB)

# 2. Allocate compute
salloc --partition=gpu_a100_short --nodes=8 --gres=gpu:4 --exclusive --time=05:00:00

# 3. Set up distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_ASYNC_ERROR_HANDLING=1

# 4. Train with DDP
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64

# 5. Comprehensive evaluation
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection \
    --n_evolution_per_type 10 \
    --n_samples 50000

# 6. Check results
ls results/baseline_1M_*/evaluation/
# Should see: ROC curves, confusion matrix, calibration, u0_dependency.png, etc.
```

### Phase 3: Topology Studies (3 hours each)

Run the same workflow for each topology:

```bash
# Distinct topology (optimal conditions)
python simulate.py --preset distinct \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000

srun torchrun ... train.py \
    --data ../data/raw/distinct.npz \
    --experiment_name distinct \
    ...

python evaluate.py --experiment_name distinct ...

# Repeat for 'planetary' and 'stellar' presets
```

### Phase 4: Analysis & Figures

```bash
cd results

# Extract summary table
python -c "
import json
from pathlib import Path

experiments = ['baseline_1M', 'distinct', 'planetary', 'stellar']

print(f'{'Experiment':<15} {'Overall%':<10} {'PSPL%':<8} {'Binary%':<8}')
print('-' * 50)

for exp in experiments:
    runs = sorted(Path('.').glob(f'{exp}_*'))
    if runs:
        summary = runs[-1] / 'evaluation/evaluation_summary.json'
        if summary.exists():
            data = json.load(open(summary))
            m = data['metrics']
            print(f'{exp:<15} {m['accuracy']*100:>8.1f}  {m['pspl_recall']*100:>6.1f}  {m['binary_recall']*100:>6.1f}')
"

# Compare u₀ dependency across topologies
# (See full plotting script in original guide)
```

---

## 🔧 Troubleshooting

### Training Issues

**Out of Memory**:
```bash
--batch_size 32  # Reduce from 64
--gradient_checkpointing  # Enable memory savings
```

**NaN Losses** (rare):
- Already handled in code (skips bad batches)
- If >10% batches skipped → check data quality

**DDP Hangs**:
```bash
# Verify master node
echo $MASTER_ADDR

# Test connectivity
srun --nodes=2 hostname

# Increase timeout
export NCCL_TIMEOUT=3600
```

### Evaluation Issues

**Missing u₀ Analysis**:
```bash
# Verify parameters saved
python -c "import numpy as np; d=np.load('data.npz'); print('params_binary_json' in d.files)"

# Re-generate data with --save_params flag
```

---
