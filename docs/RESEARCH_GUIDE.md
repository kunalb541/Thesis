# Research Guide 

**Version 14.0 - Roman Space Telescope**  
**Simplified Research Scope** | **5 Experiments Total** | **Multi-Node DDP Ready**

Complete experimental workflow for thesis research focusing on physical detection limits.

---

## 🎯 Research Questions (v14.0)

This work addresses:

1. **Baseline Performance**: What classification accuracy is achievable with Roman Space Telescope quality data?

2. **Binary Morphology**: How does binary lens topology (mass ratio, separation) affect detectability?

3. **Physical Detection Limits**: What are the fundamental u₀-dependent detection limits for binary systems?

4. **Early Detection**: How early can we classify events with Roman's high-cadence observations?

---

## 🛰️ Why Roman Space Telescope?

### Advantages Over LSST (Ground-Based)

**Roman (Space-Based)**:
- Cadence: ~15 min sampling (5% missing)
- Photometry: 0.05 mag error
- Coverage: Continuous, no weather
- Timeline: Feasible for thesis (simpler parameters)

**LSST (Ground-Based)** - v13.1:
- Cadence: ~3 days (85% missing when accounting for weather/moon)
- Photometry: 0.10 mag error
- Coverage: Weather-dependent
- Timeline: Too many experiments (11 total)

### Research Focus Simplification

**v13.1 (LSST)**:
- 4 cadence experiments (5%, 20%, 30%, 40% missing)
- 3 error experiments (0.05, 0.10, 0.20 mag)
- 4 topology experiments
- **Total: 11 experiments = 8-10 weeks**

**v14.0 (Roman)**:
- 1 baseline experiment (1M events, Roman quality)
- 4 topology experiments (150k each)
- **Total: 5 experiments = 6-8 weeks** ✅

---

## 🧪 Experimental Design (v14.0)

### 1. Baseline Experiment (1M Events)

**Purpose**: Establish Roman Space Telescope baseline performance

**Configuration**:
```python
N_events = 1,000,000 (balanced 3-class)
N_flat = 333,000
N_pspl = 333,000
N_binary = 334,000

Binary Parameters: 'baseline'
  s: 0.1 - 2.5        # Wide range
  q: 0.001 - 1.0      # Planetary to stellar
  u0: 0.001 - 0.3     # Physically detectable range
  rho: 0.001 - 0.05   # Typical sources
  tE: 20 - 40 days    # Realistic timescales
  t0: 0 - 80 days     # Temporal bias fix

Observational - Roman Space Telescope:
  Cadence: 5% missing (~15 min sampling)
  Photometry: 0.05 mag error (space-based)
  Points: 1500 per light curve
  Time window: [-120, +120] days
```

**Commands**:
```bash
cd code

# 1. Generate (1M events, Roman quality)
python simulate.py \
    --n_flat 333000 --n_pspl 333000 --n_binary 334000 \
    --output ../data/raw/roman_baseline_1M.npz \
    --binary_params baseline \
    --save_params \
    --num_workers 200 \
    --seed 42

# 2. Train (multi-node if available)
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/roman_baseline_1M.npz \
        --experiment_name roman_baseline_1M \
        --epochs 50 \
        --batch_size 64

# 3. Evaluate (includes u0 analysis automatically)
python evaluate.py \
    --experiment_name roman_baseline_1M \
    --data ../data/raw/roman_baseline_1M.npz \
    --early_detection \
    --n_evolution_per_type 10 \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Results (Roman Quality)**:
```
Overall Accuracy: 78-83%

Per-Class Recall:
  Flat:   90-95%  (easy - constant flux)
  PSPL:   73-78%  (good - better with Roman quality)
  Binary: 75-80%  (good - u0 dependent)

Improvement over LSST (v13.1):
  +3-5% overall accuracy
  +3-5% binary recall (clearer caustic crossings)
```

---

### 2. Binary Morphology Study (150k each)

**Purpose**: Characterize u₀ dependency and physical detection limits across different binary configurations

| Experiment | Description | Key Parameters | Expected Accuracy |
|------------|-------------|----------------|-------------------|
| **Distinct** | Clear caustics | s=0.7-1.5, q=0.01-0.5, u₀<0.15 | 85-90% |
| **Planetary** | Exoplanet focus | q=0.0001-0.01, small features | 80-85% |
| **Stellar** | Binary stars | q=0.3-1.0, complex caustics | 77-82% |
| **Challenging** | Physical limit | Wide u₀ (0.01-1.0) | 70-75% |

#### Experiment 2a: Distinct Topology

**Purpose**: Optimal detection conditions (clear caustics)

```bash
cd code

# Generate
python simulate.py \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/roman_distinct.npz \
    --binary_params distinct \
    --save_params \
    --num_workers 200 \
    --seed 42

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/roman_distinct.npz \
        --experiment_name roman_distinct \
        --epochs 50 \
        --batch_size 64

# Evaluate
python evaluate.py \
    --experiment_name roman_distinct \
    --data ../data/raw/roman_distinct.npz \
    --early_detection \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 85-92% (clear caustics)
- u₀ dependency: Steep drop at u₀ > 0.15
- Early detection: High confidence at 40% completeness

#### Experiment 2b: Planetary Topology

**Purpose**: Exoplanet detection (small mass ratios)

```bash
cd code

# Generate
python simulate.py \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/roman_planetary.npz \
    --binary_params planetary \
    --save_params \
    --num_workers 200 \
    --seed 42

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/roman_planetary.npz \
        --experiment_name roman_planetary \
        --epochs 50 \
        --batch_size 64

# Evaluate
python evaluate.py \
    --experiment_name roman_planetary \
    --data ../data/raw/roman_planetary.npz \
    --early_detection \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 78-85% (small features)
- Planet-host separation critical
- Roman's quality helps with small q

#### Experiment 2c: Stellar Topology

**Purpose**: Binary star systems (equal masses)

```bash
cd code

# Generate
python simulate.py \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/roman_stellar.npz \
    --binary_params stellar \
    --save_params \
    --num_workers 200 \
    --seed 42

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/roman_stellar.npz \
        --experiment_name roman_stellar \
        --epochs 50 \
        --batch_size 64

# Evaluate
python evaluate.py \
    --experiment_name roman_stellar \
    --data ../data/raw/roman_stellar.npz \
    --early_detection \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 75-82% (complex caustics)
- Symmetric caustics challenging
- u₀ dependency moderate

#### Experiment 2d: Challenging Topology

**Purpose**: Physical detection limit study (wide u₀)

```bash
cd code

# Generate
python simulate.py \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/roman_challenging.npz \
    --binary_params challenging \
    --save_params \
    --num_workers 200 \
    --seed 42

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/roman_challenging.npz \
        --experiment_name roman_challenging \
        --epochs 50 \
        --batch_size 64

# Evaluate
python evaluate.py \
    --experiment_name roman_challenging \
    --data ../data/raw/roman_challenging.npz \
    --early_detection \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 60-70% (many large u₀)
- Clear u₀ threshold at ~0.3
- Proves physical, not algorithmic limit

---

## 📊 Analysis Workflow (v14.0)

### 1. Extract Results

```bash
# Generate summary table
python -c "
import json
from pathlib import Path

experiments = [
    'roman_baseline_1M',
    'roman_distinct',
    'roman_planetary',
    'roman_stellar',
    'roman_challenging'
]

print(f'{'Experiment':<25} {'Overall':<10} {'Flat%':<8} {'PSPL%':<8} {'Binary%':<8}')
print('-' * 70)

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        eval_file = runs[-1] / 'evaluation' / 'evaluation_summary.json'
        if eval_file.exists():
            data = json.load(open(eval_file))
            
            overall = data.get('metrics', {}).get('accuracy', 0) * 100
            flat_rec = data.get('metrics', {}).get('flat_recall', 0) * 100
            pspl_rec = data.get('metrics', {}).get('pspl_recall', 0) * 100
            binary_rec = data.get('metrics', {}).get('binary_recall', 0) * 100
            
            print(f'{exp:<25} {overall:>8.1f}%  {flat_rec:>6.1f}  {pspl_rec:>6.1f}  {binary_rec:>6.1f}')
" > results_roman_v14.txt

cat results_roman_v14.txt
```

### 2. u₀ Dependency Comparison

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Compare u0 dependency across all topologies
topologies = ['roman_baseline_1M', 'roman_distinct', 'roman_planetary', 'roman_stellar', 'roman_challenging']
colors = ['purple', 'green', 'blue', 'orange', 'red']

fig, ax = plt.subplots(figsize=(14, 8))

for topo, color in zip(topologies, colors):
    exp_dirs = sorted(Path('results').glob(f'{topo}_*'))
    if exp_dirs:
        u0_report = exp_dirs[-1] / 'evaluation' / 'u0_report.json'
        
        if u0_report.exists():
            with open(u0_report) as f:
                data = json.load(f)
            
            u0_centers = data['u0_centers']
            accuracies = [a*100 if a else None for a in data['accuracies']]
            
            # Filter valid points
            valid = [(u, a) for u, a in zip(u0_centers, accuracies) if a is not None]
            if valid:
                u_vals, a_vals = zip(*valid)
                
                ax.plot(u_vals, a_vals, 'o-', linewidth=3, markersize=10, 
                       color=color, label=topo.replace('roman_', '').replace('_', ' ').title(),
                       alpha=0.8)

# Add threshold line
ax.axvline(x=0.3, color='red', linestyle='--', linewidth=3, 
          label='Physical Limit (u₀=0.3)', alpha=0.7)
ax.axhline(y=50, color='gray', linestyle=':', linewidth=2, 
          alpha=0.5, label='Random (50%)')

ax.set_xlabel('Impact Parameter u₀', fontsize=14, fontweight='bold')
ax.set_ylabel('Binary Classification Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Roman Space Telescope: Binary Detection vs. Impact Parameter', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim([30, 100])
ax.set_xlim([0, max([max([u for u, a in valid]) for valid in [[(u, a) for u, a in zip(u0_centers, accuracies) if a is not None] for u0_centers, accuracies in [...]]])])

plt.tight_layout()
plt.savefig('figures/roman_u0_comparison_all.png', dpi=300, bbox_inches='tight')
print("Saved: figures/roman_u0_comparison_all.png")
```

### 3. Performance Summary

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

experiments = ['roman_baseline_1M', 'roman_distinct', 'roman_planetary', 'roman_stellar', 'roman_challenging']
names = ['Baseline', 'Distinct', 'Planetary', 'Stellar', 'Challenging']
overall_accs = []
binary_recalls = []

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        eval_file = runs[-1] / 'evaluation' / 'evaluation_summary.json'
        with open(eval_file) as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        overall_accs.append(metrics.get('accuracy', 0) * 100)
        binary_recalls.append(metrics.get('binary_recall', 0) * 100)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Overall accuracy
x = np.arange(len(names))
bars1 = ax1.bar(x, overall_accs, color='purple', alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(y=75, color='red', linestyle='--', linewidth=2, label='75% Target')
ax1.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Roman Space Telescope: Overall Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim([60, 95])

for bar, acc in zip(bars1, overall_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 1, f'{acc:.1f}%',
            ha='center', fontsize=10, fontweight='bold')

# Binary recall
bars2 = ax2.bar(x, binary_recalls, color='darkblue', alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(y=70, color='red', linestyle='--', linewidth=2, label='70% Target')
ax2.set_ylabel('Binary Recall (%)', fontsize=13, fontweight='bold')
ax2.set_title('Binary Classification Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, ha='right')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim([50, 95])

for bar, rec in zip(bars2, binary_recalls):
    ax2.text(bar.get_x() + bar.get_width()/2, rec + 1, f'{rec:.1f}%',
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/roman_performance_summary.png', dpi=300, bbox_inches='tight')
print("Saved: figures/roman_performance_summary.png")
```

---

## 📖 Thesis Structure (v14.0)

### Chapter 4: Results

#### 4.1 Baseline Performance (Roman Space Telescope)

**Report**:
- Overall accuracy: 78-83%
- Roman advantage: +3-5% over ground-based
- Per-class performance:
  - Flat: 90-95%
  - PSPL: 73-78%
  - Binary: 75-80%

**Key Message**: "Roman Space Telescope's high cadence and photometric quality enable 80%+ three-class accuracy, with significant improvement in binary detection compared to ground-based surveys."

**Figures**:
- ROC curves (3-class)
- Confusion matrix
- Calibration curves
- Example light curves
- Early detection curves

#### 4.2 Binary Morphology Study

**Distinct Topology** (Optimal Conditions):
- Binary recall: 85-92%
- Key finding: Clear caustics enable early detection
- u₀ threshold: Sharp drop at u₀ > 0.15

**Planetary Topology** (Exoplanet Search):
- Binary recall: 78-85%
- Small mass ratios challenging but detectable
- Roman's quality critical for small q

**Stellar Topology** (Binary Stars):
- Binary recall: 75-82%
- Complex caustics
- Symmetric structures harder to classify

**Challenging Topology** (Physical Limits):
- Binary recall: 60-70%
- Wide u₀ range demonstrates physical limit
- Clear threshold at u₀ ≈ 0.3

#### 4.3 Physical Detection Limits

**u₀ Dependency Analysis**:
- All topologies show sharp degradation at u₀ > 0.3
- 15-20% of binaries have u₀ > 0.3 (fundamentally PSPL-like)
- This is physical, not algorithmic limitation

**Key Conclusion**: "The u₀ = 0.3 threshold represents a fundamental physical detection limit for binary microlensing, not a failure of the classification algorithm."

#### 4.4 Early Detection (Roman Advantages)

**High-Cadence Benefits**:
- 50% completeness: 75-80% accuracy (reliable trigger)
- 25% completeness: 55-65% accuracy (early warning)
- Roman's 15-min cadence enables 2-3 week earlier classification

**Inference Performance**:
- Latency: <1 ms per event
- Throughput: 10,000+ events/second
- Survey-scale ready

---

## 🎓 Expected Contributions (v14.0)

1. **First Roman Benchmark**: 
   - 80%+ accuracy on space-based quality data
   - Baseline for future Roman microlensing surveys

2. **Binary Morphology Characterization**:
   - Quantified performance across 4 topologies
   - Established u₀ = 0.3 as physical threshold

3. **Space-Based Advantage Quantified**:
   - +3-5% overall accuracy vs. ground-based
   - +5-8% binary recall improvement
   - Earlier classification capability

4. **Survey Operations**:
   - Roman can detect 75-80% of binaries with u₀ < 0.3
   - High-cadence enables early follow-up triggers
   - Real-time capability demonstrated

5. **Physical Interpretation**:
   - Detection limits explained by caustic geometry
   - u₀ threshold validated across topologies
   - PSPL/Binary confusion physically motivated

---

## 📅 Timeline (v14.0 - Simplified)

### Phase 0: Validation (2-3 days)
- [ ] Run quick test (300 events)
- [ ] Verify Roman parameters (5% missing, 0.05 mag)
- [ ] Validate outputs

### Phase 1: Baseline (1 week)
- [ ] Generate 1M events (Roman quality)
- [ ] Train 50 epochs (~3-5 hours)
- [ ] Full evaluation

### Phase 2: Morphology Study (3-4 weeks)
- [ ] 4 topology experiments (150k each)
- [ ] Each: Generate → Train → Evaluate

### Phase 3: Analysis (2 weeks)
- [ ] u₀ comparison plots
- [ ] Performance summaries
- [ ] Statistical tests

### Phase 4: Writing (4-6 weeks)
- [ ] Methods chapter
- [ ] Results chapter
- [ ] Discussion
- [ ] Conclusions

**Total: 10-12 weeks → Feb 1, 2025** ✅

**Advantage over v13.1**: 
- 5 experiments instead of 11
- Simpler parameter space
- Clearer physical interpretation
- More feasible timeline

---

## 🚀 Quick Start Commands

### Multi-Node Setup (AMD MI300)

```bash
# Allocation
salloc --partition=gpu_a100_short --nodes=10 --gres=gpu:4 --exclusive --time=00:30:00

# Environment
cd ~/Thesis/code
conda activate microlens

export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
```

### Quick Test (CRITICAL - DO THIS FIRST)

```bash
cd code

# Generate 300 events
python simulate.py \
    --n_flat 100 --n_pspl 100 --n_binary 100 \
    --output ../data/raw/roman_quick_test.npz \
    --binary_params baseline \
    --save_params \
    --seed 42

# Verify Roman parameters
python -c "
from config import SimulationConfig
print(f'Cadence: {SimulationConfig.CADENCE_MASK_PROB*100:.0f}% missing')
print(f'Error: {SimulationConfig.MAG_ERROR_STD:.2f} mag')
assert SimulationConfig.CADENCE_MASK_PROB == 0.05, 'Should be 0.05 for Roman!'
assert SimulationConfig.MAG_ERROR_STD == 0.05, 'Should be 0.05 for Roman!'
print('✅ Configuration correct for Roman!')
"

# Train
python train.py \
    --data ../data/raw/roman_quick_test.npz \
    --experiment_name roman_quick_test \
    --epochs 5 \
    --batch_size 32

# Evaluate
python evaluate.py \
    --experiment_name roman_quick_test \
    --data ../data/raw/roman_quick_test.npz
```

**If quick test passes → Proceed to baseline!**

---

## 📊 Key Metrics (v14.0)

For each experiment:

1. **Overall**: Accuracy, F1, Confusion matrix
2. **Per-Class**: Recall, Precision, F1 for Flat/PSPL/Binary
3. **ROC**: One-vs-rest AUC
4. **Binary-Specific**: u₀ dependency, threshold analysis
5. **Confidence**: Calibration curves

**Target Performance (Roman)**:
- Overall: >78%
- Flat: >90%
- PSPL: >73%
- Binary: >75% (u₀ < 0.3)

---

## ✅ Pre-Experiment Checklist

- [ ] config.py updated to v14.0 (CADENCE=0.05, ERROR=0.05)
- [ ] All cadence/error experiments removed from ExperimentPresets
- [ ] Only 5 experiments: baseline_1M + 4 topologies
- [ ] Quick test passes (300 events)
- [ ] Multi-node commands tested
- [ ] README updated with Roman focus

---

## 🎯 YOU ARE READY!

Version 14.0 provides:
- ✅ Simplified scope (5 experiments)
- ✅ Roman Space Telescope focus
- ✅ Clear physical interpretation
- ✅ Feasible timeline (10-12 weeks)
- ✅ Production-ready code

**Run quick test, then start baseline!** 🚀🛰️