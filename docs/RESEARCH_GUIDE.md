# Research Guide

---

## 🎯 Research Questions

This work addresses:

1. **Baseline Performance**: What classification accuracy is achievable with space-based quality data while preventing temporal shortcuts?

2. **Binary Morphology**: How does binary lens topology (mass ratio, separation) affect detectability across different parameter spaces?

3. **Physical Detection Limits**: What are the fundamental u₀-dependent detection limits for binary systems?

4. **Early Detection**: How early can we classify events with high-cadence observations?

---

## 🛰️ Why Space-Based Quality?

### Advantages Over Ground-Based Surveys

**Space-Based (Roman-quality)**:
- Cadence: ~15 min sampling (5% missing)
- Photometry: 0.05 mag error
- Coverage: Continuous, no weather
- Result: Clearer caustic crossings, better classification

**Ground-Based (LSST-typical)**:
- Cadence: ~3 days (30%+ missing with weather)
- Photometry: 0.10 mag error
- Coverage: Weather-dependent
- Result: More challenging classification

### Research Focus

**5 Core Experiments**:
1. Baseline (1M events) - Space-based quality benchmark
2. Distinct topology - Clear caustics (optimal detection)
3. Planetary topology - Exoplanet focus (small q)
4. Stellar topology - Binary stars (equal masses)
5. Baseline topology - Mixed population (physical limits)

**Total compute time**: ~12-16 hours (feasible for thesis deadline)

---

## 🧪 Experimental Design 

### 1. Baseline Experiment (1M Events)

**Purpose**: Establish space-based quality baseline with anti-cheating architecture

**Configuration**:
```python
N_events = 1,000,000 (balanced 3-class)
N_flat = 333,000
N_pspl = 333,000
N_binary = 334,000

Binary Parameters: 'baseline'
  s: 0.1 - 3.0        # Wide separation range
  q: 0.0001 - 1.0     # Planetary to stellar
  u0: 0.001 - 1.0     # Full range (includes physical limits)
  rho: 0.001 - 0.1    # Source sizes
  tE: 10 - 40 days    # Realistic timescales
  t0: -80 - 60 days   # TEMPORAL INVARIANT (peaks outside observation)

Observational - Space-based Quality:
  Cadence: 5% missing (~15 min sampling)
  Photometry: 0.05 mag error
  Points: 1500 per light curve
  Time window: [-120, +120] days

Anti-Cheating Features:
  Temporal randomization: ON (±30 days t0 shift)
  Causal attention: ON (no future peeking)
  Temporal invariance loss: 0.1 weight
  Caustic detection: 0.8 weight
```

**Commands**:
```bash
cd code

# 1. Generate (1M events, space-based quality)
python simulate.py --preset baseline_1M
# This automatically sets:
#   - 333k + 333k + 334k = 1M events
#   - Binary preset: baseline
#   - Cadence: 5% missing
#   - Error: 0.05 mag
#   - Temporal randomization: ON
#   - Saves parameters for u0 analysis

# 2. Train (multi-node DDP)
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=baseline-$(date +%s) \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8
# Note: Causal attention is ON by default
# Use --no_causal_attention to disable (NOT recommended)

# 3. Evaluate (includes all v15.0 diagnostics)
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection \
    --temporal_bias_check \
    --n_evolution_per_type 10 \
    --batch_size 64 \
    --n_samples 50000
```

**Expected Results (Space-Based Quality + Anti-Cheating)**:
```
Overall Accuracy: 80-85% (+2-3% from anti-cheating design)

Per-Class Recall:
  Flat:   92-95%  (easy - constant flux)
  PSPL:   75-80%  (good - better with space quality)
  Binary: 77-82%  (good - u0 dependent, improved robustness)

Improvement from Anti-Cheating:
  +2-3% overall accuracy (less overfitting)
  Better early detection (learns morphology, not timing)
  More robust to temporal shifts
```

---

### 2. Binary Morphology Study (150k each)

**Purpose**: Characterize u₀ dependency and physical detection limits across different binary configurations

| Experiment | Description | Key Parameters | Expected Accuracy |
|------------|-------------|----------------|-------------------|
| **Distinct** | Clear caustics | s=0.7-1.5, q=0.01-0.5, u₀<0.15 | 87-92% |
| **Planetary** | Exoplanet focus | q=0.0001-0.01, small features | 82-87% |
| **Stellar** | Binary stars | q=0.3-1.0, complex caustics | 78-83% |
| **Baseline** | Physical limit | Wide u₀ (0.001-1.0), full space | 73-78% |

#### Experiment 2a: Distinct Topology

**Purpose**: Optimal detection conditions (clear caustics)

```bash
cd code

# Generate
python simulate.py --preset distinct \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=distinct-$(date +%s) \
    train.py \
        --data ../data/raw/distinct.npz \
        --experiment_name distinct \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8

# Evaluate
python evaluate.py \
    --experiment_name distinct \
    --data ../data/raw/distinct.npz \
    --early_detection \
    --temporal_bias_check \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 87-92% (clear caustics)
- u₀ dependency: Steep drop at u₀ > 0.15
- Early detection: High confidence at 40% completeness
- Temporal bias: None (anti-cheating works)

#### Experiment 2b: Planetary Topology

**Purpose**: Exoplanet detection (small mass ratios)

```bash
cd code

# Generate
python simulate.py --preset planetary \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=planetary-$(date +%s) \
    train.py \
        --data ../data/raw/planetary.npz \
        --experiment_name planetary \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8

# Evaluate
python evaluate.py \
    --experiment_name planetary \
    --data ../data/raw/planetary.npz \
    --early_detection \
    --temporal_bias_check \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 82-87% (small features)
- Planet-host separation critical
- Space-based quality essential for small q
- Anti-cheating helps: focuses on magnification spikes

#### Experiment 2c: Stellar Topology

**Purpose**: Binary star systems (equal masses)

```bash
cd code

# Generate
python simulate.py --preset stellar \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=stellar-$(date +%s) \
    train.py \
        --data ../data/raw/stellar.npz \
        --experiment_name stellar \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8

# Evaluate
python evaluate.py \
    --experiment_name stellar \
    --data ../data/raw/stellar.npz \
    --early_detection \
    --temporal_bias_check \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 78-83% (complex caustics)
- Symmetric caustics challenging
- u₀ dependency moderate
- Caustic detection head helps with morphology

#### Experiment 2d: Baseline Topology

**Purpose**: Physical detection limit study (full parameter space)

```bash
cd code

# Generate
python simulate.py --preset baseline \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000

# Train
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=baseline-$(date +%s) \
    train.py \
        --data ../data/raw/baseline.npz \
        --experiment_name baseline \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8

# Evaluate
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline.npz \
    --early_detection \
    --temporal_bias_check \
    --batch_size 64 \
    --n_samples 10000
```

**Expected Findings**:
- Binary recall: 73-78% (many large u₀)
- Clear u₀ threshold at ~0.3
- Proves physical, not algorithmic limit
- 15-20% of binaries fundamentally undetectable (u₀ > 0.3)

---

## 📊 Analysis Workflow

### 1. Extract Results

```bash
# Generate summary table
python -c "
import json
from pathlib import Path

experiments = [
    'baseline_1M',
    'distinct',
    'planetary',
    'stellar',
]

print(f'{'Experiment':<25} {'Overall':<10} {'Flat%':<8} {'PSPL%':<8} {'Binary%':<8}')
print('-' * 70)

for exp in experiments:
    runs = sorted(Path('../results').glob(f'{exp}_*'))
    if runs:
        eval_file = runs[-1] / 'evaluation' / 'evaluation_summary.json'
        if eval_file.exists():
            data = json.load(open(eval_file))
            
            overall = data.get('metrics', {}).get('accuracy', 0) * 100
            flat_rec = data.get('metrics', {}).get('flat_recall', 0) * 100
            pspl_rec = data.get('metrics', {}).get('pspl_recall', 0) * 100
            binary_rec = data.get('metrics', {}).get('binary_recall', 0) * 100
            
            print(f'{exp:<25} {overall:>8.1f}%  {flat_rec:>6.1f}  {pspl_rec:>6.1f}  {binary_rec:>6.1f}')
" > results_summary.txt

cat results_summary.txt
```

### 2. u₀ Dependency Comparison

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Compare u0 dependency across all topologies
topologies = ['baseline_1M', 'distinct', 'planetary', 'stellar']
colors = ['purple', 'green', 'blue', 'orange']

fig, ax = plt.subplots(figsize=(14, 8))

for topo, color in zip(topologies, colors):
    exp_dirs = sorted(Path('../results').glob(f'{topo}_*'))
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
                       color=color, label=topo.title(),
                       alpha=0.8)

# Add threshold line
ax.axvline(x=0.3, color='red', linestyle='--', linewidth=3, 
          label='Physical Limit (u₀=0.3)', alpha=0.7)
ax.axhline(y=50, color='gray', linestyle=':', linewidth=2, 
          alpha=0.5, label='Random (50%)')

ax.set_xlabel('Impact Parameter u₀', fontsize=14, fontweight='bold')
ax.set_ylabel('Binary Classification Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Binary Detection vs. Impact Parameter (All Topologies)', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim([30, 100])

plt.tight_layout()
plt.savefig('../figures/u0_comparison_all.png', dpi=300, bbox_inches='tight')
print("Saved: figures/u0_comparison_all.png")
```

### 3. Performance Summary

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

experiments = ['baseline_1M', 'distinct', 'planetary', 'stellar']
names = ['Baseline', 'Distinct', 'Planetary', 'Stellar']
overall_accs = []
binary_recalls = []

for exp in experiments:
    runs = sorted(Path('../results').glob(f'{exp}_*'))
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
ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Target')
ax1.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Overall Performance', fontsize=14, fontweight='bold')
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
ax2.axhline(y=75, color='red', linestyle='--', linewidth=2, label='75% Target')
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
plt.savefig('../figures/performance_summary.png', dpi=300, bbox_inches='tight')
print("Saved: figures/performance_summary.png")
```

### 4. Temporal Bias Verification

After each experiment, check the temporal bias diagnostics:

```python
import json
from pathlib import Path

exp_name = 'baseline_1M'  # Change as needed
exp_dirs = sorted(Path('../results').glob(f'{exp_name}_*'))

if exp_dirs:
    # Check if temporal_bias_diagnosis.png exists
    bias_plot = exp_dirs[-1] / 'evaluation' / 'temporal_bias_diagnosis.png'
    
    if bias_plot.exists():
        print(f"✅ Temporal bias diagnostics available for {exp_name}")
        print(f"   Check: {bias_plot}")
        print("\nWhat to look for:")
        print("  • t0 distributions should be uniform")
        print("  • No correlation between t0 and prediction accuracy")
        print("  • KS test p-value > 0.05 (no significant difference)")
    else:
        print(f"⚠️  Run evaluate.py with --temporal_bias_check")
```

---

## 📖 Thesis Structure

### Chapter 4: Results

#### 4.1 Baseline Performance (Space-Based Quality)

**Report**:
- Overall accuracy: 80-85%
- Anti-cheating advantage: +2-3% over naive models
- Per-class performance:
  - Flat: 92-95%
  - PSPL: 75-80%
  - Binary: 77-82%

**Key Message**: "Space-based photometric quality combined with anti-cheating architecture enables 80%+ three-class accuracy. The model learns from magnification morphology rather than temporal patterns, providing robust classification across different observation schedules."

**Figures**:
- ROC curves (3-class)
- Confusion matrix
- Calibration curves
- High-res evolution plots (20 points)
- Fine-grained early detection (15 fractions)
- Temporal bias diagnostics

#### 4.2 Binary Morphology Study

**Distinct Topology** (Optimal Conditions):
- Binary recall: 87-92%
- Key finding: Clear caustics enable early detection
- u₀ threshold: Sharp drop at u₀ > 0.15
- Anti-cheating benefit: Focuses on caustic crossings, not peak timing

**Planetary Topology** (Exoplanet Search):
- Binary recall: 82-87%
- Small mass ratios challenging but detectable
- Space-based quality critical for small q
- Caustic detection head improves planet identification

**Stellar Topology** (Binary Stars):
- Binary recall: 78-83%
- Complex caustics from equal-mass systems
- Symmetric structures harder to classify
- Temporal invariance loss prevents shortcut learning

**Baseline Topology** (Physical Limits):
- Binary recall: 73-78%
- Full parameter space demonstrates physical limit
- Clear threshold at u₀ ≈ 0.3
- 15-20% of binaries fundamentally undetectable

#### 4.3 Physical Detection Limits

**u₀ Dependency Analysis**:
- All topologies show sharp degradation at u₀ > 0.3
- 15-20% of binaries have u₀ > 0.3 (fundamentally PSPL-like)
- This is physical (source doesn't cross caustics), not algorithmic limitation
- Anti-cheating design ensures this finding is robust

**Key Conclusion**: "The u₀ = 0.3 threshold represents a fundamental physical detection limit for binary microlensing, arising from caustic geometry. Our anti-cheating architecture confirms this is not an artifact of temporal pattern learning."

#### 4.4 Early Detection 

**High-Cadence Benefits**:
- 50% completeness: 75-80% accuracy (reliable trigger)
- 25% completeness: 55-65% accuracy (early warning)
- High-cadence sampling enables 2-3 week earlier classification
- Fine-grained analysis (15 fractions) shows smooth accuracy growth

**Inference Performance**:
- Latency: <1 ms per event
- Throughput: 10,000+ events/second
- Survey-scale ready

---

## 🚀 Quick Start Commands

### Multi-Node Setup (AMD MI300 / NVIDIA GPUs)

```bash
# Allocate nodes
salloc --partition=gpu_a100_short --nodes=8 --gres=gpu:4 --exclusive --time=05:00:00

# Environment
cd ~/Thesis/code
conda activate microlens

# Suppress warnings
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Set DDP variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_TIMEOUT=1800
```

### Quick Test (CRITICAL - Run This First!)

```bash
cd code

# Generate 300 events (quick_test preset)
python simulate.py --preset quick_test

# Train (single GPU, 5 epochs)
python train.py \
    --data ../data/raw/quick_test.npz \
    --experiment_name quick_test \
    --epochs 5 \
    --batch_size 32

# Evaluate
python evaluate.py \
    --experiment_name quick_test \
    --data ../data/raw/quick_test.npz

# Check output
ls -lh ../results/quick_test_*/evaluation/
```

**Expected**: ~60-70% accuracy (tiny dataset, just checking it works)  
**If this passes** → Proceed to baseline!

---

## 📊 Key Metrics

For each experiment, evaluate.py generates:

1. **Classification Metrics**:
   - Overall accuracy, F1, confusion matrix
   - Per-class: Recall, Precision, F1 for Flat/PSPL/Binary
   - ROC curves (one-vs-rest) with AUC

2. **Binary-Specific**:
   - u₀ dependency analysis (if params available)
   - Physical threshold detection
   - Caustic detection performance

3. **Anti-Cheating Verification**:
   - Temporal bias diagnostics (KS tests)
   - t₀ distribution analysis
   - Correlation between peak time and predictions

4. **Early Detection**:
   - High-res evolution (20 fractions)
   - Fine-grained early detection (15 fractions)
   - Confidence calibration

5. **Attention Diagnostics** (optional):
   - Attention pattern analysis
   - Entropy measurements
   - Temporal concentration checks

**Target Performance (Space-Based + Anti-Cheating)**:
- Overall: >80%
- Flat: >92%
- PSPL: >75%
- Binary: >77% (u₀ < 0.3)
- Temporal bias: p > 0.05 (no shortcuts)

---

## 🔍 Debugging & Troubleshooting

### If Training Fails

**Out of Memory (OOM)**:
```bash
# Reduce batch size
--batch_size 32  # instead of 64

# Or enable gradient checkpointing
--gradient_checkpointing
```

**NaN/Inf Losses**:
- Already handled in code (skips bad batches)
- Check if > 10% batches skipped → data issue
- Verify data: `python -c "import numpy as np; d=np.load('data.npz'); print(d['X'].min(), d['X'].max())"`

**DDP Hangs**:
```bash
# Check MASTER_ADDR
echo $MASTER_ADDR

# Test connectivity
srun --nodes=2 --ntasks=2 hostname

# Increase timeout
export NCCL_TIMEOUT=3600
```

### If Evaluation Fails

**Memory Issues**:
```bash
# Limit samples
--n_samples 1000

# Reduce batch size
--batch_size 32
```

**Missing Parameters** (no u0 analysis):
- Check data has params: `python -c "import numpy as np; d=np.load('data.npz'); print(d.files)"`
- Re-generate with `--save_params` flag

### Verifying Anti-Cheating Features

```bash
# Check config.json after training
cat ../results/experiment_name_*/config.json

# Should see:
# "temporal_inv_weight": 0.1
# "caustic_weight": 0.8
# "causal_attention": true (not "no_causal_attention")

# Check temporal randomization in data
python -c "
import numpy as np
d = np.load('../data/raw/yourdata.npz')
print('Temporal randomization:', d.get('temporal_randomization', 'unknown'))
print('Max time shift:', d.get('max_time_shift', 'unknown'))
"
```

---

## 🎯 Success Criteria

Your experiments are successful if:

1. **Quick test passes** (no errors)
2. **Baseline achieves**:
   - Overall: >80%
   - Binary: >77%
   - Temporal bias: p > 0.05
3. **u₀ threshold observed** at ~0.3 across topologies
4. **Early detection works**: smooth accuracy growth with completeness
5. **All visualizations generate** without errors

---

## 📝 Lab Notebook Template

Keep track of experiments:

```markdown
## Experiment: baseline_1M
**Date**: YYYY-MM-DD
**Status**: [Running/Complete/Failed]

### Configuration
- Data: baseline_1M.npz (1M events)
- Topology: baseline (full parameter space)
- Nodes: 8 × 4 GPUs (32 total)
- Epochs: 50
- Batch size: 64

### Results
- Overall accuracy: XX.X%
- Binary recall: XX.X%
- u0 threshold: ~0.X
- Temporal bias p-value: X.XX

### Observations
- [Any unexpected results]
- [Ideas for discussion]
- [Follow-up experiments needed]

### Files
- Model: results/baseline_1M_TIMESTAMP/best_model.pt
- Plots: results/baseline_1M_TIMESTAMP/evaluation/*.png
- Summary: results/baseline_1M_TIMESTAMP/evaluation/evaluation_summary.json
```

---

## ✅ Pre-Experiment Checklist

Before starting experiments:
- [ ] Environment: `conda activate microlens`
- [ ] VBBinaryLensing installed
- [ ] PyTorch 2.2.0 with GPU support (ROCm 6.0 or CUDA 12.1)
- [ ] GPU nodes available
- [ ] Quick test passed
- [ ] Backup strategy in place
- [ ] Lab notebook ready

---

*This guide is version-agnostic and reflects the current state of the anti-cheating architecture and evaluation framework. Update as needed based on experimental findings.*
