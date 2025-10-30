# Research Guide v3.0: Systematic Benchmarking of Microlensing Binary Classification

**Complete workflow for thesis research using v3.0 features**

---

## 🎯 Research Questions

Your thesis will quantitatively answer:

1. **What performance is achievable across diverse binary systems?**  
   → Test with wide parameter ranges representing realistic populations

2. **How does observing cadence affect classification?**  
   → Test 5%, 20%, 30%, 40% missing observations

3. **How early can we reliably detect binary events?**  
   → Test classification with partial light curves

4. **What's the physical detection limit?**  
   → Test easy vs hard binary topologies (caustic-crossing vs PSPL-like)

5. **How do photometric errors impact performance?**  
   → Test space-based vs ground-based quality

---

## 🆕 v3.0 Features for Research

### Experiment Tracking Made Easy

**v2.0 Problem**: Results got overwritten, hard to compare runs
**v3.0 Solution**: Every run gets its own timestamped directory

```bash
# Run same experiment with different seeds for statistics
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 42
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 123  
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 456

# Results automatically organized:
results/baseline_20251027_143022/  # seed 42
results/baseline_20251027_150315/  # seed 123
results/baseline_20251027_152748/  # seed 456
```

### Auto-Detection Saves Time

**v2.0**: Had to manually find and specify model paths  
**v3.0**: Scripts automatically find latest model

```bash
# Evaluate latest run (auto-detection)
python evaluate.py --experiment_name baseline --data data/raw/baseline_1M.npz

# Benchmark latest run (auto-detection)
python benchmark_realtime.py --experiment_name baseline --data data/raw/baseline_1M.npz
```

### Complete Reproducibility

Every experiment directory contains:
- `config.json` - Exact parameters used
- `training.log` - Full training history
- `summary.json` - Final metrics
- `best_model.pt` - Best checkpoint

---

## 📐 Understanding Binary Parameters

### The Physics Behind Detection

**Caustics** are critical curves in the lens plane where magnification becomes very large. When a source star crosses a caustic, the light curve shows:
- Sharp, dramatic spikes
- Complex, multi-peaked structure
- Features that PSPL events cannot produce

**This is the key signature for detecting binary lenses.**

### Critical Parameters (in order of importance)

#### 1. **u₀ (Impact Parameter)** - MOST IMPORTANT

**Definition**: Minimum distance between source trajectory and lens center of mass (in Einstein radii)

```
        Source trajectory
              →
    ━━━━━━━━━━━━━━━━━━━━
                         ↑ u₀
          ● ●           ┘
        Binary lens
```

**Physical significance**:
- **Small u₀ (< 0.15)**: Source passes close to lens
  - High probability of crossing caustics
  - Strong, distinctive features
  - Clearly different from PSPL
  
- **Medium u₀ (0.15 - 0.30)**: Moderate distance
  - May or may not cross caustics
  - Sometimes distinguishable, sometimes not
  
- **Large u₀ (> 0.30)**: Source passes far from lens
  - Very low caustic crossing probability
  - **Fundamentally PSPL-like**
  - No algorithm can reliably distinguish these

**Research finding**: Approximately 15-25% of binary events have large u₀ and are intrinsically indistinguishable from PSPL. This is a **fundamental physical limit**, not a machine learning limitation.

---

#### 2. **s (Separation)** - Caustic Size

**Definition**: Distance between the two lens masses (in Einstein radii)

**Physical significance**:
- **s < 0.5**: Close binary → small caustics
- **s ≈ 0.8-1.5**: Wide binary → **largest caustics** (optimal for detection)
- **s > 2.0**: Very wide binary → small, separated caustics

```
s = 0.3           s = 1.0           s = 3.0
(close)           (wide)         (very wide)

  ●─●              ●     ●          ●           ●
  tiny            LARGE            small
caustics         caustics        caustics
```

---

#### 3. **q (Mass Ratio)** - Asymmetry

**Definition**: Ratio of secondary to primary mass (m₂/m₁)

**Physical significance**:
- **q << 1 (e.g., 0.001)**: Planetary system (Jupiter/Sun ≈ 0.001)
  - Asymmetric caustic structure
  - Smaller planetary caustics
  
- **q ≈ 0.1-0.5**: Intermediate (brown dwarfs, low-mass stars)
  - Moderate asymmetry
  
- **q ≈ 0.5-1.0**: Stellar binary
  - More symmetric caustics
  - Can mimic PSPL symmetry if u₀ is large

---

#### 4. **ρ (Source Size)** - Feature Sharpness

**Definition**: Angular radius of the source star (in Einstein radii)

**Physical significance**:
- **ρ << 0.01**: Point-like source → infinitely sharp features
- **ρ ≈ 0.01-0.03**: Typical source → features visible but smoothed
- **ρ > 0.05**: Large source → heavily smoothed, may be undetectable

```
Light curve during caustic crossing:

ρ → 0             ρ = 0.01          ρ = 0.05
(point)           (typical)         (large)

    ∧                 ∧                 ∧
   ╱ ╲              ╱   ╲             ╱     ╲
  ╱   ╲            ╱     ╲           ╱       ╲
 ╱     ╲          ╱       ╲         ╱         ╲
──────────     ─────────────    ───────────────
Sharp spike     Rounded peak      Smooth bump
```

---

### Parameter Sets in This Project

#### BASELINE (Wide Range)
**Goal**: Represent realistic population from planetary to stellar

```python
{
    's': (0.1, 10.0),      # All separations
    'q': (0.001, 1.0),     # Planetary to equal-mass
    'u₀': (0.001, 1.0),    # All impact parameters
    'ρ': (0.0001, 0.1),    # All source sizes
    'tE': (10, 200),       # All timescales
}
```

**What to expect**: Mixed population with varying distinguishability

---

#### DISTINCT (Maximum Distinguishability)
**Goal**: Events guaranteed to look different from PSPL

```python
{
    's': (0.8, 1.5),       # Wide binary (largest caustics)
    'q': (0.01, 0.5),      # Asymmetric
    'u₀': (0.001, 0.15),   # MUST cross caustics
    'ρ': (0.0001, 0.01),   # Sharp features
    'tE': (20, 150),       # Good timescale
}
```

**What to expect**: Near-optimal classification possible

---

#### PLANETARY (Planet Detection)
**Goal**: Simulate planetary microlensing events

```python
{
    's': (0.5, 3.0),       # Typical planet orbits
    'q': (0.0001, 0.01),   # Jupiter-mass to super-Jupiter
    'u₀': (0.001, 0.5),    # All impacts
}
```

**Physical context**: Models planet-hosting systems

---

#### STELLAR (Binary Stars)
**Goal**: Simulate stellar binary lenses

```python
{
    's': (0.3, 5.0),       # Stellar separations
    'q': (0.3, 1.0),       # Near-equal to equal mass
    'u₀': (0.001, 0.8),    # All impacts
}
```

**Physical context**: Models binary star systems

---

## 📊 Experimental Design with v3.0

### Baseline (Reference Experiment)

**Configuration**:
```python
n_events = 1,000,000
cadence_mask_prob = 0.20  # 20% missing observations
mag_error_std = 0.10       # 0.1 mag photometric error
binary_params = 'baseline' # Mixed difficulty
```

**Commands**:
```bash
cd code

# Generate data
python simulate.py \
    --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline

# Train (creates timestamped directory automatically)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50

# Evaluate (auto-detects latest model)
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection

# Benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

**Results location**: `results/baseline_TIMESTAMP/`

---

### Systematic Experiment Suite

After baseline completes, run these experiments:

#### 1. Cadence Experiments

Test how observation frequency affects performance.

| Experiment | Missing % | Command |
|------------|-----------|---------|
| Dense | 5% | `python simulate.py --cadence_mask_prob 0.05 --output ../data/raw/cadence_05.npz` |
| Baseline | 20% | (already done) |
| Sparse | 30% | `python simulate.py --cadence_mask_prob 0.30 --output ../data/raw/cadence_30.npz` |
| Very Sparse | 40% | `python simulate.py --cadence_mask_prob 0.40 --output ../data/raw/cadence_40.npz` |

**Batch training**:
```bash
for exp in cadence_05 cadence_30 cadence_40; do
    python train.py --data ../data/raw/${exp}.npz --experiment_name ${exp}
done
```

---

#### 2. Photometric Error Experiments

Test how measurement precision affects performance.

| Experiment | Error (mag) | Quality |
|------------|-------------|---------|
| Low | 0.05 | Space-based (Roman) |
| Baseline | 0.10 | Ground-based (LSST) |
| High | 0.20 | Poor conditions |

**Hypothesis**: Photometric error matters less than cadence for caustic-crossing events (sharp features survive moderate noise).

---

#### 3. Binary Topology Experiments

Test fundamental limits set by caustic topology.

| Experiment | Description | Expected Performance |
|------------|-------------|---------------------|
| Distinct | u₀<0.15, s≈1 | High (>90% accuracy) |
| Planetary | q<<1 | Moderate (70-80%) |
| Stellar | q~1 | Challenging (60-75%) |

---

#### 4. Early Detection Analysis

Test real-time classification capability using `--early_detection` flag.

**Evaluation checkpoints**:
- 10% observed
- 25% observed
- 33% observed
- 50% observed
- 67% observed
- 83% observed
- 100% observed (complete)

```bash
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection
```

This creates plots showing accuracy vs observation completeness.

---

## 🔬 Analysis Workflow with v3.0

### 1. Individual Experiment Analysis

For each experiment, you get:

**Automatic outputs**:
- `results/EXPERIMENT_TIMESTAMP/best_model.pt`
- `results/EXPERIMENT_TIMESTAMP/config.json`
- `results/EXPERIMENT_TIMESTAMP/training.log`
- `results/EXPERIMENT_TIMESTAMP/summary.json`
- `results/EXPERIMENT_TIMESTAMP/evaluation/` (after evaluation)
- `results/EXPERIMENT_TIMESTAMP/benchmark/` (after benchmarking)

**Quick results check**:
```bash
# View training summary
cat $(ls -td results/baseline_*/ | head -1)/summary.json

# View evaluation metrics
cat $(ls -td results/baseline_*/ | head -1)/evaluation/evaluation_summary.json
```

---

### 2. Comparative Analysis

**Compare multiple runs**:
```python
import json
from pathlib import Path

experiments = ['baseline', 'cadence_05', 'cadence_30', 'cadence_40']

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        latest = runs[-1]
        summary = latest / 'summary.json'
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            print(f"{exp:15s}: Acc={data['final_test_acc']:.4f}")
```

**Create comparison plots**:
```python
import matplotlib.pyplot as plt

# Cadence comparison
cadences = [5, 20, 30, 40]
accuracies = []  # Extract from each experiment

plt.figure(figsize=(10, 6))
plt.plot(cadences, accuracies, 'o-', linewidth=2)
plt.xlabel('Missing Observations (%)')
plt.ylabel('Test Accuracy')
plt.title('Performance vs Observing Cadence')
plt.savefig('figures/cadence_comparison.png', dpi=300)
```

---

### 3. Statistical Analysis (Multiple Seeds)

**Run experiments with multiple seeds**:
```bash
for seed in 42 123 456 789 101112; do
    python train.py \
        --data data/raw/baseline_1M.npz \
        --experiment_name baseline \
        --seed $seed
done
```

**Analyze variance**:
```python
import numpy as np
import json
from pathlib import Path

accuracies = []
for run_dir in Path('results').glob('baseline_*'):
    summary = run_dir / 'summary.json'
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
        accuracies.append(data['final_test_acc'])

print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
```

---

## 📝 Thesis Structure

### Chapter 1: Introduction
- Gravitational microlensing basics
- Binary vs PSPL distinction
- Importance for exoplanet science
- Motivation for ML approach

### Chapter 2: Theoretical Background
- Microlensing theory (PSPL and binary)
- Caustic topology and crossing physics
- Role of u₀, s, q, ρ parameters
- Observational constraints (cadence, photometry)

### Chapter 3: Methodology
- Dataset generation (VBMicrolensing)
- Observational realism (errors, gaps)
- CNN architecture (TimeDistributed design)
- Training procedure and hyperparameters
- **v3.0 Feature**: Experiment management and reproducibility

### Chapter 4: Results
- **4.1 Baseline Performance**: 1M event training results
- **4.2 Cadence Experiments**: Impact of observation frequency
- **4.3 Photometric Error**: Impact of measurement precision  
- **4.4 Binary Difficulty**: Detection across different configurations
- **4.5 Early Detection**: Real-time classification capability
- **4.6 Comparative Analysis**: Summary plots and tables

### Chapter 5: Discussion
- **Physical interpretation**: Connect to caustic crossing physics
- **Survey implications**: LSST vs Roman strategies
- **Detection limits**: Understanding the u₀ threshold
- **Comparison to literature**: How does this compare to other ML studies?

### Chapter 6: Conclusion
- Summary of findings
- Practical recommendations for survey design
- Future work (other lens types, real data, etc.)

---

## 🚀 Execution Timeline

### Phase 1: Baseline (Weeks 1-2)
- ✅ 1M dataset generated
- 🔄 Baseline training
- ⏳ Baseline evaluation

### Phase 2: Systematic Experiments (Weeks 3-6)
- Cadence experiments (4 configurations)
- Photometric error experiments (3 configurations)
- Binary difficulty experiments (4 configurations)
- Early detection analysis

### Phase 3: Analysis (Weeks 7-8)
- Generate all comparison plots
- Physical interpretation (u₀ analysis)
- Statistical analysis
- **v3.0 Feature**: Easy comparison across runs

### Phase 4: Writing (Weeks 9-12)
- Draft all chapters
- Refine figures and tables
- Proofreading and revision

---

## 💡 Key Research Questions to Answer

1. **What fraction of realistic binaries are detectable?**
   - Analyze baseline results by u₀ distribution
   
2. **How dense must survey cadence be?**
   - Compare performance across cadence experiments
   
3. **Can we trigger follow-up observations early?**
   - Analyze early detection results
   
4. **Do planetary systems differ from stellar?**
   - Compare planetary vs stellar experiments
   
5. **What's the intrinsic physical limit?**
   - Identify u₀ threshold where performance drops

---

## 📊 Results Documentation Template

### Master Comparison Table

Use this template for your thesis:

```
| Experiment       | Accuracy | ROC AUC | PR AUC | Training Time | Notes        |
|------------------|----------|---------|--------|---------------|--------------|
| Baseline         | XX.X%    | X.XXX   | X.XXX  | Xh           | Reference    |
| Dense (5%)       | XX.X%    | X.XXX   | X.XXX  | Xh           | LSST-like    |
| Sparse (30%)     | XX.X%    | X.XXX   | X.XXX  | Xh           | Poor         |
| V.Sparse (40%)   | XX.X%    | X.XXX   | X.XXX  | Xh           | Minimal      |
| Low Error (0.05) | XX.X%    | X.XXX   | X.XXX  | Xh           | Space-based  |
| High Error (0.20)| XX.X%    | X.XXX   | X.XXX  | Xh           | Poor quality |
| Distinct         | XX.X%    | X.XXX   | X.XXX  | Xh           | Easy         |
| Planetary        | XX.X%    | X.XXX   | X.XXX  | Xh           | Moderate     |
| Stellar          | XX.X%    | X.XXX   | X.XXX  | Xh           | Challenging  |
```

**Extract automatically**:
```bash
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_05', 'cadence_30', 'cadence_40',
               'error_05', 'error_20', 'distinct', 'planetary', 'stellar']

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        latest = runs[-1]
        eval_file = latest / 'evaluation' / 'evaluation_summary.json'
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            metrics = data['metrics']
            print(f\"{exp:15s} | {metrics['accuracy']*100:5.1f}% | {metrics['roc_auc']:5.3f} | {metrics['pr_auc']:5.3f}\")
"
```

---

## 💬 Key Insights to Communicate

In your thesis, emphasize:

1. **Why ML?** Traditional light curve fitting is computationally expensive and requires good initial guesses. ML enables real-time classification.

2. **Why TimeDistributed?** Enables early detection—critical for triggering follow-up observations.

3. **Physical limits matter**: Some binaries (large u₀) are fundamentally indistinguishable from PSPL. This is not a failure of the algorithm—it's physics.

4. **Survey design implications**: Your results inform how LSST and Roman should allocate observing time.

5. **v3.0 makes research easier**: Timestamped directories, auto-detection, and complete reproducibility streamline the research process.

---

## 🎯 v3.0 Advantages for Thesis

### Better Organization
- Every experiment run is preserved
- Easy to revisit and compare old results
- No confusion about which model was which

### Faster Iteration
- Auto-detection saves typing and time
- Quick comparison scripts work seamlessly
- Batch processing is simpler

### Complete Reproducibility
- Config files show exact parameters
- Logs show full training history
- Easy to regenerate any result

### Publication-Ready
- Clean, organized results structure
- Easy to extract figures and tables
- Straightforward to share with advisors

---

## 📚 Recommended Reading

**Background**:
- Gaudi (2012): "Microlensing Surveys for Exoplanets" (comprehensive review)
- Paczynski (1986): Original microlensing paper
- Mao & Paczynski (1991): Binary lensing introduction

**Caustic Topology**:
- Erdl & Schneider (1993): "Classification of gravitational lens models"
- Dominik (1999): "Binary lensing and its extreme cases"

**Machine Learning Applications**:
- Recent papers on ML for microlensing (search Google Scholar)
- Papers on real-time event classification

---

## 🎯 Expected Contributions

Your thesis will provide:

1. **Systematic benchmarking** of binary classification under realistic conditions
2. **Quantitative guidance** for LSST and Roman survey strategies
3. **Physical interpretation** of detection limits
4. **Open-source pipeline** for the community
5. **Real-time classification** architecture for triggering follow-up

**These are publishable results!** Consider writing a paper after thesis submission.

---

**v3.0 makes your research workflow smoother and more organized!** 🚀🔭

Remember: Focus on clarity, physical intuition, and practical implications for survey design.