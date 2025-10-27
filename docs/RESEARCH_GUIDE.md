# Research Guide: Systematic Benchmarking of Microlensing Binary Classification

**Complete workflow for thesis research**

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

#### 5. **tE (Einstein Crossing Time)** - Event Timescale

**Definition**: Time for source to cross one Einstein radius

**Physical significance**:
- **tE ~ 10-50 days**: Typical for bulge events
- **tE ~ 50-200 days**: Long events (good for detailed study)
- Longer tE → more observations → better characterization

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

## 📊 Experimental Design

### Baseline (Current - In Progress)

**Configuration**:
```python
n_events = 1,000,000
cadence_mask_prob = 0.20  # 20% missing observations
mag_error_std = 0.10       # 0.1 mag photometric error
binary_params = 'baseline' # Mixed difficulty
```

**Status**: ✅ Data ready, 🔄 Training in progress

---

### Experiment Suite

After baseline completes, run these systematic experiments:

#### 1. Cadence Experiments

Test how observation frequency affects performance.

**Configurations**:
```python
# Dense cadence (LSST-like)
'cadence_dense': {
    'cadence_mask_prob': 0.05,  # 95% coverage
    'n_events': 200_000,
}

# Baseline
'cadence_baseline': {
    'cadence_mask_prob': 0.20,  # 80% coverage
    'n_events': 1_000_000,
}

# Sparse
'cadence_sparse': {
    'cadence_mask_prob': 0.30,  # 70% coverage  
    'n_events': 200_000,
}

# Very sparse
'cadence_very_sparse': {
    'cadence_mask_prob': 0.40,  # 60% coverage
    'n_events': 200_000,
}
```

**Commands**:
```bash
# Generate data
python code/simulate.py --n_pspl 100000 --n_binary 100000 \
       --output data/raw/events_cadence_05.npz --cadence 0.05

# Train
python code/train.py --data data/raw/events_cadence_05.npz \
       --output models/cadence_05.pt --experiment_name cadence_05

# Evaluate
python code/evaluate.py --model results/cadence_05_*/best_model.pt \
       --data data/raw/events_cadence_05.npz --output_dir results/cadence_05_eval
```

---

#### 2. Photometric Error Experiments

Test how measurement precision affects performance.

**Configurations**:
```python
# Space-based quality (Roman)
'error_low': {
    'mag_error_std': 0.05,
}

# Ground-based quality (LSST)
'error_baseline': {
    'mag_error_std': 0.10,
}

# Poor conditions
'error_high': {
    'mag_error_std': 0.20,
}
```

**Hypothesis**: Photometric error matters less than cadence for caustic-crossing events (sharp features survive moderate noise).

---

#### 3. Binary Difficulty Experiments

Test fundamental limits set by caustic topology.

**Configurations**:
```python
# Distinct: Small u₀, s≈1, small ρ (crosses caustics)
'binary_distinct': {
    'binary_params': 'distinct',
}

# Baseline: Mixed population
'binary_baseline': {
    'binary_params': 'baseline',
}

# Planetary vs Stellar
'planetary': {
    'binary_params': 'planetary',
}

'stellar': {
    'binary_params': 'stellar',
}
```

---

#### 4. Early Detection Experiments

Test real-time classification capability.

**Concept**: For triggering follow-up observations, we need to classify events **before** they complete. The TimeDistributed architecture outputs probabilities at each timestep.

**Evaluation**: Use `--early_detection` flag in `evaluate.py` to test performance at:
- 10% observed
- 25% observed
- 33% observed
- 50% observed
- 67% observed
- 83% observed
- 100% observed (complete)

```bash
python evaluate.py \
    --model results/baseline_*/best_model.pt \
    --data data/raw/events_baseline_1M.npz \
    --output_dir results/baseline_eval \
    --early_detection
```

---

## 🔬 Analysis Plan

### 1. Individual Experiment Analysis

For each experiment, report:

**Performance Metrics**:
- Accuracy (train/val/test)
- Precision, Recall, F1 per class
- ROC AUC and PR AUC
- Confusion matrix

**Visualizations**:
- Training curves (loss and accuracy)
- ROC curve
- Precision-Recall curve
- Confusion matrix heatmap

**Computational Cost**:
- Training time
- GPU memory usage
- Inference speed (events/second)

---

### 2. Comparative Analysis

Create comparison plots across experiments:

**Accuracy vs Cadence**:
```python
import matplotlib.pyplot as plt
import json

experiments = ['cadence_05', 'cadence_10', 'cadence_20', 'cadence_30', 'cadence_40']
cadences = [5, 10, 20, 30, 40]
accuracies = []

for exp in experiments:
    with open(f'results/{exp}/metrics.json') as f:
        metrics = json.load(f)
        accuracies.append(metrics['accuracy'])

plt.figure(figsize=(10, 6))
plt.plot(cadences, accuracies, 'o-', linewidth=2, markersize=10)
plt.xlabel('Missing Observations (%)', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Classification Performance vs Observing Cadence', fontsize=16)
plt.grid(alpha=0.3)
plt.savefig('figures/cadence_comparison.png', dpi=300, bbox_inches='tight')
```

**Binary Difficulty Comparison**:
```python
# Bar plot comparing distinct/baseline/planetary/stellar
difficulties = ['Distinct', 'Baseline', 'Planetary', 'Stellar']
accuracies = [load_accuracy(exp) for exp in difficulties]

plt.figure(figsize=(10, 6))
bars = plt.bar(difficulties, accuracies, alpha=0.7)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Performance vs Binary Configuration', fontsize=16)
plt.grid(alpha=0.3, axis='y')
plt.savefig('figures/difficulty_comparison.png', dpi=300, bbox_inches='tight')
```

**Early Detection Curve**:
```python
# Plot accuracy vs fraction of event observed
# (Already generated by evaluate.py with --early_detection flag)
```

---

### 3. Physical Interpretation

Connect results to caustic crossing physics:

**u₀ Distribution Analysis**:
```python
# Load binary parameters from simulation
data = np.load('data/raw/events_baseline_1M.npz', allow_pickle=True)
binary_params = data['binary_params']
u0_values = [p['u0'] for p in binary_params]

# Load predictions from evaluation
# Analyze which u₀ values lead to misclassification

# Plot distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(u0_values, bins=50, alpha=0.7, label='All binaries')
plt.xlabel('Impact Parameter u₀')
plt.ylabel('Count')
plt.title('Binary u₀ Distribution')

plt.subplot(1, 2, 2)
plt.hist(misclassified_u0, bins=50, alpha=0.7, color='red')
plt.xlabel('Impact Parameter u₀')
plt.ylabel('Count')
plt.title('Misclassified Binary u₀ Distribution')

plt.tight_layout()
plt.savefig('figures/u0_analysis.png', dpi=300)
```

**Hypothesis to test**: Misclassifications concentrate at u₀ > 0.3 (PSPL-like regime).

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
- 🔄 Baseline training (current)
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

These are **publishable results**! Consider writing a paper after thesis submission.

---

## 📊 Results Documentation

### Master Comparison Table Template

Create a table like this in your thesis:

| Experiment | Configuration | Accuracy | ROC AUC | PR AUC | Training Time | Notes |
|------------|---------------|----------|---------|--------|---------------|-------|
| Baseline | 20% missing, 0.1 mag error | - | - | - | - | Reference |
| Dense cadence | 5% missing | - | - | - | - | LSST-like |
| Sparse cadence | 40% missing | - | - | - | - | Poor coverage |
| Low error | 0.05 mag | - | - | - | - | Space-based |
| High error | 0.20 mag | - | - | - | - | Poor conditions |
| Distinct | u₀<0.15, s≈1 | - | - | - | - | Caustic crossers |
| Planetary | q << 1 | - | - | - | - | Planet systems |
| Stellar | q ~ 1 | - | - | - | - | Binary stars |

Fill this in as experiments complete.

---

## 💬 Key Insights to Communicate

In your thesis, emphasize:

1. **Why ML?** Traditional light curve fitting is computationally expensive and requires good initial guesses. ML enables real-time classification.

2. **Why TimeDistributed?** Enables early detection—critical for triggering follow-up observations.

3. **Physical limits matter**: Some binaries (large u₀) are fundamentally indistinguishable from PSPL. This is not a failure of the algorithm—it's physics.

4. **Survey design implications**: Your results inform how LSST and Roman should allocate observing time.

5. **Open questions remain**: Real data will have additional complexities (blending, parallax, etc.)

---

**Remember**: Your thesis tells a story about understanding the limits of binary detection. Focus on clarity, physical intuition, and practical implications for survey design.

Good luck! 🚀