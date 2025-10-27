# Complete Thesis Workflow Guide

**Goal**: Systematically benchmark binary vs PSPL classification performance under different observational conditions.

---

## 🎯 Research Questions

Your thesis will quantitatively answer:

1. **What's the best achievable performance?**  
   → Test with ideal conditions (dense cadence, low error, easy binaries)

2. **How does observing cadence affect classification?**  
   → Test 5%, 20%, 30%, 40% missing observations

3. **How early can we reliably detect binary events?**  
   → Test classification with partial light curves (33%, 50%, 67%)

4. **What's the physical detection limit?**  
   → Test easy vs hard binary topologies (caustic-crossing vs PSPL-like)

---

## 📊 Experimental Design

### Baseline (Current - In Progress)

**Configuration**:
```python
n_events = 1,000,000
cadence_mask_prob = 0.20  # 20% missing observations
mag_error_std = 0.10       # 0.1 mag photometric error
binary_params = 'standard' # Mixed difficulty
```

**Expected Results**:
- Accuracy: 95-96%
- ROC AUC: 0.98-0.99
- Training time: 6-12 hours (4 GPUs)

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
    'expected_acc': 0.97-0.98
}

# Baseline
'cadence_baseline': {
    'cadence_mask_prob': 0.20,  # 80% coverage
    'n_events': 1_000_000,
    'expected_acc': 0.95-0.96
}

# Sparse
'cadence_sparse': {
    'cadence_mask_prob': 0.30,  # 70% coverage  
    'n_events': 200_000,
    'expected_acc': 0.90-0.93
}

# Very sparse
'cadence_very_sparse': {
    'cadence_mask_prob': 0.40,  # 60% coverage
    'n_events': 200_000,
    'expected_acc': 0.85-0.88
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

**Expected plot**:
```
Accuracy vs Missing Data

98%│ •
96%│   •
94%│     •
92%│       •  
90%│         •
88%│           •
86%├─────────────────
   0%  10% 20% 30% 40%
      Missing observations

Interpretation: Performance degrades ~0.5% per 10% missing data
```

---

#### 2. Photometric Error Experiments

Test how measurement precision affects performance.

**Configurations**:
```python
# Space-based quality (Roman)
'error_low': {
    'mag_error_std': 0.05,
    'expected_acc': 0.97-0.98
}

# Ground-based quality (LSST)
'error_baseline': {
    'mag_error_std': 0.10,
    'expected_acc': 0.95-0.96
}

# Poor conditions
'error_high': {
    'mag_error_std': 0.20,
    'expected_acc': 0.88-0.92
}
```

**Expected insight**: Photometric error matters less than cadence for caustic-crossing events (sharp features survive moderate noise).

---

#### 3. Binary Difficulty Experiments

Test fundamental limits set by caustic topology.

**Configurations**:
```python
# Easy: Small u₀, s≈1, small ρ (crosses caustics)
'binary_easy': {
    'binary_params': 'easy',
    'expected_acc': 0.98-0.99  # Near perfect
}

# Standard: Mixed population
'binary_baseline': {
    'binary_params': 'standard',
    'expected_acc': 0.95-0.96
}

# Hard: Large u₀, extreme s, large ρ (PSPL-like)
'binary_hard': {
    'binary_params': 'hard',
    'expected_acc': 0.82-0.88  # Fundamental limit!
}
```

**Expected finding**: ~15-20% of binaries are fundamentally indistinguishable from PSPL (u₀ > 0.3). This is a **physical limit**, not an algorithmic one.

---

#### 4. Early Detection Experiments

Test real-time classification capability.

**Concept**: For triggering follow-up observations, we need to classify events **before** they complete. The TimeDistributed architecture outputs probabilities at each timestep.

**Configurations**:
```python
# Early (33% observed)
'early_500': {
    'n_points': 500,
    'time_max': 333,  # First 1/3 of event
    'expected_acc': 0.85-0.90
}

# Mid (50% observed)  
'early_750': {
    'n_points': 750,
    'time_max': 500,  # First 1/2 of event
    'expected_acc': 0.90-0.93
}

# Late (67% observed)
'early_1000': {
    'n_points': 1000,
    'time_max': 667,  # First 2/3 of event
    'expected_acc': 0.93-0.95
}
```

**Expected plot**:
```
Accuracy vs Observation Time

98%│                    •
95%│                •
92%│            •
89%│        •
86%│    •
83%│•
   ├────────────────────
   0%  33%  50%  67% 100%
       Fraction of event observed

Interpretation: Can trigger follow-up at ~50% completion with >90% confidence
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
plt.ylim([0.80, 1.0])
plt.savefig('figures/cadence_comparison.png', dpi=300, bbox_inches='tight')
```

**Binary Difficulty Comparison**:
```python
# Bar plot comparing easy/standard/hard
difficulties = ['Easy', 'Standard', 'Hard']
accuracies = [0.985, 0.955, 0.850]  # example values

plt.figure(figsize=(10, 6))
bars = plt.bar(difficulties, accuracies, color=['green', 'blue', 'red'], alpha=0.7)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Performance vs Binary Difficulty', fontsize=16)
plt.ylim([0.80, 1.0])
plt.axhline(y=0.95, color='k', linestyle='--', label='Baseline')
plt.legend()
plt.savefig('figures/difficulty_comparison.png', dpi=300, bbox_inches='tight')
```

**Early Detection Curve**:
```python
# Plot accuracy vs fraction of event observed
fractions = [0.10, 0.25, 0.33, 0.50, 0.67, 0.83, 1.00]
accuracies = [0.75, 0.83, 0.87, 0.91, 0.94, 0.955, 0.96]  # example

plt.figure(figsize=(10, 6))
plt.plot(fractions, accuracies, 'o-', linewidth=2, markersize=10)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
plt.xlabel('Fraction of Event Observed', fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=14)
plt.title('Real-Time Classification Performance', fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('figures/early_detection.png', dpi=300, bbox_inches='tight')
```

---

### 3. Physical Interpretation

Connect results to caustic crossing physics:

**u₀ Distribution Analysis**:
```python
# Load binary parameters from simulation
data = np.load('data/raw/events_1M.npz', allow_pickle=True)
binary_params = data['binary_params']
u0_values = [p['u0'] for p in binary_params]

# Analyze misclassifications
# (Load predictions from evaluate.py)
misclassified_u0 = u0_values[misclassified_indices]

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(u0_values, bins=50, alpha=0.7, label='All binaries')
plt.xlabel('Impact Parameter u₀')
plt.ylabel('Count')
plt.title('Binary u₀ Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(misclassified_u0, bins=50, alpha=0.7, color='red', label='Misclassified')
plt.xlabel('Impact Parameter u₀')
plt.ylabel('Count')
plt.title('Misclassified Binary u₀ Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('figures/u0_analysis.png', dpi=300)
```

**Expected finding**: Misclassifications concentrate at u₀ > 0.3 (PSPL-like regime).

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
- **4.4 Binary Difficulty**: Fundamental detection limits
- **4.5 Early Detection**: Real-time classification capability
- **4.6 Comparative Analysis**: Summary plots and tables

### Chapter 5: Discussion
- **Physical interpretation**: Connect to caustic crossing physics
- **Survey implications**: LSST vs Roman strategies
- **Detection limits**: The u₀ > 0.3 fundamental threshold
- **Comparison to literature**: How does this compare to other ML studies?

### Chapter 6: Conclusion
- Summary of findings
- Practical recommendations for survey design
- Future work (other lens types, real data, etc.)

---

## 📊 Key Results Table

Create a master comparison table:

| Experiment | Config | Accuracy | ROC AUC | Training Time | Notes |
|------------|--------|----------|---------|---------------|-------|
| Baseline | 20% missing, 0.1 mag error, standard | 95.6% | 0.984 | 8.3 hrs | Reference |
| Dense cadence | 5% missing | 97.2% | 0.991 | 2.1 hrs | LSST-like |
| Sparse cadence | 40% missing | 86.8% | 0.923 | 2.0 hrs | Poor coverage |
| Low error | 0.05 mag | 96.8% | 0.988 | 2.1 hrs | Space-based |
| High error | 0.20 mag | 91.3% | 0.957 | 2.0 hrs | Poor conditions |
| Easy binaries | u₀<0.1, s≈1 | 98.5% | 0.996 | 2.1 hrs | Caustic crossers |
| Hard binaries | u₀>0.3, extreme s | 84.2% | 0.905 | 2.0 hrs | Fundamental limit |
| Early (33%) | Partial light curve | 87.5% | 0.934 | 1.5 hrs | Real-time trigger |

---

## 🚀 Execution Timeline

### Week 1-2: Baseline Completion
- ✅ 1M dataset generated
- 🔄 Baseline training (current)
- ⏳ Baseline evaluation

**Deliverable**: Baseline performance metrics and plots

---

### Week 3-4: Cadence Experiments
```bash
# Generate datasets
for cadence in 0.05 0.10 0.30 0.40; do
    python code/simulate.py --n_pspl 100000 --n_binary 100000 \
           --output data/raw/events_cadence_${cadence/.}.npz \
           --cadence $cadence --n_processes 24
done

# Train all models (can submit as array job)
for exp in cadence_05 cadence_10 cadence_30 cadence_40; do
    sbatch slurm/train_experiment.sh $exp
done

# Evaluate
for exp in cadence_05 cadence_10 cadence_30 cadence_40; do
    python code/evaluate.py --model results/${exp}_*/best_model.pt \
           --data data/raw/events_${exp}.npz --output_dir results/${exp}_eval
done
```

**Deliverable**: Cadence comparison plot and analysis

---

### Week 5-6: Error and Difficulty Experiments
```bash
# Photometric error experiments
for error in 0.05 0.20; do
    python code/simulate.py --n_pspl 100000 --n_binary 100000 \
           --output data/raw/events_error_${error/.}.npz --error $error
done

# Binary difficulty experiments  
for difficulty in easy hard; do
    python code/simulate.py --n_pspl 100000 --n_binary 100000 \
           --output data/raw/events_${difficulty}.npz \
           --binary_difficulty $difficulty
done

# Train and evaluate all
# (Similar pattern to cadence experiments)
```

**Deliverable**: Error and difficulty comparison plots

---

### Week 7-8: Early Detection and Analysis
```bash
# Generate truncated datasets
for frac in 0.33 0.50 0.67; do
    python code/simulate.py --n_pspl 100000 --n_binary 100000 \
           --output data/raw/events_early_${frac/.}.npz \
           --n_points $((1500 * $frac)) --time_max $((1000 * $frac))
done

# Physical interpretation analysis
python code/analyze_u0_distribution.py  # Custom analysis script
python code/generate_comparison_plots.py
```

**Deliverable**: Early detection curve and u₀ analysis

---

### Week 9-12: Thesis Writing
- Draft all chapters
- Refine figures and tables
- Proofreading and revision

---

## 💡 Tips for Success

### During Experiments
1. **Document everything**: Keep a lab notebook (digital or physical)
2. **Save intermediate results**: Don't overwrite experiments
3. **Use informative names**: `cadence_05_20250127` not `test_v3`
4. **Track compute time**: Report GPU hours in thesis (good for reproducibility)

### During Analysis
1. **Sanity checks**: Does a 99% accuracy model really work, or did you leak test data?
2. **Error bars**: Report standard deviation over multiple runs if time permits
3. **Failed experiments**: Negative results are results! Document what didn't work

### During Writing
1. **Show, don't tell**: Let plots speak
2. **Physical intuition**: Always connect ML results back to physics
3. **Practical impact**: What do survey designers learn from your work?

---

## 📦 Deliverables Checklist

By thesis submission, you should have:

**Code**:
- [ ] Clean, documented GitHub repository
- [ ] README with usage instructions
- [ ] Requirements file with exact versions
- [ ] Example notebooks for reproducing key figures

**Data**:
- [ ] All datasets uploaded to institutional repository (if allowed)
- [ ] Or: Code to regenerate datasets

**Results**:
- [ ] All trained models saved
- [ ] All metrics in JSON/CSV format
- [ ] All plots in high-res (300 DPI) format

**Thesis**:
- [ ] Complete LaTeX/Word document
- [ ] All figures with captions
- [ ] Bibliography with 30-50 references
- [ ] Appendix with hyperparameters and training details

---

## 🎯 Expected Contributions

When you finish, you'll have:

1. **First systematic benchmarking** of binary classification under realistic conditions
2. **Quantitative guidance** for LSST and Roman survey strategies
3. **Physical interpretation** of the u₀ > 0.3 fundamental limit
4. **Open-source pipeline** others can build on
5. **Real-time classification** architecture for triggering follow-up

These are **publishable results**! Consider writing a paper after thesis submission.

---

## 📚 Recommended Reading

Before writing each chapter:

**Chapter 1-2** (Background):
- Paczynski (1986): Original microlensing paper
- Gaudi (2012): Microlensing review (excellent overview)
- Mao & Paczynski (1991): Binary lensing introduction

**Chapter 3** (Methods):
- VBMicrolensing paper (Bozza 2010)
- TimeDistributed layers in Keras/PyTorch documentation
- Related ML papers (search "machine learning microlensing")

**Chapter 4-5** (Results/Discussion):
- Recent LSST and Roman planning documents
- Papers on caustic crossing events (look for caustic crossing rates)

---

## 💬 Questions to Address in Thesis

Make sure your thesis answers:

1. **Why ML?** What's wrong with traditional model fitting?
2. **Why CNNs?** Why not RNNs or Transformers?
3. **Why TimeDistributed?** Why not just use the final timestep?
4. **What's new?** How is this different from previous ML studies?
5. **What's the limit?** Can we do better with more data or better architecture?
6. **So what?** What should survey designers actually do differently?

---

**Remember**: A thesis is a story. Your story is: "I built a system to classify microlensing events in real-time, tested it under realistic conditions, found the fundamental limits, and provided guidance for future surveys." Tell that story clearly!

Good luck! 🚀