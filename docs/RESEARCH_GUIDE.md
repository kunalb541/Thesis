# Research Guide: Real-Time Binary Microlensing Classification

## Scientific Objectives

This thesis investigates automated binary microlensing detection for next-generation astronomical surveys. The primary research questions are:

1. **Baseline Performance**: What classification accuracy can a transformer-based neural network achieve on Roman Space Telescope quality photometry (5% missing observations, 0.05 mag photometric precision)?

2. **Binary Morphology Dependence**: How does binary lens geometry (mass ratio q, projected separation s) affect detectability? Can we identify parameter space regions where binary classification reliability exceeds 85%?

3. **Physical Detection Limits**: Does the impact parameter u₀ impose a fundamental detectability threshold? At what u₀ do binary events become indistinguishable from single-lens events due to the absence of caustic crossings?

4. **Early Classification Capability**: At what observation completeness fractions (10%, 25%, 50% of expected observations) can we achieve classification accuracy sufficient to trigger follow-up observations?

---

## Observational Context

### Space-Based vs. Ground-Based Photometry

**Nancy Grace Roman Space Telescope** (baseline configuration):
- Temporal sampling: ~15 minute cadence
- Photometric precision: 0.05 mag
- Coverage: Continuous galactic bulge monitoring
- Key advantage: Temporal resolution sufficient to detect brief caustic crossings (hours to days)

**Ground-Based Surveys** (OGLE, MOA, KMTNet):
- Temporal sampling: ~1-3 days (weather-dependent)
- Photometric precision: 0.10 mag
- Coverage: Seasonal observing windows with weather gaps
- Limitation: May undersample or miss brief caustic features characteristic of binary lensing

### Scientific Motivation

Roman's improved cadence and photometric precision should significantly enhance binary lens detection rates, particularly for planetary-mass companions (q < 0.01) where caustic crossing durations are brief (<1 day). Quantifying this improvement requires systematic comparison of classification performance across observational parameter space.

---

## Experimental Design

### Data Generation Philosophy

We generate synthetic microlensing events using VBBinaryLensing (Bozza 2010), sampling physically realistic parameter ranges informed by OGLE and MOA survey discoveries. Each experimental dataset probes a specific region of binary lens parameter space to test detectability hypotheses.

### Physical Parameters

| Parameter | Symbol | Physical Interpretation | Sampling Range |
|-----------|--------|------------------------|----------------|
| Mass ratio | q | m₂/m₁ (companion/primary) | 10⁻⁴ to 1.0 |
| Projected separation | s | d/θ_E (caustic geometry) | 0.1 to 3.0 |
| Impact parameter | u₀ | Minimum source-lens separation | 0.001 to 1.0 |
| Source size | ρ | θ*/θ_E (finite source effects) | 0.001 to 0.1 |
| Einstein timescale | t_E | Characteristic event duration | 10-40 days |
| Peak time | t₀ | Time of maximum magnification | Uniformly randomized |

### Core Experimental Series

#### Experiment 1: Baseline Performance (1M events)

**Objective**: Establish classification benchmark on Roman-quality photometry across full binary parameter space.

**Configuration**:
```
N = 1,000,000 events (333k Baseline / 333k PSPL / 334k Binary)

Binary Parameter Ranges:
  s: 0.1 - 3.0      # Complete caustic geometry range
  q: 10⁻⁴ - 1.0     # Planetary to equal-mass binaries  
  u₀: 0.001 - 1.0   # Full impact parameter space
  t_E: 10-40 days   # Typical bulge timescales

Observational Configuration:
  Missing fraction: 5% (uniform random gaps)
  Photometric error: 0.05 mag (Gaussian)
  Monitoring duration: 240 days (centered on t₀)
  Temporal sampling: Roman-like cadence
```

**Success Criteria**:
- Overall three-class accuracy > 80%
- PSPL recall > 75%
- Binary recall > 77% (for events with u₀ < 0.3)
- Inference latency < 1 ms per event

**Hypothesis**: A well-trained transformer architecture should achieve >80% classification accuracy on Roman-quality data, with the primary confusion occurring between PSPL and high-u₀ binary events (physically similar light curves).

#### Experiment 2: Distinct Topology (150k events)

**Objective**: Measure optimal detection performance when source trajectories cross well-separated caustics.

**Configuration**:
```
Focus on favorable caustic configurations:
  s: 0.7 - 1.5      # Central and planetary caustics well-separated
  q: 0.01 - 0.5     # Moderate mass ratios
  u₀: 0.001 - 0.15  # Guaranteed caustic crossings

Expected Performance:
  Binary recall: 87-92%
  Overall accuracy: 85-88%
```

**Scientific Goal**: Establish upper performance bound under ideal geometric conditions. This topology represents the most detectable binary lens configurations where both central and planetary caustics produce distinct light curve features.

#### Experiment 3: Planetary Topology (150k events)

**Objective**: Assess exoplanet detection capability via microlensing with Roman cadence.

**Configuration**:
```
Planetary mass ratio regime:
  s: 0.5 - 2.0      # Planet within Einstein ring
  q: 10⁻⁴ - 0.01    # Earth to super-Jupiter masses
  u₀: 0.001 - 0.3   # Wider u₀ range (planetary caustics smaller)

Expected Performance:
  Binary recall: 82-87%
  Overall accuracy: 81-84%
```

**Scientific Goal**: Determine minimum detectable mass ratio q for reliable classification. Brief planetary caustic crossings (hours to 1 day) test Roman's temporal resolution advantage over ground-based surveys.

#### Experiment 4: Stellar Topology (150k events)

**Objective**: Evaluate classification performance for binary star systems with comparable masses.

**Configuration**:
```
Equal-mass binary regime:
  s: 0.3 - 3.0      # Wide range of caustic separations
  q: 0.3 - 1.0      # Comparable primary and companion masses
  u₀: 0.001 - 0.3   # Standard impact parameter range

Expected Performance:
  Binary recall: 78-83%
  Overall accuracy: 78-81%
```

**Scientific Goal**: Assess whether symmetric caustic structures (characteristic of equal-mass systems) present increased classification difficulty compared to asymmetric planetary configurations.

---

## Analysis Framework

### Primary Evaluation Metrics

**1. Three-Class Classification Performance**
- Overall accuracy (macro-averaged)
- Per-class precision, recall, F1-score
- Confusion matrix analysis (identify systematic misclassifications)
- ROC curves and AUC scores (one-vs-rest)

**2. Impact Parameter Dependence (u₀ Analysis)**
- Bin classification accuracy by u₀ (10 bins from 0 to 1.0)
- Identify u₀ threshold where binary recall drops below 60%
- Compare PSPL-Binary confusion rate vs. u₀
- Validate physical vs. algorithmic detection limits

**3. Early Classification Analysis**
- Accuracy as a function of observation completeness (15 fractions: 5%-100%)
- Determine completeness threshold for 70% accuracy (follow-up trigger criterion)
- Measure prediction evolution (stability over time)
- Temporal bias diagnostics (Kolmogorov-Smirnov tests)

**4. Computational Performance**
- Single-event inference latency (milliseconds)
- Batch throughput (events per second)
- Memory footprint (GB per GPU)
- Scaling efficiency (multi-GPU speedup)

### Physical Detection Limit Hypothesis

**u₀ Threshold Prediction**: We hypothesize a sharp detectability threshold at u₀ ≈ 0.3, beyond which classification accuracy approaches random guessing (~50% between PSPL and Binary classes).

**Physical Basis**: For u₀ > 0.3, source trajectories do not intersect caustic curves. The resulting magnification profiles become morphologically indistinguishable from single-lens (Paczynski) light curves, independent of binary parameters (s, q). This represents a fundamental geometric limit, not a machine learning deficiency.

**Validation Test**: Consistent u₀ ≈ 0.3 threshold across all topology experiments (distinct, planetary, stellar) would confirm geometric caustic-crossing requirement rather than topology-specific training artifacts.

---

## Neural Network Architecture

### Transformer Model (Version 1.0)

The architecture implements a causal transformer encoder with temporal encoding specialized for irregular astronomical time series.

```
Input: Flux [B, N], Time Intervals Δt [B, N]
  │
  ├─ Flux Embedding: Linear(1 → d_model=128)
  │   • Maps normalized flux to latent space
  │
  └─ Temporal Encoding: Adaptive log-normalized Δt
      • Learn distribution during training
      • Warn on out-of-distribution inputs
  │
  ↓ Element-wise sum: flux_emb + temp_emb
  │
Transformer Encoder Blocks (×4 layers)
  │
  ├─ Multi-Head Self-Attention (8 heads, head_dim=16)
  │   • Semi-causal mask (sliding window size 64)
  │   • No future observation access
  │   • Q, K, V projections with residual connections
  │
  ├─ Feed-Forward Network (expansion factor 4×)
  │   • Linear(128 → 512) + GELU
  │   • Linear(512 → 128)
  │   • Residual connection
  │
  └─ Layer Normalization (pre-norm configuration)
  │
  ↓ Apply to all 4 layers sequentially
  │
Global Pooling Layer
  │
  ├─ Average pooling (over valid observations)
  ├─ Max pooling (over valid observations)
  └─ Concatenate → [B, 2×d_model]
  │
  ↓
Classification Head (2-layer MLP)
  │
  ├─ Hidden layer: Linear(256 → 128) + GELU + Dropout(0.1)
  └─ Output layer: Linear(128 → 3)
  │
  ↓
Output
  ├─ Logits: [B, 3] (class scores)
  ├─ Probabilities: [B, 3] (temperature-calibrated softmax)
  └─ Confidence: [B] (maximum probability)
```

**Model Specifications**:
- Total parameters: ~808,000 (all trainable)
- Embedding dimension: d_model = 128
- Attention heads: 8 (head dimension = 16)
- Encoder layers: 4
- Feed-forward dimension: 512 (4× expansion)
- Attention window: 64 observations (causal)
- Dropout rate: 0.1
- Maximum sequence length: 1500 observations

### Key Design Features

**Adaptive Temporal Encoding**:
- Tracks distribution of observation intervals (Δt) during training
- Applies log-scale normalization with 10% margin for robustness
- Issues warnings during inference on out-of-distribution time intervals
- Prevents temporal shortcuts by encoding relative intervals, not absolute timestamps

**Semi-Causal Attention**:
- Sliding window mechanism (size W=64) limits computational complexity to O(N×W)
- Masks future observations to enable real-time classification
- Maintains causality: predictions at time t use only observations ≤ t
- Supports incremental processing with key-value cache

**Streaming Inference**:
- Processes observations incrementally as they arrive
- Maintains internal state (cached keys and values)
- Updates classification probabilities with each new observation
- Enables real-time decision making in survey pipelines

**Uncertainty Quantification**:
- Temperature scaling calibration on validation set
- Confidence scores derived from softmax probabilities
- Expected calibration error (ECE) measurement
- Enables threshold-based follow-up triggers

---

## Expected Results and Interpretation

### Performance Benchmarks

**Baseline Experiment (1M events, Roman-quality)**:
- Overall accuracy: 80-83%
- PSPL precision/recall: 88-90% / 87-90%
- Binary precision/recall: 68-72% / 88-92%
- Training time: 3-5 hours (32× A100 GPUs)
- Inference throughput: >10,000 events/sec (single GPU)

**Topology-Dependent Performance**:

| Topology | Binary Recall | Overall Accuracy | Primary Challenge |
|----------|---------------|------------------|-------------------|
| Distinct | 87-92% | 85-88% | Minimal (optimal geometry) |
| Planetary | 82-87% | 81-84% | Brief caustic features |
| Stellar | 78-83% | 78-81% | Symmetric caustics |

**u₀ Dependency** (expected across all topologies):
- u₀ < 0.15: Binary recall >85% (clear caustic crossings)
- u₀ = 0.15-0.3: Binary recall 70-80% (marginal caustic features)
- u₀ > 0.3: Binary recall <60% (morphologically PSPL-like)

**Early Classification** (typical learning curve):
- 50% completeness: 75-80% accuracy (sufficient for follow-up trigger)
- 25% completeness: 60-65% accuracy (early alert capability)
- 10% completeness: 40-45% accuracy (insufficient data)

### Astrophysical Implications

**Survey Automation**: 
Classification throughput >10,000 events/sec enables real-time processing during peak galactic bulge observing season. Roman Space Telescope could process ~10⁶ events over 5-year mission with automated binary identification.

**Exoplanet Detection**: 
Binary recall >82% in planetary regime (q ~ 10⁻⁴ to 10⁻³) demonstrates Roman's capability for systematic planetary companion surveys via microlensing. Detection threshold of q ~ 10⁻⁴ corresponds to Earth-to-Jupiter mass planets.

**Follow-up Prioritization**: 
Early classification at 50% completeness (typical at t₀ - 20 days for t_E = 30 days) enables targeted high-resolution spectroscopy (Keck, VLT) before peak magnification. This increases yield of detailed mass measurements.

**Physical Detection Limits**: 
Consistent u₀ ≈ 0.3 threshold across topologies validates geometric caustic-crossing interpretation. This quantifies the fraction of binary lens events detectable by any method, regardless of algorithm sophistication.

---

## Experimental Workflow

### Phase 1: Rapid Validation (30 minutes, single GPU)

Purpose: Verify complete pipeline functionality before large-scale experiments.

```bash
cd code

# Generate minimal test dataset (300 events)
python simulate.py --preset quick_test

# Train for 5 epochs (convergence not expected)
python train.py \
    --data ../data/raw/quick_test.npz \
    --experiment_name quick_test_validation \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3

# Run evaluation
python evaluate.py \
    --experiment_name quick_test_validation \
    --data ../data/raw/quick_test.npz \
    --n_samples 300

# Expected: 60-70% accuracy (limited by dataset size)
# Goal: Verify all scripts execute without errors
```

### Phase 2: Baseline Experiment (5 hours, 32 GPUs)

Purpose: Establish primary performance benchmark on full parameter space.

```bash
cd code

# Step 1: Generate 1M event dataset
python simulate.py --preset baseline_1M
# Output: ../data/raw/baseline_1M.npz (~4 GB)
# Parameters saved: ../data/raw/baseline_1M_params.npz

# Step 2: Allocate computational resources
salloc --partition=gpu_a100_short \
       --nodes=8 \
       --gres=gpu:4 \
       --exclusive \
       --time=05:00:00

# Step 3: Configure distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Step 4: Distributed training (DDP)
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=baseline_$(date +%s) \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-3 \
        --weight_decay 1e-3 \
        --warmup_epochs 5 \
        --patience 15

# Step 5: Comprehensive evaluation
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --batch_size 128 \
    --n_samples 50000 \
    --early_detection \
    --temporal_bias_check \
    --n_evolution_per_type 10 \
    --u0_threshold 0.3 \
    --u0_bins 10

# Step 6: Verify outputs
ls results/baseline_1M_*/evaluation/
# Expected files:
#   - confusion_matrix.png
#   - roc_curves.png
#   - confidence_distribution.png
#   - calibration_curve.png
#   - u0_dependency.png (if parameters available)
#   - early_detection_analysis.png
#   - evolution_trajectories.png
#   - evaluation_summary.json
#   - u0_report.json
```

### Phase 3: Topology Studies (3 hours each, 32 GPUs)

Purpose: Systematic parameter space mapping.

```bash
# Execute for each topology: distinct, planetary, stellar
for topology in distinct planetary stellar; do
    
    # Generate topology-specific dataset
    python simulate.py --preset ${topology} \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --output ../data/raw/topology_${topology}.npz \
        --save_params \
        --seed 42
    
    # Distributed training (same configuration as baseline)
    srun torchrun \
        --nnodes=8 \
        --nproc_per_node=4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_id=${topology}_$(date +%s) \
        train.py \
            --data ../data/raw/topology_${topology}.npz \
            --experiment_name topology_${topology} \
            --epochs 50 \
            --batch_size 64
    
    # Comprehensive evaluation
    python evaluate.py \
        --experiment_name topology_${topology} \
        --data ../data/raw/topology_${topology}.npz \
        --early_detection \
        --temporal_bias_check \
        --n_evolution_per_type 10
    
done
```

### Phase 4: Comparative Analysis

Purpose: Aggregate results and generate thesis figures.

```bash
cd results

# Extract performance summary table
python -c "
import json
from pathlib import Path
import numpy as np

experiments = ['baseline_1M', 'topology_distinct', 'topology_planetary', 'topology_stellar']

print(f'{'Experiment':<20} {'Overall':<8} {'PSPL':<8} {'Binary':<8} {'Training (h)':<12}')
print('-' * 70)

for exp in experiments:
    runs = sorted(Path('.').glob(f'{exp}_*'))
    if runs:
        latest_run = runs[-1]
        summary_file = latest_run / 'evaluation' / 'evaluation_summary.json'
        config_file = latest_run / 'config.json'
        
        if summary_file.exists() and config_file.exists():
            summary = json.load(open(summary_file))
            config = json.load(open(config_file))
            
            metrics = summary['metrics']
            overall_acc = metrics['accuracy'] * 100
            pspl_recall = metrics.get('class_1_recall', 0) * 100
            binary_recall = metrics.get('class_2_recall', 0) * 100
            
            # Estimate training time from config (epochs * samples / throughput)
            epochs = config.get('epochs', 50)
            training_hours = 'N/A'  # Would need to parse from logs
            
            print(f'{exp:<20} {overall_acc:>6.1f}% {pspl_recall:>6.1f}% {binary_recall:>6.1f}% {training_hours:>10}')
"

# Generate u₀ comparison plot across topologies
python -c "
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

topologies = ['distinct', 'planetary', 'stellar']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, topology in enumerate(topologies):
    runs = sorted(Path('.').glob(f'topology_{topology}_*'))
    if runs:
        u0_file = runs[-1] / 'evaluation' / 'u0_report.json'
        if u0_file.exists():
            data = json.load(open(u0_file))
            u0_centers = data['u0_bin_centers']
            binary_accuracies = data['binary_accuracy_by_u0']
            
            axes[idx].plot(u0_centers, binary_accuracies, 'o-', linewidth=2)
            axes[idx].axvline(x=0.3, color='r', linestyle='--', label='u₀=0.3 threshold')
            axes[idx].set_xlabel('Impact Parameter u₀')
            axes[idx].set_ylabel('Binary Classification Accuracy')
            axes[idx].set_title(f'{topology.capitalize()} Topology')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
            axes[idx].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('u0_comparison_across_topologies.png', dpi=300, bbox_inches='tight')
print('Saved: u0_comparison_across_topologies.png')
"
```

---

## Visualization and Diagnostic Analysis

### Purpose

Visual inspection of model internals provides insights that aggregate metrics cannot capture:
- Attention mechanism behavior (temporal dependencies)
- Classification decision dynamics (evolution over observations)
- Embedding space structure (class separability)
- Confidence calibration (prediction certainty)

### Workflow

**Phase 5: Model Visualization (30 minutes per experiment)**

```bash
cd code

# Visualize trained model
python visualize_transformer.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --output_dir ../results/visualizations/baseline_1M \
    --n_examples 3 \
    --event_indices 42 137 289

# Expected outputs (per event):
#   - attention_patterns_event*.png
#   - temporal_encoding_event*.png
#   - classification_evolution_event*.png
#
# Global outputs:
#   - binary_vs_pspl_comparison.png
#   - embedding_space_pca.png
#   - confidence_evolution_by_class.png
```

### Diagnostic Checklist

**1. Attention Pattern Verification**

Examine `attention_patterns_event*.png` for each layer:

- [ ] Causal boundary visible (no future observation access)
- [ ] Diagonal dominance (temporal locality)
- [ ] Smooth attention distribution (no isolated spikes)
- [ ] Layer-wise evolution (deeper layers more global)

**Red flags:**
- Uniform attention (not learning temporal structure)
- Strong off-diagonal without physical justification
- Layer-independent patterns (insufficient depth)

**2. Temporal Encoding Validation**

Examine `temporal_encoding_event*.png`:

- [ ] Encoding dimensions vary smoothly with time
- [ ] PCA shows temporal ordering (not random)
- [ ] Handles irregular observation intervals
- [ ] No discontinuities or artifacts

**Red flags:**
- Flat encoding dimensions (not using temporal information)
- Random PCA structure (temporal ordering lost)
- Sensitivity to small interval perturbations

**3. Classification Dynamics**

Examine `classification_evolution_event*.png`:

- [ ] Early convergence for clear events (< 50% observations)
- [ ] Stable predictions after convergence
- [ ] Confidence increases monotonically
- [ ] Correct class probability dominates

**Red flags:**
- Late flip-flopping (unstable predictions)
- Confidence peaks early then drops
- Incorrect class probability increasing

**4. Class Discrimination**

Examine `binary_vs_pspl_comparison.png`:

- [ ] PSPL events maintain low binary probability
- [ ] Binary events achieve high binary probability
- [ ] Clear separation by 50% completeness
- [ ] Smooth probability evolution (not jumpy)

**Red flags:**
- PSPL events misclassified as binary
- Binary events remain ambiguous throughout
- Noisy probability trajectories

**5. Embedding Space Structure**

Examine `embedding_space_pca.png`:

- [ ] Three visible clusters (Flat, PSPL, Binary)
- [ ] Flat well-separated from lensing classes
- [ ] PSPL-Binary overlap acceptable (physical similarity)
- [ ] PC1 explains > 20% variance

**Red flags:**
- No visible clustering
- Flat overlaps with lensing events
- Single dominant cluster

### Integration with Thesis

**Figure Selection Criteria:**

For thesis inclusion, select visualizations that:

1. **Demonstrate attention mechanism**: Show causal masking and temporal locality
2. **Illustrate classification dynamics**: Binary event with clear caustic features vs. PSPL-like binary
3. **Validate design choices**: Temporal encoding handling irregular cadences
4. **Show failure modes**: High-u₀ binary event misclassified as PSPL (physical limit)

**Recommended figures:**

- Figure 4.X: Attention patterns across all layers for representative binary event
- Figure 4.Y: Classification evolution comparison (distinct vs. challenging binary)
- Figure 4.Z: Embedding space PCA showing class clustering
- Figure 5.X: Failure case analysis (u₀ > 0.3 binary classified as PSPL)

### Troubleshooting

**Visualization crashes or errors:**

```bash
# Check model checkpoint exists
ls results/experiment_name_*/best_model.pt

# Verify data compatibility
python -c "
import numpy as np
data = np.load('../data/raw/test.npz')
print('Keys:', list(data.keys()))
print('Flux shape:', data['flux'].shape)
"

# Run on CPU if GPU memory issues
python visualize_transformer.py \
    --experiment_name experiment \
    --data ../data/raw/test.npz \
    --no_cuda
```

**Missing visualizations:**

If certain plots are not generated:
- `--no_attention`: Attention visualizations skipped (intended)
- `--no_embedding`: Embedding space skipped (intended)
- Errors during generation: Check console output for warnings

**Performance optimization:**

For faster iteration during analysis:
```bash
# Skip attention computation (slowest component)
--no_attention

# Reduce number of examples
--n_examples 1

# Focus on specific events
--event_indices 42
```

---

## Troubleshooting

### Training Diagnostics

**GPU Out-of-Memory Errors**:
```bash
# Reduce batch size
--batch_size 32  # Down from default 64

# Enable gradient checkpointing (trades compute for memory)
--gradient_checkpointing

# Disable mixed precision (rare cases)
--no_amp
```

**Loss Divergence (NaN or Inf)**:
- Check data normalization (flux should be mean-centered, unit variance)
- Verify temporal encoding range (Δt should be positive)
- Reduce learning rate: `--lr 5e-4`
- Enable gradient clipping (automatic in train.py)
- Code automatically skips NaN batches; if >10% skipped, investigate data quality

**Distributed Training Hangs**:
```bash
# Verify master node is accessible
echo $MASTER_ADDR
ping $MASTER_ADDR

# Test SLURM job communication
srun --nodes=2 hostname

# Increase NCCL timeout for slow interconnects
export NCCL_TIMEOUT=3600

# Check for firewall issues blocking ports
# Default MASTER_PORT=29500 should be open

# Verify CUDA devices visible
srun --nodes=2 --gres=gpu:4 python -c "import torch; print(torch.cuda.device_count())"
```

**Slow Training Convergence**:
- Increase learning rate: `--lr 3e-3` (but watch for instability)
- Extend warmup: `--warmup_epochs 10`
- Verify class balance in dataset
- Check if validation accuracy improves (may need more epochs)

### Evaluation Diagnostics

**Missing u₀ Analysis Plots**:
```bash
# Verify physical parameters were saved during simulation
python -c "
import numpy as np
data = np.load('../data/raw/your_dataset.npz')
print('Available keys:', data.files)
print('Has params_binary_json:', 'params_binary_json' in data.files)
"

# If missing, regenerate dataset with --save_params flag
python simulate.py --preset your_preset \
    --save_params \
    --output ../data/raw/your_dataset.npz
```

**Calibration Warnings**:
- Temperature calibration may fail if validation set too small (<1000 samples)
- Not critical for classification accuracy, only affects confidence scores
- Can disable with `--skip_calibration` flag

**Memory Issues During Evaluation**:
```bash
# Reduce batch size
--batch_size 64  # Down from default 128

# Subsample test set for faster iteration
--n_samples 10000  # Instead of full test set

# Disable memory-intensive analyses
--no_early_detection
--no_temporal_bias_check
```

### Data Quality Checks

**Verify Dataset Integrity**:
```bash
python -c "
import numpy as np

data = np.load('../data/raw/your_dataset.npz')
flux = data['flux']
delta_t = data['delta_t']
labels = data['labels']

print(f'Flux shape: {flux.shape}')
print(f'Flux range: [{flux.min():.3f}, {flux.max():.3f}]')
print(f'NaN count: {np.isnan(flux).sum()}')
print(f'Inf count: {np.isinf(flux).sum()}')
print()
print(f'Delta_t range: [{delta_t.min():.3f}, {delta_t.max():.3f}]')
print(f'Negative delta_t: {(delta_t < 0).sum()}')
print()
print(f'Label distribution:')
for c in range(3):
    count = (labels == c).sum()
    print(f'  Class {c}: {count} ({100*count/len(labels):.1f}%)')
"
```

---

## Additional Experimental Directions

### Cadence Sensitivity Study

Purpose: Quantify classification robustness to observation frequency.

```bash
# Generate datasets with varying missing fractions
for cadence in 05 15 30 50; do
    python simulate.py --preset cadence_${cadence} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/raw/cadence_${cadence}.npz
done

# Train and evaluate each
for cadence in 05 15 30 50; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/cadence_${cadence}.npz \
        --experiment_name cadence_${cadence} \
        --epochs 50
    
    python evaluate.py \
        --experiment_name cadence_${cadence} \
        --data ../data/raw/cadence_${cadence}.npz
done
```

Expected trend: Accuracy degrades with increased missing fraction
- 5% missing: 80-83% (Roman baseline)
- 15% missing: 75-78% (high-cadence ground)
- 30% missing: 68-72% (typical ground)
- 50% missing: 55-62% (sparse monitoring)

### Photometric Error Study

Purpose: Determine noise tolerance limits.

```bash
# Generate datasets with varying photometric precision
for error in 003 005 010 015 020; do
    python simulate.py --preset error_${error} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/raw/error_${error}.npz
done

# Train and evaluate
for error in 003 005 010 015 020; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/error_${error}.npz \
        --experiment_name error_${error} \
        --epochs 50
    
    python evaluate.py \
        --experiment_name error_${error} \
        --data ../data/raw/error_${error}.npz
done
```

Expected trend: Accuracy degrades with increased noise
- 0.03 mag: 82-85% (JWST-quality)
- 0.05 mag: 80-83% (Roman, HST)
- 0.10 mag: 75-78% (large ground telescopes)
- 0.15 mag: 68-72% (standard surveys)
- 0.20 mag: 60-65% (limiting case)

---

## References

### Key Publications

1. **VBBinaryLensing**: Bozza, V. (2010). "VBBinaryLensing: A public available C++ library for microlensing light curve computation." MNRAS, 408, 2188-2196.

2. **Roman Exoplanet Survey**: Johnson, S. A., et al. (2020). "Predictions of the Nancy Grace Roman Space Telescope Galactic Exoplanet Survey. II. Free-floating Planet Detection Rates." AJ, 160, 123.

3. **Binary Lens Theory**: Schneider, P., & Weiss, A. (1986). "The gravitational lens equation near cusps." A&A, 164, 237-259.

4. **OGLE Survey Results**: Udalski, A., et al. (2015). "OGLE-IV: Fourth Phase of the Optical Gravitational Lensing Experiment." Acta Astronomica, 65, 1-38.

### Survey Resources

- **OGLE**: Optical Gravitational Lensing Experiment (http://ogle.astrouw.edu.pl/)
- **MOA**: Microlensing Observations in Astrophysics (https://www.massey.ac.nz/~iabond/moa/)
- **KMTNet**: Korea Microlensing Telescope Network (https://kmtnet.kasi.re.kr/)
- **Roman**: Nancy Grace Roman Space Telescope (https://roman.gsfc.nasa.gov/)

---

## Thesis Integration

### Chapter Organization

**Chapter 3: Methodology**
- Section 3.1: Dataset generation (VBBinaryLensing simulation protocol)
- Section 3.2: Neural network architecture (transformer encoder design)
- Section 3.3: Training procedure (distributed optimization, hyperparameters)
- Section 3.4: Evaluation framework (metrics, u₀ analysis, early detection)

**Chapter 4: Results**
- Section 4.1: Baseline performance (1M event benchmark)
- Section 4.2: Topology dependence (distinct, planetary, stellar comparison)
- Section 4.3: Physical detection limits (u₀ threshold validation)
- Section 4.4: Early classification (completeness-accuracy trade-off)
- Section 4.5: Computational performance (inference latency, throughput)

**Chapter 5: Discussion**
- Section 5.1: Comparison with literature (previous ML approaches)
- Section 5.2: Astrophysical implications (survey automation, exoplanet detection)
- Section 5.3: Limitations (degeneracies, noise sensitivity)
- Section 5.4: Future work (real data validation, architecture improvements)

### Key Figures for Thesis

1. **Dataset schematic**: Example light curves for each class (Flat, PSPL, Binary) with parameter annotations
2. **Architecture diagram**: Transformer encoder with attention mask visualization
3. **Baseline confusion matrix**: Three-class classification results on 1M event test set
4. **Topology comparison**: Bar chart of binary recall across distinct/planetary/stellar experiments
5. **u₀ dependency**: Binary classification accuracy vs. impact parameter (all topologies overlaid)
6. **Early detection**: Accuracy vs. observation completeness with confidence intervals
7. **Computational scaling**: Inference throughput vs. batch size, training time vs. dataset size

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Active research