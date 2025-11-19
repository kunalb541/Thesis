## From Light Curves to Labels: Machine Learning in Microlensing

**Transformer-based Real-time Classification of Gravitational Microlensing Events**

MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a transformer-based neural network for real-time classification of gravitational microlensing events, capable of distinguishing between single-lens (PSPL) and binary-lens events with sub-millisecond inference latency. The architecture addresses key challenges in time-series astronomical data: variable observation cadences, photometric noise, and the need for causal inference without temporal shortcuts.

### Classification Task

**Three-class discrimination:**
- **Class 0**: Baseline (no lensing event)
- **Class 1**: Point Source Point Lens (PSPL, single lens)
- **Class 2**: Binary lens (planetary or stellar companion)

---

## Installation

### Environment Setup

```bash
# Clone repository
git clone https://github.com/kunalb541/Thesis.git
cd Thesis

# Create conda environment
conda env create -f environment.yml
conda activate microlens
```

### GPU Configuration

**AMD GPUs (ROCm 6.0)** - MI300 series:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

**NVIDIA GPUs (CUDA 12.1)** - A100, H100:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only** (not recommended for production):
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Test Workflow

### Minimal Validation (5 minutes, single GPU)

This workflow validates the complete pipeline on a minimal dataset:

```bash
cd code

# Step 1: Generate test dataset (300 events)
python simulate.py --preset quick_test

# Step 2: Train model (5 epochs)
python train.py \
    --data ../data/raw/quick_test.npz \
    --experiment_name quick_test \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3

# Step 3: Evaluate performance
python evaluate.py \
    --experiment_name quick_test \
    --data ../data/raw/quick_test.npz \
    --n_samples 300
```

**Expected output:**
- Training: ~60 seconds
- Validation accuracy: 75-85% (limited by small dataset)
- Generated artifacts: confusion matrix, ROC curves, example classifications

### Small-Scale Test (30 minutes, 4 GPUs)

```bash
# Allocate GPU resources (SLURM example)
salloc --partition=gpu --nodes=1 --gres=gpu:4 --time=01:00:00

# Setup environment
cd ~/Thesis/code
conda activate microlens
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Generate 9K event dataset
python simulate.py \
    --n_flat 3000 --n_pspl 3000 --n_binary 3000 \
    --binary_preset distinct \
    --cadence_mask_prob 0.05 \
    --output ../data/raw/test_9k.npz \
    --seed 42

# Distributed training
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --data ../data/raw/test_9k.npz \
        --experiment_name test_9k \
        --epochs 20 \
        --batch_size 64

# Comprehensive evaluation
python evaluate.py \
    --experiment_name test_9k \
    --data ../data/raw/test_9k.npz \
    --early_detection \
    --temporal_bias_check \
    --n_evolution_per_type 10
```

**Expected performance (distinct topology):**
- Overall accuracy: 80-85%
- Binary precision: 65-75%
- Binary recall: 85-90%
- Inference time: <1 ms/event

---

## Architecture

### Transformer Model (Version 1.0)

The architecture implements a causal transformer with specialized components for astronomical time-series:

```
Input: [B, N]
  ├─ Flux: Normalized light curve measurements
  └─ Δt: Time intervals between observations (days)
    ↓
Embedding Layer
  ├─ Flux embedding: Linear(1 → d_model)
  └─ Temporal encoding: Adaptive normalization + Linear(1 → d_model)
    ↓
Transformer Encoder (×4 layers)
  ├─ Semi-causal attention (window size: 64)
  │  • No future observation peeking
  │  • Multi-head attention (8 heads)
  │  • Head dimension: d_model / n_heads
  ├─ Feed-forward network (4× expansion)
  └─ Layer normalization + residual connections
    ↓
Global Pooling
  ├─ Average pooling over valid observations
  └─ Max pooling over valid observations
  └─ Concatenation: [avg; max] → 2×d_model
    ↓
Classification Head
  ├─ Hidden layer: Linear(2×d_model → d_model) + GELU
  └─ Output layer: Linear(d_model → 3 classes)
    ↓
Output
  ├─ Logits: [B, 3] (class scores)
  ├─ Probabilities: [B, 3] (softmax)
  └─ Confidence: [B] (max probability)
```

**Model specifications:**
- Parameters: ~808,000 (trainable)
- d_model: 128
- n_heads: 8
- n_layers: 4
- Feed-forward dimension: 512
- Attention window: 64 observations
- Dropout: 0.1

### Key Design Features

**1. Adaptive Temporal Encoding**
- Learns distribution of observation intervals during training
- Applies log-scale normalization with 10% margin
- Warns on out-of-distribution inputs during inference
- Prevents temporal shortcuts through relative encoding

**2. Semi-Causal Attention**
- Implements sliding window attention (size 64)
- Masks future observations to enable real-time classification
- Maintains O(N×W) complexity instead of O(N²)
- Supports streaming inference with KV caching

**3. Streaming Inference**
- Processes observations incrementally
- Maintains internal state (key-value caches)
- Updates predictions as new data arrives
- Enables real-time decision making

**4. Uncertainty Quantification**
- Temperature scaling for probability calibration
- Post-hoc calibration on validation set
- Confidence scores based on maximum softmax probability

---

## Data Generation

### Simulation Parameters

The `simulate.py` script generates synthetic microlensing events using VBBinaryLensing:

```bash
python simulate.py \
    --n_flat 50000 \
    --n_pspl 50000 \
    --n_binary 50000 \
    --binary_preset [distinct|planetary|stellar|baseline] \
    --cadence_mask_prob 0.05 \
    --mag_error_std 0.05 \
    --output ../data/raw/dataset.npz \
    --save_params \
    --seed 42
```

**Key arguments:**
- `--n_flat/pspl/binary`: Number of events per class
- `--binary_preset`: Binary lens topology configuration
- `--cadence_mask_prob`: Fraction of missing observations
- `--mag_error_std`: Photometric uncertainty (magnitudes)
- `--save_params`: Store physical parameters for analysis
- `--num_workers`: Parallel simulation workers

### Binary Lens Topologies

| Preset | Mass Ratio (q) | Separation (s) | Description |
|--------|----------------|----------------|-------------|
| `distinct` | 10⁻⁴ - 10⁻¹ | 0.3 - 3.0 | Clear caustic structures |
| `planetary` | 10⁻⁵ - 10⁻³ | 0.5 - 2.0 | Exoplanet detection regime |
| `stellar` | 0.1 - 1.0 | 0.5 - 2.5 | Binary star systems |
| `baseline` | 10⁻⁵ - 1.0 | 0.3 - 5.0 | Full parameter space |

### Observational Configurations

**Cadence studies** (observation frequency):

| Preset | Missing % | Typical Δt | Facility |
|--------|-----------|------------|----------|
| `cadence_05` | 5% | ~15 min | Roman Space Telescope |
| `cadence_15` | 15% | ~1 day | High-cadence ground |
| `cadence_30` | 30% | ~3 days | Typical ground surveys |
| `cadence_50` | 50% | ~5 days | Sparse monitoring |

**Photometric error studies**:

| Preset | σ (mag) | Facility |
|--------|---------|----------|
| `error_003` | 0.03 | JWST |
| `error_005` | 0.05 | Roman, HST |
| `error_010` | 0.10 | Large ground telescopes |
| `error_015` | 0.15 | Standard ground surveys |

---

## Training

### Single GPU Training

```bash
python train.py \
    --data ../data/raw/dataset.npz \
    --experiment_name experiment_name \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --warmup_epochs 5 \
    --patience 15
```

### Distributed Training (Multi-GPU)

**Single node, multiple GPUs:**
```bash
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/dataset.npz \
    --experiment_name multi_gpu \
    --batch_size 64
```

**Multi-node training (SLURM):**
```bash
salloc --partition=gpu --nodes=8 --gres=gpu:4 --time=05:00:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

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
```

### Training Features

**Optimization:**
- AdamW optimizer (β₁=0.9, β₂=0.999)
- Cosine annealing learning rate schedule
- Linear warmup (default: 5 epochs)
- Gradient clipping (max norm: 1.0)
- Mixed precision training (automatic)
- Class-balanced loss weighting

**Regularization:**
- Weight decay: 1e-3
- Dropout: 0.1
- Early stopping (patience: 15 epochs)

**Distributed training:**
- DistributedDataParallel (DDP)
- Gradient aggregation across GPUs
- Synchronized batch normalization
- Temporal encoding distribution tracking

---

## Evaluation

### Comprehensive Analysis

```bash
python evaluate.py \
    --experiment_name experiment_name \
    --data ../data/raw/test_dataset.npz \
    --batch_size 128 \
    --n_samples 50000 \
    --early_detection \
    --temporal_bias_check \
    --n_evolution_per_type 10 \
    --u0_threshold 0.3 \
    --u0_bins 10
```

### Generated Outputs

**Diagnostic plots:**
1. ROC curves (one-vs-rest, per-class AUC)
2. Confusion matrix (absolute counts and normalized)
3. Confidence score distributions
4. Calibration curves (reliability diagrams)
5. Example classifications per class
6. Evolution trajectories (accuracy vs. observation count)
7. Early detection analysis (accuracy vs. completeness)
8. Temporal bias diagnostics (position of observations)

**Quantitative metrics:**
- Overall accuracy, precision, recall, F1-score
- Per-class metrics
- Confidence calibration error
- Expected calibration error (ECE)
- u₀ dependency (impact parameter analysis, if parameters available)

**Output files:**
- `evaluation_summary.json`: Complete metrics
- `u0_report.json`: Impact parameter analysis
- `config.json`: Experiment configuration
- `*.png`: All diagnostic plots

### Early Detection Analysis

Evaluates classification accuracy as a function of observation completeness:

```bash
python evaluate.py \
    --experiment_name experiment \
    --data ../data/raw/test.npz \
    --early_detection \
    --n_evolution_per_type 10
```

Tests 15 completeness fractions: [5%, 10%, 15%, ..., 95%, 100%]

**Typical performance:**
- 50% completeness: 75-80% accuracy
- 25% completeness: 55-65% accuracy
- 10% completeness: ~40% accuracy

### Physical Detection Limits

**Impact parameter (u₀) analysis:**

The minimum separation between source and lens (u₀) fundamentally limits binary lens detectability:

- **u₀ < 0.15**: High accuracy (85-90%) — source crosses caustics
- **u₀ ~ 0.2-0.3**: Moderate accuracy (70-80%) — caustic signatures present
- **u₀ > 0.3**: Low accuracy (55-60%) — morphologically PSPL-like

This threshold reflects astrophysical reality: distant sources do not exhibit binary lens signatures regardless of algorithm sophistication.

---

## Visualization

### Model Internals and Diagnostic Analysis

The `visualize_transformer.py` script provides comprehensive visualization of model behavior and internal representations:

```bash
python visualize_transformer.py \
    --experiment_name experiment_name \
    --data ../data/raw/test.npz \
    --output_dir ../results/visualizations \
    --n_examples 2
```

**Key arguments:**
- `--experiment_name`: Trained model experiment directory
- `--data`: Test dataset for visualization
- `--output_dir`: Directory for output plots
- `--event_indices`: Specific events to visualize (optional)
- `--n_examples`: Number of examples per class (default: 2)
- `--no_attention`: Skip attention visualization (faster)
- `--no_embedding`: Skip embedding space visualization

### Generated Visualizations

**Per-event diagnostics:**

1. **Attention patterns** (`attention_patterns_event*.png`)
   - Attention matrices for each transformer layer
   - Temporal attention distribution across observations
   - Visualization of causal masking structure
   - Average attention received by each time step

2. **Temporal encoding** (`temporal_encoding_event*.png`)
   - Light curve in magnitude space
   - Observation interval distribution
   - Temporal encoding dimensions (first 6 components)
   - PCA projection of temporal representations

3. **Classification evolution** (`classification_evolution_event*.png`)
   - Class probabilities vs. observation completeness (100 points)
   - Confidence progression over time
   - Light curve with observation markers
   - High-resolution tracking of prediction dynamics

**Global analyses:**

4. **Binary vs PSPL comparison** (`binary_vs_pspl_comparison.png`)
   - Evolution of binary class probability for true PSPL events
   - Evolution of binary class probability for true Binary events
   - Demonstrates discrimination capability between similar classes

5. **Embedding space** (`embedding_space_pca.png`)
   - PCA projection of final layer embeddings
   - Visualization of class clustering in latent space
   - Explained variance by principal components

6. **Confidence evolution by class** (`confidence_evolution_by_class.png`)
   - Mean confidence trajectories for each class
   - Standard deviation bands
   - Comparison across observation completeness

### Example Usage

**Visualize specific events:**
```bash
python visualize_transformer.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --event_indices 42 137 289
```

**Quick visualization without attention (faster):**
```bash
python visualize_transformer.py \
    --experiment_name test_experiment \
    --data ../data/raw/test.npz \
    --no_attention \
    --n_examples 3
```

**CPU execution:**
```bash
python visualize_transformer.py \
    --experiment_name experiment \
    --data ../data/raw/test.npz \
    --no_cuda
```

### Interpretation Guidelines

**Attention patterns:**
- Strong diagonal indicates temporal locality (recent observations most important)
- Off-diagonal attention suggests long-range dependencies
- Causal boundary should be clearly visible (no future peeking)

**Temporal encoding:**
- Smooth encoding dimensions indicate learned temporal structure
- PCA clustering by time suggests proper temporal ordering
- Irregular intervals should be handled gracefully

**Classification evolution:**
- Stable predictions indicate robust classification
- Early convergence suggests strong signal
- Late changes may indicate confusion or ambiguous morphology

**Embedding space:**
- Clear class separation indicates effective feature learning
- Overlap between PSPL and Binary is expected (physical similarity)
- Flat events should be well-separated from lensing classes

---

## Performance Benchmarks

### Baseline Results (1M events, balanced classes)

**Overall performance:**
- Accuracy: 80.5%
- Macro F1-score: 0.78
- Training time: 3-5 hours (32 GPUs)
- Inference: <1 ms/event

**Per-class metrics:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Flat (0) | 0.92 | 0.95 | 0.93 |
| PSPL (1) | 0.88 | 0.90 | 0.89 |
| Binary (2) | 0.70 | 0.90 | 0.79 |

**Topology-dependent accuracy:**

| Topology | Description | Accuracy |
|----------|-------------|----------|
| Distinct | Clear caustics | 83-86% |
| Planetary | q ~ 10⁻⁴ - 10⁻³ | 81-84% |
| Stellar | q ~ 0.1 - 1.0 | 78-81% |
| Baseline | Full range | 73-76% |

### Computational Efficiency

**Training:**
- Throughput: ~10,000 events/sec (32× A100 40GB)
- Memory: ~8 GB/GPU (batch size 64)
- Convergence: 30-50 epochs

**Inference:**
- Single observation processing: <1 ms
- Batch processing: 10,000+ events/sec (single GPU)
- Streaming latency: <2 ms (real-time updates)

---

## Project Structure

```
thesis-microlensing/
├── code/
│   ├── simulate.py          # Dataset generation (VBBinaryLensing)
│   ├── train.py             # Distributed training pipeline
│   ├── evaluate.py          # Comprehensive evaluation suite
│   ├── transformer.py       # Model architecture (v1.0)
│   └── visualize_transformer.py  # Visualization suite
│
├── data/
│   └── raw/                 # Generated datasets (.npz format)
│
├── results/
│   └── experiment_*/        # Training outputs
│       ├── best_model.pt    # Trained model checkpoint
│       ├── config.json      # Experiment configuration
│       ├── normalizer.pkl   # Input normalization parameters
│       ├── results.json     # Training metrics
│       └── evaluation/      # Diagnostic plots and analysis
│
├── visualizations/          # Model visualization outputs
│
├── docs/
│   └── RESEARCH_GUIDE.md    # Experimental protocols
│
├── environment.yml          # Conda environment specification
├── README.md               # This file
├── LICENSE                 # MIT License
└── .gitignore
```

---

## Methodological Notes

### Model Design Considerations

**Temporal encoding:**
- Relative time differences (Δt) instead of absolute timestamps prevent information leakage
- Adaptive normalization ensures robust handling of variable cadences
- Log-scale transformation accommodates wide range of observation intervals

**Causal attention:**
- Semi-causal mask prevents model from accessing future observations
- Sliding window (size 64) balances context and efficiency
- Enables deployment in real-time survey pipelines

**Architecture choices:**
- Transformer selected for: (1) variable-length sequences, (2) long-range dependencies, (3) parallelizable training
- Global pooling aggregates information across valid observations only
- Two-layer classifier provides sufficient capacity without overfitting

### Training Strategy

**Class imbalance:**
- Inverse frequency weighting in loss function
- Balanced sampling during evaluation
- Stratified train-validation-test splits

**Regularization:**
- Early stopping on validation loss (patience: 15 epochs)
- Weight decay prevents overfitting to training topology
- Dropout in attention and feed-forward layers

**Distributed training:**
- Synchronous gradient updates across GPUs
- Learning rate scales with effective batch size
- Temporal encoding distribution tracked across all workers

### Evaluation Methodology

**Stratified sampling:**
- Equal representation of all classes in test set
- Subsampling for fast iteration during development
- Full dataset evaluation for final benchmarks

**Temporal bias diagnostics:**
- Kolmogorov-Smirnov test for observation position uniformity
- Prevents model exploiting t₀ alignment artifacts
- Validates robustness to different event phases

**Uncertainty quantification:**
- Temperature scaling on validation set
- Expected calibration error (ECE) measurement
- Confidence-accuracy correlation analysis

---

## Limitations and Future Work

### Current Limitations

1. **Binary lens degeneracies**: High mass-ratio systems (q → 1) difficult to distinguish from PSPL
2. **Extreme noise regimes**: Performance degrades beyond σ > 0.20 mag
3. **Sparse cadences**: Accuracy drops significantly for >50% missing observations
4. **Training data**: Synthetic simulations may not capture all real-world systematics

### Planned Improvements

1. **Architecture enhancements**:
   - Caustic-aware feature extraction modules
   - Explicit u₀ prediction for binary detection confidence
   - Multi-scale temporal attention mechanisms

2. **Training improvements**:
   - Curriculum learning (simple → complex topologies)
   - Data augmentation (noise injection, gap patterns)
   - Transfer learning from real survey data

3. **Deployment optimization**:
   - Model quantization for faster inference
   - ONNX export for production pipelines
   - Integration with alert systems (e.g., LSST Broker)

---

## Experimental Workflows

### Topology Study

Systematic evaluation across binary lens parameter space:

```bash
# Generate datasets for each topology
for topology in distinct planetary stellar baseline; do
    python simulate.py --preset ${topology} \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --output ../data/raw/topology_${topology}.npz
done

# Train models
for topology in distinct planetary stellar baseline; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/topology_${topology}.npz \
        --experiment_name topology_${topology} \
        --epochs 50
done

# Comparative evaluation
for topology in distinct planetary stellar baseline; do
    python evaluate.py \
        --experiment_name topology_${topology} \
        --data ../data/raw/topology_${topology}.npz
done
```

### Cadence Study

Assess robustness to observation frequency:

```bash
# Generate datasets with varying cadences
for cadence in 05 15 30 50; do
    python simulate.py --preset cadence_${cadence} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/raw/cadence_${cadence}.npz
done

# Train and evaluate
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

### Error Study

Evaluate photometric noise tolerance:

```bash
# Generate datasets with varying noise levels
for error in 003 005 010 015; do
    python simulate.py --preset error_${error} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/raw/error_${error}.npz
done

# Train and evaluate
for error in 003 005 010 015; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/error_${error}.npz \
        --experiment_name error_${error} \
        --epochs 50
    
    python evaluate.py \
        --experiment_name error_${error} \
        --data ../data/raw/error_${error}.npz
done
```

---

## Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={From Light Curves to Labels: Machine Learning in Microlensing},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis},
  note={Transformer-based real-time classification}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Supervisor**: Prof. Dr. Joachim Wambsganß (University of Heidelberg)
- **VBBinaryLensing**: Valerio Bozza
- **Computing resources**: bwHPC (BW 3.0), AMD MI300 and NVIDIA A100 clusters
- **Survey collaborations**: OGLE, MOA, KMTNet teams

---

## Technical Support

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg

**For technical issues:**
- Implementation questions → GitHub Issues
- Methodology questions → See thesis document
- Collaboration inquiries → Contact via institutional email

---

## References

**Key publications:**
1. Bozza, V. (2010). "VBBinaryLensing: A C++ library for microlensing light curve computation." MNRAS, 408, 2188-2196.
2. Zhu, W., et al. (2017). "Mass Measurements of Isolated Objects from Space-based Microlensing." ApJ, 849, L31.
3. Johnson, S. A., et al. (2020). "Predictions of the Nancy Grace Roman Space Telescope Galactic Exoplanet Survey." AJ, 160, 123.

**Survey resources:**
- OGLE: http://ogle.astrouw.edu.pl/
- MOA: https://www.massey.ac.nz/~iabond/moa/
- Nancy Grace Roman Space Telescope: https://roman.gsfc.nasa.gov/
- LSST: https://www.lsst.org/

---

**Version**: 1.0  
**Last updated**: November 2025  
**Status**: Active development