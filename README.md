# Gravitational Microlensing Event Classification

**Causal Hybrid GRU-Transformer for Real-time Binary Lens Detection**

MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a strictly causal hybrid architecture combining GRU and Transformer layers for real-time classification of gravitational microlensing events. The model distinguishes between baseline observations (no lensing), single-lens (PSPL) events, and binary-lens events with sub-millisecond inference latency.

### Key Features

- **Strict Causality**: No future observation peeking through explicit masking and incremental state management
- **Hybrid Architecture**: GRU handles sequential dependencies, Transformer captures long-range patterns
- **Temporal Encoding**: Continuous sinusoidal encoding based on observation intervals (Δt), not absolute timestamps
- **Streaming Inference**: Supports real-time predictions with cached key-value states
- **Anti-Bias Design**: Temporal invariance through relative time encoding and extensive validation checks

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

**CPU only**:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Minimal Test (5 minutes, single GPU)

Validate the complete pipeline on a small dataset:

```bash
cd code

# 1. Generate test dataset (300 events)
python simulate.py --preset quick_test

# 2. Train model (5 epochs)
python train.py \
    --data ../data/dataset.npz \
    --epochs 5 \
    --batch_size 32

# 3. Evaluate
python evaluate.py \
    --experiment_name results_* \
    --data ../data/dataset.npz
```

### Full Workflow

```bash
# 1. Generate dataset
python simulate.py \
    --n_flat 10000 \
    --n_pspl 10000 \
    --n_binary 10000 \
    --binary_preset distinct \
    --output ../data/dataset.npz

# 2. Train model
python train.py \
    --data ../data/dataset.npz \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4

# 3. Evaluate with diagnostics
python evaluate.py \
    --experiment_name results_* \
    --data ../data/dataset.npz

# 4. Optional: Visualize model internals
python visualize.py \
    --experiment_name results_* \
    --data ../data/dataset.npz
```

---

## Architecture

### Causal Hybrid Model

The architecture implements a strictly causal design that prevents temporal information leakage:

```
Input: [B, N]
  ├─ Flux: Light curve measurements (normalized)
  └─ Δt: Time intervals between observations
    ↓
Embedding Layer
  ├─ Flux embedding: Linear(1 → d_model)
  ├─ Layer normalization + Padding mask enforcement
  └─ Temporal encoding: Continuous sinusoidal (Δt-based)
    ↓
GRU Encoder (1-2 layers)
  ├─ Packed sequences (handles variable lengths)
  ├─ State caching for incremental inference
  └─ Layer normalization + Dropout + Padding mask
    ↓
Transformer Encoder (2-4 layers)
  ├─ Strict causal attention (with padding mask caching)
  │  • Sliding window (size 64)
  │  • Multi-head attention (8 heads)
  │  • No future peeking (col_idx ≤ row_idx)
  ├─ Feed-forward network (4× expansion)
  ├─ Layer normalization + residual connections
  └─ Padding mask enforcement after each block
    ↓
Classification Head
  ├─ Extract final valid timestep output
  ├─ Hidden layer: Linear(d_model → d_model) + GELU
  └─ Output layer: Linear(d_model → 3 classes)
    ↓
Output
  ├─ Logits: [B, 3] (or [B, SeqLen, 3] if return_all_timesteps=True)
  ├─ Probabilities: Softmax with temperature scaling
  └─ Predictions: argmax(probabilities)
```

**Default specifications:**
- d_model: 128
- n_heads: 8
- n_gru_layers: 1
- n_transformer_layers: 2
- Feed-forward dimension: 512 (4× d_model)
- Attention window: 64 observations
- Dropout: 0.1

### Design Principles

**1. Strict Causality**
- Causal attention mask: `col_idx ≤ row_idx` (keys must exist before queries)
- Sliding window prevents O(N²) complexity while maintaining locality
- Padding mask cached in incremental state to prevent "ghost" signals
- Post-normalization masking eliminates LayerNorm shift artifacts on padding tokens

**2. Temporal Encoding**
- Continuous sinusoidal encoding based on Δt (observation intervals)
- Prevents model from using absolute timestamps as shortcuts
- Handles irregular cadences naturally (no interpolation needed)
- Log-scale normalization for wide range of timescales

**3. Hybrid Architecture Benefits**
- GRU: Efficient sequential processing, handles variable-length sequences
- Transformer: Captures long-range dependencies, parallel training
- Complementary strengths: Local (GRU) + Global (Transformer) context

**4. Streaming Inference**
- Incremental state management with key-value caching
- GRU hidden state preservation across observations
- Padding mask history tracking for correct attention computation
- Enables real-time classification as new observations arrive

**5. Anti-Bias Safeguards**
- Temperature clamping (0.5-5.0) prevents confidence collapse
- GRU state freezing for completed sequences in streaming mode
- Explicit padding mask re-application after every normalization layer

---

## Data Generation

### Simulation Parameters

Generate synthetic microlensing events using VBBinaryLensing (or fallback approximations):

```bash
python simulate.py \
    --n_flat 50000 \
    --n_pspl 50000 \
    --n_binary 50000 \
    --binary_preset distinct \
    --cadence_mask_prob 0.05 \
    --mag_error_std 0.05 \
    --output ../data/dataset.npz \
    --seed 42
```

**Key arguments:**
- `--n_flat/pspl/binary`: Number of events per class
- `--binary_preset`: Binary lens topology (see below)
- `--cadence_mask_prob`: Fraction of missing observations (0.0-1.0)
- `--mag_error_std`: Photometric uncertainty (magnitudes)
- `--seed`: Random seed for reproducibility
- `--num_workers`: Parallel processes for generation

### Binary Lens Topologies

| Preset | Mass Ratio (q) | Separation (s) | Description |
|--------|----------------|----------------|-------------|
| `distinct` | 0.1 - 1.0 | 0.90 - 1.10 | Resonant caustics (s≈1), guaranteed crossings |
| `planetary` | 10⁻⁴ - 10⁻² | 0.5 - 2.0 | Exoplanet detection regime |
| `stellar` | 0.3 - 1.0 | 0.3 - 3.0 | Binary star systems |
| `baseline` | 10⁻⁴ - 1.0 | 0.1 - 3.0 | Full parameter space |

**Anti-Bias Sampling Strategy:**
- PSPL and Binary share identical t₀ ranges to prevent temporal shortcuts
- Log-uniform sampling for u₀ (impact parameter) favors high-magnification events
- Ensures high-mag PSPLs exist to prevent "high mag = binary" bias

### Observational Presets

**Cadence studies:**

| Preset | Missing % | Description |
|--------|-----------|-------------|
| `cadence_05` | 5% | Space-based high cadence |
| `cadence_15` | 15% | High-quality ground surveys |
| `cadence_30` | 30% | Typical ground surveys |
| `cadence_50` | 50% | Sparse monitoring |

**Photometric error studies:**

| Preset | σ (mag) | Description |
|--------|---------|-------------|
| `error_003` | 0.03 | JWST-quality |
| `error_005` | 0.05 | Roman/HST quality |
| `error_010` | 0.10 | Professional observatories |
| `error_015` | 0.15 | Wide-field surveys |

**List all presets:**
```bash
python simulate.py --list_presets
```

---

## Training

### Single GPU

```bash
python train.py \
    --data ../data/dataset.npz \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

### Distributed Training (Multi-GPU)

**Single node:**
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=4 train.py \
    --data ../data/dataset.npz \
    --batch_size 64
```

**Multi-node (SLURM):**
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
        --data ../data/dataset.npz \
        --epochs 50
```

### Training Features

**Optimization:**
- AdamW optimizer (β₁=0.9, β₂=0.999)
- Weight decay: 1e-4
- ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- Gradient clipping (max norm: 1.0)
- Mixed precision training (AMP)
- Gradient accumulation (4 steps)

**Regularization:**
- Dropout: 0.1
- Early stopping (patience: 15 epochs)
- Class-balanced loss weighting
- Loss spike detection with automatic learning rate reduction

**Robustness:**
- NaN gradient detection and skipping
- Rank-0 only logging (clean DDP output)
- Graceful distributed cleanup on exit

**Causality Auditing:**
- Periodic early detection checks (every 5 epochs)
- Accuracy at 20%, 50%, 100% completeness
- Validates that predictions stabilize progressively

---

## Evaluation

### Comprehensive Analysis

```bash
python evaluate.py \
    --experiment_name results_* \
    --data ../data/test.npz \
    --batch_size 128
```

### Generated Outputs

**Diagnostic plots:**
1. **roc_curve.png**: One-vs-rest ROC curves with per-class AUC
2. **confusion_matrix.png**: Normalized confusion matrix heatmap
3. **calibration.png**: Reliability diagram + confidence histograms
4. **fine_early_detection.png**: Accuracy vs observation completeness (50 points)
5. **evolution_[type]_[idx].png**: Per-event classification trajectory with:
   - Light curve with observation markers
   - Class probability evolution over time
   - Confidence progression
6. **temporal_bias_check.png**: t₀ distribution comparison (KS-test)
7. **u0_dependency.png**: Binary accuracy vs impact parameter

**Quantitative metrics (summary.json):**
- Overall accuracy, precision, recall, F1-score
- Per-class metrics (Flat, PSPL, Binary)
- AUROC (macro and weighted)
- Model configuration snapshot

### Early Detection Analysis

The evaluation automatically computes fine-grained accuracy across 50 observation completeness fractions (5%-100%). This validates:
- Model doesn't require full light curves for classification
- Predictions stabilize as more data arrives (causal consistency)
- No sudden accuracy jumps that would indicate temporal shortcuts

### Physical Detection Limits

**Impact parameter (u₀) dependency:**
Binary lens detectability is fundamentally limited by source-lens separation:
- **u₀ < 0.15**: Strong caustic signatures, high accuracy achievable
- **u₀ ~ 0.2-0.3**: Weak caustic features, moderate accuracy
- **u₀ > 0.3**: Morphologically PSPL-like, low accuracy (physically unavoidable)

This analysis confirms the model respects astrophysical constraints rather than exploiting data artifacts.

---

## Visualization

### Model Internals

The `visualize.py` script provides detailed visualization of model behavior:

```bash
python visualize.py \
    --experiment_name results_* \
    --data ../data/test.npz \
    --n_examples 3
```

**Generated visualizations:**
1. **attention_patterns_event*.png**: Attention weight matrices per layer
2. **temporal_encoding_event*.png**: Δt encoding analysis
3. **classification_evolution_event*.png**: High-resolution prediction trajectories
4. **binary_vs_pspl_comparison.png**: Class separation over time
5. **embedding_space_pca.png**: Final layer embeddings in 2D
6. **confidence_evolution_by_class.png**: Mean confidence trajectories with std bands

---

## Project Structure

```
thesis-microlensing/
├── code/
│   ├── simulate.py          # Dataset generation
│   ├── train.py             # Distributed training
│   ├── evaluate.py          # Comprehensive evaluation
│   ├── visualize.py         # Model visualization
│   └── transformer.py       # Causal hybrid architecture
│
├── data/
│   └── *.npz                # Generated datasets
│
├── results/
│   └── results_*/           # Training outputs per experiment
│       ├── best_model.pt    # Best checkpoint (highest val accuracy)
│       ├── summary.json     # Metrics summary
│       └── eval_*/          # Evaluation outputs with timestamp
│
├── environment.yml          # Conda environment
├── README.md
└── LICENSE
```

---

## Methodological Notes

### Temporal Bias Prevention

**Problem:** Models can exploit data artifacts (e.g., peak always at t=0) rather than learning physical signatures.

**Solutions implemented:**
1. **Wide t₀ sampling:** PSPL and Binary events share identical t₀ ranges
2. **Relative time encoding:** Model receives Δt (intervals), not absolute timestamps
3. **KS-test validation:** Compares t₀ distributions between classes
4. **Evolution auditing:** Checks prediction stability over observation phases

### Causality Enforcement

**Problem:** Transformers can attend to future observations during training if not properly masked.

**Solutions implemented:**
1. **Strict causal mask:** `col_idx ≤ row_idx` in attention (keys ≤ queries in time)
2. **Sliding window:** Limits context to recent 64 observations
3. **Padding mask caching:** Prevents attention to padding in incremental inference
4. **Post-normalization masking:** Eliminates "ghost" signals from LayerNorm on padding
5. **Verification tests:** Online vs batch consistency checks

### Data Quality Safeguards

**Simulation:**
- Numba JIT acceleration for Δt computation (>10× speedup)
- VBBinaryLensing for accurate binary magnification (fallback approximation if unavailable)
- Photometric noise applied to flux (not magnitude) for physical realism

**Preprocessing:**
- Log1p normalization handles wide dynamic range
- NaN/Inf sanitization with explicit clamping
- Sequence length computed from padding mask, not assumed

---

## Experimental Workflows

### Topology Study

Compare performance across binary lens parameter spaces:

```bash
for topology in distinct planetary stellar baseline; do
    python simulate.py --preset ${topology} \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --output ../data/topology_${topology}.npz
    
    python train.py \
        --data ../data/topology_${topology}.npz \
        --epochs 50
    
    python evaluate.py \
        --experiment_name results_* \
        --data ../data/topology_${topology}.npz
done
```

### Cadence Study

Assess robustness to observation frequency:

```bash
for cadence in 05 15 30 50; do
    python simulate.py --preset cadence_${cadence} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/cadence_${cadence}.npz
    
    python train.py \
        --data ../data/cadence_${cadence}.npz \
        --epochs 50
    
    python evaluate.py \
        --experiment_name results_* \
        --data ../data/cadence_${cadence}.npz
done
```

### Error Study

Evaluate photometric noise tolerance:

```bash
for error in 003 005 010 015; do
    python simulate.py --preset error_${error} \
        --n_flat 30000 --n_pspl 30000 --n_binary 30000 \
        --output ../data/error_${error}.npz
    
    python train.py \
        --data ../data/error_${error}.npz \
        --epochs 50
    
    python evaluate.py \
        --experiment_name results_* \
        --data ../data/error_${error}.npz
done
```

---

## Limitations and Future Work

### Current Limitations

1. **Training data:** Synthetic simulations may not capture all observational systematics
2. **High mass ratios:** q → 1 binary systems difficult to distinguish from PSPL
3. **Extreme noise:** Performance degrades beyond σ > 0.20 mag
4. **Sparse cadences:** Accuracy drops for >50% missing observations

### Planned Improvements

**Architecture:**
- Caustic-aware feature extraction modules
- Explicit u₀ regression head for uncertainty quantification
- Multi-scale temporal attention (short/long timescales)

**Training:**
- Curriculum learning (simple → complex topologies)
- Data augmentation (synthetic noise, gap patterns)
- Semi-supervised learning with unlabeled real survey data

**Deployment:**
- Model quantization (INT8) for faster inference
- ONNX export for production pipelines
- Integration with survey alert systems (LSST, ZTF)

---

## Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={Gravitational Microlensing Event Classification with Causal Transformers},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis}
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

**For issues:**
- Implementation questions → GitHub Issues
- Methodology questions → Thesis document
- Collaboration inquiries → Contact via institutional email

---

## References

**Key publications:**
1. Bozza, V. (2010). "VBBinaryLensing: A C++ library for microlensing." MNRAS, 408, 2188-2196.
2. Zhu, W., et al. (2017). "Mass Measurements from Space-based Microlensing." ApJ, 849, L31.
3. Johnson, S. A., et al. (2020). "Nancy Grace Roman Space Telescope Predictions." AJ, 160, 123.

**Survey resources:**
- OGLE: http://ogle.astrouw.edu.pl/
- MOA: https://www.massey.ac.nz/~iabond/moa/
- Nancy Grace Roman: https://roman.gsfc.nasa.gov/
- LSST: https://www.lsst.org/

---

**Version**: 1.0  
**Last updated**: December 2025  
**Status**: Active development
