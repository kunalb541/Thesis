# Gravitational Microlensing Event Classification

MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Three-class classification of gravitational microlensing events using a GRU-based recurrent neural network:
- **Class 0**: Baseline (flat light curve, no lensing)
- **Class 1**: Point Source Point Lens (PSPL, single lens)
- **Class 2**: Binary lens (planetary or stellar companion)

The model uses depthwise separable convolutions, flash attention pooling, hierarchical classification, and processes variable-length sequences with causal masking. Temporal encoding based on observation intervals (Δt) rather than absolute timestamps.

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

Basic pipeline validation:

```bash
cd code

# 1. Generate test dataset (300 events)
python simulate.py --n_flat 100 --n_pspl 100 --n_binary 100 --output ../data/raw/test.h5

# 2. Train model (5 epochs, no convergence expected)
python train.py --data ../data/raw/test.h5 --epochs 5 --batch-size 32

# 3. Evaluate
python evaluate.py --experiment-name rom --data ../data/raw/test.h5
```

---

## Data Generation

### Simulation Command

```bash
python simulate.py \
    --n_flat 100000 \
    --n_pspl 500000 \
    --n_binary 500000 \
    --binary_preset baseline \
    --output ../data/raw/baseline.h5 \
    --num_workers 32 \
    --seed 42
```

**Binary Lens Presets:**

| Preset | Mass Ratio (q) | Separation (s) | Impact (u₀) | Description |
|--------|----------------|----------------|-------------|-------------|
| `distinct` | 0.1 - 1.0 | 0.90 - 1.10 | 0.0001 - 0.4 | Resonant caustics, guaranteed crossings |
| `planetary` | 10⁻⁴ - 10⁻² | 0.5 - 2.0 | 0.001 - 0.3 | Exoplanet detection regime |
| `stellar` | 0.3 - 1.0 | 0.3 - 3.0 | 0.001 - 0.3 | Binary star systems |
| `baseline` | 10⁻⁴ - 1.0 | 0.1 - 3.0 | 0.001 - 1.0 | Full parameter space |

**Output:** `<output>.h5` (HDF5 format, ~3-4 GB per 1M events)
- Datasets: `flux`, `delta_t`, `labels`, `timestamps`
- Structured parameters: `params_flat`, `params_pspl`, `params_binary`

---

## Training

### Single GPU

```bash
python train.py \
    --data ../data/raw/baseline.h5 \
    --experiment-name baseline \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-3 \
    --warmup-epochs 5
```

### Distributed Training (Multi-GPU)

**SLURM Environment (bwHPC):**

```bash
# Allocate resources
salloc --partition=gpu_a100_short --nodes=12 --gres=gpu:4 --exclusive --time=00:30:00

# Setup environment
cd ~/Thesis/code
conda activate microlens

export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE
export TORCH_DISTRIBUTED_ACK_TIMEOUT=1800
export TORCH_DISTRIBUTED_SEND_TIMEOUT=1200
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=5
export NCCL_MIN_NCHANNELS=16
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_ALGO=TREE

# Distributed training
srun torchrun \
  --nnodes=12 \
  --nproc-per-node=4 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="train-$(date +%s)" \
  train.py \
  --data ../data/raw/baseline.h5 \
  --prefetch-factor 12 \
  --experiment-name baseline \
```

**Training Outputs** (saved in `results/<experiment-name>_<timestamp>/`):
- `best_model.pt` - Best checkpoint (highest validation accuracy)
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `training.log` - Training log with loss/accuracy per epoch
- `final_predictions.npz` - Predictions on validation set
- `confusion_matrix.npy` - Final confusion matrix
- `classification_report.txt` - Per-class metrics

**Training Features:**
- AdamW optimizer with linear warmup
- Gradient clipping (max norm 1.0)
- Mixed precision training (AMP) with bfloat16
- Gradient accumulation (default 1 step)
- ReduceLROnPlateau scheduler
- Early stopping (configurable patience)
- Class-balanced loss weighting
- DDP with gradient-as-bucket-view optimization

---

## Evaluation

### Standard Evaluation

```bash
python evaluate.py \
    --experiment-name baseline \
    --data ../data/raw/baseline.h5 \
    --batch-size 512 \
    --n-samples 100000 \
    --early-detection \
    --n-evolution-per-type 10
```

**Evaluation Outputs** (saved in `results/<experiment-name>_*/eval_<dataset>_<timestamp>/`):
- `evaluation_summary.json` - Overall metrics and configuration
- `confusion_matrix.png` - Normalized confusion matrix heatmap
- `roc_curves.png` - One-vs-rest ROC curves with AUC scores
- `calibration.png` - Reliability diagram with confidence histograms
- `u0_dependency.png` - Binary accuracy vs. impact parameter (if params available)
- `temporal_bias_check.png` - t₀ distribution comparison (KS-test)
- `fine_early_detection.png` - Accuracy vs. observation completeness (50 points)
- `evolution_<class>_<idx>.png` - Per-event classification trajectories (light curve + probabilities + confidence)

**Metrics Computed:**
- Overall accuracy, precision, recall, F1-score
- Per-class metrics (Flat, PSPL, Binary)
- AUROC (macro and weighted)
- Calibration error
- Early detection accuracy at multiple completeness fractions

### Batch Evaluation Across Datasets

```bash
# Evaluate on multiple test sets
python evaluate.py --data ../data/raw/baseline.h5   --experiment-name baseline --batch-size 512 --n-samples 100000 --early-detection --n-evolution-per-type 10
python evaluate.py --data ../data/raw/stellar.h5    --experiment-name baseline --batch-size 512 --n-samples 100000 --early-detection --n-evolution-per-type 10
python evaluate.py --data ../data/raw/planetary.h5  --experiment-name baseline --batch-size 512 --n-samples 100000 --early-detection --n-evolution-per-type 10
python evaluate.py --data ../data/raw/distinct.h5   --experiment-name baseline --batch-size 512 --n-samples 100000 --early-detection --n-evolution-per-type 10
```

---

## Parallel Data Generation

Generate multiple datasets simultaneously on SLURM:

```bash
cd ~/Thesis/code
conda activate microlens

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Launch 4 parallel jobs
srun --partition=gpu_a100_short --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gres=gpu:4 --exclusive --time=00:30:00 \
  python -u simulate.py --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
  --binary_preset distinct --output ../data/raw/distinct.h5 \
  --num_workers 32 --seed 42 > log_distinct.txt 2>&1 &

srun --partition=gpu_a100_short --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gres=gpu:4 --exclusive --time=00:30:00 \
  python -u simulate.py --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
  --binary_preset stellar --output ../data/raw/stellar.h5 \
  --num_workers 32 --seed 43 > log_stellar.txt 2>&1 &

srun --partition=gpu_a100_short --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gres=gpu:4 --exclusive --time=00:30:00 \
  python -u simulate.py --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
  --binary_preset planetary --output ../data/raw/planetary.h5 \
  --num_workers 32 --seed 44 > log_planetary.txt 2>&1 &

srun --partition=gpu_a100_short --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --gres=gpu:4 --exclusive --time=00:30:00 \
  python -u simulate.py --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
  --binary_preset baseline --output ../data/raw/baseline.h5 \
  --num_workers 32 --seed 45 > log_baseline.txt 2>&1 &

# Wait for all jobs
wait
echo "All datasets generated."
```

---

## Architecture

### Model: RomanMicrolensingGRU

```
Input: Flux [B, N], Time Intervals Δt [B, N]
  │
  ├─ Flux Embedding: Linear(1 → d_model/2)
  └─ Temporal Encoding: Sinusoidal (Δt-based)
  │
Input Mixing: Linear(d_model → d_model) + LayerNorm + SiLU + Dropout
  │
Feature Extraction: Depthwise Separable Conv (2 blocks)
  │
Window Processing: Depthwise Separable Conv (causal padding)
  │
Multi-scale: Concatenate [features, windowed] → [B, N, 2×d_model]
  │
Recurrent Core: Stacked GRU (CuDNN-fused, multiple layers)
  │
Final Normalization: LayerNorm
  │
Pooling: Flash Attention Pooling (or sequence-end extraction)
  │
Hierarchical Classification:
  ├─ Shared Trunk: Linear + LayerNorm + SiLU + Dropout
  ├─ Stage 1: Flat vs. Deviation (2 classes)
  └─ Stage 2: PSPL vs. Binary (2 classes)
  │
Output: Logits [B, 3], Probabilities [B, 3]
```

**Default Configuration:**
- d_model: 64
- n_layers: 2
- dropout: 0.3
- window_size: 5
- max_seq_len: 2400
- Hierarchical: True
- Attention pooling: True
- Feature extraction: Depthwise separable convolutions
- Activation: SiLU (Swish)

**Parameter Count:** ~200K (varies with d_model and n_layers)

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py          # VBBinaryLensing-based data generation
│   ├── train.py             # DDP training with AMP and gradient accumulation
│   ├── evaluate.py          # Comprehensive metrics, u₀ analysis, early detection
│   └── model.py             # RomanMicrolensingGRU architecture
│
├── data/
│   ├── raw/                 # Generated datasets (*.h5)
│
├── results/
│   └── <experiment>_*/      # Training outputs per experiment
│       ├── best_model.pt    # Best checkpoint
│       ├── training.log     # Training log
│       ├── eval_*/          # Evaluation outputs (timestamped)
│       └── checkpoint_*.pt  # Periodic checkpoints
│
├── docs/
│   └── RESEARCH_GUIDE.md    # Detailed experimental methodology
│
├── environment.yml          # Conda environment
├── README.md
└── LICENSE
```

---

## Model Configuration Options

**Architecture variants:**
```bash
# Default: Hierarchical with attention pooling
python train.py --data <data> --hierarchical --attention-pooling

# Flat classification (no hierarchy)
python train.py --data <data> --no-hierarchical

# Without attention pooling (use sequence-end)
python train.py --data <data> --no-attention-pooling

# Change feature extraction method
python train.py --data <data> --feature-extraction mlp  # or 'conv' (default)

# Disable residual connections
python train.py --data <data> --no-residual

# Disable layer normalization
python train.py --data <data> --no-layer-norm
```

**Optimization options:**
```bash
# Disable mixed precision
python train.py --data <data> --no-amp

# Disable gradient checkpointing
python train.py --data <data> --no-gradient-checkpointing

# Enable torch.compile (PyTorch 2.0+)
python train.py --data <data> --compile --compile-mode max-autotune
```

---

## Troubleshooting

### GPU Out-of-Memory
```bash
# Reduce batch size
--batch-size 32

# Enable gradient checkpointing (trades compute for memory)
--use-gradient-checkpointing

# Disable mixed precision
--no-amp
```

### Training Divergence (NaN Loss)
- Check data normalization: `python -c "import h5py; f=h5py.File('data.h5'); print(f['flux'][:100].mean(), f['flux'][:100].std())"`
- Reduce learning rate: `--lr 5e-4`
- Increase warmup: `--warmup-epochs 10`
- Code automatically skips NaN batches; if >10% skipped, investigate data quality

### Distributed Training Hangs
```bash
# Verify master node accessibility
echo $MASTER_ADDR
ping $MASTER_ADDR

# Check SLURM communication
srun --nodes=2 hostname

# Increase NCCL timeout
export NCCL_TIMEOUT=3600

# Verify CUDA devices
srun --nodes=2 --gres=gpu:4 python -c "import torch; print(torch.cuda.device_count())"
```

### Missing u₀ Analysis Plots
```bash
# Verify parameters exist in dataset
python -c "import h5py; f=h5py.File('data.h5'); print(list(f.keys()))"

# Parameters should include: params_flat, params_pspl, params_binary
# If missing, regenerate data with simulate.py (parameters are auto-saved)
```

### Data Quality Check
```bash
python -c "
import h5py
import numpy as np

with h5py.File('data.h5', 'r') as f:
    flux = f['flux'][:]
    delta_t = f['delta_t'][:]
    labels = f['labels'][:]
    
    print(f'Flux shape: {flux.shape}')
    print(f'Flux range: [{flux.min():.3f}, {flux.max():.3f}]')
    print(f'NaN count: {np.isnan(flux).sum()}')
    print(f'Label distribution: {np.bincount(labels)}')
"
```

---

## Physical Parameter Ranges

**Observational Configuration:**
- Temporal sampling: ~12.1 minutes (Roman-like cadence)
- Missing observations: 5% (uniform random)
- Photometric error: 0.05 mag (Gaussian)
- Mission duration: 200 days
- Sequence length: 2400 observations max

**Microlensing Parameters:**
- Einstein timescale (t_E): 5-30 days
- Peak time (t₀): 0.2-0.8 × mission duration
- Source baseline magnitude: 18-24 AB mag
- AB zeropoint: 3631 Jy

---

## Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={Gravitational Microlensing Event Classification with Recurrent Neural Networks},
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

## References

**Key Publications:**
1. Bozza, V. (2010). "VBBinaryLensing: A C++ library for microlensing." MNRAS, 408, 2188-2196.
2. Zhu, W., et al. (2017). "Mass Measurements from Space-based Microlensing." ApJ, 849, L31.
3. Johnson, S. A., et al. (2020). "Nancy Grace Roman Space Telescope Predictions." AJ, 160, 123.

**Survey Resources:**
- OGLE: http://ogle.astrouw.edu.pl/
- MOA: https://www.massey.ac.nz/~iabond/moa/
- Nancy Grace Roman: https://roman.gsfc.nasa.gov/

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Active Development
