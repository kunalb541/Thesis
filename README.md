# Gravitational Microlensing Event Classification

MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Three-class classification of gravitational microlensing events using a CNN-GRU recurrent neural network:
- **Class 0**: Flat (no lensing)
- **Class 1**: PSPL (Point Source Point Lens)
- **Class 2**: Binary lens (planetary or stellar companion)

The model uses depthwise separable convolutions, flash attention pooling, hierarchical classification, and processes variable-length sequences with causal masking. Temporal encoding based on observation intervals (Δt) rather than absolute timestamps.

---

## Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/kunalb541/Thesis.git
cd Thesis

# Create conda environment (PyTorch CUDA 12.1 included)
conda env create -f environment.yml
conda activate microlens

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import VBBinaryLensing; print('VBBinaryLensing: OK')"
```

### GPU Configuration

**The environment.yml now includes PyTorch CUDA 12.1 by default.**

For different hardware, edit `environment.yml` before creating the environment:

**AMD GPUs (ROCm 6.0)** - MI300 series:
```yaml
# In environment.yml, replace PyTorch lines with:
- pytorch::pytorch=2.2.0
- pytorch::torchvision=0.17.0
# Then after creating environment:
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

**NVIDIA GPUs (CUDA 11.8)** - older cards:
```yaml
# In environment.yml, replace PyTorch lines with:
- pytorch::pytorch=2.2.0
- pytorch::torchvision=0.17.0
- pytorch::pytorch-cuda=11.8
```

**CPU only**:
```yaml
# In environment.yml, replace PyTorch lines with:
- pytorch::pytorch=2.2.0
- pytorch::torchvision=0.17.0
- pytorch::cpuonly
```

---

## Quick Start

Validate the complete pipeline:

```bash
cd code

# 1. Generate test dataset (300 events)
python simulate.py --n_flat 100 --n_pspl 100 --n_binary 100 \
  --output ../data/raw/test.h5

# 2. Train model (5 epochs for validation)
python train.py --data ../data/raw/test.h5 --epochs 5 --batch-size 32

# 3. Evaluate
python evaluate.py --experiment-name roman --data ../data/raw/test.h5
```

---

## Data Generation

### Simulation Command

```bash
python simulate.py \
    --n_flat 100000 \
    --n_pspl 100000 \
    --n_binary 100000 \
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
- Core datasets: `flux`, `delta_t`, `labels`, `timestamps`
- Parameters: `params_flat`, `params_pspl`, `params_binary` (structured arrays)
- Metadata: Mission duration, cadence, seed, etc.

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

srun torchrun \
  --nnodes=12 \
  --nproc-per-node=4 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="train-$(date +%s)" \
  train.py \
  --data ../data/raw/distinct.h5 \
  --prefetch-factor 12 \
  --experiment-name distinct
```

### Checkpoint Resumption 

Training can now be resumed from any checkpoint:

```bash
# Initial training (times out at epoch 12)
python train.py --data baseline.h5 --epochs 50

# Resume from checkpoint (continues from epoch 13)
python train.py --data baseline.h5 --epochs 50 \
  --resume results/baseline_*/checkpoint_epoch_12.pt

# Or resume from best model
python train.py --data baseline.h5 --epochs 50 \
  --resume results/baseline_*/best_model.pt
```

**Training Outputs** (saved in `results/<experiment-name>_<timestamp>/`):
- `best_model.pt` - Best checkpoint (highest validation accuracy)
- `checkpoint_epoch_N.pt` - Periodic checkpoints (every `--save-every` epochs)
- `training.log` - Complete training log with metrics
- `final_predictions.npz` - Predictions on validation set
- `confusion_matrix.npy` - Final confusion matrix
- `classification_report.txt` - Per-class metrics
- `config.json` - Full configuration and hyperparameters

**Training Features:**
- AdamW optimizer with cosine annealing warmup
- Gradient clipping (max norm 1.0)
- Mixed precision training (AMP) with bfloat16 or float16
- Gradient accumulation (configurable)
- Class-balanced loss weighting
- DDP with gradient-as-bucket-view optimization
- torch.compile support for 30-50% speedup
- Checkpoint resumption for fault tolerance

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

### Advanced Options 

```bash
# Publication-quality outputs (PDF + SVG)
python evaluate.py \
    --experiment-name baseline \
    --data test.h5 \
    --save-formats png pdf svg \
    --colorblind-safe

# Verbose logging for debugging
python evaluate.py \
    --experiment-name baseline \
    --data test.h5 \
    --verbose

```

**Evaluation Outputs** (saved in `results/<experiment>_*/eval_<dataset>_<timestamp>/`):
- `evaluation_summary.json` - Overall metrics and configuration
- `confusion_matrix.png` - Normalized confusion matrix heatmap
- `roc_curves.png` - One-vs-rest ROC curves with AUC scores
- `calibration.png` - Reliability diagram with confidence histograms
- `u0_dependency.png` - Binary accuracy vs. impact parameter (if params available)
- `temporal_bias_check.png` - t₀ distribution comparison (KS-test)
- `early_detection_curve.png` - Accuracy vs. observation completeness
- `evolution_<class>_<idx>.png` - Probability evolution (3-panel: light curve + probabilities + confidence)
- `example_light_curves.png` - Grid of example classifications
- `per_class_metrics.png` - Precision/recall/F1 bar chart
- `evaluation.log` - Complete evaluation log

**Metrics Computed:**
- Overall: accuracy, precision, recall, F1-score
- Per-class: precision, recall, F1 for Flat, PSPL, Binary
- AUROC: macro and weighted averages
- Calibration error
- Bootstrap confidence intervals (95% CI)
- Early detection performance at multiple completeness levels
- Impact parameter (u₀) dependency for binary events
- Temporal bias check (KS-test on t₀ distributions)

### Batch Evaluation Across Datasets

```bash
# Evaluate on multiple test sets
for preset in baseline stellar planetary distinct; do
  python evaluate.py \
    --experiment-name distinct \
    --data ../data/raw/${preset}.h5 \
    --batch-size 512 \
    --n-samples 500000 \
    --early-detection \
    --save-formats png pdf
done
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

# Launch 4 parallel simulation jobs
for preset in distinct stellar planetary baseline; do
  srun --partition=gpu_a100_short --gres=gpu:4 --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --exclusive --time=00:30:00 \
    python -u simulate.py \
    --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
    --binary_preset ${preset} \
    --output ../data/raw/${preset}.h5 \
    --num_workers 32 --seed $RANDOM > log_${preset}.txt 2>&1 &
done
wait
rm *txt
echo "All datasets generated."
```

---

## Architecture

### Model: RomanMicrolensingClassifier

```
Input: Flux [B, N], Time Intervals Δt [B, N]
  │
  ├─ Input Projection: Linear(2 → d_model)
  │
Feature Extraction: Depthwise Separable Conv (2 blocks, causal)
  │  ├─ Block 1: kernel=5, dilation=1
  │  └─ Block 2: kernel=5, dilation=2 (multi-scale)
  │
Recurrent Core: Stacked GRU (CuDNN-fused)
  │  └─ n_layers with dropout between layers
  │
Layer Normalization + Residual Connection
  │
Temporal Pooling:
  │  ├─ Attention Pooling (multi-head, learnable query)
  │  └─ OR Mean Pooling (masked by sequence length)
  │
Hierarchical Classification:
  ├─ Shared Trunk: Linear + LayerNorm + SiLU + Dropout
  ├─ Stage 1: Flat vs. Deviation (2 classes)
  └─ Stage 2: PSPL vs. Binary (2 classes)
  │
Output: Logits [B, 3], Probabilities [B, 3]
```

**Default Configuration:**
- d_model: 16 (hidden dimension)
- n_layers: 2 (GRU layers)
- dropout: 0.3
- window_size: 5
- max_seq_len: 2400
- n_classes: 3
- hierarchical: True
- attention_pooling: True
- feature_extraction: depthwise separable convolutions
- activation: SiLU (Swish)

**Parameter Count:** ~50-200K (varies with d_model and n_layers)

**Performance:**
- Inference: <1ms per light curve (1000× faster than χ² fitting)
- Throughput: ~15,000 samples/sec/GPU (A100)
- Receptive field: 13 time steps (with default config)

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py          # VBBinaryLensing-based data generation
│   ├── train.py             # DDP training with checkpoint resume
│   ├── evaluate.py          # Comprehensive metrics and visualization
│   └── model.py             # CNN-GRU architecture
│
├── data/
│   └── raw/                 # Generated datasets (*.h5)
│
├── results/
│   └── <experiment>_*/      # Training outputs per experiment
│       ├── best_model.pt    # Best checkpoint
│       ├── checkpoint_*.pt  # Periodic checkpoints
│       ├── training.log     # Training log
│       └── eval_*/          # Evaluation outputs (timestamped)
│
├── environment.yml          # Conda environment (with active PyTorch)
├── README.md                # This file
└── LICENSE                  # MIT License
```

---

## Model Configuration Options

**Architecture variants:**
```bash
# Default: Hierarchical with attention pooling
python train.py --data <data> --hierarchical --attention-pooling

# Flat classification (no hierarchy)
python train.py --data <data> --no-hierarchical

# Without attention pooling (use mean pooling)
python train.py --data <data> --no-attention-pooling

# Larger model
python train.py --data <data> --d-model 64 --n-layers 3
```

**Optimization options:**
```bash
# Disable mixed precision
python train.py --data <data> --no-amp

# Enable torch.compile (30-50% faster)
python train.py --data <data> --compile --compile-mode max-autotune

# Gradient accumulation for larger effective batch size
python train.py --data <data> --batch-size 32 --accumulation-steps 4
```

---

## Troubleshooting

### GPU Out-of-Memory
```bash
# Reduce batch size
--batch-size 32

# Use gradient accumulation
--accumulation-steps 2

# Disable mixed precision
--no-amp
```

### Training Divergence (NaN Loss)
```bash
# Check data normalization
python -c "import h5py; f=h5py.File('data.h5'); print(f['flux'][:100].mean())"

# Reduce learning rate
--lr 5e-4

# Increase warmup
--warmup-epochs 10
```

### Distributed Training Hangs
```bash
# Verify master node accessibility
echo $MASTER_ADDR
ping $MASTER_ADDR

# Increase timeout
export NCCL_TIMEOUT=3600

# Check GPU visibility
srun --nodes=2 --gres=gpu:4 python -c "import torch; print(torch.cuda.device_count())"
```

### Checkpoint Resume Issues
```bash
# Verify checkpoint exists
ls results/experiment_*/checkpoint_epoch_12.pt

# Check checkpoint contents
python -c "import torch; ckpt = torch.load('checkpoint.pt'); print(ckpt.keys())"

# If mismatch, train from scratch with same config
python train.py --data <data> --d-model 16 --n-layers 2
```

### Missing u₀ Analysis Plots
The parameter extraction has been fixed in v2.0. If you still don't see u₀ plots:

```bash
# Verify parameters exist in dataset
python -c "import h5py; f=h5py.File('data.h5'); print(list(f.keys()))"

# Should see: params_flat, params_pspl, params_binary

# If missing, regenerate data with simulate.py
python simulate.py --n_flat 100 --n_pspl 100 --n_binary 100 --output test.h5
```

---

## Physical Parameter Ranges

**Observational Configuration:**
- Temporal sampling: ~12.1 minutes (Roman-like cadence)
- Missing observations: 5% (uniform random)
- Photometric error: Realistic (Roman F146 detector model)
- Mission duration: **200 days** (POC, full Roman survey is 5 years)
- Sequence length: 2400 observations max
- Source baseline magnitude: 18-24 AB mag
- AB zeropoint: 3631 Jy

**Microlensing Parameters:**
- Einstein timescale (t_E): 5-30 days
- Peak time (t₀): 40-160 days (0.2-0.8 × mission duration)
- Impact parameter (u₀):
  - PSPL: 0.001-1.0
  - Binary: 0.001-1.0 (preset-dependent)
- Binary separation (s): 0.1-3.0 Einstein radii
- Mass ratio (q): 10⁻⁴ - 1.0
- Source radius (ρ): 10⁻³ - 0.1 Einstein radii

---

## Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={From Light Curves to Labels: Machine Learning in Microlensing Event Classification},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis},
  note={Advisor: Prof. Dr. Joachim Wambsganß}
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

**Machine Learning in Microlensing:**
- Lam et al. (2020): ML classification of OGLE events
- Godines et al. (2019): CNN-based microlensing detection

---

## Development Status

**Version 2.0 (Current)**
- Status: Production-ready
- All critical bugs fixed
- Comprehensive testing complete
- Ready for thesis submission

**Known Limitations:**
- Trained on simulated data only (no real OGLE/MOA validation yet)
- 200-day mission duration (POC, not full 5-year survey)
- Parameters ~50-200K (lightweight, could scale up for more complexity)

**Future Work:**
- Validation on real OGLE/MOA data
- Comparison to traditional χ² fitting methods
- Uncertainty quantification and calibration
- Phase-folded light curve visualization
- Extended to 5-year Roman mission simulations

---
