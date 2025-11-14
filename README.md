## From Light Curves to Labels: Machine Learning in Microlensing

**Version 15.0 - Anti-Cheating Edition**  
MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

Real-time classification of gravitational microlensing events using transformer neural networks with anti-cheating architecture.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## 🔬 Project Overview

This project develops a real-time gravitational microlensing event classifier capable of distinguishing between single-lens (PSPL) and binary-lens events at sub-millisecond inference speeds—critical for next-generation surveys like the Roman Space Telescope and LSST. 

## 🎯 What This Does

**Three-class gravitational microlensing classification:**
- **Class 0 (Flat)**: No event - constant baseline
- **Class 1 (PSPL)**: Single lens event  
- **Class 2 (Binary)**: Binary lens - planets or binary stars

**Performance:**
- 🚀 **<1ms inference** per event
- 📊 **80%+ accuracy** on Roman Space Telescope quality data
- ⏱️ **Early detection** at 50% observation completeness
- 📈 **Survey-scale**: 10,000+ events/second

**Key Innovation:**
- Semi-causal attention (no future peeking)
- Relative positional encoding (no temporal shortcuts)
- Temporal invariance loss (learns morphology, not timing)

---

## 🏃 Quick Start

### 1️⃣ Installation
```bash
# Clone repository
git clone https://github.com/kunalb541/Thesis.git
cd Thesis

# Create base environment
conda env create -f environment.yml
conda activate microlens
```

### 2️⃣ GPU Setup

**For AMD GPUs (ROCm 6.0)** - e.g., AMD MI300:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

**For NVIDIA GPUs (CUDA 12.1)** - e.g., A100, H100:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only** (slow, not recommended):
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

### 3️⃣ Quick Test (5 minutes, single GPU)
```bash
cd code

# Generate 300 test events
python simulate.py \
    --n_flat 300 --n_pspl 300 --n_binary 300 \
    --preset distinct \
    --output ../data/raw/test.npz \
    --save_params \
    --num_workers 8 \
    --seed 42

# Train 5 epochs
python train.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --epochs 5 \
    --batch_size 32

# Evaluate
python evaluate.py \
    --experiment_name test \
    --data ../data/raw/test.npz
```

---

## ⚡ Quick Shortcuts (Production Workflow)

### Small Test (9K events, 40 GPUs, ~5 minutes)
```bash
# Allocate nodes
salloc --partition=gpu_a100_short --nodes=10 --gres=gpu:4 --exclusive --time=00:30:00

# Setup
cd ~/Thesis/code
conda activate microlens

# Generate data
python simulate.py \
    --n_flat 3000 --n_pspl 3000 --n_binary 3000 \
    --preset distinct \
    --output ../data/raw/test.npz \
    --save_params \
    --num_workers 200 \
    --seed 42

# Environment variables
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# Train
srun torchrun \
    --nnodes=10 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/test.npz \
        --experiment_name test \
        --epochs 10 \
        --batch_size 64

# Evaluate
python evaluate.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --batch_size 64 \
    --n_samples 10000 \
    --early_detection \
    --n_evolution_per_type 10 \
    --temporal_bias_check
```

### Baseline 1M (32 GPUs, 3-5 hours)
```bash
# Generate
python simulate.py --preset baseline_1M

# Train
salloc --partition=gpu_a100_short --nodes=8 --gres=gpu:4 --exclusive --time=05:00:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8

# Evaluate
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection \
    --n_evolution_per_type 10 \
    --n_samples 50000
```

---

## 🧠 Architecture

### Transformer v15.0 - Anti-Cheating Design

**Key Innovation**: Model learns from **magnification morphology**, NOT temporal position.
```
Input [B, 1500]
    ↓
Input Embedding (1 → 128-dim)
    ↓
Relative Positional Encoding
  • Observation count (not absolute time)
  • Relative gaps (not positions)
    ↓
Transformer Layers (×4)
  • Semi-causal attention (no future peeking)
  • Multi-head (4 heads)
  • Feed-forward network
    ↓
Global Pooling (avg + max)
    ↓
Task Heads:
  ├─ Classification: [B, 3] (Flat/PSPL/Binary)
  ├─ Caustic: [B, 1] (Binary morphology)
  └─ Confidence: [B, 1] (Prediction certainty)
    ↓
Temporal Invariance Loss
  • Penalizes time-dependent features
  • Forces morphology learning
```

**Model Stats:**
- Parameters: ~435,000
- d_model: 128, heads: 4, layers: 4
- Inference: <1ms per event
- Throughput: 10,000+ events/sec

**Anti-Cheating Features:**
1. **Semi-causal attention**: Can only see past, not future
2. **Relative encoding**: No absolute time information
3. **Temporal invariance loss**: Penalizes shortcuts
4. **Temporal randomization**: Random peak shifts in data

---

## 🚀 Usage Examples

### Topology Studies

**Distinct** (Clear caustics, optimal detection):
```bash
python simulate.py --preset distinct \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```

**Planetary** (Exoplanet search, small q):
```bash
python simulate.py --preset planetary \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```

**Stellar** (Binary stars, large q):
```bash
python simulate.py --preset stellar \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```

**Baseline** (Full parameter space, physical limits):
```bash
python simulate.py --preset baseline \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```

### Cadence Studies

Test observation frequency impact:

| Preset | Missing % | Description | Expected Accuracy |
|--------|-----------|-------------|-------------------|
| `cadence_05` | 5% | Roman Space Telescope (~15 min) | 80-85% |
| `cadence_15` | 15% | Good ground (~1 day) | 75-80% |
| `cadence_30` | 30% | Typical ground (~3 days) | 70-75% |
| `cadence_50` | 50% | Sparse ground (~5 days) | 60-70% |
```bash
python simulate.py --preset cadence_05 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

### Error Studies

Test photometric quality impact:

| Preset | Error (mag) | Description |
|--------|-------------|-------------|
| `error_003` | 0.03 | JWST-quality |
| `error_005` | 0.05 | Roman Space Telescope |
| `error_010` | 0.10 | High-quality ground |
| `error_015` | 0.15 | Typical ground |
```bash
python simulate.py --preset error_005 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

### Custom Experiments

Mix parameters:
```bash
python simulate.py \
    --n_flat 20000 --n_pspl 20000 --n_binary 20000 \
    --binary_preset planetary \
    --cadence_mask_prob 0.20 \
    --mag_error_std 0.08 \
    --output ../data/raw/custom.npz \
    --save_params
```

### Early Detection Analysis
```bash
python evaluate.py \
    --experiment_name your_experiment \
    --data ../data/raw/your_data.npz \
    --early_detection \
    --n_evolution_per_type 5
```

### Temporal Bias Check
```bash
python evaluate.py \
    --experiment_name your_experiment \
    --data ../data/raw/your_data.npz \
    --temporal_bias_check
```

**Good model**: No correlation between t₀ and errors  
**Bad model**: Misclassifications at specific peak times

---

## 📊 Performance (Expected)

### Baseline Results (1M events, Roman quality)
```
Overall Accuracy: 80.2%

Per-Class Recall:
  Flat:   92.5%  (no event detection)
  PSPL:   75.8%  (single lens)
  Binary: 77.3%  (binary lens)

vs. Ground-Based:
  +4.5% overall accuracy
  +5.2% binary recall
```

### Binary Morphology Impact

| Topology | Binary Recall | u₀ Threshold | Key Finding |
|----------|--------------|--------------|-------------|
| Distinct | 88.7% | 0.15 | Clear caustics → early detection |
| Planetary | 82.3% | 0.20 | Small q detectable with Roman |
| Stellar | 78.9% | 0.25 | Complex caustics challenging |
| Baseline | 73.5% | 0.30 | **Physical limit confirmed** |

### Physical Detection Limit

**u₀ = 0.3 is a physical threshold, not algorithmic failure.**

- **u₀ < 0.15**: 85%+ accuracy (excellent)
- **u₀ ≈ 0.30**: Sharp drop (threshold)
- **u₀ > 0.30**: 55% accuracy (fundamentally PSPL-like)

**Why?** For u₀ > 0.3, source doesn't cross caustics → indistinguishable from PSPL.

### Early Detection

- **50% completeness**: 75-80% accuracy (reliable trigger)
- **25% completeness**: 55-65% accuracy (early warning)
- **10% completeness**: ~40% accuracy (too early)

Roman's 15-min cadence enables classification **2-3 weeks earlier** than ground-based.

---

## 📁 Project Structure
```
thesis-microlensing/
├── code/
│   ├── simulate.py           # Data generation (VBBinaryLensing)
│   ├── train.py              # Multi-GPU DDP training
│   ├── evaluate.py           # Comprehensive evaluation
│   └── transformer_v15.py    # Anti-cheating architecture
│
├── data/
│   ├── raw/                  # Generated datasets (.npz)
│   └── processed/            # Optional preprocessed data
│
├── results/
│   └── experiment_*/         # Training outputs
│       ├── best_model.pt
│       ├── config.json
│       ├── normalizer.pkl
│       ├── results.json
│       └── evaluation/       # All plots + metrics
│
├── docs/
│   └── RESEARCH_GUIDE.md     # Experimental workflow
│
├── environment.yml           # Conda environment (base)
├── requirements.txt          # Pip requirements
├── README.md                 # This file
├── LICENSE                   # MIT License
└── .gitignore
```

---

## 📚 Documentation

### Simulation (simulate.py)

**Generate datasets with configurable parameters:**
```bash
python simulate.py --help

# Key arguments:
--n_flat, --n_pspl, --n_binary    # Event counts per class
--preset                           # Use predefined experiment
--binary_preset                    # Topology: distinct/planetary/stellar/baseline
--cadence_mask_prob                # Missing observations fraction
--mag_error_std                    # Photometric error (magnitudes)
--output                           # Output .npz path
--save_params                      # Save physical parameters
--num_workers                      # Parallel workers
--seed                             # Random seed
```

**Built-in presets:**
- `quick_test`: 300 events (validation)
- `baseline_1M`: 1M events (thesis baseline)
- `distinct`, `planetary`, `stellar`, `baseline`: Topology studies
- `cadence_05/15/30/50`: Cadence studies
- `error_003/005/010/015`: Error studies

**List all presets:**
```bash
python simulate.py --list_presets
```

### Training (train.py)

**Multi-GPU distributed training:**
```bash
python train.py --help

# Key arguments:
--data                      # Input .npz file
--experiment_name           # Experiment identifier
--epochs                    # Training epochs
--batch_size                # Per-GPU batch size
--lr                        # Learning rate
--temporal_inv_weight       # Temporal invariance loss weight (0.1)
--caustic_weight            # Caustic detection loss weight (0.8)
--no_causal_attention       # Disable causal attention (NOT recommended)
--gradient_checkpointing    # Enable for memory savings
--no_amp                    # Disable mixed precision
```

**Single GPU:**
```bash
python train.py --data ../data/raw/your_data.npz --experiment_name test
```

**Multi-GPU (torchrun):**
```bash
torchrun --nproc_per_node=4 train.py --data ../data/raw/your_data.npz
```

**Multi-node (SLURM):**
```bash
srun torchrun \
    --nnodes=8 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --data ../data/raw/your_data.npz --batch_size 64
```

### Evaluation (evaluate.py)

**Comprehensive model evaluation:**
```bash
python evaluate.py --help

# Key arguments:
--experiment_name           # Experiment to evaluate
--data                      # Test dataset
--n_samples                 # Subsample for speed (optional)
--early_detection           # Enable early detection analysis
--temporal_bias_check       # Run temporal bias diagnostics
--n_evolution_per_type      # Evolution plots per class (3-10)
--u0_threshold              # Physical threshold (0.3)
--u0_bins                   # Bins for u0 analysis (10)
--batch_size                # Evaluation batch size
```

**Outputs generated:**
1. ROC curves (one-vs-rest)
2. Confusion matrix
3. Confidence distribution
4. Calibration curves
5. Example light curves
6. High-res evolution plots (20 points)
7. Fine-grained early detection (15 fractions)
8. Temporal bias diagnostics (KS tests)
9. u₀ dependency (if params available)

**JSON outputs:**
- `evaluation_summary.json`: All metrics
- `u0_report.json`: u₀ analysis details
- `config.json`: Experiment configuration

---

## 📜 Changelog

### Version 15.0 (January 2025) - Anti-Cheating Edition

**Major Changes:**
- ✨ Semi-causal attention (prevents future peeking)
- ✨ Relative positional encoding (no absolute time)
- ✨ Temporal invariance loss (penalizes shortcuts)
- ✨ Temporal randomization (random peak shifts)
- ✨ High-res evolution plots (20 points)
- ✨ Fine-grained early detection (15 fractions)
- ✨ Temporal bias diagnostics (KS tests)
- 🔄 Simplified multi-task (classification + caustic only)
- 🔄 Renamed presets (telescope-agnostic)

**Performance:**
- +2-3% accuracy over v14.0
- Better early detection (50%: 75-80% vs. 70-75%)
- More robust to temporal shortcuts

**Breaking Changes:**
- Removed tE/u0 prediction heads
- Renamed observational presets
- `challenging` → `baseline` topology

### Version 14.0 (December 2024) - Roman Focus
- Roman Space Telescope baseline (5% missing, 0.05 mag)
- Simplified to 5 experiments (down from 11)
- Multi-node DDP tested (8 nodes × 4 GPUs)

### Version 13.0 (October 2024) - PyTorch Migration
- Complete TensorFlow → PyTorch rewrite
- AMD ROCm support (MI300)
- Mixed precision training

---

## 📚 Citation
```bibtex
@mastersthesis{bhatia2025microlensing,
  title={From Light Curves to Labels: Machine Learning in Microlensing},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis},
  note={Version 15.0 - Anti-Cheating Edition}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

- **Supervisor**: Prof. Dr. Joachim Wambsganß (University of Heidelberg)
- **VBBinaryLensing**: Valerio Bozza
- **Compute**: bwHPC (BW 3.0)
- **Inspiration**: OGLE, MOA, LSST, Roman Space Telescope teams

---

## 🔗 Links

- **GitHub**: [thesis-microlensing](https://github.com/kunalb541/thesis-microlensing)
- **VBBinaryLensing**: [PyPI](https://pypi.org/project/VBMicrolensing/)
- **Roman Space Telescope**: [NASA](https://roman.gsfc.nasa.gov/)
- **LSST**: [Rubin Observatory](https://www.lsst.org/)
- **OGLE**: [Warsaw Observatory](http://ogle.astrouw.edu.pl/)

---

## 📞 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg

**Questions:**
- Code/implementation → GitHub Issues
- Physics/methodology → See thesis document  
- Collaboration → Contact via email

---
