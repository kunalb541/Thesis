# Real-Time Binary Microlensing Classification using Deep Learning

**Master's Thesis Project - WORKING VERSION**  
**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Last Updated**: October 2025

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

**Research Question**: Can deep learning enable real-time classification of binary microlensing events for next-generation surveys (LSST, Roman)?

**Approach**: TimeDistributed 1D CNN trained on 1M+ synthetic light curves from VBMicrolensing.

**Key Innovation**: Temporal aggregation across full light curve captures distributed caustic features → enables early detection.

---

## 🔧 Critical Fix (October 2025)

**Bug**: Original implementation used only final timestep for classification  
**Fix**: Aggregate predictions across all timesteps using `logits = outputs.mean(dim=1)`  
**Impact**: +18.7% accuracy improvement (54.8% → 73.5%)

All code in this repository includes the fix.

---

## 📊 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Environment Setup | ✅ Complete | Tested on NVIDIA/AMD GPUs |
| Data Simulation | ✅ Complete | VBMicrolensing pipeline working |
| Training Pipeline | ✅ Complete | Multi-GPU support, mixed precision |
| Evaluation Pipeline | ✅ Complete | Early detection analysis included |
| Baseline Experiment | 🔄 In Progress | 1M events, wide parameter range |
| Cadence Experiments | ⏳ Queued | 4 experiments (5%-40% missing) |
| Error Experiments | ⏳ Queued | 3 experiments (0.05-0.20 mag) |
| Topology Experiments | ⏳ Queued | 4 experiments (distinct/planetary/stellar) |
| Real-time Benchmarking | ⏳ Queued | Inference speed tests |
| Thesis Writing | ⏳ Not Started | Awaiting experiment completion |

**Tracking**: See `EXPERIMENTS_LOG.md` for detailed progress.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU (CUDA 12.1) or AMD GPU (ROCm 6.0)
- 64 GB RAM recommended
- 100 GB free disk space

### Setup (5 minutes)

```bash
# Clone and navigate
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch (choose your GPU)
# NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# AMD:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install -r requirements.txt

# Verify
python code/preflight_check.py
```

### Run Complete Workflow

```bash
cd code

# 1. Generate data (30 min for 100K events)
python simulate.py \
    --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/test_100k.npz \
    --binary_params baseline

# 2. Train (2-3 hours on 4 GPUs)
python train.py \
    --data ../data/raw/test_100k.npz \
    --output ../models/test_100k.pt \
    --epochs 50 \
    --experiment_name test_100k

# 3. Evaluate
RESULTS_DIR=$(ls -td ../results/test_100k_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/test_100k.npz \
    --output_dir $RESULTS_DIR/evaluation \
    --early_detection

# 4. Real-time benchmark
python benchmark_realtime.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/test_100k.npz \
    --output_dir $RESULTS_DIR/benchmark
```

---

## 📁 Repository Structure

```
Thesis/
├── code/
│   ├── simulate.py              # Dataset generation (VBMicrolensing)
│   ├── train.py                 # Training with temporal aggregation
│   ├── evaluate.py              # Evaluation + early detection
│   ├── benchmark_realtime.py    # Inference speed benchmarking
│   ├── model.py                 # TimeDistributedCNN architecture
│   ├── config.py                # All experiment configurations
│   └── utils.py                 # GPU detection, dataset loading
│
├── data/
│   └── raw/                     # Simulated light curves (.npz)
│
├── models/                      # Saved model checkpoints
├── results/                     # Training logs, evaluation outputs
│
├── docs/
│   ├── SETUP_GUIDE.md          # Installation instructions
│   ├── RESEARCH_GUIDE.md       # Thesis workflow and physics
│   └── QUICK_REFERENCE.md      # Command cheatsheet
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔬 Systematic Experiments

All experiments use the same architecture; only data conditions vary.

### 1. Baseline (Reference)
**Goal**: Establish performance on realistic population  
**Config**: 1M events, 20% missing, 0.10 mag error, wide u₀ range  
**Command**:
```bash
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz --binary_params baseline
python train.py --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline
```

### 2. Cadence Studies (4 experiments)
**Goal**: Quantify observing frequency requirements

| Experiment | Missing % | Coverage | Survey Type |
|------------|-----------|----------|-------------|
| Dense | 5% | 95% | LSST-like |
| Baseline | 20% | 80% | Reference |
| Sparse | 30% | 70% | Poor |
| Very Sparse | 40% | 60% | Minimal |

### 3. Photometric Error Studies (3 experiments)
**Goal**: Test robustness to measurement uncertainty

| Experiment | Error (mag) | Quality |
|------------|-------------|---------|
| Low | 0.05 | Space-based (Roman) |
| Baseline | 0.10 | Ground-based (LSST) |
| High | 0.20 | Poor conditions |

### 4. Binary Topology Studies (3 experiments)
**Goal**: Understand physical detection limits

| Experiment | u₀ range | s range | q range | Expected |
|------------|----------|---------|---------|----------|
| Distinct | 0.001-0.15 | 0.8-1.5 | 0.01-0.5 | High accuracy |
| Planetary | 0.001-0.5 | 0.5-3.0 | 0.0001-0.01 | Moderate |
| Stellar | 0.001-0.8 | 0.3-5.0 | 0.3-1.0 | Challenging |

---

## 🧮 Physics Background (Brief)

### Why Binary Lenses Are Hard to Detect

**Point-Source Point-Lens (PSPL)**: Simple, symmetric magnification curve  
**Binary Lens**: Complex caustic topology creates sharp features

**Key Parameter: u₀ (impact parameter)**
- **u₀ < 0.15**: Source crosses caustics → clearly binary
- **u₀ = 0.15-0.30**: Marginal → sometimes detectable
- **u₀ > 0.30**: No caustic crossing → fundamentally PSPL-like

**Hypothesis**: Model failures will concentrate at large u₀ (physical limit, not algorithmic).

**For detailed physics**: See `docs/RESEARCH_GUIDE.md`

---

---

## 📝 Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Complete installation guide (local + HPC)
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Physics background, experiment design, thesis structure
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Command cheatsheet for all experiments

---

## 🧪 Reproducibility

All experiments are fully reproducible:

1. **Fixed random seeds**: Set in `config.py` and enforced in all scripts
2. **Saved configurations**: All experiment parameters logged
3. **Data permutations**: Saved and reapplied consistently
4. **Exact versions**: See `requirements.txt` for pinned dependencies
5. **Hardware-agnostic**: Works on NVIDIA and AMD GPUs

**To reproduce any experiment**:
```bash
# Use exact configuration from EXPERIMENTS_LOG.md
python simulate.py [EXACT_PARAMS_FROM_LOG]
python train.py [EXACT_PARAMS_FROM_LOG]
```

---

## 📧 Contact

**Author**: Kunal Bhatia  
**Email**: kunal29bhatia@gmail.com  
**Institution**: University of Heidelberg

**For Issues**:
- Code bugs: Open GitHub issue
- Physics questions: See `docs/RESEARCH_GUIDE.md`
- Setup problems: See `docs/SETUP_GUIDE.md`

---

## 📚 Citation

```bibtex
@mastersthesis{bhatia2025realtime,
  title={Real-Time Binary Microlensing Classification using Deep Learning for Survey Operations},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  note={Code available at https://github.com/YOUR_USERNAME/Thesis}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.