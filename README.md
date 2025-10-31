# Real-Time Binary Microlensing Classification

**Deep Learning for Next-Generation Survey Operations**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Date**: October 2025

---

## Overview

This project implements an automated classification system for binary gravitational microlensing events using deep learning. With upcoming surveys like LSST and Roman expected to detect 20,000+ microlensing events annually, automated real-time classification becomes essential for triggering follow-up observations.

### Key Features

- **TimeDistributed CNN** architecture for sequential classification
- **Real-time capable**: Sub-millisecond inference per event
- **Early detection**: Classification with partial light curves
- **Comprehensive benchmarking**: Performance across diverse observing conditions
- **Production-ready**: Saved normalization parameters ensure reproducible inference

### Research Questions

1. What classification accuracy is achievable across realistic binary systems?
2. How does observing cadence impact detection performance?
3. How early can we reliably identify binary events?
4. What are the fundamental physical limits for distinguishing binary from PSPL events?

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended) or AMD GPU
- 64 GB RAM recommended
- 100 GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch (choose based on your GPU)
# NVIDIA CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm 6.0:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install -r requirements.txt

# Verify installation
python code/utils.py
```

### Basic Usage

```bash
cd code

# 1. Generate synthetic dataset (1M events, ~2 hours)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline

# 2. Train model (~6-8 hours on 4 GPUs)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50

# 3. Evaluate
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection

# 4. Benchmark real-time performance
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py           # Dataset generation (VBMicrolensing)
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   ├── benchmark_realtime.py # Inference benchmarking
│   ├── plot_samples.py       # Visualization
│   ├── model.py              # CNN architecture
│   ├── config.py             # Configuration
│   └── utils.py              # Utilities
│
├── data/
│   └── raw/                  # Simulated light curves (.npz)
│
├── results/                  # Auto-generated experiment outputs
│   └── {experiment}_{timestamp}/
│       ├── best_model.pt     # Trained model
│       ├── config.json       # Experiment configuration
│       ├── training.log      # Training logs
│       ├── summary.json      # Metrics summary
│       ├── scaler_standard.pkl  # Normalization parameters
│       ├── scaler_minmax.pkl    # Normalization parameters
│       └── evaluation/       # Evaluation results
│
├── docs/
│   ├── SETUP_GUIDE.md        # Detailed installation
│   ├── RESEARCH_GUIDE.md     # Experimental workflow
│   └── QUICK_REFERENCE.md    # Command cheatsheet
│
└── README.md                 # This file
```

---

## Systematic Experiments

### Baseline Experiment

**Purpose**: Establish performance across mixed binary parameter ranges

```bash
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz --binary_params baseline

python train.py --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline --epochs 50
```

**Expected Results**:
- Test Accuracy: 70-75%
- ROC AUC: 0.78-0.82
- Inference time: <1 ms per event

### Cadence Experiments

Test how observation frequency affects performance:

| Experiment | Missing Obs | Expected Accuracy |
|------------|-------------|-------------------|
| Dense      | 5%          | 75-80%            |
| Baseline   | 20%         | 70-75%            |
| Sparse     | 30%         | 65-70%            |
| Very Sparse| 40%         | 60-65%            |

```bash
# Example: Dense cadence
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/cadence_dense.npz \
    --cadence_mask_prob 0.05

python train.py --data ../data/raw/cadence_dense.npz \
    --experiment_name cadence_dense --epochs 50
```

### Photometric Error Experiments

Test how measurement precision affects performance:

```bash
# Low error (space-based quality)
python simulate.py --mag_error_std 0.05 \
    --output ../data/raw/error_low.npz

# High error (poor ground conditions)
python simulate.py --mag_error_std 0.20 \
    --output ../data/raw/error_high.npz
```

### Binary Topology Experiments

Test performance across different binary mass ratios:

```bash
# Distinct caustic-crossing events (easiest)
python simulate.py --binary_params distinct \
    --output ../data/raw/distinct.npz

# Planetary systems
python simulate.py --binary_params planetary \
    --output ../data/raw/planetary.npz

# Stellar binaries (hardest)
python simulate.py --binary_params stellar \
    --output ../data/raw/stellar.npz
```

---

## Model Architecture

### TimeDistributed CNN

The model processes light curves sequentially, making predictions at each timestep:

```
Input: [batch, 1, sequence_length] 
   ↓
Conv1D (9, 128) + BatchNorm + ReLU + Dropout
   ↓
Conv1D (7, 64) + BatchNorm + ReLU + Dropout
   ↓
Conv1D (5, 32) + BatchNorm + ReLU + Dropout
   ↓
TimeDistributed(FC 64 + ReLU + Dropout)
   ↓
TimeDistributed(FC 2)
   ↓
Output: [batch, sequence_length, 2]
```

**Key Design Choices**:
- Per-timestep predictions enable early detection
- Mean/max pooling aggregates across full light curve
- Captures distributed caustic features

---

## Dataset Generation

Synthetic light curves are generated using [VBMicrolensing](https://github.com/valboz/VBBinaryLensing):

### PSPL (Point-Source Point-Lens) Events

Parameters sampled uniformly:
- Impact parameter u₀: 0.01 - 1.0
- Einstein timescale tE: 10 - 150 days
- Event time t₀: 0 - 1000 days

### Binary Events

Configurable parameter ranges for different experiments:

**Baseline** (mixed difficulty):
- Separation s: 0.1 - 2.5
- Mass ratio q: 0.1 - 1.0
- Impact parameter u₀: 0.01 - 0.5
- Source size ρ: 0.01 - 0.1

**Distinct** (clear caustics):
- Separation s: 0.8 - 1.5
- Mass ratio q: 0.1 - 0.5
- Impact parameter u₀: 0.01 - 0.15

See `config.py` for full parameter definitions.

---

## Evaluation Metrics

The system provides comprehensive performance metrics:

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC
- Precision-Recall Curve
- Confusion Matrix

### Early Detection Analysis
- Performance at 10%, 25%, 50%, 67%, 83%, 100% observation completeness
- Shows real-time classification capability

### Decision Time Analysis
- Average timesteps until confident classification
- Accuracy vs. decision time curves
- Confidence threshold sweep (0.5 - 1.0)

### Real-Time Performance
- Inference latency per event
- Throughput (events/second)
- GPU memory usage

---

## Results

Results for each experiment are automatically saved in timestamped directories:

```
results/baseline_20251027_143022/
├── best_model.pt              # Best model checkpoint
├── config.json                # Experiment parameters
├── training.log               # Training history
├── summary.json               # Final metrics
├── scaler_standard.pkl        # StandardScaler parameters
├── scaler_minmax.pkl          # MinMaxScaler parameters
└── evaluation/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── accuracy_vs_decision_time.png
    ├── early_detection.png
    └── evaluation_summary.json
```

**Key Files**:
- `scaler_*.pkl`: Normalization parameters fitted on training data
  - Critical for consistent evaluation and deployment
  - Loaded automatically during evaluation
- `evaluation_summary.json`: Complete metrics and analysis

---

## Data Preprocessing

The pipeline uses two-stage normalization for stable training:

1. **StandardScaler**: Z-score normalization (zero mean, unit variance)
2. **MinMaxScaler**: Scale to [0, 1] range

**Important**: 
- Scalers are fitted **only** on training data (no data leakage)
- Same scalers applied to validation, test, and evaluation sets
- Scalers saved with model for consistent inference

```python
# During training (automatic)
X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = \
    two_stage_normalize(X_train, X_val, X_test)

# During evaluation (automatic)
scaler_std, scaler_mm = load_scalers(model_dir)
X_test_scaled = apply_scalers_to_data(X_test, scaler_std, scaler_mm)
```

---

## Configuration

All experiments are configured in `config.py`:

```python
# Model architecture
CONV1_FILTERS = 128
CONV2_FILTERS = 64
CONV3_FILTERS = 32
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

# Data splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Binary parameter sets
BINARY_PARAM_SETS = {
    'baseline': {...},
    'distinct': {...},
    'planetary': {...},
    'stellar': {...},
}
```

---

## Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Complete installation instructions for local and HPC
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Systematic experimental workflow for thesis
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Command cheatsheet for all experiments

---

## Performance Tips

### For Training

- **Multi-GPU**: Automatically uses DataParallel for 2+ GPUs
- **Mixed Precision**: Enabled by default with torch.cuda.amp
- **Batch Size**: 128 works well on 24GB GPUs; adjust for your hardware
- **Data Loading**: Uses persistent workers for efficiency

### For Inference

- **Batch Processing**: Process 128-256 events simultaneously
- **GPU Warmup**: First few batches are slower (GPU initialization)
- **CPU Fallback**: Works without GPU but ~100× slower

---

## Reproducing Results

All experiments are fully reproducible:

1. **Fixed random seeds**: Set in config.py and enforced throughout
2. **Saved configurations**: Every run logs all parameters to config.json
3. **Saved scalers**: Normalization parameters preserved for consistent evaluation
4. **Data permutations**: Shuffling applied consistently via saved permutation arrays
5. **Exact versions**: See requirements.txt for all dependencies

To reproduce a specific experiment:

```bash
# Load configuration from previous run
CONFIG_FILE=results/baseline_20251027_143022/config.json

# Extract parameters and re-run
python train.py \
    --data $(python -c "import json; print(json.load(open('$CONFIG_FILE'))['data_path'])") \
    --experiment_name baseline_reproduction \
    --batch_size $(python -c "import json; print(json.load(open('$CONFIG_FILE'))['batch_size'])")
```

---

## Citation

```bibtex
@mastersthesis{bhatia2025realtime,
  title={Real-Time Binary Microlensing Classification using Deep Learning},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  note={Code: https://github.com/YOUR_USERNAME/Thesis}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contact

**Kunal Bhatia**  
kunal29bhatia@gmail.com  
University of Heidelberg

**For Issues**:
- Installation problems: See [SETUP_GUIDE.md](docs/SETUP_GUIDE.md)
- Research workflow: See [RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)
- Code bugs: Open GitHub issue

---

## Acknowledgments

- VBMicrolensing library by Valerio Bozza
- PyTorch team for deep learning framework
- University of Heidelberg for computational resources