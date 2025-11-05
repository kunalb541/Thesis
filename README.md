# Real-Time Binary Microlensing Classification with Transformers

**Deep Learning for Next-Generation Survey Operations - Version 5.3**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Date**: November 2025

---

## Overview

This project implements an automated classification system for binary gravitational microlensing events using **Transformer neural networks with self-attention mechanisms**. With upcoming surveys like LSST and Roman expected to detect 20,000+ microlensing events annually, automated real-time classification becomes essential for triggering follow-up observations.

### Key Features (v5.3)

- **Transformer Architecture**: Self-attention for temporal sequence modeling
- **Sequential Classification**: Per-timestep predictions enabling early detection
- **Distributed Training (DDP)**: Multi-node, multi-GPU support with PyTorch DDP
- **Ultra-Fast Pipeline**: Complete 1M event workflow in ~1 hour
- **Real-time capable**: Sub-millisecond inference per event
- **Production-ready**: Saved normalization parameters ensure reproducible inference
- **Comprehensive Evaluation**: Three-panel visualizations with decision-time analysis
- **Robust DDP**: Fixed checkpoint saving, scaler persistence, and metric aggregation

### What's New in v5.3

- ✅ **Fixed DDP Training**: Proper checkpoint saving with all necessary state
- ✅ **Fixed Scaler Saving**: Scalers saved correctly on rank 0 with proper structure
- ✅ **Consistent Normalization**: Evaluate.py uses same normalization pipeline as training
- ✅ **Fixed Model Loading**: Checkpoint includes model_state_dict, optimizer, and config
- ✅ **Better Documentation**: Corrected architecture description throughout

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

# Validate VBMicrolensing
python code/test_vbm.py
```

### Basic Usage

```bash
cd code

# 1. Generate dataset (1M events, ~10-15 min with 200 workers)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --num_workers 200

# 2. Train with DDP (~30-45 min on 4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4

# 3. Evaluate
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --n_samples 20

# 4. Benchmark real-time performance
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

## Model Architecture

### Transformer Classifier

```
Input: [batch, 1, T=1500]
   ↓
1D Convolution Downsampling (factor: 3)
   [batch, 1, 1500] → [batch, d_model, 500]
   Purpose: Reduce sequence length for computational efficiency
   ↓
Positional Encoding
   Adds temporal position information
   ↓
Transformer Encoder (L=2 layers)
   - Multi-Head Self-Attention (H=4 heads)
     Captures long-range temporal dependencies
   - Feed-Forward Network (d_ff=256)
   - Layer Normalization
   - Residual connections
   ↓
Classification Heads:
   - Per-Timestep Head: For sequential decision-making
   - Global Pooling Head: For final classification
   ↓
Output: [batch, 2] → {PSPL, Binary}
```

**Key Design Choices**:
- **Conv1D Downsampling**: Efficiently reduces sequence length 3× while preserving temporal structure
- **Self-Attention**: Captures complex temporal patterns in light curves without recurrence
- **Dual Output Modes**: 
  - `return_sequence=False`: Final classification (global pooling)
  - `return_sequence=True`: Per-timestep predictions for early detection
- **Padding Masking**: Properly handles missing observations (-1.0 padding) throughout pipeline

**Why Transformer?**
- **Long-range dependencies**: Binary caustic features can span hundreds of timesteps
- **Parallel processing**: Unlike RNNs, processes entire sequence in parallel
- **Attention maps**: Interpretable - can visualize which timesteps drive classification

---

## Distributed Training (DDP)

### Single Node (4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4
```

### Multi-Node (SLURM)

```bash
#!/bin/bash
# On SLURM cluster
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4
```

### DDP Debugging

```bash
# Enable comprehensive debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check GPU visibility
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Test with single GPU first
torchrun --nproc_per_node=1 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline_test \
    --epochs 2 \
    --batch_size 64

# Network interface issues (SLURM)
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
```

---

## Evaluation & Visualization

The evaluation script generates all visualizations:

```bash
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --n_samples 20 \
    --confidence_threshold 0.8
```

### Generated Outputs

**1. Three-Panel Sample Plots** (`samples/sample_XXXX.png`):
- Panel 1: Original light curve data with decision line
- Panel 2: Model input view (normalized) with decision line  
- Panel 3: Class probabilities over time (clamped after decision)

**2. Confusion Matrix** (`confusion_matrix.png`)

**3. Decision Time Distribution** (`decision_time_distribution.png`)

**4. Accuracy vs Decision Time** (`accuracy_vs_decision_time.png`)

**5. ROC Curve** (`roc_curve.png`)

**6. Evaluation Summary** (`evaluation_summary.json`):
```json
{
  "accuracy": 0.7234,
  "roc_auc": 0.7891,
  "decision_time_mean": 245.3,
  "decision_time_median": 198.0,
  "confidence_threshold": 0.8
}
```

---

## Research Questions Addressed

### 1. **Baseline Performance**
What classification accuracy is achievable across realistic binary parameter distributions?

**Answer**: 70-75% accuracy on mixed population (planetary + stellar binaries)

### 2. **Early Detection Capability**
How early can we reliably identify binary events with partial light curves?

**Answer**: 68-72% accuracy with only 50% of observations, enabling follow-up trigger decisions hours to days earlier

### 3. **Physical Detection Limits**
What are the fundamental limits imposed by binary topology?

**Answer**: Impact parameter u₀ > 0.3 represents physical boundary—these events are intrinsically PSPL-like (~20% of binaries)

### 4. **Observational Dependence**
How do cadence and photometric quality affect performance?

**Cadence Study**:
- Dense (5% missing): 75-80% accuracy
- Standard (20% missing): 70-75% accuracy  
- Sparse (40% missing): 60-65% accuracy

**Photometric Quality**:
- Space-based (0.05 mag): 75-80% accuracy
- Ground-based (0.10 mag): 70-75% accuracy
- Poor conditions (0.20 mag): 65-70% accuracy

### 5. **Real-Time Feasibility**
Can this run in production survey pipelines?

**Answer**: <1 ms inference per event → 10,000+ LSST alerts/night processable on single GPU (~1000× faster than traditional fitting)

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py           # Fast parallel simulation
│   ├── train.py              # DDP Transformer training (v5.3 fixed)
│   ├── evaluate.py           # Complete evaluation + plots (v5.3 fixed)
│   ├── benchmark_realtime.py # Performance testing
│   ├── model.py              # Transformer architecture
│   ├── config.py             # Configuration
│   ├── utils.py              # Utilities (v5.3 fixed)
│   └── visualize.py          # Visualization functions
│
├── data/
│   └── raw/                  # Simulated datasets (.npz)
│
├── results/                  # Experiment outputs
│   └── {experiment}_{timestamp}/
│       ├── best_model.pt     # Includes model_state_dict, optimizer, config
│       ├── config.json
│       ├── scaler_standard.pkl
│       ├── scaler_minmax.pkl
│       └── evaluation/
│
├── docs/
│   ├── SETUP_GUIDE.md
│   ├── RESEARCH_GUIDE.md
│   └── QUICK_REFERENCE.md
│
└── README.md
```

---

## Expected Performance (v5.3)

### Timing (4 GPUs)

| Task | Time | Notes |
|------|------|-------|
| Simulate 1M | 10-15 min | 200 workers |
| Train DDP 1M | 30-45 min | 4 GPUs |
| Evaluate | 2-5 min | Includes all plots |

### Accuracy (Expected Ranges)

| Experiment | Test Accuracy | Notes |
|------------|---------------|-------|
| Baseline (1M) | 70-75% | Mixed parameters |
| Dense (5%) | 75-80% | Intensive cadence |
| Sparse (30%) | 65-70% | Poor coverage |
| Low Error (0.05) | 75-80% | Space-based |
| Distinct | 80-90% | Clear caustics |

---

## Troubleshooting

### DDP Issues

**Only using 1 GPU**:
```bash
# Verify torchrun detects all GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Ensure proper torchrun call
torchrun --nproc_per_node=4 train.py [args]
```

**Training hangs**:
```bash
# Enable debug
export NCCL_DEBUG=INFO

# Check network
export NCCL_SOCKET_IFNAME=eth0
```

### Normalization Issues

**Data range incorrect**:
```bash
# Check raw data
python -c "
import numpy as np
data = np.load('../data/raw/baseline_1M.npz')
X = data['X']
print(f'Range: [{X.min():.3f}, {X.max():.3f}]')
"

# Verify scalers exist
ls -lh ../results/baseline_*/scaler_*.pkl
```

---

## Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Installation + DDP setup
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Systematic experiments
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Command cheatsheet

---

## Citation

```bibtex
@mastersthesis{bhatia2025realtime,
  title={Real-Time Binary Microlensing Classification using Transformers},
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

---

## Acknowledgments

- VBMicrolensing library by Valerio Bozza
- PyTorch team for Transformer and DDP frameworks
- University of Heidelberg for computational resources