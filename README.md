# Real-Time Binary Microlensing Classification with Transformers

**Deep Learning for Next-Generation Survey Operations - Version 5.5**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Date**: November 2025

---

## Overview

This project implements an automated classification system for binary gravitational microlensing events using **Transformer neural networks**. With upcoming surveys like LSST and Roman expected to detect 20,000+ microlensing events annually, automated real-time classification becomes essential for triggering follow-up observations.

### Key Features (v5.5)

- **Transformer Architecture**: Encoder-based architecture for temporal classification
- **Sequential Classification**: Per-timestep predictions enabling early detection
- **Distributed Training (DDP)**: Multi-node, multi-GPU support with PyTorch DDP
- **Ultra-Fast Pipeline**: Complete 1M event workflow in ~1 hour
- **Real-time capable**: Sub-millisecond inference per event
- **Production-ready**: Saved normalization parameters ensure reproducible inference
- **Comprehensive Evaluation**: Three-panel visualizations with decision-time analysis
- **Robust DDP**: Fixed checkpoint saving, scaler persistence, and metric aggregation

### What's New in v5.5

- ✅ **Fixed Path Import**: evaluate.py now imports Path correctly
- ✅ **Clearer Documentation**: Consistent terminology (Conv1D preprocessing vs Transformer model)
- ✅ **Minor Code Cleanup**: Improved variable naming consistency

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
Input: [batch, 1, T=1500] light curve time series
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREPROCESSING LAYER (downsampling only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1D Convolution (Reduce sequence length)
   [batch, 1, 1500] → [batch, d_model=64, 500]
   
   Purpose: Reduce sequence length by 3×
   - Makes computation tractable
   - Projects to embedding dimension
   - NOT the classification model itself
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSFORMER ENCODER (the actual model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Positional Encoding
   Adds position information to embeddings
   ↓
Transformer Encoder Stack (L=2 layers)
   Each layer contains:
   • Multi-Head Attention (H=4 heads)
     - Computes attention across all timesteps
     - Captures temporal dependencies
   • Feed-Forward Network (d_ff=256)
     - Two linear layers with GELU activation
   • Layer Normalization + Residual Connections
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT LAYERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two classification modes:

Mode 1: Sequential (return_sequence=True)
   Per-Timestep Head → [batch, 500, 2]
   Used for: Early detection analysis
   
Mode 2: Final (return_sequence=False)  
   Global Pooling → [batch, 64]
   Classification Head → [batch, 2]
   Used for: Final PSPL vs Binary decision
   ↓
Output: Class probabilities {PSPL, Binary}
```

**Key Points**:
1. **Conv1D is just preprocessing** - reduces 1500 → 500 timesteps for efficiency
2. **Transformer is the actual model** - processes the 500-length sequences
3. **Multi-head attention** captures temporal patterns in light curves
4. **No recurrence** - entire sequence processed in parallel (unlike RNNs)

**Why This Architecture?**
- **Transformers**: Better at long-range dependencies than RNNs/LSTMs
- **Downsampling**: Makes full sequence tractable (500 vs 1500 timesteps)
- **Attention**: Can focus on caustic crossings regardless of position
- **Parallel**: Much faster than recurrent models

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
# Enable debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check GPUs
nvidia-smi

# Test single GPU first
torchrun --nproc_per_node=1 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name test \
    --epochs 2 \
    --batch_size 64

# Network interface (SLURM)
export NCCL_SOCKET_IFNAME=eth0
```

---

## Evaluation & Visualization

```bash
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --n_samples 20 \
    --confidence_threshold 0.8
```

### Generated Outputs

1. **Three-Panel Sample Plots** (`samples/sample_XXXX.png`)
2. **Confusion Matrix** (`confusion_matrix.png`)
3. **Decision Time Distribution** (`decision_time_distribution.png`)
4. **Accuracy vs Decision Time** (`accuracy_vs_decision_time.png`)
5. **ROC Curve** (`roc_curve.png`)
6. **Evaluation Summary** (`evaluation_summary.json`)

---

## Research Questions Addressed

### 1. Baseline Performance
**Answer**: 70-75% accuracy on mixed population (planetary + stellar binaries)

### 2. Early Detection Capability
**Answer**: 68-72% accuracy with only 50% of observations

### 3. Physical Detection Limits
**Answer**: Impact parameter u₀ > 0.3 represents physical boundary (~20% of binaries)

### 4. Observational Dependence

**Cadence**:
- Dense (5% missing): 75-80%
- Standard (20% missing): 70-75%  
- Sparse (40% missing): 60-65%

**Photometric Quality**:
- Space-based (0.05 mag): 75-80%
- Ground-based (0.10 mag): 70-75%
- Poor (0.20 mag): 65-70%

### 5. Real-Time Feasibility
**Answer**: <1 ms per event → 10,000+ LSST alerts/night on single GPU

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py           # Fast parallel simulation
│   ├── train.py              # DDP Transformer training (v5.5)
│   ├── evaluate.py           # Evaluation + plots (v5.5)
│   ├── benchmark_realtime.py # Performance testing
│   ├── model.py              # Transformer architecture
│   ├── config.py             # Configuration
│   ├── utils.py              # Utilities (v5.5)
│   └── visualize.py          # Visualization
│
├── data/raw/                 # Simulated datasets
├── results/                  # Experiment outputs
├── docs/                     # Documentation
└── README.md
```

---

## Expected Performance (v5.5)

### Timing (4 GPUs)

| Task | Time | Notes |
|------|------|-------|
| Simulate 1M | 10-15 min | 200 workers |
| Train DDP 1M | 30-45 min | 4 GPUs |
| Evaluate | 2-5 min | All plots |

### Accuracy

| Experiment | Test Accuracy | Notes |
|------------|---------------|-------|
| Baseline | 70-75% | Mixed parameters |
| Dense (5%) | 75-80% | Intensive cadence |
| Sparse (30%) | 65-70% | Poor coverage |
| Space (0.05 mag) | 75-80% | Low noise |
| Distinct | 80-90% | Clear caustics |

---

## Troubleshooting

### DDP Issues

```bash
# Verify GPU count
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Enable debug
export NCCL_DEBUG=INFO

# Network interface
export NCCL_SOCKET_IFNAME=eth0
```

### Normalization Issues

```bash
# Check data
python -c "
import numpy as np
data = np.load('../data/raw/baseline_1M.npz')
X = data['X']
print(f'Range: [{X.min():.3f}, {X.max():.3f}]')
"

# Verify scalers
ls -lh ../results/baseline_*/scaler_*.pkl
```

---

## Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Installation + DDP
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Experiments
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Commands

---

## Citation

```bibtex
@mastersthesis{bhatia2025realtime,
  title={Real-Time Binary Microlensing Classification using Transformers},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg}
}
```

---

## License

MIT License - See [LICENSE](LICENSE)

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