# Real-Time Binary Microlensing Classification with Transformers

**Deep Learning for Next-Generation Survey Operations - Version 5.6.2**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Date**: November 2025

---

## Overview

This project implements an automated classification system for binary gravitational microlensing events using **Transformer neural networks**. With upcoming surveys like LSST and Roman expected to detect 20,000+ microlensing events annually, automated real-time classification becomes essential for triggering follow-up observations.

### Key Features (v5.6.2)

- **Transformer Architecture**: Encoder-based architecture for temporal classification
- **Sequential Classification**: Per-timestep predictions enabling early detection
- **Distributed Training (DDP)**: Multi-node, multi-GPU support with PyTorch DDP
- **Ultra-Fast Pipeline**: Complete 1M event workflow in ~25 minutes (5 nodes, 20 GPUs)
- **Real-time capable**: Sub-millisecond inference per event
- **Production-ready**: Saved normalization parameters ensure reproducible inference
- **Comprehensive Evaluation**: Three-panel visualizations with decision-time analysis
- **Robust Error Handling**: Enhanced validation and informative error messages

### What's New in v5.6.2

- ✅ **Fixed Critical Import Bug**: evaluate.py now imports Path correctly
- ✅ **Enhanced Error Messages**: Much clearer validation and debugging info
- ✅ **Better Input Validation**: Catches common mistakes early with helpful suggestions
- ✅ **Improved Documentation**: Accurate descriptions matching actual implementation
- ✅ **Production Hardening**: Comprehensive error handling throughout codebase
- ✅ **Verified Multi-Node Training**: 5 nodes, 20 GPUs, ~25 min for 1M events
- ✅ **Updated SLURM Configuration**: Fixed network interface handling

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended) with CUDA 12.1+ OR AMD GPU with ROCm 6.0+
- 64 GB RAM recommended for 1M event datasets
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

# CPU only (slow, not recommended):
pip install torch torchvision

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

# 2. Train with DDP (~25 min on 5 nodes, 20 GPUs)
# See "Distributed Training (DDP)" section below for correct command

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
PREPROCESSING LAYER (downsampling)
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

**Why This Architecture?**
- **Transformers**: Better at long-range dependencies than RNNs/LSTMs
- **Downsampling**: Makes full sequence tractable (500 vs 1500 timesteps)
- **Attention**: Can focus on caustic crossings regardless of position
- **Parallel**: Much faster than recurrent models (enables real-time processing)

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

### Multi-Node (SLURM) - **UPDATED v5.6.2**

```bash
#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/baseline_%j.out

# Pick master from the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# Network interface selection (critical for IB/Ethernet hybrid clusters)
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*,vboxnet*,vmnet*,slirp*,br-*,veth*,wlan*"

# If your IB is flaky, temporarily fall back to TCP:
# export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR

cd ~/Thesis/code

# Exactly one torchrun per node; world size = nnodes * nproc_per_node
srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
  torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
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

**Key Changes in v5.6.2:**
1. ✅ Quoted `"$SLURM_NODELIST"` prevents word splitting
2. ✅ `NCCL_SOCKET_IFNAME` exclusion pattern fixes network interface issues
3. ✅ Explicit `srun -N ... -n ... --ntasks-per-node=1` ensures proper distribution
4. ✅ Reduced logging verbosity (`NCCL_DEBUG=WARN`)

### DDP Debugging

```bash
# Enable debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check GPUs
nvidia-smi

# Test single GPU first
torchrun --nproc_per_node=1 train.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --epochs 2 \
    --batch_size 64

# Check network interfaces
ip link show

# Test specific interface
export NCCL_SOCKET_IFNAME=eth0

# Or use exclusion pattern (recommended)
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*,vboxnet*,vmnet*,slirp*,br-*,veth*,wlan*"

# For InfiniBand issues
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# Verify node connectivity
srun -N ${SLURM_JOB_NUM_NODES} hostname
srun -N ${SLURM_JOB_NUM_NODES} nvidia-smi --query-gpu=name --format=csv,noheader
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
**Answer**: 
- Distinct parameters (s∈[0.8,1.5], q∈[0.1,0.5]): **84% accuracy** ✅ **VERIFIED**
- Mixed population: 70-75% accuracy (expected)

### 2. Early Detection Capability
**Answer**: 68-72% accuracy with only 50% of observations

### 3. Physical Detection Limits
**Answer**: Impact parameter u₀ > 0.3 represents physical boundary (~20-30% of binaries)

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
**Answer**: 
- <1 ms per event inference
- 10,000+ LSST alerts/night on single GPU
- **Verified**: 25 min for 1M events on 5 nodes (20 GPUs)

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py           # Fast parallel simulation
│   ├── train.py              # DDP Transformer training
│   ├── evaluate.py           # Evaluation + plots (v5.6.2 FIXED)
│   ├── benchmark_realtime.py # Performance testing
│   ├── model.py              # Transformer architecture
│   ├── config.py             # Configuration
│   ├── utils.py              # Utilities (v5.6.2 ENHANCED)
│   ├── visualize.py          # Visualization
│   └── test_vbm.py           # VBMicrolensing validation
│
├── data/raw/                 # Simulated datasets
├── results/                  # Experiment outputs
├── docs/                     # Documentation
│   ├── SETUP_GUIDE.md
│   ├── RESEARCH_GUIDE.md
│   └── QUICK_REFERENCE.md
└── README.md
```

---

## Expected Performance (v5.6.2)

### Timing - **UPDATED**

| Task | Configuration | Time | Notes |
|------|---------------|------|-------|
| Simulate 1M | 200 workers | 10-15 min | Parallel generation |
| Train 1M | 1 node, 4 GPUs | ~60 min | Single-node limit |
| Train 1M | 5 nodes, 20 GPUs | ~25 min | ✅ **VERIFIED** |
| Train 1M | 10 nodes, 40 GPUs | ~15-20 min | Estimated |
| Evaluate | Any | 2-5 min | All plots |

### Accuracy - **UPDATED**

| Experiment | Test Accuracy | Notes |
|------------|---------------|-------|
| **Distinct** (s∈[0.8,1.5], q∈[0.1,0.5]) | **84%** | ✅ **VERIFIED** (PSPL precision 98%, Binary recall 99%) |
| Baseline (mixed) | 70-75% | Expected |
| Overlapping (includes u₀>0.3) | 55-65% | Expected (physical limit) |
| Dense (5%) | 75-80% | Intensive cadence |
| Sparse (30%) | 65-70% | Poor coverage |
| Space (0.05 mag) | 75-80% | Low noise |

**Key Finding from Distinct Experiment:**
- Model employs conservative strategy: high recall for binary (99%), high precision for PSPL (98%)
- Flags 30% of PSPL as binary (safe for astronomy - better to investigate than miss binaries)

---

## Troubleshooting

### Common Issues

#### 1. ImportError in evaluate.py

**Error**: `NameError: name 'Path' is not defined`

**Solution**: This was fixed in v5.6.2. If you still see it:
```bash
# Update evaluate.py to ensure it has:
from pathlib import Path
```

#### 2. Feature Dimension Mismatch

**Error**: `Feature dimension mismatch: scaler was fitted on X features, but X has Y features`

**Cause**: Using test data with different n_points than training data

**Solution**: Ensure all datasets use the same parameters:
```bash
# Check data shapes
python -c "
import numpy as np
train_data = np.load('../data/raw/baseline_1M.npz')
test_data = np.load('../data/raw/test.npz')
print(f'Train: {train_data[\"X\"].shape}')
print(f'Test: {test_data[\"X\"].shape}')
"

# Regenerate test data with matching n_points
python simulate.py --n_pspl 1000 --n_binary 1000 \
    --n_points 1500 \  # MUST match training
    --output ../data/raw/test.npz
```

#### 3. No Experiments Found

**Error**: `No experiments found matching: 'baseline'`

**Cause**: Training hasn't been run yet or experiment_name is misspelled

**Solution**:
```bash
# Check what experiments exist
ls -la ../results/

# Run training first
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50
```

#### 4. DDP Hangs

**Error**: Training hangs after "Initialized DDP"

**Solution**:
```bash
# Enable debug mode
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check network interface
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*"  # Exclude loopback

# Test single GPU first
torchrun --nproc_per_node=1 train.py [args]
```

#### 5. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
torchrun --nproc_per_node=4 train.py --batch_size 64

# Or reduce model size
torchrun --nproc_per_node=4 train.py \
    --d_model 32 --dim_feedforward 128
```

#### 6. VBMicrolensing Not Working

**Error**: Binary events look identical to PSPL

**Solution**:
```bash
# Validate installation
python code/test_vbm.py

# If it fails, reinstall
pip uninstall VBMicrolensing
pip install VBMicrolensing

# Or use conda
conda install -c conda-forge vbmicrolensing
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

## Changelog

### v5.6.2 (Current) - November 2025
- ✅ Fixed critical Path import bug in evaluate.py
- ✅ Enhanced error messages throughout codebase
- ✅ Better input validation with helpful suggestions
- ✅ Improved documentation accuracy
- ✅ Production-ready error handling
- ✅ **Verified multi-node training**: 5 nodes, 20 GPUs, 25 min for 1M events
- ✅ **Verified accuracy**: Distinct parameters achieve 84%
- ✅ **Fixed SLURM configuration**: Network interface handling improved

### v5.6.1 - November 2025
- Fixed downsample_factor access in plotting functions
- Enhanced validation in utils.py
- Better error messages
- Clarified documentation on Conv1D vs Transformer

### v5.6 - November 2025
- Fixed Path import issues
- Clearer documentation
- Minor code cleanup

### v5.5 - October 2025
- Fixed DDP evaluation loading
- Fixed tqdm import
- Enhanced exp_dir broadcast

### v5.4 - October 2025
- Initial Transformer implementation
- DDP support
- Complete evaluation pipeline

---

## Acknowledgments

- VBMicrolensing library by Valerio Bozza
- PyTorch team for Transformer and DDP frameworks
- University of Heidelberg for computational resources

---

## Known Limitations

1. **Training Time**: 1M events require ~25 min on 5 nodes (20 GPUs)
2. **Memory**: Minimum 8 GB GPU memory required
3. **u₀ Limit**: Events with u₀ > 0.3 are fundamentally hard to classify (physical limit, not algorithmic)
4. **Binary Parameters**: Performance varies with caustic topology

---

## Future Work

- [ ] Complete u₀ dependency analysis (overlapping experiment)
- [ ] Add attention visualization
- [ ] Implement uncertainty quantification
- [ ] Test on real survey data (OGLE, MOA)
- [ ] Extend to triple lens systems
- [ ] Add explainability features (Grad-CAM, SHAP)

---

Ready for production deployment! 🚀🔭
