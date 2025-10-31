# Real-Time Binary Microlensing Classification

**Deep Learning for Next-Generation Survey Operations - Version 4.0**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Date**: October 2025

---

## Overview

This project implements an automated classification system for binary gravitational microlensing events using deep learning. With upcoming surveys like LSST and Roman expected to detect 20,000+ microlensing events annually, automated real-time classification becomes essential for triggering follow-up observations.

### Key Features (v4.0)

- **Distributed Training (DDP)**: Multi-node, multi-GPU support with PyTorch DDP
- **Ultra-Fast Pipeline**: Complete 1M event workflow in ~1 hour
- **TimeDistributed CNN** architecture for sequential classification
- **Real-time capable**: Sub-millisecond inference per event
- **Production-ready**: Saved normalization parameters ensure reproducible inference
- **Comprehensive benchmarking**: Performance across diverse observing conditions

### What's New in v4.0

- ⚡ **10× faster simulation**: Parallel processing with 200+ workers
- 🚀 **DDP training**: Multi-node distributed training support
- 🎯 **Cleaner output**: Suppressed warnings, better logging
- 📊 **Fixed evaluation**: Complete metrics and visualization pipeline
- 🔧 **Optimized defaults**: Learning rate 1e-4, batch size 128

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
# Single node:
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4

# 3. Evaluate
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz

# 4. Benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

## Distributed Training

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

### Performance Tips

- **Batch size**: Use 128 for 4 GPUs (32 per GPU)
- **Learning rate**: 1e-4 works best for DDP training
- **Workers**: Set `--num_workers 4` per GPU
- **Memory**: Each GPU needs ~6-8 GB

---

## Systematic Experiments

### Complete Pipeline (~1 hour total)

```bash
cd code

# 1. Simulate all datasets (~15-20 min)
# Baseline
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline --num_workers 200

# Cadence experiments
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence --num_workers 200
done

# Error experiments
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${name}.npz \
        --mag_error_std $error --num_workers 200
done

# Topology experiments
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} --num_workers 200
done

# 2. Train all models with DDP (~40-50 min)
# Baseline
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 --batch_size 128 --lr 1e-4

# Cadence
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Error
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/error_${name}.npz \
        --experiment_name error_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Topology
for topo in distinct planetary stellar; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/${topo}.npz \
        --experiment_name ${topo} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# 3. Evaluate all (~5-10 min)
for exp in baseline cadence_* error_* distinct planetary stellar; do
    python evaluate.py --experiment_name $exp \
        --data ../data/raw/${exp}.npz 2>/dev/null || \
    python evaluate.py --experiment_name $exp \
        --data ../data/raw/baseline_1M.npz 2>/dev/null
done
```

---

## Expected Performance (v4.0)

### Timing (4 GPUs)

| Task | Time | Notes |
|------|------|-------|
| Simulate 1M | 10-15 min | 200 workers |
| Simulate 200K | 2-3 min | Per experiment |
| Train 1M | 30-45 min | DDP, 4 GPUs |
| Train 200K | 8-12 min | Per experiment |
| Evaluate | 1-2 min | Per experiment |
| **Total Pipeline** | **~1 hour** | All experiments |

### Accuracy

| Experiment | Test Accuracy | Notes |
|------------|---------------|-------|
| Baseline (1M) | 70-75% | Mixed parameters |
| Dense (5%) | 75-80% | Intensive cadence |
| Sparse (30%) | 65-70% | Poor coverage |
| Very Sparse (40%) | 60-65% | Limited data |
| Low Error (0.05) | 75-80% | Space-based |
| High Error (0.20) | 65-70% | Ground-based |
| Distinct | 80-90% | Clear caustics |
| Planetary | 70-80% | Planet systems |
| Stellar | 60-75% | Equal mass |

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py           # Fast parallel simulation
│   ├── train.py              # DDP training
│   ├── evaluate.py           # Complete evaluation
│   ├── benchmark_realtime.py # Performance testing
│   ├── plot_samples.py       # Visualization
│   ├── test_vbm.py           # VBMicrolensing validation
│   ├── model.py              # CNN architecture
│   ├── config.py             # Configuration
│   └── utils.py              # Utilities
│
├── data/
│   └── raw/                  # Simulated datasets (.npz)
│
├── results/                  # Experiment outputs
│   └── {experiment}_{timestamp}/
│       ├── best_model.pt
│       ├── config.json
│       ├── scaler_*.pkl
│       └── evaluation/
│
├── docs/
│   ├── SETUP_GUIDE.md        # Installation & DDP setup
│   ├── RESEARCH_GUIDE.md     # Experimental workflow
│   └── QUICK_REFERENCE.md    # Command cheatsheet
│
└── README.md
```

---

## Model Architecture

### TDConvClassifier

```
Input: [batch, 1, 1500]
   ↓
Conv1D (9, 128) + BatchNorm + ReLU + Dropout
   ↓
Conv1D (7, 64) + BatchNorm + ReLU + Dropout
   ↓
Conv1D (5, 32) + BatchNorm + ReLU + Dropout
   ↓
Mean + Max Pooling
   ↓
FC (64) + ReLU + Dropout
   ↓
FC (2) → Binary classification
```

---

## Results Visualization

```bash
# Plot sample predictions
python plot_samples.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --n_samples 12

# Generate comparison table
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_dense', 'cadence_sparse',
               'error_low', 'error_high', 'distinct', 'planetary', 'stellar']

print(f'{'Experiment':<20} {'Test Acc':<12}')
print('-' * 35)

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        summary = runs[-1] / 'summary.json'
        if summary.exists():
            data = json.load(open(summary))
            acc = data.get('final_test_acc', 0) * 100
            print(f'{exp:<20} {acc:>10.2f}%')
"
```

---

## Troubleshooting

### Training Issues

**DDP hangs or errors**:
```bash
# Check NCCL debug
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=4 train.py [args]

# Use gloo backend instead of nccl
export NCCL_SOCKET_IFNAME=eth0  # or your network interface
```

**Out of memory**:
```bash
# Reduce batch size
torchrun --nproc_per_node=4 train.py --batch_size 64

# Or use gradient accumulation
torchrun --nproc_per_node=4 train.py --batch_size 32
```

**Slow training**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Ensure using all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Simulation Issues

**Too slow**:
```bash
# Increase workers (up to CPU cores)
python simulate.py --num_workers 200

# Check CPU usage
htop
```

---

## Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Complete installation + DDP setup
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Systematic experiments
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Command cheatsheet

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

---

## Acknowledgments

- VBMicrolensing library by Valerio Bozza
- PyTorch team for DDP framework
- University of Heidelberg for computational resources