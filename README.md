# Real-Time Binary Microlensing Detection with Streaming Transformers

**MSc Thesis Project - From Light Curves to Labels: Machine Learning in Microlensing**

Author: Kunal Bhatia, University of Heidelberg  
Supervisor: Prof. Dr. Joachim Wambsganß  
Date: November 2025

---

## Overview

This repository implements a **streaming transformer architecture** for real-time detection of binary microlensing events. The system addresses the critical challenge of distinguishing planetary and stellar binary events from simple Point-Source Point-Lens (PSPL) events in the era of large-scale surveys (LSST, Roman).

**Key Innovation**: Unlike traditional approaches requiring complete light curves, our streaming transformer processes observations sequentially, enabling early detection with <50% of data while maintaining >70% accuracy.

### Critical Findings

- **Physical Detection Limit**: Events with impact parameter u₀ > 0.3 are fundamentally indistinguishable (not an algorithmic limitation)
- **Caustic Preservation**: Custom normalization preserves caustic spikes (>20× magnification) critical for binary detection  
- **Real-Time Capable**: <1ms latency per observation, 10,000+ events/second throughput
- **Early Detection**: 70% accuracy with only 50% of observations

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/thesis-microlensing.git
cd thesis-microlensing

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch (choose based on your GPU)
# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD GPU (ROCm 6.0):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Installation

```bash
cd code
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import VBBinaryLensing; print('VBMicrolensing: OK')"
```

### 3. Generate Test Dataset

```bash
# Small test (2K events, ~2 min)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --binary_params critical \
    --output ../data/raw/test_2k.npz \
    --num_workers 4

# Validate dataset
python validate_dataset.py \
    --data ../data/raw/test_2k.npz \
    --output_dir ../results/validation
```

### 4. Train Model

```bash
# Single GPU
python train_ddp.py \
    --data ../data/raw/test_2k.npz \
    --experiment_name test \
    --epochs 10 \
    --batch_size 32

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train_ddp.py \
    --data ../data/raw/test_2k.npz \
    --experiment_name test_ddp \
    --epochs 10
```

---

## Project Structure

```
Thesis/
├── code/                          # Core implementation
│   ├── config.py                  # Configuration (all hyperparameters)
│   ├── simulate.py                # Data generation with caustic validation
│   ├── normalization.py           # Caustic-preserving normalization
│   ├── streaming_transformer.py   # Model: causal attention, multi-head output
│   ├── streaming_losses.py        # Custom losses (early detection, caustic focal)
│   ├── train_ddp.py              # Multi-node distributed training
│   ├── validate_dataset.py       # Dataset validation & visualization
│   └── streaming_inference.py    # Real-time inference pipeline
│
├── data/
│   └── raw/                      # Generated datasets (.npz files)
│
├── results/                      # Experiment outputs
│   └── experiment_TIMESTAMP/
│       ├── best_model.pt        # Trained model checkpoint
│       ├── normalizer.pkl       # Fitted normalizer
│       ├── config.json          # Experiment configuration
│       └── results.json         # Final metrics
│
├── docs/                         # Documentation
│   ├── RESEARCH_GUIDE.md         # Experiment design
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Key Components

### 1. Binary Event Simulation (`simulate.py`)

**Critical**: Enforces caustic crossings with u₀ < 0.05 and magnification > 20×

```python
# Binary parameter sets optimized for detection
BINARY_CRITICAL = {
    'u0_min': 0.001,    # Force close approach
    'u0_max': 0.05,     # Maximum for guaranteed caustics
    's_min': 0.7,       # Optimal separation
    's_max': 1.5,
    'q_min': 0.01,      # Clear perturbations
    'q_max': 0.5
}
```

### 2. Streaming Transformer (`streaming_transformer.py`)

- **No downsampling**: Processes at full 1500-point resolution
- **Causal attention**: Strictly no future information leakage
- **Multi-head output**: Binary classification + Anomaly detection + Caustic detection

```python
model = StreamingTransformer(
    d_model=256,        # Embedding dimension
    nhead=8,            # Attention heads
    num_layers=6,       # Transformer layers
    window_size=200,    # Sliding window
    use_multi_head=True # Enable all outputs
)
```

### 3. Caustic-Preserving Normalization (`normalization.py`)

Preserves sharp caustic features critical for binary detection:

1. Works in flux space (not magnitude)
2. Uses robust statistics (median/MAD not mean/std)
3. Log transform preserves dynamic range
4. Per-event normalization maintains caustic spikes

### 4. Multi-Node Training (`train_ddp.py`)

Flexible distributed training supporting n nodes × m GPUs:

```bash
# SLURM submission for 4 nodes × 4 GPUs = 16 total
sbatch scripts/train_multinode.sh
```

---

## Thesis Experiments

### Baseline Experiment (1M events)

```bash
# Generate
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --binary_params baseline \
    --output ../data/raw/baseline_1M.npz \
    --num_workers 32

# Train (16 GPUs, ~30 min)
sbatch scripts/train_multinode.sh \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline
```

### Critical Experiment: u₀ Dependency

Demonstrates the physical detection limit at u₀ > 0.3:

```bash
# Generate with overlapping parameters
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --binary_params overlapping \  # u0 up to 1.0
    --output ../data/raw/overlapping.npz \
    --save_params  # Save for u0 analysis

# Analyze u0 dependency
python analyze_u0.py \
    --data ../data/raw/overlapping.npz \
    --experiment_name overlapping
```

---

## Performance Benchmarks

### Accuracy Results

| Dataset | u₀ Range | Test Accuracy | Notes |
|---------|----------|---------------|-------|
| Critical | < 0.05 | 92-95% | Strong caustics |
| Distinct | < 0.1 | 85-90% | Clear signatures |
| Baseline | < 0.3 | 70-75% | Realistic mix |
| Overlapping | < 1.0 | 55-65% | Includes hard cases |

### Computational Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Inference latency | <1 ms/event | Single GPU |
| Training (1M events) | ~30 min | 16× A100 GPUs |
| Throughput | 10,000+ events/sec | RTX 4090 |
| Memory usage | ~4 GB | Per GPU |

---


## Multi-Node Training Guide

### SLURM Script Template

Create `scripts/train_multinode.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=microlensing
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00

# Setup environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME="^lo,docker"

# Run training
cd ~/thesis-microlensing/code

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_ddp.py "$@"
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{bhatia2025microlensing,
    title={From Light Curves to Labels: Machine Learning in Microlensing},
    author={Bhatia, Kunal},
    school={University of Heidelberg},
    year={2025},
    supervisor={Wambsganß, Joachim}
}
```

---

## License

MIT License - See LICENSE file

---

## Contact

Kunal Bhatia  
Email: kunal29bhatia@gmail.com  
University of Heidelberg

---