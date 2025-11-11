## Real-Time Binary Microlensing Classification with Transformers

**MSc Thesis Project - From Light Curves to Labels: Machine Learning in Microlensing**

Author: Kunal Bhatia  
Supervisor: Prof. Dr. Joachim Wambsganß  
Institution: University of Heidelberg  
Version: 10.0 - Production Ready  
Date: November 2025

---

## Overview

This repository implements a **transformer architecture** for real-time classification of binary microlensing events (planetary systems and stellar binaries) versus simple Point-Source Point-Lens (PSPL) events. 


Designed for next-generation survey operations (LSST, Roman Space Telescope) requiring sub-second inference on alert streams.

### Key Features

- **Multi-Task Learning**: Binary classification + Anomaly detection + Caustic detection
- **Distributed Training**: Optimized for multi-GPU (H100, A100, MI300) with DDP
- **Real-Time Capability**: <1ms inference, 10,000+ events/second
- **Early Detection**: 70%+ accuracy with only 50% of observations
- **Robust Architecture**: Stable gradient flow, handles missing data
- **AMD Compatible**: Full ROCm support for AMD GPUs
- **Multi-Node DDP**: Scales to 32+ GPUs across multiple nodes

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch (choose based on your hardware)
# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD GPU (ROCm 6.0):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# CPU only:
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt

# CRITICAL: Install VBMicrolensing for realistic simulations
pip install VBMicrolensing
```

### 2. Generate Test Dataset

```bash
cd code

# Small test dataset (2K events, ~2 minutes)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --binary_params critical \
    --output ../data/raw/test_2k.npz \
    --num_workers 4 \
    --save_params
```

### 3. Train Model

```bash
python train.py \
    --data ../data/raw/test_2k.npz \
    --experiment_name test \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --grad_clip 5.0
```

**Note**: Training automatically saves:
- Best model checkpoint (`best_model.pt`)
- Normalizer parameters (`normalizer.pkl`)
- Configuration (`config.json`)
- Final results (`results.json`)

### 4. Evaluate Model (Includes u0 Analysis)

```bash
# Single command for complete evaluation + u0 analysis
python evaluate.py \
    --experiment_name test \
    --data ../data/raw/test_2k.npz
```

**Outputs**: `results/test_TIMESTAMP/evaluation/`
- `roc_curve.png` - ROC curve and AUC
- `confusion_matrix.png` - Classification breakdown
- `confidence_distribution.png` - Confidence histogram
- `u0_dependency.png` - Accuracy vs. impact parameter (if parameter data available)
- `evaluation_summary.json` - All metrics
- `u0_report.json` - u0 analysis results (if available)
- `real_time_evolution_*.png` - Probability evolution plots 

**Note**: u0 analysis automatically runs if parameter data exists (datasets generated with `--save_params`). To skip u0 analysis, use `--no_u0` flag.

---

## 📁 Project Structure

```
Thesis/
├── code/                          # Core implementation
│   ├── config.py                  # Configuration parameters (v10.0)
│   ├── simulate.py                # Data generation (v10.0)
│   ├── transformer.py             # MicrolensingTransformer model (v10.0) 
│   ├── evaluate.py                # Complete evaluation + u0 analysis (v10.0)
│   └── train.py                   # Training with DDP support (v10.0)
│
├── data/
│   └── raw/                       # Generated datasets (.npz)
│
├── results/                       # Experiment outputs
│   └── experiment_TIMESTAMP/
│       ├── best_model.pt          # Model checkpoint
│       ├── normalizer.pkl         # Normalizer parameters
│       ├── config.json            # Experiment config
│       ├── results.json           # Training metrics
│       └── evaluation/            # All evaluation outputs
│           ├── roc_curve.png
│           ├── confusion_matrix.png
│           ├── confidence_distribution.png
│           ├── u0_dependency.png
│           ├── real_time_evolution_*.png  
│           ├── evaluation_summary.json
│           └── u0_report.json
│
├── docs/
│   ├── RESEARCH_GUIDE.md          # Complete experimental workflow (v10.0)
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file (v10.0)
│
```

---

## 🔬 Complete Workflow

### Baseline Experiment (100K events)

```bash
cd code

# 1. Generate dataset with parameters
python simulate.py \
    --n_pspl 50000 \
    --n_binary 50000 \
    --binary_params baseline \
    --output ../data/raw/baseline_100k.npz \
    --num_workers 8 \
    --save_params

# 2. Train model
python train.py \
    --data ../data/raw/baseline_100k.npz \
    --experiment_name baseline_100k \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-5

# 3. Complete evaluation (metrics + u0 analysis)
python ../evaluate.py \
    --experiment_name baseline_100k \
    --data ../data/raw/baseline_100k.npz
```

**Expected Results**:
- Test Accuracy: 70-75%
- ROC AUC: 0.78-0.82
- u0 dependency: Clear accuracy drop at u₀ > 0.3

---

## 🏗️ Model Architecture

### MicrolensingTransformer

**Architecture Features**:
- **Stable Multi-Head Attention**: Normalized Q/K projections prevent gradient explosion
- **Pre-Norm Transformer Blocks**: Improved training stability
- **Learnable Positional Encoding**: Adapts to light curve structure
- **Gap Embedding**: Explicitly handles missing observations
- **Multi-Task Heads**: 
  - Binary classification (main task)
  - Anomaly detection (auxiliary, weight=0.1)
  - Caustic detection (auxiliary, weight=0.1)

**Default Configuration**:
```python
MicrolensingTransformer(
    n_points=1500,        # Full temporal resolution
    d_model=256,          # Embedding dimension
    nhead=8,              # Attention heads
    num_layers=6,         # Transformer layers
    dim_feedforward=1024, # FFN dimension
    dropout=0.1           # Dropout rate
)
```

**Parameters**: ~2-10M depending on configuration

---

## 📊 Data Generation

### Binary Parameter Sets

**Critical** (u₀ < 0.05):
- Forces caustic crossings (>80% with mag > 20×)
- Used for demonstrating clear binary signatures

**Baseline** (u₀ < 0.3):
- Realistic mixed population
- Expected: 70-75% accuracy

**Overlapping** (u₀ < 1.0):
- Includes fundamentally hard cases
- Expected: 55-65% accuracy

**Commands**:
```bash
# Critical dataset (easy)
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --binary_params critical \
    --output ../data/raw/critical_200k.npz \
    --save_params

# Baseline dataset (realistic)
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --binary_params baseline \
    --output ../data/raw/baseline_200k.npz \
    --save_params
```

**Note**: Always use `--save_params` to enable u0 dependency analysis.

---

## 🎯 Training

### Single GPU

```bash
python train.py \
    --data ../data/raw/baseline_200k.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-5 \
    --grad_clip 5.0
```

### Multi-GPU (8 GPUs, Single Node)

```bash
torchrun --nproc_per_node=8 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline_8gpu \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --grad_clip 1.0
```

### Multi-Node (4 nodes, 32 GPUs total)

**Node 0 (Master)**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name multinode \
    --epochs 50 \
    --batch_size 32
```

**Nodes 1-3**: Same command but with `--node_rank=1`, `--node_rank=2`, `--node_rank=3`

**Training Features**:
- Multi-task learning with auxiliary losses
- Learning rate warmup + cosine annealing
- Gradient clipping for stability
- Mixed precision training (AMP)
- Automatic normalizer saving
- Early stopping (patience=15)
- Full multi-node DDP support

---

## 📈 Evaluation

### Basic Evaluation

```bash
# Automatically includes u0 analysis if parameter data available
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_200k.npz
```

### Custom Options

```bash
# Skip u0 analysis
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_200k.npz \
    --no_u0

# Include early detection analysis
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_200k.npz \
    --early_detection

# Custom u0 threshold and bins
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_200k.npz \
    --u0_threshold 0.35 \
    --u0_bins 15

# Use CPU
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_200k.npz \
    --no_cuda
```

---

## 📊 Performance Benchmarks

### Classification Accuracy

| Dataset | u₀ Range | Test Accuracy | ROC AUC | Notes |
|---------|----------|---------------|---------|-------|
| Critical | < 0.05 | 92-95% | 0.95-0.97 | Strong caustics guaranteed |
| Baseline | < 0.3 | 70-75% | 0.78-0.82 | Realistic mixed population |
| Overlapping | < 1.0 | 55-65% | 0.65-0.75 | Includes hard cases (u₀>0.3) |

### Computational Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Inference latency | <1 ms/event | Single GPU |
| Training time (100K) | ~10 min | Single GPU |
| Training time (1M) | ~60-90 min | Single GPU |
| Training time (1M) | ~10-15 min | 8× H100 |
| Training time (1M) | ~3-5 min | 32× H100 (4 nodes) |
| Throughput | >10,000 events/sec | Single GPU |

### Early Detection

| Observation Completeness | Accuracy | Use Case |
|--------------------------|----------|----------|
| 10% | 50-55% | Too early |
| 25% | 60-65% | Marginal |
| 50% | 68-72% | **Trigger follow-up** |
| 100% | 70-75% | Full baseline |

---

## 🔧 GPU Compatibility

### NVIDIA GPUs (Tested ✅)

```bash
# CUDA 12.1 (RTX 40-series, A100, H100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (Older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### AMD GPUs (Fully Compatible ✅)

```bash
# ROCm 6.0 (RX 7900 XTX, MI200/MI300 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Verify (reports as "CUDA available" through ROCm)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check device count
python -c "import torch; print(f'Devices: {torch.cuda.device_count()}')"
```

**AMD-Specific Optimizations**:
```bash
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=8
export PYTORCH_ROCM_ARCH=gfx90a  # For MI250X
export PYTORCH_ROCM_ARCH=gfx942  # For MI300A
```

### CPU Only

```bash
pip install torch torchvision
```

---

## 🐛 Bug Fixes in Version 10.0

### Critical Fixes:

1. **evaluate.py - Tensor Creation Efficiency**
   - **Before**: Created list, then numpy array (slow)
   - **After**: Pre-allocate numpy array directly (2-3x faster)
   - **Impact**: Faster evaluation, especially for early detection

2. **evaluate.py - Array Indexing**
   - **Before**: `self.X_norm[j][:n_points]` (wrong)
   - **After**: `self.X_norm[j, :n_points]` (correct 2D indexing)
   - **Impact**: Fixes potential indexing errors

3. **Version Consistency**
   - **Before**: Different versions across files (9.0, 18.0, 8.0, 13.0, 10.0, 7.1)
   - **After**: All files now v10.0
   - **Tool**: Use `python update_versions.py --version 10.0`


---

## 🔍 Troubleshooting

### Common Issues

**1. VBMicrolensing not installed**
```bash
pip install VBMicrolensing
```

**2. CUDA out of memory**
```bash
python train.py --batch_size 8  # Reduce batch size
```

**3. "No experiment found"**
```bash
# Check what exists
ls ../results/

# Use exact name
python evaluate.py --experiment_name baseline_20250109_143022 --data ...
```

**4. Training instability (NaN loss)**
```bash
python train.py --lr 1e-5 --grad_clip 1.0 --warmup_epochs 10
```

**5. "No u0 analysis" message**
- Dataset was generated without `--save_params`
- Re-generate with: `python simulate.py ... --save_params`
- Or use `--no_u0` flag to suppress message

**6. Multi-node DDP not working**
- Check firewall: `ping <MASTER_ADDR>`
- Verify port open: Port 29500 (or your MASTER_PORT)
- Check environment variables: `echo $RANK $LOCAL_RANK $WORLD_SIZE`
- Enable debug: `export NCCL_DEBUG=INFO`

---

## 📚 Additional Documentation

- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Systematic experimental design and thesis workflow

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{bhatia2025microlensing,
    title={From Light Curves to Labels: Machine Learning in Microlensing},
    author={Bhatia, Kunal},
    school={University of Heidelberg},
    year={2026},
    month={February},
    supervisor={Wambsganß, Joachim},
    type={Master's Thesis}
}
```

---

## 📄 License

MIT License - See LICENSE file

---

## 📧 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg  
Email: kunal29bhatia@gmail.com

---

## 📝 Changelog

### Version 10.0 (Current) - Production Ready
- ✅ Fixed tensor creation efficiency in evaluate.py (2-3x faster)
- ✅ Fixed array indexing bugs
- ✅ Standardized version numbers across all files
- ✅ Added comprehensive AMD GPU support documentation
- ✅ Added multi-node DDP setup guide
- ✅ Enhanced real-time evolution plots (shows BOTH PSPL and Binary probabilities)
- ✅ All files updated to v10.0

### Version 9.0 (Previous)
- ✅ Combined evaluate.py and analyze_u0.py into single script
- ✅ Automatic u0 analysis when parameter data available
- ✅ Simplified workflow (3 commands instead of 4)
- ✅ Improved documentation and project structure

### Version 8.0
- ✅ Fixed evaluate.py model loading
- ✅ Fixed analyze_u0.py imports
- ✅ Added normalizer saving in train.py
- ✅ Updated documentation
- ✅ Added AMD/NVIDIA compatibility guide

