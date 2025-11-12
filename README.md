## Real-Time Microlensing Classification with Transformers

**MSc Thesis Project: From Light Curves to Labels - Machine Learning in Microlensing**

**Author**: Kunal Bhatia  
**Supervisor**: Prof. Dr. Joachim Wambsganß  
**Institution**: University of Heidelberg  
**Submission**: February 2025

---

## Overview

Transformer-based system for real-time three-class classification of gravitational microlensing events.

### Classification Task

- **Class 0 (Flat)**: No microlensing event, baseline flux variations only
- **Class 1 (PSPL)**: Point Source Point Lens - single lens microlensing
- **Class 2 (Binary)**: Binary lens system - complex caustic structures


### Key Capabilities

- **Fast**: <1ms inference, 10,000+ events/second on single GPU
- **Early Detection**: Reliable classification at 50% observation completeness
- **Physically Grounded**: Naturally captures detection limit at u₀ > 0.3
- **Production Ready**: Distributed training, mixed precision, gradient checkpointing
- **Hardware Agnostic**: Tested on AMD MI250X/MI300A (ROCm) and NVIDIA GPUs (CUDA)

---

## 📋 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch (choose your hardware)
# NVIDIA (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD (ROCm 6.0):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# CPU only:
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt

# CRITICAL: Install VBMicrolensing for physically accurate simulations
pip install VBMicrolensing
```

### Generate Dataset

```bash
cd code

python simulate.py \
    --n_flat 100 \
    --n_pspl 100 \
    --n_binary 100 \
    --binary_params baseline \
    --output ../data/raw/test.npz \
    --num_workers 4 \
    --save_params
```

### Train Model

**Single GPU:**
```bash
python train.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --epochs 10 \
    --batch_size 64
```

**Multi-GPU (DDP):**
```bash

export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes=n \
  --nproc_per_node=n \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$(date +%s) \
  train.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --epochs 10 \
    --batch_size 64
```

### Evaluate Model

```bash
python evaluate.py \
    --experiment_name test \
    --data ../data/raw/test.npz \
    --early_detection \
    --n_samples 10000
```

**Outputs** (`results/test/evaluation/`):
- `roc_curve.png` - One-vs-rest ROC curves
- `confusion_matrix.png` - 3×3 confusion matrix
- `calibration.png` - Model calibration analysis
- `u0_dependency.png` - Accuracy vs. impact parameter (Binary class)
- `early_detection.png` - Performance vs. observation completeness
- `real_time_evolution_*.png` - All 3 class probabilities evolving
- `example_grid_3class.png` - Example light curves
- `evaluation_summary.json` - Complete metrics
- `u0_report.json` - Impact parameter analysis

---

## 🏗️ Model Architecture

### Transformer Design

```
Input: Light curve [B, T=1500, 1]
  ↓
Input Embedding [B, T, D=128]
  + Relative Positional Encoding (observation count + gaps)
  + Gap Features (sparse sampling information)
  ↓
Transformer Layers ×4:
  - Multi-Head Attention (4 heads, Flash Attention when available)
  - Feed-Forward Network (4×D = 512)
  - Pre-norm + Residual Connections
  - Dropout (0.1)
  ↓
Global Pooling (Average + Max)
  ↓
Classification Head [B, 3] → CrossEntropy Loss
Auxiliary Heads [B, 1] each → BCEWithLogits Loss:
  - Flat Detection (weight=0.5)
  - PSPL Detection (weight=0.5)
  - Anomaly Detection (weight=0.2)
  - Caustic Detection (weight=0.2)
  - Confidence Estimation
```

**Model Size**: ~100K parameters

**Key Features**:
- **Relative encoding**: No absolute time information (prevents leakage)
- **Variable-length sequences**: Handles missing observations naturally
- **Multi-task learning**: Auxiliary tasks improve feature learning
- **Flash Attention**: 2-4× speedup when available (PyTorch 2.0+)

---

## 📊 Expected Performance

### Baseline Performance (100% Observed)

| Dataset | Overall | Flat | PSPL | Binary | Physical Regime |
|---------|---------|------|------|--------|-----------------|
| Baseline 1M | 70-75% | 80-85% | 65-70% | 70-75% | Realistic mix |
| Distinct | 85-90% | 90-95% | 80-85% | 85-90% | Clear caustics |
| Planetary | 75-80% | 85-90% | 70-75% | 75-80% | Exoplanet focus |
| Challenging | 60-65% | 75-80% | 55-60% | 55-65% | Near physical limit |

### Impact Parameter Dependency (Binary Class)

| u₀ Range | Accuracy | Physical Regime |
|----------|----------|----------------|
| < 0.1 | 90-95% | Close approach, clear caustics |
| 0.1-0.2 | 80-85% | Detectable binary features |
| 0.2-0.3 | 70-75% | Subtle features, challenging |
| 0.3-0.5 | 50-60% | Mostly PSPL-like |
| > 0.5 | 30-40% | Indistinguishable from PSPL |

**Physical Interpretation**: The accuracy drop at u₀ > 0.3 is a fundamental detection limit, not an algorithmic failure. High impact parameter events have minimal caustic interactions and are intrinsically PSPL-like.

### Early Detection (Baseline Dataset)

| Completeness | Overall Acc | Binary Recall | Use Case |
|--------------|-------------|---------------|----------|
| 10% | ~40% | ~30% | Too early for reliable classification |
| 25% | ~55% | ~45% | Some confident predictions possible |
| 50% | ~70% | ~65% | **Recommended trigger point** |
| 75% | ~73% | ~70% | High confidence |
| 100% | ~75% | ~75% | Full event |

---

## 🧪 Systematic Experimental Plan

### Experiment: Baseline Benchmark (1M events)

**Purpose**: Establish performance on realistic parameter distributions

```bash
# Generate
python simulate.py \
    --n_flat 333000 --n_pspl 333000 --n_binary 334000 \
    --binary_params baseline \
    --output ../data/raw/baseline_1M.npz \
    --save_params --num_workers 8 --seed 42

# Train
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline_1M \
    --epochs 50 --batch_size 64

# Evaluate
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection
```

**Expected**: 70-75% overall accuracy, clear u₀ dependency

---

### Experiments

Test generalization across different binary configurations:

#### Distinct Topology (Clear Caustics)

```bash
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params distinct --output ../data/raw/distinct.npz \
    --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/distinct.npz
```

**Expected**: 85-90% accuracy (model trained on baseline tested on easy cases)

#### Planetary Systems

```bash
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params planetary --output ../data/raw/planetary.npz \
    --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/planetary.npz
```

**Expected**: 75-80% accuracy

#### Stellar Binaries

```bash
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params stellar --output ../data/raw/stellar.npz \
    --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/stellar.npz
```

**Expected**: 65-75% accuracy

#### Challenging (Near Physical Limit)

```bash
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params challenging --output ../data/raw/challenging.npz \
    --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/challenging.npz
```

**Expected**: 60-65% accuracy (demonstrates physical limit)

---

### Experiments: Observational Conditions

Test robustness to observing conditions:

#### Cadence Experiments

**Scientific Question**: How does observation frequency affect performance?

| Cadence | Missing Obs | Survey Context | Command Suffix |
|---------|-------------|----------------|----------------|
| Dense | 5% | Intensive follow-up | `cadence_05` |
| LSST Nominal | 20% | Standard survey | `cadence_20` |
| Sparse | 30% | Poor weather | `cadence_30` |
| Very Sparse | 40% | Limited coverage | `cadence_40` |

**Example**:
```bash
# Dense cadence (5% missing)
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params baseline --cadence_mask_prob 0.05 \
    --output ../data/raw/cadence_05.npz --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/cadence_05.npz
```

**Repeat for 0.20, 0.30, 0.40**

**Expected Results**:
- 5% missing: 75-80% accuracy
- 20% missing: 70-75% accuracy (baseline)
- 30% missing: 65-70% accuracy
- 40% missing: 60-65% accuracy

**Finding**: Performance degrades gracefully; LSST nominal cadence sufficient

#### Photometric Error Experiments

**Scientific Question**: How does measurement precision affect classification?

| Error | σ (mag) | Quality | Survey Context | Command Suffix |
|-------|---------|---------|----------------|----------------|
| Low | 0.05 | Space-based (Roman) | Excellent | `error_05` |
| Medium | 0.10 | Ground-based (LSST) | Good | `error_10` |
| High | 0.20 | Poor conditions | Challenging | `error_20` |

**Example**:
```bash
# Low photometric error (0.05 mag)
python simulate.py --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
    --binary_params baseline --mag_error_std 0.05 \
    --output ../data/raw/error_05.npz --save_params --seed 42

python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/error_05.npz
```

**Repeat for 0.10, 0.20**

**Expected Results**:
- 0.05 mag: 75-80% accuracy
- 0.10 mag: 70-75% accuracy (baseline)
- 0.20 mag: 65-70% accuracy

**Finding**: Caustic features robust to moderate noise; space-based quality provides modest benefit
---


## 📚 Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
    title={From Light Curves to Labels: Machine Learning in Microlensing},
    author={Bhatia, Kunal},
    school={University of Heidelberg},
    year={2025},
    month={February},
    supervisor={Wambsganß, Joachim},
    type={Master's Thesis}
}
```

---

## 📧 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg  
Email: kunal29bhatia@gmail.com

---

## 🎯 Quick Reference

**Essential Commands**:
```bash
# Generate baseline dataset (1M events)
python simulate.py --n_flat 333000 --n_pspl 333000 --n_binary 334000 \
    --binary_params baseline --output ../data/raw/baseline_1M.npz \
    --save_params --num_workers 8 --seed 42

# Train on baseline
python train.py --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline_1M --epochs 50 --batch_size 64

# Test on different configuration
python evaluate.py --experiment_name baseline_1M \
    --data ../data/raw/challenging.npz
```

**File Locations**:
- Data: `data/raw/*.npz`
- Models: `results/*/best_model.pt`
- Evaluation: `results/*/evaluation/`
- Config: `code/config.py`

**Key Parameters** (`config.py`):
- Time window: [-100, 100] days
- t₀ range: [-50, 50] days (PSPL and Binary)
- Points: 1500 per light curve
- Cadence: 20% missing (baseline)
- Photometry: 0.10 mag error (baseline)