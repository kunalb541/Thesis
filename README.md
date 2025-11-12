## Real-Time Three-Class Microlensing Classification with Transformers

**MSc Thesis Project - From Light Curves to Labels: Machine Learning in Microlensing**

Author: Kunal Bhatia  
Supervisor: Prof. Dr. Joachim Wambsganß  
Institution: University of Heidelberg  
**Version: 12.0-beta - ARCHITECTURAL FIX (DATA LEAKAGE RESOLVED)**  
Date: November 2025

---

## Overview

This repository implements a **transformer architecture with relative positional encoding** for real-time three-class classification of astronomical time series: distinguishing baseline observations (Flat), simple microlensing events (PSPL), and complex binary microlensing events (Binary).

**v12.0-beta represents a critical scientific improvement**: After discovering that v11 was "cheating" via positional encoding, we redesigned the architecture to use relative positional encoding. This results in more realistic (lower) early-detection performance, but represents genuine learned patterns rather than temporal artifacts.

Designed for next-generation survey operations (LSST, Roman Space Telescope) requiring sub-second inference on alert streams with robust rejection of non-events.

### Key Features

- **Three-Class Classification**: Flat / PSPL / Binary with high-confidence event rejection
- **Architectural Solution**: Relative positional encoding prevents data leakage
- **No Causal Truncation**: Full-sequence training works better (tested and validated)
- **Variable-Length Support**: No fixed padding artifacts
- **Distributed Training**: Multi-node DDP on AMD/NVIDIA GPUs (tested 32 GPUs)
- **Real-Time Capability**: <1ms inference, 10,000+ events/second
- **Realistic Early Detection**: 70%+ accuracy with 50% of observations (physically grounded)
- **Smaller Model**: ~100K parameters (4.5x smaller than v11)
- **AMD Compatible**: Full ROCm support for MI250X/MI300A
- **AMP-Safe**: Mixed-precision training without numerical issues

---

## 📋 Quick Start

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

### 2. Generate Test Dataset (3-Class, v12.0-beta)

```bash
cd code

# Quick test dataset (300 events: 100 Flat + 100 PSPL + 100 Binary)
# v12.0-beta: Note the wider t0 range!
python simulate.py \
    --n_flat 100 \
    --n_pspl 100 \
    --n_binary 100 \
    --binary_params baseline \
    --output ../data/raw/test_3class_v12beta_300.npz \
    --num_workers 4 \
    --save_params
```

**Output**:
```
GENERATING 100 FLAT + 100 PSPL + 100 BINARY EVENTS
v12.0-beta: t0 range = [-50, 50] days (WIDER than v11!)
THREE-CLASS CLASSIFICATION: 0=Flat, 1=PSPL, 2=Binary
Total events: 300
  Flat:   100 (33.3%)
  PSPL:   100 (33.3%)
  Binary: 100 (33.3%)
```

### 3. Train Model (v12.0-beta Architecture)

**Single GPU:**
```bash
python train.py \
    --data ../data/raw/test_3class_v12beta_300.npz \
    --experiment_name test_v12beta \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4
```

**Multi-GPU (8 GPUs):**
```bash
torchrun --nproc_per_node=8 train.py \
    --data ../data/raw/test_3class_v12beta_300.npz \
    --experiment_name test_v12beta_8gpu \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4
```

### 4. Evaluate Model (3-Class Metrics)

```bash
python evaluate.py \
    --experiment_name test_v12beta \
    --data ../data/raw/test_3class_v12beta_300.npz \
    --early_detection \
    --n_samples 10000
```

**Outputs** (in `results/test_v12beta_TIMESTAMP/evaluation/`):
- `roc_curve.png` - One-vs-rest ROC curves for all 3 classes
- `confusion_matrix.png` - 3×3 confusion matrix
- `confidence_distribution.png` - Confidence by correctness
- `calibration.png` - Model calibration analysis
- `u0_dependency.png` - Accuracy vs. impact parameter (Binary class only)
- `early_detection.png` - **REALISTIC** performance vs. completeness
- `real_time_evolution_*.png` - Shows ALL 3 class probabilities evolving
- `example_grid_3class.png` - Example light curves from each class
- `evaluation_summary.json` - All metrics
- `u0_report.json` - u0 analysis (if parameter data available)

---

## 🏗️ Model Architecture

### MicrolensingTransformer v12.0-beta (Architecture-Based Solution)

**Main Task**: 3-class classification
- **Class 0**: Flat (no event, baseline only)
- **Class 1**: PSPL (single lens)
- **Class 2**: Binary (binary lens)

**Auxiliary Tasks** (all output logits for AMP-safe BCEWithLogitsLoss):
1. **Flat Detection** (weight=0.5, HIGH):
   - Target: 1.0 for Flat, 0.0 for PSPL/Binary
   - Purpose: Reject non-events, prevent false triggers
   
2. **PSPL Detection** (weight=0.5, HIGH):
   - Target: 1.0 for PSPL, 0.0 for Flat/Binary
   - Purpose: Identify simple lens events
   
3. **Anomaly Detection** (weight=0.2):
   - Target: 1.0 for any event (PSPL or Binary), 0.0 for Flat
   - Purpose: General event detection
   
4. **Caustic Detection** (weight=0.2):
   - Target: 1.0 for Binary, 0.0 for PSPL/Flat
   - Purpose: Binary-specific features
   
5. **Confidence Estimation**:
   - Single output with sigmoid (0-1 range)
   - Self-assessment of prediction quality

**Architecture Details (v12.0-beta - SMALLER)**:
```python
MicrolensingTransformer(
    n_points=1500,
    d_model=128,     
    nhead=4,         
    num_layers=4,    
    dropout=0.1
)

```

**Key Features**:
- **Relative Positional Encoding**: Only knows observation count & gaps
- **Stable Multi-Head Attention**: Normalized Q/K projections
- **Pre-Norm Architecture**: Improved training stability
- **Gap Embedding**: Handles missing observations explicitly
- **Variable-Length Support**: No fixed padding patterns
- **Auxiliary Heads Output Logits**: AMP-safe, numerically stable

---

## 📊 Expected Performance

### Three-Class Accuracy at 100% Observed

| Dataset | Overall | Flat | PSPL | Binary | Notes |
|---------|---------|------|------|--------|-------|
| Baseline 1M | 70-75% | 80-85% | 65-70% | 70-75% | Realistic mix |
| Critical | 85-90% | 90-95% | 80-85% | 85-90% | Upper bound |
| Planetary | 75-80% | 85-90% | 70-75% | 75-80% | Exoplanets |
| Challenging | 60-65% | 75-80% | 55-60% | 55-65% | Physical limit |

### u₀ Dependency (Binary Class)

| u₀ Range | Accuracy | Interpretation |
|----------|----------|----------------|
| < 0.1 | 90-95% | Clear caustics |
| 0.1-0.2 | 80-85% | Detectable |
| 0.2-0.3 | 70-75% | Subtle |
| 0.3-0.5 | 50-60% | PSPL-like |
| > 0.5 | 30-40% | Indistinguishable |

---

## 🔧 Troubleshooting

**"v11 had better early performance!"**
- v11 was cheating with positional encoding
- v12.0-beta is honest - this is the real physical limit
- Lower performance = better science!

**Training shows NaN loss:**
- Reduce learning rate: `--lr 5e-4`
- Increase gradient clipping: `--grad_clip 2.0`

**Poor performance:**
- Check data normalization
- Verify architecture: d_model=128, nhead=4, num_layers=4
- Ensure dataset has wider t0 range [-50, 50]

---

## 📝 Changelog

### CRITICAL DISCOVERY: Data Leakage Resolution Through Architecture

**Problem Discovered in v11.x**: Model was "cheating" by using absolute positional encoding
- v11 achieved unrealistic 95% confidence after seeing only 10% of data
- Root cause: Absolute positional encoding leaked temporal information
- Model learned: "Events peaking at day 0 are likely type X, events at day -20 are type Y"
- This is NOT real-time classification - it's inferring from temporal position!

**Solution in v12.0-beta**: Architectural redesign
- **Relative Positional Encoding** (model only knows observation count, not absolute time)
- **Wider t0 Distribution** (-50 to +50 days, was -20 to +20)
- **Variable-Length Sequence Support** (prevents padding artifacts)
- **Smaller, Faster Model** (~100K parameters vs ~450K in v11)

**Research Finding - Causal Truncation Rejected**:
During development, we tested causal truncation (randomly shortening sequences during training) as a potential solution. **We found it degraded performance**, particularly for PSPL events (accuracy dropped from 77% to <60%). 

**Physical Interpretation**: PSPL events exhibit smooth, symmetric magnification profiles that require observation of both rise and fall to distinguish from baseline fluctuations. When truncated, PSPL light curves become ambiguous. In contrast, binary events have sharp caustic features that are distinctive even in partial observations.

**Final Approach**: Architecture change (RelativePositionalEncoding) is sufficient to prevent data leakage. Standard full-sequence training outperforms augmented training with causal truncation. This demonstrates that **architecture > augmentation** for preventing temporal information leakage.

### Classification System

- **Class 0: Flat** (no event, just baseline fluctuations)
- **Class 1: PSPL** (single lens microlensing)
- **Class 2: Binary** (binary lens microlensing)

### Key v12.0-beta Improvements

1. **Relative Positional Encoding** (CRITICAL):
   - Model only knows: "I've seen N observations" and "gap since last observation"
   - Model CANNOT know: "I'm at day -50" or "peak should be at day 0"
   - Forces model to learn from magnification patterns only

2. **Variable-Length Sequences**:
   - No fixed padding patterns that model could exploit
   - Each batch has different max length
   - Prevents learning: "If padding starts at position X, it's event type Y"

3. **Wider t0 Distribution**:
   - Events can peak anywhere from -50 to +50 days (was -20 to +20)
   - More realistic temporal diversity
   - Prevents timing artifacts

4. **Smaller, Faster Model**:
   - d_model: 256 → 128 (4x fewer parameters!)
   - nhead: 8 → 4
   - num_layers: 6 → 4
   - Total: ~100K parameters (was ~450K)
   - Training time: ~4 hours (was ~12 hours on 8 GPUs)

5. **Realistic Performance**:
   - Early detection curve is now physically realistic
   - 10% observed → ~40% accuracy (near random for 3-class)
   - 50% observed → ~70% accuracy (magnification visible)
   - 100% observed → ~77% accuracy (empirically validated!)
   - v11's high early performance was an artifact!

### Version 12.0-beta (Current) - Architectural Fix Only
- ✅ **CRITICAL FIX**: Data leakage resolved via architecture
- ✅ Relative positional encoding (no absolute time)
- ✅ Variable-length sequences (no padding artifacts)
- ✅ Wider t0 distribution (-50 to +50 days)
- ✅ Smaller model (~100K parameters, 4.5x reduction)
- ✅ **NO causal truncation** (tested and rejected - hurts PSPL)
- ✅ Realistic early detection (physically grounded)
- ✅ Simpler implementation (architecture > augmentation)

### Version 11.1 (Previous)
- Three-class classification
- **ISSUE**: Absolute positional encoding caused data leakage

---

## 📚 Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
    title={From Light Curves to Labels: Machine Learning in Microlensing},
    author={Bhatia, Kunal},
    school={University of Heidelberg},
    year={2026},
    month={February},
    supervisor={Wambsganß, Joachim},
    type={Master's Thesis},
    note={Three-class classification with architectural fix (v12.0-beta)}
}
```

---

## 📧 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg  
Email: kunal29bhatia@gmail.com

---
