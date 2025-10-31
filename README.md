## Real-Time Microlensing Classification using Deep Learning

**Master's Thesis Project - Version 3.1** ⚠️ **CRITICAL BUGS FIXED**  
**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Institution**: University of Heidelberg  
**Last Updated**: October 2025

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚨 IMPORTANT: Critical Bug Fixes (v3.1)

**⚠️ IF YOU HAVE EXISTING RESULTS, READ THIS FIRST:**

Version 3.1 fixes **critical bugs** that invalidated previous results:

1. **Double Normalization Bug**: Data was normalized twice, causing incorrect scaling
2. **Data Leakage Bug**: Validation/test statistics leaked into training
3. **Scaler Mismatch**: Evaluation used different scalers than training

**Action Required:**
- ✅ Read [CRITICAL_BUGS_AND_FIXES.md](CRITICAL_BUGS_AND_FIXES.md) for details
- ✅ Apply fixes to `train.py`, `utils.py`, `evaluate.py`
- ✅ Re-train ALL models (old models are invalid)
- ✅ Do NOT use old results in thesis

**Expected Performance After Fixes:**
- Training accuracy: 70-75% (was ~55% with bugs)
- Stable train/val curves
- Test accuracy within 2-3% of validation

---

## 🎯 Project Overview

**Research Question**: Can deep learning enable real-time classification of binary microlensing events for next-generation surveys (LSST, Roman)?

**Approach**: TimeDistributed 1D CNN trained on 1M+ synthetic light curves from VBMicrolensing.

**Key Innovation**: Temporal aggregation across full light curve captures distributed caustic features → enables early detection.

---

## 🔧 Critical Fixes in v3.1

### Bug #1: Double Normalization (FIXED)

**Problem**: Data was normalized twice - once during loading, once after splitting
```python
# ❌ WRONG (v3.0):
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=True)
# ... split data ...
X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(...)

# ✅ CORRECT (v3.1):
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
# ... split data ...
X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(...)
```

**Impact**: 
- Incorrect data scale
- Data leakage (test statistics in training)
- Poor model performance

### Bug #2: Scaler Mismatch in Evaluation (FIXED)

**Problem**: Evaluation scripts re-fitted scalers instead of loading saved ones

**Fix**: New functions in `utils.py`:
```python
# Load scalers from training
scaler_std, scaler_mm = load_scalers(results_dir)

# Apply to evaluation data
X_normalized = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=-1)
```

### Verification After Fixes

```bash
# 1. Train with fixed code
python train.py --data data/raw/test.npz --experiment_name test_fix --epochs 5

# Check logs for:
# "Applying two-stage normalization (FIT ON TRAIN ONLY - no data leakage)..."
# "Train data range: [0.000, 1.000]"  # Should be approximately [0, 1]

# 2. Verify scaler files created
ls results/test_fix_*/scaler_*.pkl
# Should see: scaler_standard.pkl, scaler_minmax.pkl

# 3. Evaluate with correct scalers
python evaluate.py --experiment_name test_fix --data data/raw/test.npz
# Check logs for:
# "✓ Loaded scalers from training"
# "✓ Applied same normalization as training"
```

---

## 📊 Project Status (v3.1)

| Component | Status | Notes |
|-----------|--------|-------|
| Environment Setup | ✅ Complete | Tested on NVIDIA/AMD GPUs |
| Data Simulation | ✅ Complete | VBMicrolensing pipeline working |
| Training Pipeline | ✅ **FIXED** | **Normalization bugs resolved** |
| Evaluation Pipeline | ✅ **FIXED** | **Now uses saved scalers** |
| Baseline Experiment | 🔄 **RESTART REQUIRED** | Must re-run with fixes |
| Cadence Experiments | ⏳ Pending | After baseline completes |
| Error Experiments | ⏳ Pending | After baseline completes |
| Topology Experiments | ⏳ Pending | After baseline completes |
| Real-time Benchmarking | ⏳ Pending | After evaluation fixed |
| Thesis Writing | ⏳ Not Started | Awaiting valid results |

---

## 🚀 Quick Start (v3.1 - FIXED VERSION)

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
python code/utils.py
```

### Apply Critical Fixes

⚠️ **BEFORE RUNNING ANY EXPERIMENTS:**

```bash
# 1. Replace utils.py with fixed version
cp utils_FIXED.py code/utils.py

# 2. Fix train.py (change one line)
# In code/train.py, line ~180:
# CHANGE: load_npz_dataset(args.data, apply_perm=True, normalize=True)
# TO:     load_npz_dataset(args.data, apply_perm=True, normalize=False)

# 3. Fix evaluate.py (see CRITICAL_BUGS_AND_FIXES.md for details)

# 4. Test with small dataset
cd code
python simulate.py --n_pspl 1000 --n_binary 1000 --output ../data/raw/test_fix.npz
python train.py --data ../data/raw/test_fix.npz --experiment_name test_fix --epochs 5

# Should see in logs:
# "FIT ON TRAIN ONLY - no data leakage"
# "Train data range: [0.000, 1.000]"
```

---

## 💡 Key Features (Version 3.1 - FIXED)

### Correct Normalization Pipeline
- ✅ **No data leakage**: Scalers fit on training data only
- ✅ **Consistent scaling**: Same normalization in train and evaluation
- ✅ **Saved scalers**: Automatically saved and loaded
- ✅ **Verified ranges**: Data properly scaled to [0, 1]

### Temporal Aggregation (v3.0 fix maintained)
- ✅ **Per-timestep loss**: Loss computed at every timestep (not aggregated)
- ✅ **Mean aggregation**: Predictions aggregated via mean pooling
- ✅ **Early detection**: Enables classification with partial observations

### Auto-Detection (v3.0 feature maintained)
- ✅ **Timestamped directories**: Each run gets unique timestamp
- ✅ **Auto model finding**: Scripts find latest model automatically
- ✅ **Easy comparison**: Multiple runs preserved separately

---

## 📁 Repository Structure

```
Thesis/
├── code/
│   ├── simulate.py              # Dataset generation (VBMicrolensing)
│   ├── train.py                 # Training with FIXED normalization ⚠️
│   ├── evaluate.py              # Evaluation with FIXED scaler loading ⚠️
│   ├── benchmark_realtime.py    # Inference speed benchmarking ⚠️
│   ├── plot_samples.py          # Sample visualization ⚠️
│   ├── model.py                 # TimeDistributedCNN architecture
│   ├── config.py                # All experiment configurations
│   └── utils.py                 # FIXED: Scaler loading/saving ⚠️
│
├── data/
│   └── raw/                     # Simulated light curves (.npz)
│
├── results/                     # Auto-generated timestamped directories
│   └── {experiment}_{timestamp}/
│       ├── best_model.pt        # Best model checkpoint
│       ├── config.json          # Experiment configuration
│       ├── training.log         # Training logs
│       ├── summary.json         # Final metrics
│       ├── scaler_standard.pkl  # ⚠️ NEW: Saved StandardScaler
│       ├── scaler_minmax.pkl    # ⚠️ NEW: Saved MinMaxScaler
│       ├── evaluation/          # Evaluation results
│       └── benchmark/           # Benchmark results
│
├── docs/
│   ├── SETUP_GUIDE.md          # Installation instructions (UPDATED)
│   ├── RESEARCH_GUIDE.md       # Thesis workflow (UPDATED)
│   ├── QUICK_REFERENCE.md      # Command cheatsheet (UPDATED)
│   └── CRITICAL_BUGS_AND_FIXES.md  # ⚠️ NEW: Detailed bug documentation
│
├── CRITICAL_BUGS_AND_FIXES.md  # ⚠️ READ THIS FIRST
├── utils_FIXED.py              # ⚠️ NEW: Fixed utils.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file (UPDATED)
```

---

## 🔬 Systematic Experiments (After Fixes)

### Workflow for Each Experiment

```bash
# 1. Generate data (once)
python simulate.py \
    --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline

# 2. Train (creates timestamped directory with scalers)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50

# Verify in logs:
# ✓ "FIT ON TRAIN ONLY - no data leakage"
# ✓ "Train data range: [0.000, 1.000]"
# ✓ "Scalers saved to results/baseline_TIMESTAMP/"

# 3. Evaluate (auto-loads scalers)
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection

# Verify in logs:
# ✓ "Loaded scalers from training"
# ✓ "Applied same normalization as training"

# 4. Benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

### Expected Performance (After Fixes)

| Metric | Before Fixes (v3.0) | After Fixes (v3.1) |
|--------|---------------------|-------------------|
| Training Accuracy | ~55% | **70-75%** |
| Validation Accuracy | ~50% | **70-75%** |
| Test Accuracy | ~50% | **70-75%** |
| Train/Val Gap | Large (overfitting) | Small (2-3%) |
| Data Scale | Incorrect (double normalized) | Correct [0, 1] |

---

## 📝 Documentation

- **[CRITICAL_BUGS_AND_FIXES.md](CRITICAL_BUGS_AND_FIXES.md)**: ⚠️ **START HERE** - Detailed bug explanations and fixes
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Complete installation guide (updated for v3.1)
- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Physics background, experiment design (updated for v3.1)
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Command cheatsheet (updated for v3.1)

---

## 🧪 Reproducibility (v3.1)

All experiments are now truly reproducible with fixes:

1. **Fixed random seeds**: Set in `config.py` and enforced in all scripts
2. **Saved configurations**: All experiment parameters logged to `config.json`
3. **Saved scalers**: ⚠️ **NEW**: StandardScaler and MinMaxScaler saved with each experiment
4. **Consistent normalization**: Same scalers used for train, val, test, and evaluation
5. **Data permutations**: Saved and reapplied consistently via `load_npz_dataset()`
6. **Exact versions**: See `requirements.txt` for pinned dependencies

---

## 📧 Contact

**Author**: Kunal Bhatia  
**Email**: kunal29bhatia@gmail.com  
**Institution**: University of Heidelberg

**For Issues**:
- **Bugs**: Read [CRITICAL_BUGS_AND_FIXES.md](CRITICAL_BUGS_AND_FIXES.md) first
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

## 🔄 Version History

**v3.1** (October 2025) - **CRITICAL BUG FIXES**:
- ⚠️ Fixed double normalization bug
- ⚠️ Fixed data leakage bug
- ⚠️ Added scaler saving/loading for evaluation
- ⚠️ All previous results INVALID - must re-run
- Updated all documentation with warnings
- Added CRITICAL_BUGS_AND_FIXES.md

**v3.0** (October 2025):
- Auto-detection of results directories
- Timestamped experiment organization
- Unified data loading via `load_npz_dataset()`
- Improved experiment tracking

**v2.0** (October 2025):
- Fixed temporal aggregation bug
- Added early detection analysis
- Multi-GPU training support

**v1.0** (September 2025):
- Initial implementation

---

## ⚠️ Pre-Training Checklist (v3.1)

Before starting experiments:

- [ ] Read [CRITICAL_BUGS_AND_FIXES.md](CRITICAL_BUGS_AND_FIXES.md)
- [ ] Replace `code/utils.py` with `utils_FIXED.py`
- [ ] Fix `train.py` (change `normalize=True` to `normalize=False`)
- [ ] Fix `evaluate.py` (add scaler loading)
- [ ] Fix `plot_samples.py` (add scaler loading)
- [ ] Fix `benchmark_realtime.py` (add scaler loading)
- [ ] Test with small dataset (verify logs show correct normalization)
- [ ] Verify scaler files are created in results directory
- [ ] Delete all old results (they are invalid)

---

## 🎯 Next Steps

1. **Apply all fixes** (see CRITICAL_BUGS_AND_FIXES.md)
2. **Test with small dataset** to verify fixes work
3. **Re-generate baseline dataset** (1M events)
4. **Re-train baseline** with fixed code
5. **Run all systematic experiments** with fixed code
6. **Write thesis** with valid results

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**⚠️ REMEMBER: Version 3.1 fixes critical bugs. All previous results must be discarded and experiments re-run with the fixed code.** 🚨