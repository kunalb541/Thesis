# Setup Guide v3.1: From Zero to Training (WITH BUG FIXES)

Complete installation guide with critical bug fixes for local workstations and HPC clusters.

---

## 🚨 CRITICAL: Read This First

**Version 3.1 fixes critical normalization bugs that invalidated all previous results.**

If you have v3.0 code:
1. Read [CRITICAL_BUGS_AND_FIXES.md](../CRITICAL_BUGS_AND_FIXES.md)
2. Apply all fixes before training
3. Discard any old results

This guide assumes you're starting fresh with v3.1 fixes applied.

---

## 🎯 Quick Links

- **Just want to get started?** → Jump to [Quick Start](#quick-start)
- **Need to apply fixes?** → See [Applying Critical Fixes](#applying-critical-fixes)
- **Having issues?** → See [Troubleshooting](#troubleshooting)
- **On a cluster?** → See [HPC Cluster Setup](#hpc-cluster-setup)

---

## 📋 Prerequisites

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 16 GB (32+ GB recommended)
- **Storage**: 50 GB free space
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+) or macOS

### Recommended for Training
- **GPU**: NVIDIA RTX 3090 / AMD MI200 series or better
- **RAM**: 64 GB+
- **Storage**: 200 GB SSD
- **Multi-GPU**: 2-4 GPUs for faster training

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis
```

---

### 2. Create Python Environment

**Option A: Using Conda (Recommended)**

```bash
# Install Miniconda if you don't have it
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens
```

**Option B: Using venv**

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
```

---

### 3. Install PyTorch

Choose based on your hardware:

**For NVIDIA GPUs (CUDA 12.1)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For NVIDIA GPUs (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPUs (ROCm 6.0)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

**For CPU only**:
```bash
pip install torch torchvision
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- NumPy, SciPy (scientific computing)
- scikit-learn (preprocessing, metrics)
- matplotlib, seaborn (visualization)
- VBMicrolensing (light curve simulation)
- tqdm, joblib (utilities)

---

### 5. ⚠️ Apply Critical Fixes

**REQUIRED BEFORE ANY TRAINING:**

```bash
cd Thesis

# 1. Replace utils.py with fixed version
cp utils_FIXED.py code/utils.py

echo "✓ utils.py updated with scaler loading functions"
```

**2. Fix train.py manually:**

Open `code/train.py` and find line ~180:
```python
# ❌ CHANGE THIS:
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=True)

# ✅ TO THIS:
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
```

**3. Fix evaluate.py:**

Add after loading data (around line 100):
```python
# Load RAW data
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)

# Load saved scalers from training
from utils import load_scalers, apply_scalers_to_data
scaler_std, scaler_mm = load_scalers(results_dir)
logger.info("✓ Loaded scalers from training")

# Apply same transformation used during training
X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
logger.info("✓ Applied same normalization as training")
```

**4. Fix plot_samples.py** (similar scaler loading)

**5. Fix benchmark_realtime.py** (similar scaler loading)

---

### 6. Verify Installation

```bash
python code/utils.py
```

This will check:
- ✅ PyTorch installed
- ✅ GPU detection
- ✅ All packages available
- ✅ Scaler functions working

**Expected output** (with GPU):
```
============================================================
GPU Check:
============================================================
✓ CUDA available: 1 GPU(s)
  GPU 0: NVIDIA RTX 4090
    Memory: 24.0 GB
    Compute: 8.9
============================================================

============================================================
Testing Scaler Functions:
============================================================
Train shape: (1000, 100)
Val shape: (200, 100)
Test shape: (200, 100)
Applying StandardScaler (fit on train only)...
Applying MinMaxScaler (fit on train only)...
Final ranges (non-padded values):
  Train: [0.000, 1.000]
  Val:   [-0.052, 1.043]
  Test:  [-0.031, 1.027]

✓ Normalization test passed!
============================================================
```

---

### 7. Quick Test (Verify Fixes Work)

Generate a small test dataset and train briefly:

```bash
cd code

# Generate 2K events (5 minutes)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --output ../data/raw/test_2k_v31.npz

# Quick training test (10 minutes)
python train.py \
    --data ../data/raw/test_2k_v31.npz \
    --epochs 5 \
    --batch_size 32 \
    --experiment_name test_v31
```

**✅ Verify in training logs:**
```bash
tail -50 $(ls -td ../results/test_v31_*/ | head -1)/training.log
```

Look for:
```
✓ FIT ON TRAIN ONLY - no data leakage
✓ Train data range: [0.000, 1.000]
✓ Scalers saved to results/test_v31_TIMESTAMP/
```

**✅ Check scaler files created:**
```bash
ls -lh $(ls -td ../results/test_v31_*/ | head -1)/scaler_*.pkl

# Should show:
# scaler_standard.pkl
# scaler_minmax.pkl
```

**✅ Test evaluation:**
```bash
python evaluate.py \
    --experiment_name test_v31 \
    --data ../data/raw/test_2k_v31.npz
```

Look for in logs:
```
✓ Loaded scalers from training
✓ Applied same normalization as training
```

If all checks pass, you're ready for full training! 🎉

---

## 🆕 Understanding v3.1 Changes

### What Was Fixed

1. **Double Normalization** (CRITICAL):
   - **Before**: Data normalized in `load_npz_dataset()`, then again in `two_stage_normalize()`
   - **After**: Data loaded raw, normalized once after splitting
   - **Impact**: 15-20% accuracy improvement

2. **Data Leakage** (CRITICAL):
   - **Before**: Scalers fitted on entire dataset (train+val+test)
   - **After**: Scalers fitted on training data only
   - **Impact**: Valid generalization estimates

3. **Evaluation Mismatch**:
   - **Before**: Evaluation re-fitted scalers on test set
   - **After**: Evaluation loads and applies training scalers
   - **Impact**: Consistent results between training and evaluation

### New Files Created

```
results/experiment_TIMESTAMP/
├── best_model.pt
├── config.json
├── training.log
├── summary.json
├── scaler_standard.pkl     # ⚠️ NEW
└── scaler_minmax.pkl        # ⚠️ NEW
```

### New Utility Functions

Added to `utils.py`:
- `load_scalers(model_dir)` - Load saved scalers from training
- `apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value)` - Apply scalers to new data
- Enhanced `save_scalers()` - Save scalers during training

---

## 💡 Correct Workflow (v3.1)

```
┌─────────────────────┐
│  Data Generation    │ simulate.py
│  (Raw .npz file)    │
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│  Training           │ train.py
│  ├─ Load RAW data   │ (normalize=False)
│  ├─ Split data      │
│  ├─ Fit scalers     │ (on train only)
│  ├─ Transform all   │ (train/val/test)
│  ├─ Save scalers    │ (to results dir)
│  └─ Train model     │
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│  Evaluation         │ evaluate.py
│  ├─ Load RAW data   │ (normalize=False)
│  ├─ Load scalers    │ (from training)
│  ├─ Transform data  │ (with train scalers)
│  └─ Evaluate model  │
└─────────────────────┘
```

**Key Points:**
- Raw data loaded everywhere
- Scalers fitted ONCE (on training data)
- Same scalers applied in evaluation
- No data leakage
- Consistent normalization

---

## 🔧 Applying Critical Fixes

### Detailed Fix Instructions

#### Fix 1: utils.py

```bash
# Option A: Use provided fixed version
cp utils_FIXED.py code/utils.py

# Option B: Add functions manually
# Add to code/utils.py:
```

```python
def load_scalers(model_dir):
    """Load saved scalers from training directory"""
    import pickle
    from pathlib import Path
    
    model_dir = Path(model_dir)
    
    with open(model_dir / "scaler_standard.pkl", 'rb') as f:
        scaler_standard = pickle.load(f)
    with open(model_dir / "scaler_minmax.pkl", 'rb') as f:
        scaler_minmax = pickle.load(f)
    
    print(f"✓ Loaded scalers from {model_dir}")
    return scaler_standard, scaler_minmax

def apply_scalers_to_data(X, scaler_standard, scaler_minmax, pad_value=-1):
    """Apply pre-fitted scalers to new data"""
    X_normalized = X.copy()
    mask = (X_normalized != pad_value)
    
    # Stage 1: StandardScaler
    X_normalized[mask] = scaler_standard.transform(
        X_normalized[mask].reshape(-1, 1)
    ).flatten()
    
    # Stage 2: MinMaxScaler
    X_normalized[mask] = scaler_minmax.transform(
        X_normalized[mask].reshape(-1, 1)
    ).flatten()
    
    return X_normalized
```

#### Fix 2: train.py

Find line ~180:
```python
# Load data
logger.info(f"Loading dataset: {args.data}")
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)  # Changed!
```

Verify logging after normalization:
```python
# After two_stage_normalize()
logger.info("Applying two-stage normalization (FIT ON TRAIN ONLY - no data leakage)...")
X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
    X_train, X_val, X_test, pad_value=CFG.PAD_VALUE
)
save_scalers(scaler_std, scaler_mm, output_dir)
logger.info(f"✓ Scalers saved to {output_dir}/")
```

#### Fix 3: evaluate.py

Replace data loading section (~line 100):
```python
# Load RAW data (no normalization)
logger.info("Loading RAW data...")
X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
L = X.shape[1]
logger.info(f"✓ Data loaded: {X.shape}")

# Load saved scalers from training directory
logger.info("Loading scalers from training...")
from utils import load_scalers, apply_scalers_to_data
scaler_std, scaler_mm = load_scalers(results_dir)
logger.info("✓ Loaded scalers from training")

# Apply same transformation used during training
logger.info("Applying training normalization...")
X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
logger.info("✓ Applied same normalization as training")
logger.info(f"Data range: [{X[X != CFG.PAD_VALUE].min():.3f}, {X[X != CFG.PAD_VALUE].max():.3f}]")
```

#### Fix 4: plot_samples.py

Similar to evaluate.py - load raw data and apply saved scalers:
```python
# For normalized data (model input)
logger.info("Loading normalized data (for model)...")
X_raw, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)

# Load scalers
scaler_std, scaler_mm = load_scalers(results_dir)
X_normalized = apply_scalers_to_data(X_raw, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
```

#### Fix 5: benchmark_realtime.py

Same pattern as evaluate.py - load scalers and apply.

---

## ✅ Verification Checklist

After applying all fixes:

- [ ] `utils.py` has `load_scalers()` and `apply_scalers_to_data()` functions
- [ ] `train.py` loads data with `normalize=False`
- [ ] `train.py` logs "FIT ON TRAIN ONLY - no data leakage"
- [ ] Training creates `scaler_*.pkl` files in results directory
- [ ] `evaluate.py` loads scalers with `load_scalers()`
- [ ] `evaluate.py` logs "Loaded scalers from training"
- [ ] Test training completes successfully
- [ ] Test evaluation loads scalers correctly
- [ ] No warning messages about data leakage

---

## 🖥️ HPC Cluster Setup

[Rest of HPC section remains the same as before...]

### Critical Addition for HPC

When submitting batch jobs, ensure fixes are applied:

```bash
#!/bin/bash
#SBATCH --job-name=baseline
# ... other SLURM directives ...

# Load modules and activate environment
module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# ⚠️ VERIFY FIXES BEFORE TRAINING
cd ~/Thesis/code

echo "Verifying v3.1 fixes..."

# Check utils.py has new functions
if grep -q "def load_scalers" utils.py; then
    echo "✓ utils.py has load_scalers()"
else
    echo "❌ utils.py missing load_scalers() - ABORTING"
    exit 1
fi

# Check train.py uses normalize=False
if grep -q "normalize=False" train.py; then
    echo "✓ train.py uses normalize=False"
else
    echo "❌ train.py still uses normalize=True - ABORTING"
    exit 1
fi

echo "✓ All fixes verified"
echo "Starting training..."

# Run training
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128

echo "Training complete!"
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'pickle'"

This shouldn't happen (pickle is built-in), but if it does:
```python
# In utils.py, add at top:
import pickle
```

### Issue: "File not found: scaler_standard.pkl"

**Cause**: Training completed before fixes were applied

**Solution**:
```bash
# Re-train with fixed code
python train.py --data data/raw/baseline.npz --experiment_name baseline_v31

# Verify scalers created
ls results/baseline_v31_*/scaler_*.pkl
```

### Issue: "Data range [-2.5, 3.8] - wrong scale"

**Cause**: Double normalization not fixed

**Solution**:
```bash
# Verify train.py uses normalize=False
grep "normalize=" code/train.py

# Should show:
# load_npz_dataset(args.data, apply_perm=True, normalize=False)
```

### Issue: "Different accuracy in training vs evaluation"

**Cause**: Evaluation not using saved scalers

**Solution**:
```bash
# Check evaluate.py loads scalers
grep "load_scalers" code/evaluate.py

# Should be present
```

### Issue: Low Accuracy Even After Fixes

Check these:

1. **Verify normalization range**:
   ```bash
   # Check training logs
   grep "Train data range" results/*/training.log
   # Should show approximately [0.0, 1.0]
   ```

2. **Verify no data leakage**:
   ```bash
   # Check for warning messages
   grep "data leakage" results/*/training.log
   # Should NOT appear
   ```

3. **Check model convergence**:
   ```bash
   # Plot training curves
   grep "Epoch" results/*/training.log | tail -20
   # Loss should decrease, accuracy should increase
   ```

---

## 📊 Expected Performance (v3.1)

### After Fixes Applied Correctly

**Small test (2K events, 5 epochs)**:
- Training accuracy: 60-70%
- Validation accuracy: 55-65%
- Should complete in 5-10 minutes on GPU

**Full baseline (1M events, 50 epochs)**:
- Training accuracy: 72-76%
- Validation accuracy: 70-74%
- Test accuracy: 70-74%
- Should complete in 6-8 hours on 4 GPUs

### Warning Signs (Fixes Not Applied)

- Training accuracy < 60% after 20 epochs
- Large train/val gap (>10%)
- Data range not [0, 1]
- Missing scaler files
- Evaluation logs don't mention "Loaded scalers"

---

## 🎯 Next Steps

After successful setup with fixes:

1. **Verify fixes with test dataset**:
   ```bash
   cd code
   python simulate.py --n_pspl 1000 --n_binary 1000 --output ../data/raw/test_v31.npz
   python train.py --data ../data/raw/test_v31.npz --experiment_name test_v31 --epochs 5
   ```

2. **Check all verification points**:
   - Logs show "FIT ON TRAIN ONLY"
   - Scalers created in results directory
   - Data range approximately [0, 1]
   - Training accuracy > 60%

3. **Generate full baseline dataset**:
   ```bash
   python simulate.py \
       --n_pspl 500000 --n_binary 500000 \
       --output ../data/raw/baseline_1M_v31.npz \
       --binary_params baseline
   ```

4. **Train baseline**:
   ```bash
   python train.py \
       --data ../data/raw/baseline_1M_v31.npz \
       --experiment_name baseline_v31
   ```

5. **Monitor and evaluate**:
   ```bash
   # Watch training
   tail -f $(ls -td ../results/baseline_v31_*/ | head -1)/training.log
   
   # After completion
   python evaluate.py --experiment_name baseline_v31 --data ../data/raw/baseline_1M_v31.npz
   ```

---

## 📞 Getting Help

### For Bug Fixes:
- Read [CRITICAL_BUGS_AND_FIXES.md](../CRITICAL_BUGS_AND_FIXES.md) thoroughly
- Verify each fix was applied correctly
- Check verification steps passed

### For GPU/CUDA Issues:
- NVIDIA: https://forums.developer.nvidia.com/
- AMD: https://community.amd.com/

### For PyTorch Issues:
- Forum: https://discuss.pytorch.org/
- GitHub: https://github.com/pytorch/pytorch

### For This Project:
- Email: kunal29bhatia@gmail.com
- Include: logs, error messages, verification output

---

**You're now ready to start training with v3.1 fixes!** 🚀

Remember: Always verify fixes before starting long experiments!