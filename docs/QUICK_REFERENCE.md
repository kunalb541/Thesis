# Quick Reference Guide - v3.1 (WITH CRITICAL BUG FIXES)

**Purpose**: Command cheatsheet for all experiments with v3.1 fixes  
**Usage**: Copy-paste commands for running experiments  
**Last Updated**: October 2025

---

## 🚨 CRITICAL: Use v3.1 Commands Only

**⚠️ ALL v3.0 COMMANDS ARE OBSOLETE** due to critical bug fixes.

**Before using any commands:**
1. ✅ Apply all v3.1 fixes (see [CRITICAL_BUGS_AND_FIXES.md](../CRITICAL_BUGS_AND_FIXES.md))
2. ✅ Verify fixes with test dataset
3. ✅ Check logs show "FIT ON TRAIN ONLY - no data leakage"
4. ✅ Confirm scaler files created

---

## 🆕 v3.1 Features

### Verification After Each Command

Every training command should be followed by verification:

```bash
# After training
EXP=baseline_v31
LATEST=$(ls -td results/${EXP}_*/ | head -1)

# 1. Check scalers created
ls $LATEST/scaler_*.pkl
# Should show: scaler_standard.pkl, scaler_minmax.pkl

# 2. Check normalization range
grep "Train data range" $LATEST/training.log
# Should show: approximately [0.000, 1.000]

# 3. Check no data leakage
grep "data leakage" $LATEST/training.log
# Should show: "FIT ON TRAIN ONLY - no data leakage"
```

### New Naming Convention

All experiments now use `_v31` suffix to distinguish from invalid v3.0 results:

```
OLD (v3.0 - INVALID):  baseline_20251027_143022/
NEW (v3.1 - VALID):    baseline_v31_20251027_143022/
```

---

## 🚀 All Experiments - Command List (v3.1)

### Pre-Flight Check

**ALWAYS run this before starting experiments:**

```bash
cd code

# Test normalization with dummy data
python -c "
import numpy as np
from utils import two_stage_normalize

# Create test data
X_train = np.random.randn(100, 50)
X_val = np.random.randn(20, 50)
X_test = np.random.randn(20, 50)

# Add padding
X_train[:, -5:] = -1
X_val[:, -5:] = -1
X_test[:, -5:] = -1

# Normalize
X_train_scaled, X_val_scaled, X_test_scaled, scaler_std, scaler_mm = two_stage_normalize(
    X_train, X_val, X_test, pad_value=-1
)

print('✓ Normalization test passed')
"

# Verify utils.py has new functions
grep -q "def load_scalers" utils.py && echo "✓ load_scalers() present" || echo "❌ MISSING"
grep -q "def apply_scalers_to_data" utils.py && echo "✓ apply_scalers_to_data() present" || echo "❌ MISSING"

# Verify train.py uses normalize=False
grep "normalize=False" train.py && echo "✓ train.py fixed" || echo "❌ NEEDS FIX"

echo ""
echo "If all checks passed, you're ready to run experiments!"
```

---

### E1: Baseline (1M events) - v3.1

```bash
cd code

# ========================================
# 1. DATA GENERATION (2-3 hours on 24 cores)
# ========================================
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M_v31.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.20 \
    --mag_error_std 0.10 \
    --seed 42 \
    --num_workers 24

echo "✓ Data generated"

# ========================================
# 2. TRAINING (6-8 hours on 4 GPUs)
# ========================================
python train.py \
    --data ../data/raw/baseline_1M_v31.npz \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --experiment_name baseline_v31

# ========================================
# 3. VERIFICATION (IMMEDIATELY AFTER TRAINING)
# ========================================
LATEST=$(ls -td ../results/baseline_v31_*/ | head -1)
echo "Verifying: $LATEST"

# Check scalers
ls $LATEST/scaler_*.pkl || echo "❌ SCALERS MISSING - TRAINING INVALID"

# Check normalization
echo "Checking normalization range:"
grep "Train data range" $LATEST/training.log

# Check data leakage
grep -q "FIT ON TRAIN ONLY" $LATEST/training.log && echo "✓ No data leakage" || echo "❌ DATA LEAKAGE"

# ========================================
# 4. EVALUATION (auto-detects latest model)
# ========================================
python evaluate.py \
    --experiment_name baseline_v31 \
    --data ../data/raw/baseline_1M_v31.npz \
    --early_detection

# Verify scaler loading
EVAL_LOG=$(ls -td ../results/baseline_v31_*/ | head -1)/evaluation/evaluation.log
grep -q "Loaded scalers from training" $EVAL_LOG && echo "✓ Scalers loaded" || echo "❌ SCALER MISMATCH"

# ========================================
# 5. REAL-TIME BENCHMARK
# ========================================
python benchmark_realtime.py \
    --experiment_name baseline_v31 \
    --data ../data/raw/baseline_1M_v31.npz

echo ""
echo "✓ Baseline complete!"
echo "Results in: $LATEST"
```

---

### E2: Dense Cadence (5% missing) - v3.1

```bash
# ========================================
# DATA GENERATION
# ========================================
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/cadence_05_v31.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.05 \
    --seed 42 \
    --num_workers 24

# ========================================
# TRAINING
# ========================================
python train.py \
    --data ../data/raw/cadence_05_v31.npz \
    --experiment_name cadence_05_v31 \
    --epochs 50

# ========================================
# VERIFICATION
# ========================================
LATEST=$(ls -td ../results/cadence_05_v31_*/ | head -1)
ls $LATEST/scaler_*.pkl && echo "✓ Scalers saved" || echo "❌ INVALID"
grep "Train data range" $LATEST/training.log

# ========================================
# EVALUATION
# ========================================
python evaluate.py \
    --experiment_name cadence_05_v31 \
    --data ../data/raw/cadence_05_v31.npz \
    --early_detection
```

---

### E3-E4: Sparse Cadence (30%, 40% missing) - v3.1

```bash
# ========================================
# BATCH GENERATION AND TRAINING
# ========================================
for cadence in 30 40; do
    echo "========================================="
    echo "Processing cadence ${cadence}% missing"
    echo "========================================="
    
    # Generate
    python simulate.py \
        --n_pspl 100000 \
        --n_binary 100000 \
        --output ../data/raw/cadence_${cadence}_v31.npz \
        --binary_params baseline \
        --cadence_mask_prob 0.${cadence} \
        --seed 42
    
    # Train
    python train.py \
        --data ../data/raw/cadence_${cadence}_v31.npz \
        --experiment_name cadence_${cadence}_v31 \
        --epochs 50
    
    # Verify
    LATEST=$(ls -td ../results/cadence_${cadence}_v31_*/ | head -1)
    echo "Verifying: $LATEST"
    ls $LATEST/scaler_*.pkl && echo "✓ Scalers OK" || echo "❌ INVALID"
    
    # Evaluate
    python evaluate.py \
        --experiment_name cadence_${cadence}_v31 \
        --data ../data/raw/cadence_${cadence}_v31.npz \
        --early_detection
    
    echo "✓ cadence_${cadence}_v31 complete"
    echo ""
done
```

---

### E5-E6: Photometric Error Experiments - v3.1

```bash
# ========================================
# LOW ERROR (0.05 mag - Space-based)
# ========================================
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/error_05_v31.npz \
    --binary_params baseline \
    --mag_error_std 0.05 \
    --seed 42

python train.py \
    --data ../data/raw/error_05_v31.npz \
    --experiment_name error_05_v31 \
    --epochs 50

# Verify
LATEST=$(ls -td ../results/error_05_v31_*/ | head -1)
ls $LATEST/scaler_*.pkl || echo "❌ INVALID"

python evaluate.py \
    --experiment_name error_05_v31 \
    --data ../data/raw/error_05_v31.npz \
    --early_detection

# ========================================
# HIGH ERROR (0.20 mag - Poor conditions)
# ========================================
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/error_20_v31.npz \
    --binary_params baseline \
    --mag_error_std 0.20 \
    --seed 42

python train.py \
    --data ../data/raw/error_20_v31.npz \
    --experiment_name error_20_v31 \
    --epochs 50

# Verify
LATEST=$(ls -td ../results/error_20_v31_*/ | head -1)
ls $LATEST/scaler_*.pkl || echo "❌ INVALID"

python evaluate.py \
    --experiment_name error_20_v31 \
    --data ../data/raw/error_20_v31.npz \
    --early_detection
```

---

### E7-E9: Binary Topology Experiments - v3.1

```bash
# ========================================
# BATCH PROCESSING FOR ALL TOPOLOGIES
# ========================================
for topo in distinct planetary stellar; do
    echo "========================================="
    echo "Processing ${topo} topology"
    echo "========================================="
    
    # Generate
    python simulate.py \
        --n_pspl 100000 \
        --n_binary 100000 \
        --output ../data/raw/${topo}_v31.npz \
        --binary_params ${topo} \
        --seed 42 \
        --num_workers 24
    
    # Train
    python train.py \
        --data ../data/raw/${topo}_v31.npz \
        --experiment_name ${topo}_v31 \
        --epochs 50
    
    # Verify
    LATEST=$(ls -td ../results/${topo}_v31_*/ | head -1)
    echo "Verifying: $LATEST"
    
    # Critical checks
    if [ ! -f "$LATEST/scaler_standard.pkl" ]; then
        echo "❌ CRITICAL: Missing scaler_standard.pkl"
        echo "   Training is INVALID - do not use results!"
        continue
    fi
    
    if [ ! -f "$LATEST/scaler_minmax.pkl" ]; then
        echo "❌ CRITICAL: Missing scaler_minmax.pkl"
        echo "   Training is INVALID - do not use results!"
        continue
    fi
    
    echo "✓ Scalers present"
    grep "Train data range" $LATEST/training.log
    
    # Evaluate
    python evaluate.py \
        --experiment_name ${topo}_v31 \
        --data ../data/raw/${topo}_v31.npz \
        --early_detection
    
    echo "✓ ${topo}_v31 complete"
    echo ""
done
```

---

## 🔄 Complete Experiment Suite (One Script)

**Run all experiments sequentially with verification:**

```bash
#!/bin/bash
# run_all_experiments_v31.sh

set -e  # Exit on any error

cd code

echo "========================================="
echo "STARTING ALL v3.1 EXPERIMENTS"
echo "========================================="
echo ""

# Pre-flight check
echo "Running pre-flight checks..."
python -c "from utils import load_scalers, apply_scalers_to_data; print('✓ Utils OK')"
grep -q "normalize=False" train.py && echo "✓ train.py fixed" || (echo "❌ train.py not fixed"; exit 1)
echo ""

# Baseline
echo "1/9: Baseline (1M events)..."
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M_v31.npz --binary_params baseline --seed 42
python train.py --data ../data/raw/baseline_1M_v31.npz --experiment_name baseline_v31 --epochs 50
LATEST=$(ls -td ../results/baseline_v31_*/ | head -1)
ls $LATEST/scaler_*.pkl || (echo "❌ Baseline INVALID"; exit 1)
python evaluate.py --experiment_name baseline_v31 --data ../data/raw/baseline_1M_v31.npz --early_detection
echo "✓ Baseline complete"
echo ""

# Cadence experiments
for cadence in 05 30 40; do
    echo "$((cadence+1))/9: Cadence ${cadence}%..."
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${cadence}_v31.npz \
        --cadence_mask_prob 0.${cadence} --seed 42
    python train.py --data ../data/raw/cadence_${cadence}_v31.npz \
        --experiment_name cadence_${cadence}_v31 --epochs 50
    LATEST=$(ls -td ../results/cadence_${cadence}_v31_*/ | head -1)
    ls $LATEST/scaler_*.pkl || (echo "❌ Cadence ${cadence} INVALID"; exit 1)
    python evaluate.py --experiment_name cadence_${cadence}_v31 \
        --data ../data/raw/cadence_${cadence}_v31.npz --early_detection
    echo "✓ Cadence ${cadence} complete"
    echo ""
done

# Error experiments
for error in 05 20; do
    echo "Error ${error}..."
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${error}_v31.npz \
        --mag_error_std 0.${error} --seed 42
    python train.py --data ../data/raw/error_${error}_v31.npz \
        --experiment_name error_${error}_v31 --epochs 50
    LATEST=$(ls -td ../results/error_${error}_v31_*/ | head -1)
    ls $LATEST/scaler_*.pkl || (echo "❌ Error ${error} INVALID"; exit 1)
    python evaluate.py --experiment_name error_${error}_v31 \
        --data ../data/raw/error_${error}_v31.npz --early_detection
    echo "✓ Error ${error} complete"
    echo ""
done

# Topology experiments
for topo in distinct planetary stellar; do
    echo "Topology: ${topo}..."
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}_v31.npz \
        --binary_params ${topo} --seed 42
    python train.py --data ../data/raw/${topo}_v31.npz \
        --experiment_name ${topo}_v31 --epochs 50
    LATEST=$(ls -td ../results/${topo}_v31_*/ | head -1)
    ls $LATEST/scaler_*.pkl || (echo "❌ ${topo} INVALID"; exit 1)
    python evaluate.py --experiment_name ${topo}_v31 \
        --data ../data/raw/${topo}_v31.npz --early_detection
    echo "✓ ${topo} complete"
    echo ""
done

echo "========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================="
echo ""
echo "Generating summary..."
python -c "
import json
from pathlib import Path

experiments = [
    'baseline_v31', 'cadence_05_v31', 'cadence_30_v31', 'cadence_40_v31',
    'error_05_v31', 'error_20_v31', 'distinct_v31', 'planetary_v31', 'stellar_v31'
]

print(f'{'Experiment':<20} {'Test Acc':<12} {'Scalers':<10}')
print('-' * 45)

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        latest = runs[-1]
        
        # Check validity
        valid = (latest / 'scaler_standard.pkl').exists() and (latest / 'scaler_minmax.pkl').exists()
        status = '✓' if valid else '❌ INVALID'
        
        summary_file = latest / 'summary.json'
        if summary_file.exists() and valid:
            with open(summary_file) as f:
                data = json.load(f)
            acc = data.get('final_test_acc', 0) * 100
            print(f'{exp:<20} {acc:>10.2f}% {status:<10}')
        else:
            print(f'{exp:<20} {'N/A':<12} {status:<10}')
"
```

**Save as `run_all_experiments_v31.sh` and run:**
```bash
chmod +x run_all_experiments_v31.sh
./run_all_experiments_v31.sh
```

---

## 🐛 Debugging Commands

### Check If v3.1 Fixes Applied

```bash
cd code

echo "Checking v3.1 compliance..."
echo ""

# 1. Check utils.py
echo "1. Checking utils.py..."
grep -q "def load_scalers" utils.py && echo "  ✓ load_scalers() present" || echo "  ❌ MISSING"
grep -q "def apply_scalers_to_data" utils.py && echo "  ✓ apply_scalers_to_data() present" || echo "  ❌ MISSING"

# 2. Check train.py
echo "2. Checking train.py..."
grep -n "load_npz_dataset" train.py | grep "normalize="
echo "  (Should show normalize=False, not normalize=True)"

# 3. Check evaluate.py
echo "3. Checking evaluate.py..."
grep -q "load_scalers" evaluate.py && echo "  ✓ Loads scalers" || echo "  ❌ DOESN'T load scalers"

# 4. Check for old results
echo "4. Checking for old (invalid) results..."
OLD_RESULTS=$(ls -d ../results/* 2>/dev/null | grep -v "_v31_" | wc -l)
if [ $OLD_RESULTS -gt 0 ]; then
    echo "  ⚠️  Found $OLD_RESULTS old result directories"
    echo "  These should be deleted (they're invalid)"
else
    echo "  ✓ No old results found"
fi

echo ""
echo "Summary: If all checks passed, you're using v3.1"
```

### Validate Existing Results

```bash
# Check if existing results are valid (v3.1)
for dir in ../results/*/; do
    EXP=$(basename $dir)
    
    # Skip if not a results directory
    [ ! -f "$dir/best_model.pt" ] && continue
    
    echo "Checking: $EXP"
    
    # Must have scalers
    if [ -f "$dir/scaler_standard.pkl" ] && [ -f "$dir/scaler_minmax.pkl" ]; then
        echo "  ✓ Scalers present"
        
        # Check logs
        if grep -q "FIT ON TRAIN ONLY" "$dir/training.log" 2>/dev/null; then
            echo "  ✓ Proper normalization"
            echo "  ✅ VALID v3.1 results"
        else
            echo "  ⚠️  Old logs (might be v3.0)"
            echo "  ❓ VERIFY MANUALLY"
        fi
    else
        echo "  ❌ Missing scalers - INVALID (v3.0 or broken)"
    fi
    
    echo ""
done
```

### Find Data Normalization Issues

```bash
# Check data normalization in training logs
for log in ../results/*/training.log; do
    EXP=$(dirname $log | xargs basename)
    
    echo "Checking: $EXP"
    
    # Extract normalization range
    RANGE=$(grep "Train data range" $log 2>/dev/null | tail -1)
    
    if [ -n "$RANGE" ]; then
        echo "  $RANGE"
        
        # Check if range is approximately [0, 1]
        if echo "$RANGE" | grep -q "\[0\."; then
            echo "  ✓ Looks correct"
        else
            echo "  ❌ WRONG RANGE - double normalization likely"
        fi
    else
        echo "  ⚠️  No normalization range found"
    fi
    
    echo ""
done
```

---

## 📊 Analysis Commands (v3.1)

### Extract Results Table

```bash
python -c "
import json
from pathlib import Path

experiments = {
    'baseline_v31': 'Baseline (1M, 20% missing, 0.10 error)',
    'cadence_05_v31': 'Dense Cadence (5% missing)',
    'cadence_30_v31': 'Sparse Cadence (30% missing)',
    'cadence_40_v31': 'Very Sparse (40% missing)',
    'error_05_v31': 'Low Error (0.05 mag)',
    'error_20_v31': 'High Error (0.20 mag)',
    'distinct_v31': 'Distinct Binaries',
    'planetary_v31': 'Planetary Systems',
    'stellar_v31': 'Stellar Binaries',
}

print(f'{'Experiment':<50} {'Test Acc':<12} {'Valid':<8}')
print('-' * 75)

for exp, desc in experiments.items():
    runs = sorted(Path('results').glob(f'{exp}_*'))
    
    if runs:
        latest = runs[-1]
        
        # Check validity (v3.1)
        valid = (
            (latest / 'scaler_standard.pkl').exists() and
            (latest / 'scaler_minmax.pkl').exists()
        )
        
        status = '✓ v3.1' if valid else '❌ v3.0'
        
        summary_file = latest / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            acc = data.get('final_test_acc', 0) * 100
            print(f'{desc:<50} {acc:>10.2f}% {status:<8}')
        else:
            print(f'{desc:<50} {'N/A':<12} {status:<8}')
    else:
        print(f'{desc:<50} {'Not run':<12} {'-':<8}')
" > results_summary_v31.txt

cat results_summary_v31.txt
```

### Compare v3.0 vs v3.1 (If You Have Both)

```bash
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_05', 'cadence_30', 'error_05']

print(f'{'Experiment':<20} {'v3.0 (buggy)':<15} {'v3.1 (fixed)':<15} {'Diff':<10}')
print('-' * 65)

for exp in experiments:
    # v3.0 results (no _v31)
    v30_runs = sorted(Path('results').glob(f'{exp}_*'))
    v30_runs = [r for r in v30_runs if '_v31_' not in str(r)]
    
    # v3.1 results (with _v31)
    v31_runs = sorted(Path('results').glob(f'{exp}_v31_*'))
    
    v30_acc = None
    v31_acc = None
    
    if v30_runs:
        summary = v30_runs[-1] / 'summary.json'
        if summary.exists():
            with open(summary) as f:
                v30_acc = json.load(f).get('final_test_acc', 0) * 100
    
    if v31_runs:
        summary = v31_runs[-1] / 'summary.json'
        if summary.exists():
            with open(summary) as f:
                v31_acc = json.load(f).get('final_test_acc', 0) * 100
    
    if v30_acc and v31_acc:
        diff = v31_acc - v30_acc
        print(f'{exp:<20} {v30_acc:>13.2f}% {v31_acc:>13.2f}% {diff:>+8.2f}%')
    else:
        v30_str = f'{v30_acc:.2f}%' if v30_acc else 'N/A'
        v31_str = f'{v31_acc:.2f}%' if v31_acc else 'N/A'
        print(f'{exp:<20} {v30_str:>13} {v31_str:>13} {\"N/A\":>10}')
"
```

---

## 🔧 Maintenance Commands (v3.1)

### Clean Up Old (Invalid) Results

```bash
# List old v3.0 results (no _v31)
echo "Old (invalid) results to delete:"
ls -d results/*/ | grep -v "_v31_"

# Delete them (BE CAREFUL!)
# Uncomment after reviewing:
# ls -d results/*/ | grep -v "_v31_" | xargs rm -rf

echo ""
echo "✓ Cleaned up v3.0 results"
```

### Archive v3.1 Results

```bash
# Archive all v3.1 results for thesis
tar -czf results_v31_$(date +%Y%m%d).tar.gz \
    results/*_v31_* \
    data/raw/*_v31.npz

echo "✓ Archived to: results_v31_$(date +%Y%m%d).tar.gz"
```

---

## ⚡ Performance Tips (v3.1)

### Monitor Training in Real-Time

```bash
# Watch training progress
watch -n 2 "tail -20 \$(ls -td results/baseline_v31_*/ | head -1)/training.log | grep Epoch"

# Monitor GPU usage
watch -n 1 nvidia-smi  # or rocm-smi for AMD
```

### Quick Accuracy Check

```bash
# Get current accuracy without full evaluation
EXP=baseline_v31
LATEST=$(ls -td results/${EXP}_*/ | head -1)

echo "Latest run: $LATEST"
echo ""

# From training logs
echo "Training performance:"
grep "Epoch" $LATEST/training.log | tail -1

# From summary
if [ -f "$LATEST/summary.json" ]; then
    echo "Final test accuracy:"
    python -c "import json; print(f\"{json.load(open('$LATEST/summary.json'))['final_test_acc']*100:.2f}%\")"
fi
```

---

## ✅ Pre-Experiment Checklist (v3.1)

**Run this before EVERY experiment:**

```bash
#!/bin/bash
# pre_experiment_check_v31.sh

echo "========================================="
echo "PRE-EXPERIMENT CHECK (v3.1)"
echo "========================================="
echo ""

PASS=0
FAIL=0

# 1. Check utils.py
echo "1. Checking utils.py..."
if grep -q "def load_scalers" code/utils.py && \
   grep -q "def apply_scalers_to_data" code/utils.py; then
    echo "   ✓ PASS"
    PASS=$((PASS+1))
else
    echo "   ❌ FAIL - Missing scaler functions"
    FAIL=$((FAIL+1))
fi

# 2. Check train.py
echo "2. Checking train.py normalization..."
if grep -A 2 "load_npz_dataset" code/train.py | grep -q "normalize=False"; then
    echo "   ✓ PASS"
    PASS=$((PASS+1))
else
    echo "   ❌ FAIL - Still uses normalize=True"
    FAIL=$((FAIL+1))
fi

# 3. Check evaluate.py
echo "3. Checking evaluate.py scaler loading..."
if grep -q "load_scalers" code/evaluate.py; then
    echo "   ✓ PASS"
    PASS=$((PASS+1))
else
    echo "   ❌ FAIL - Doesn't load scalers"
    FAIL=$((FAIL+1))
fi

# 4. Check disk space
echo "4. Checking disk space..."
SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ $(echo "$SPACE > 50" | bc -l) -eq 1 ]; then
    echo "   ✓ PASS - ${SPACE}GB available"
    PASS=$((PASS+1))
else
    echo "   ⚠️  WARNING - Only ${SPACE}GB available"
    FAIL=$((FAIL+1))
fi

# 5. Check GPU
echo "5. Checking GPU..."
if nvidia-smi &>/dev/null || rocm-smi &>/dev/null; then
    echo "   ✓ PASS - GPU detected"
    PASS=$((PASS+1))
else
    echo "   ⚠️  WARNING - No GPU detected"
    # Don't fail, CPU training is possible
    PASS=$((PASS+1))
fi

echo ""
echo "========================================="
echo "RESULT: $PASS passed, $FAIL failed"
echo "========================================="

if [ $FAIL -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - Ready to train!"
    exit 0
else
    echo "❌ SOME CHECKS FAILED - Fix issues before training!"
    exit 1
fi
```

**Usage:**
```bash
chmod +x pre_experiment_check_v31.sh
./pre_experiment_check_v31.sh || exit 1  # Exit if checks fail
python train.py ...  # Proceed with training
```

---

## 💡 v3.1 Pro Tips

### 1. Always Verify Scalers

```bash
# After ANY training, immediately check:
LATEST=$(ls -td results/*_v31_*/ | head -1)
[ -f "$LATEST/scaler_standard.pkl" ] && \
[ -f "$LATEST/scaler_minmax.pkl" ] && \
echo "✓ Valid training" || echo "❌ INVALID - redo training"
```

### 2. Use Experiment Log File

```bash
# Create a log of all experiments
touch experiment_log_v31.txt

# After each experiment:
echo "$(date): baseline_v31 - Test Acc: 73.5% - Valid" >> experiment_log_v31.txt
```

### 3. Quick Results Comparison

```bash
# One-liner to see all v3.1 accuracies
for d in results/*_v31_*/summary.json; do
    echo "$(dirname $d | xargs basename): $(python -c \"import json; print(f\\\"{json.load(open('$d'))['final_test_acc']*100:.2f}%\\\")\")"
done
```

---

**Remember: v3.1 commands only! Never mix v3.0 and v3.1 results!** 🚀