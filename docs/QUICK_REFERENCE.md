# Quick Reference Guide - v3.0

**Purpose**: Command cheatsheet for all experiments  
**Usage**: Copy-paste commands for running experiments  
**Last Updated**: October 2025

---

## 🆕 v3.0 Features

### Auto-Detection
No need to manually specify model paths:
```bash
# Old way (still works):
python evaluate.py --model results/baseline_*/best_model.pt \
    --data data/raw/baseline_1M.npz --output_dir results/baseline_eval

# New way (easier):
python evaluate.py --experiment_name baseline \
    --data data/raw/baseline_1M.npz
```

### Timestamped Results
Every training run creates a unique directory:
```
results/baseline_20251027_143022/  # First run
results/baseline_20251027_150532/  # Second run (different seed or params)
```

---

## 🚀 All Experiments - Command List

### E1: Baseline (1M events)

```bash
# Data generation (2-3 hours on 24 cores)
cd code
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.20 \
    --mag_error_std 0.10 \
    --seed 42 \
    --num_workers 24

# Training (6-8 hours on 4 GPUs)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --experiment_name baseline

# Evaluation (auto-detects latest model)
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection

# Real-time benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

### E2: Dense Cadence (5% missing)

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/cadence_05.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.05 \
    --seed 42

# Training
python train.py \
    --data ../data/raw/cadence_05.npz \
    --experiment_name cadence_05

# Evaluation
python evaluate.py \
    --experiment_name cadence_05 \
    --data ../data/raw/cadence_05.npz \
    --early_detection
```

---

### E3: Sparse Cadence (30% missing)

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/cadence_30.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.30 \
    --seed 42

# Training
python train.py \
    --data ../data/raw/cadence_30.npz \
    --experiment_name cadence_30

# Evaluation
python evaluate.py \
    --experiment_name cadence_30 \
    --data ../data/raw/cadence_30.npz \
    --early_detection
```

---

### E4: Very Sparse Cadence (40% missing)

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/cadence_40.npz \
    --binary_params baseline \
    --cadence_mask_prob 0.40 \
    --seed 42

# Training
python train.py \
    --data ../data/raw/cadence_40.npz \
    --experiment_name cadence_40

# Evaluation
python evaluate.py \
    --experiment_name cadence_40 \
    --data ../data/raw/cadence_40.npz \
    --early_detection
```

---

### E5: Low Photometric Error (0.05 mag)

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/error_05.npz \
    --binary_params baseline \
    --mag_error_std 0.05 \
    --seed 42

# Training
python train.py \
    --data ../data/raw/error_05.npz \
    --experiment_name error_05

# Evaluation
python evaluate.py \
    --experiment_name error_05 \
    --data ../data/raw/error_05.npz \
    --early_detection
```

---

### E6: High Photometric Error (0.20 mag)

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/error_20.npz \
    --binary_params baseline \
    --mag_error_std 0.20 \
    --seed 42

# Training
python train.py \
    --data ../data/raw/error_20.npz \
    --experiment_name error_20

# Evaluation
python evaluate.py \
    --experiment_name error_20 \
    --data ../data/raw/error_20.npz \
    --early_detection
```

---

### E7: Distinct Binary Topology

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/distinct.npz \
    --binary_params distinct \
    --seed 42

# Training
python train.py \
    --data ../data/raw/distinct.npz \
    --experiment_name distinct

# Evaluation
python evaluate.py \
    --experiment_name distinct \
    --data ../data/raw/distinct.npz \
    --early_detection
```

---

### E8: Planetary Systems

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/planetary.npz \
    --binary_params planetary \
    --seed 42

# Training
python train.py \
    --data ../data/raw/planetary.npz \
    --experiment_name planetary

# Evaluation
python evaluate.py \
    --experiment_name planetary \
    --data ../data/raw/planetary.npz \
    --early_detection
```

---

### E9: Stellar Binaries

```bash
# Data generation
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/stellar.npz \
    --binary_params stellar \
    --seed 42

# Training
python train.py \
    --data ../data/raw/stellar.npz \
    --experiment_name stellar

# Evaluation
python evaluate.py \
    --experiment_name stellar \
    --data ../data/raw/stellar.npz \
    --early_detection
```

---

## 🔄 Batch Processing

### Run All Data Generation

```bash
cd code

# Baseline
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz --binary_params baseline --seed 42

# Cadence experiments
for cadence in 05 30 40; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${cadence}.npz \
        --cadence_mask_prob 0.${cadence} --seed 42
done

# Error experiments
for error in 05 20; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${error}.npz \
        --mag_error_std 0.${error} --seed 42
done

# Topology experiments
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} --seed 42
done
```

### Run All Training (Sequential)

```bash
for exp in baseline cadence_05 cadence_30 cadence_40 \
           error_05 error_20 distinct planetary stellar; do
    echo "Training $exp..."
    python train.py \
        --data ../data/raw/${exp}*.npz \
        --experiment_name ${exp}
done
```

### Run All Evaluation (Sequential)

```bash
for exp in baseline cadence_05 cadence_30 cadence_40 \
           error_05 error_20 distinct planetary stellar; do
    echo "Evaluating $exp..."
    python evaluate.py \
        --experiment_name ${exp} \
        --data ../data/raw/${exp}*.npz \
        --early_detection
done
```

---

## 🐛 Debugging Commands

### Check Data

```bash
# Inspect NPZ file
python -c "
import numpy as np
d = np.load('data/raw/baseline_1M.npz', allow_pickle=False)
print('Keys:', d.files)
print('X shape:', d['X'].shape)
print('y shape:', d['y'].shape)
print('Unique labels:', np.unique(d['y'], return_counts=True))
"

# Check for NaN values
python -c "
import numpy as np
d = np.load('data/raw/baseline_1M.npz')
X = d['X']
print('NaN count:', np.isnan(X).sum())
print('Inf count:', np.isinf(X).sum())
print('PAD_VALUE count:', (X == -1).sum())
"
```

### Find Latest Results

```bash
# List all runs for an experiment
ls -ltr results/baseline_*/

# Get most recent
ls -td results/baseline_*/ | head -1

# Check contents
ls -lh $(ls -td results/baseline_*/ | head -1)
```

### Monitor Training

```bash
# Watch training progress
watch -n 1 "tail -20 $(ls -td results/baseline_*/ | head -1)/training.log"

# Check GPU usage
watch -n 1 nvidia-smi  # or rocm-smi for AMD

# Monitor disk space
watch -n 60 df -h
```

### Test Model

```bash
# Quick inference test
python -c "
import torch
import numpy as np
from model import TimeDistributedCNN

model = TimeDistributedCNN(sequence_length=1500, num_channels=1, num_classes=2)
x = torch.randn(4, 1, 1500)  # Batch of 4
output = model(x)
print('Output shape:', output.shape)
print('Expected: [4, 1500, 2]')
"
```

---

## 📊 Analysis Commands

### Compare Multiple Runs

```bash
# Compare accuracy across runs of same experiment
cd code
python -c "
import json
from pathlib import Path

exp = 'baseline'
runs = sorted(Path('../results').glob(f'{exp}_*'))

for run in runs:
    summary_file = run / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
        print(f'{run.name}: Accuracy = {data[\"final_test_acc\"]:.4f}')
"
```

### Extract Results Table

```bash
# Generate results table for all experiments
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_05', 'cadence_30', 'cadence_40',
               'error_05', 'error_20', 'distinct', 'planetary', 'stellar']

print(f'{'Experiment':<15} {'Accuracy':>10} {'Val Acc':>10} {'Epoch':>8}')
print('-' * 50)

for exp in experiments:
    runs = sorted(Path('results').glob(f'{exp}_*'))
    if runs:
        latest = runs[-1]
        summary_file = latest / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            acc = data['final_test_acc'] * 100
            val_acc = data['best_val_acc'] * 100
            epoch = data['best_epoch']
            print(f'{exp:<15} {acc:>9.2f}% {val_acc:>9.2f}% {epoch:>8}')
"
```

---

## 🔧 Maintenance Commands

### Clean Up Old Results

```bash
# Keep only the 3 most recent runs for each experiment
for exp in baseline cadence_05 cadence_30; do
    ls -td results/${exp}_*/ | tail -n +4 | xargs -r rm -rf
done
```

### Archive Experiment

```bash
# Archive completed experiment (latest run)
EXP=baseline
LATEST=$(ls -td results/${EXP}_*/ | head -1)
tar -czf ${EXP}_archive_$(date +%Y%m%d).tar.gz \
    data/raw/${EXP}*.npz \
    ${LATEST}
```

### Backup to Remote

```bash
# Backup to cluster/server
rsync -avz --progress \
    data/ results/ \
    username@cluster:/path/to/backup/
```

---

## ⚡ Performance Tips

### Speed Up Data Generation
```bash
# Use maximum CPU cores
python simulate.py --num_workers $(nproc) ...

# Use tmpfs for intermediate files
mkdir /tmp/microlens_tmp
python simulate.py --output /tmp/microlens_tmp/data.npz ...
```

### Speed Up Training
```bash
# Increase batch size (if memory allows)
python train.py --batch_size 256 ...

# Use mixed precision (automatically enabled in v3.0)

# Pin memory for faster data loading (automatically enabled in v3.0)
```

### Reduce Disk Usage
```bash
# Compress old datasets
gzip data/raw/*.npz

# Remove old checkpoints (keep best_model.pt only)
find results/ -name "checkpoint_*.pt" -delete
```

---

## 📋 Experiment Checklist

### Before Starting Experiment
- [ ] Check disk space (50+ GB free)
- [ ] Check GPU availability (`nvidia-smi` or `rocm-smi`)
- [ ] Activate conda environment
- [ ] Review config.py settings
- [ ] Generate data if not already present

### After Experiment Completes
- [ ] Check results directory created
- [ ] Review training.log for issues
- [ ] Run evaluation
- [ ] Run benchmark
- [ ] Document observations
- [ ] Archive or clean up old runs

---

## 💡 Pro Tips

### Using Auto-Detection

```bash
# Train multiple runs for statistical comparison
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 42
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 123
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 456

# Evaluate all runs
for run in results/baseline_*/; do
    echo "Evaluating $run"
    python evaluate.py --model ${run}best_model.pt \
        --data data/raw/baseline_1M.npz \
        --output_dir ${run}evaluation
done
```

### Comparing Specific Runs

```bash
# If you want to evaluate a specific run (not the latest)
# Just specify the full model path:
python evaluate.py \
    --model results/baseline_20251027_143022/best_model.pt \
    --data data/raw/baseline_1M.npz \
    --output_dir results/baseline_20251027_143022/evaluation_v2
```

---

**v3.0 makes everything easier with auto-detection!** 🚀