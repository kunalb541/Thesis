# Quick Reference Guide

**Purpose**: Command cheatsheet for all experiments  
**Usage**: Copy-paste commands for running experiments  
**Last Updated**: October 2025

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
    --output ../models/baseline.pt \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --experiment_name baseline

# Evaluation
RESULTS_DIR=$(ls -td ../results/baseline_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/baseline_1M.npz \
    --output_dir $RESULTS_DIR/evaluation \
    --early_detection

# Real-time benchmark
python benchmark_realtime.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/baseline_1M.npz \
    --output_dir $RESULTS_DIR/benchmark
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
RESULTS_DIR=$(ls -td ../results/cadence_05_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/cadence_05.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/cadence_30_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/cadence_30.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/cadence_40_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/cadence_40.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/error_05_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/error_05.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/error_20_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/error_20.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/distinct_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/distinct.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/planetary_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/planetary.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
RESULTS_DIR=$(ls -td ../results/stellar_* | head -1)
python evaluate.py \
    --model $RESULTS_DIR/best_model.pt \
    --data ../data/raw/stellar.npz \
    --output_dir $RESULTS_DIR/evaluation \
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
for exp in baseline_1M cadence_05 cadence_30 cadence_40 \
           error_05 error_20 distinct planetary stellar; do
    echo "Training $exp..."
    python train.py \
        --data ../data/raw/${exp}.npz \
        --experiment_name ${exp}
done
```

### Run All Evaluation (Sequential)

```bash
for exp in baseline cadence_05 cadence_30 cadence_40 \
           error_05 error_20 distinct planetary stellar; do
    echo "Evaluating $exp..."
    RESULTS_DIR=$(ls -td ../results/${exp}_* | head -1)
    python evaluate.py \
        --model $RESULTS_DIR/best_model.pt \
        --data ../data/raw/${exp}.npz \
        --output_dir $RESULTS_DIR/evaluation \
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

### Monitor Training

```bash
# Watch training progress
watch -n 1 "tail -20 results/baseline_*/training.log"

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

### Generate Comparison Plots

```bash
# After all experiments complete
cd code

# Create comparison plots script
cat > compare_experiments.py << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all results
experiments = ['baseline', 'cadence_05', 'cadence_30', 'cadence_40',
               'error_05', 'error_20', 'distinct', 'planetary', 'stellar']

results = {}
for exp in experiments:
    result_dirs = sorted(Path('../results').glob(f'{exp}_*'))
    if result_dirs:
        eval_path = result_dirs[-1] / 'evaluation' / 'evaluation_summary.json'
        if eval_path.exists():
            with open(eval_path) as f:
                results[exp] = json.load(f)

# Cadence comparison
cadence_exps = ['cadence_05', 'baseline', 'cadence_30', 'cadence_40']
cadences = [5, 20, 30, 40]
accs = [results[e]['metrics']['accuracy']*100 for e in cadence_exps if e in results]

plt.figure(figsize=(10, 6))
plt.plot(cadences, accs, 'o-', linewidth=2, markersize=10)
plt.xlabel('Missing Observations (%)', fontsize=14)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.title('Classification Performance vs Observing Cadence', fontsize=16)
plt.grid(alpha=0.3)
plt.savefig('../figures/cadence_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: figures/cadence_comparison.png')

# Error comparison
error_exps = ['error_05', 'baseline', 'error_20']
errors = [0.05, 0.10, 0.20]
accs = [results[e]['metrics']['accuracy']*100 for e in error_exps if e in results]

plt.figure(figsize=(10, 6))
plt.plot(errors, accs, 's-', linewidth=2, markersize=10, color='red')
plt.xlabel('Photometric Error (mag)', fontsize=14)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.title('Classification Performance vs Photometric Error', fontsize=16)
plt.grid(alpha=0.3)
plt.savefig('../figures/error_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: figures/error_comparison.png')

# Topology comparison
topo_exps = ['distinct', 'baseline', 'planetary', 'stellar']
labels = ['Distinct\n(u₀<0.15)', 'Baseline\n(mixed)', 'Planetary\n(q<<1)', 'Stellar\n(q~1)']
accs = [results[e]['metrics']['accuracy']*100 for e in topo_exps if e in results]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accs, alpha=0.7, color=['green', 'blue', 'orange', 'red'])
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.title('Classification Performance vs Binary Topology', fontsize=16)
plt.ylim([0, 100])
plt.grid(alpha=0.3, axis='y')
for i, (bar, acc) in enumerate(zip(bars, accs)):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 2, 
             f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
plt.savefig('../figures/topology_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: figures/topology_comparison.png')
EOF

python compare_experiments.py
```

### Extract Results Table

```bash
# Generate LaTeX table
python -c "
import json
from pathlib import Path

experiments = {
    'Baseline': 'baseline',
    'Dense (5%)': 'cadence_05',
    'Sparse (30%)': 'cadence_30',
    'V.Sparse (40%)': 'cadence_40',
    'Low Error': 'error_05',
    'High Error': 'error_20',
    'Distinct': 'distinct',
    'Planetary': 'planetary',
    'Stellar': 'stellar',
}

print('\\begin{table}[h]')
print('\\centering')
print('\\begin{tabular}{lcccc}')
print('\\hline')
print('Experiment & Accuracy & Precision & Recall & F1 \\\\')
print('\\hline')

for name, exp in experiments.items():
    result_dirs = sorted(Path('results').glob(f'{exp}_*'))
    if result_dirs:
        eval_path = result_dirs[-1] / 'evaluation' / 'evaluation_summary.json'
        if eval_path.exists():
            with open(eval_path) as f:
                r = json.load(f)['metrics']
            acc = r['accuracy'] * 100
            tp, fp, fn = r['tp'], r['fp'], r['fn']
            prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f'{name} & {acc:.1f} & {prec:.1f} & {rec:.1f} & {f1:.2f} \\\\\\\\')

print('\\hline')
print('\\end{tabular}')
print('\\caption{Classification performance across systematic experiments.}')
print('\\label{tab:results}')
print('\\end{table}')
" > results_table.tex

cat results_table.tex
```

---

## 🔧 Maintenance Commands

### Clean Up

```bash
# Remove old results (keep only best)
find results/ -name "*.pt" -not -name "best_model.pt" -delete

# Clean old data files
rm data/raw/test_*.npz

# Remove temporary files
rm -rf logs/*.tmp
rm -rf __pycache__/
```

### Archive Experiment

```bash
# Archive completed experiment
EXP=baseline
tar -czf ${EXP}_archive.tar.gz \
    data/raw/${EXP}*.npz \
    results/${EXP}_*/ \
    models/${EXP}.pt
```

### Backup to Remote

```bash
# Backup to cluster/server
rsync -avz --progress \
    data/ results/ models/ \
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

# Use mixed precision (if supported)
# (automatically enabled in config.py)

# Pin memory for faster data loading
# (automatically enabled in train.py)
```

### Reduce Disk Usage
```bash
# Don't save intermediate checkpoints
# (set SAVE_BEST_ONLY=True in config.py)

# Compress old datasets
gzip data/raw/*.npz

# Use compressed saves (already default)
```

---

## 📋 Experiment Checklist

### Before Starting Experiment
- [ ] Check disk space (50+ GB free)
- [ ] Check GPU availability
- [ ] Activate conda environment
- [ ] Review config.py settings
- [ ] Note start time in EXPERIMENTS_LOG.md

### After Experiment Completes
- [ ] Copy results to EXPERIMENTS_LOG.md
- [ ] Generate evaluation plots
- [ ] Check for anomalies
- [ ] Document observations in NOTES.md
- [ ] Archive data and results
- [ ] Update status in README.md

---

**Bookmark this file - you'll use it constantly!**