# Quick Reference - Command Cheatsheet

Fast command reference for all experiments.

---

## Essential Commands

### Setup Verification

```bash
# Check GPU
python code/utils.py

# Quick test (2K events, 5 min)
cd code
python simulate.py --n_pspl 1000 --n_binary 1000 --output ../data/raw/test.npz
python train.py --data ../data/raw/test.npz --experiment_name test --epochs 5
```

---

## Baseline Experiment (1M events)

```bash
cd code

# Generate data (~2 hours on 24 cores)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --seed 42 \
    --num_workers 24

# Train (~6-8 hours on 4 GPUs)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128

# Evaluate (auto-finds latest run)
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --early_detection

# Benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

## Cadence Experiments

```bash
# Dense (5% missing)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/cadence_dense.npz \
    --cadence_mask_prob 0.05
python train.py --data ../data/raw/cadence_dense.npz \
    --experiment_name cadence_dense --epochs 50

# Sparse (30% missing)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/cadence_sparse.npz \
    --cadence_mask_prob 0.30
python train.py --data ../data/raw/cadence_sparse.npz \
    --experiment_name cadence_sparse --epochs 50

# Very Sparse (40% missing)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/cadence_vsparse.npz \
    --cadence_mask_prob 0.40
python train.py --data ../data/raw/cadence_vsparse.npz \
    --experiment_name cadence_vsparse --epochs 50
```

---

## Photometric Error Experiments

```bash
# Low error (0.05 mag - space-based)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/error_low.npz \
    --mag_error_std 0.05
python train.py --data ../data/raw/error_low.npz \
    --experiment_name error_low --epochs 50

# High error (0.20 mag - poor conditions)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/error_high.npz \
    --mag_error_std 0.20
python train.py --data ../data/raw/error_high.npz \
    --experiment_name error_high --epochs 50
```

---

## Binary Topology Experiments

```bash
# Distinct caustic-crossing (easiest)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/distinct.npz \
    --binary_params distinct
python train.py --data ../data/raw/distinct.npz \
    --experiment_name distinct --epochs 50

# Planetary systems
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/planetary.npz \
    --binary_params planetary
python train.py --data ../data/raw/planetary.npz \
    --experiment_name planetary --epochs 50

# Stellar binaries (hardest)
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/stellar.npz \
    --binary_params stellar
python train.py --data ../data/raw/stellar.npz \
    --experiment_name stellar --epochs 50
```

---

## Batch Processing

### All Cadence Experiments

```bash
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence
    python train.py --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} --epochs 50
    python evaluate.py --experiment_name cadence_${name} \
        --data ../data/raw/cadence_${name}.npz --early_detection
done
```

### All Topology Experiments

```bash
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo}
    python train.py --data ../data/raw/${topo}.npz \
        --experiment_name ${topo} --epochs 50
    python evaluate.py --experiment_name ${topo} \
        --data ../data/raw/${topo}.npz --early_detection
done
```

---

## Monitoring & Analysis

### Watch Training

```bash
# Follow training logs
tail -f $(ls -td results/baseline_*/ | head -1)/training.log

# Monitor GPU
watch -n 1 nvidia-smi  # or rocm-smi for AMD
```

### Quick Results Check

```bash
# Get test accuracy
EXP=baseline
LATEST=$(ls -td results/${EXP}_*/ | head -1)
python -c "import json; print(f\"{json.load(open('$LATEST/summary.json'))['final_test_acc']*100:.2f}%\")"

# List all experiments and accuracies
for dir in results/*/summary.json; do
    exp=$(dirname $dir | xargs basename)
    acc=$(python -c "import json; print(f\"{json.load(open('$dir'))['final_test_acc']*100:.2f}%\")")
    echo "$exp: $acc"
done
```

### Generate Comparison Table

```bash
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_dense', 'cadence_sparse', 
               'error_low', 'error_high', 'distinct', 'planetary', 'stellar']

print(f'{'Experiment':<20} {'Test Acc':<12} {'ROC AUC':<10}')
print('-' * 45)

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

## Visualization

```bash
# Plot sample predictions
python plot_samples.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz \
    --n_samples 12

# Results in: results/baseline_TIMESTAMP/sample_plots/
```

---

## Troubleshooting

### Low Accuracy

```bash
# Check data range (should be ~[0, 1])
grep "Train data range" results/*/training.log

# Check training convergence
grep "Epoch" results/*/training.log | tail -20
```

### GPU Memory Issues

```bash
# Reduce batch size
python train.py --data data.npz --batch_size 64  # instead of 128

# Use gradient accumulation
python train.py --data data.npz --accumulate_steps 2
```

### Slow Training

```bash
# Check if using GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use more workers
python train.py --data data.npz --num_workers 16
```

---

## File Locations

```
results/experiment_TIMESTAMP/
├── best_model.pt           # Trained model
├── config.json             # Parameters
├── training.log            # Training history
├── summary.json            # Final metrics
├── scaler_standard.pkl     # Normalization params
├── scaler_minmax.pkl       # Normalization params
└── evaluation/             # Evaluation results
```

---

## Common Options

### simulate.py
- `--n_pspl`: Number of PSPL events
- `--n_binary`: Number of binary events
- `--n_points`: Time series length (default: 1500)
- `--cadence_mask_prob`: Fraction of missing observations
- `--mag_error_std`: Photometric error (mag)
- `--binary_params`: Parameter set (baseline/distinct/planetary/stellar)
- `--num_workers`: Parallel processes

### train.py
- `--data`: Path to .npz dataset
- `--experiment_name`: Experiment identifier
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)

### evaluate.py
- `--experiment_name`: Auto-find latest run
- `--data`: Path to test data
- `--early_detection`: Run partial observation analysis
- `--batch_size`: Inference batch size

---

## Expected Performance

| Experiment       | Test Accuracy | Training Time |
|------------------|---------------|---------------|
| Baseline (1M)    | 70-75%        | 6-8 hours     |
| Dense (5%)       | 75-80%        | 2-3 hours     |
| Sparse (30%)     | 65-70%        | 2-3 hours     |
| Very Sparse (40%)| 60-65%        | 2-3 hours     |
| Low Error (0.05) | 75-80%        | 2-3 hours     |
| High Error (0.20)| 65-70%        | 2-3 hours     |
| Distinct         | 80-90%        | 2-3 hours     |
| Planetary        | 70-80%        | 2-3 hours     |
| Stellar          | 60-75%        | 2-3 hours     |

*Times on 4× NVIDIA A100 GPUs*