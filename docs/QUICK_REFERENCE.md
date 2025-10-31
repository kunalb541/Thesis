# Quick Reference - Command Cheatsheet (v4.0)

Fast command reference for all experiments with **DDP support**.

---

## Essential Commands

### Setup Verification

```bash
# Check GPU
python code/utils.py

# Validate VBMicrolensing
python code/test_vbm.py

# Quick test (2K events, 5 min)
cd code
python simulate.py --n_pspl 1000 --n_binary 1000 \
    --output ../data/raw/test.npz --num_workers 8
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/test.npz \
    --experiment_name test --epochs 5 --batch_size 128 --lr 1e-4
```

---

## Baseline Experiment (1M events)

```bash
cd code

# Generate data (~10-15 min with 200 workers)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --seed 42 \
    --num_workers 200

# Train with DDP (~30-45 min on 4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4

# Evaluate
python evaluate.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz

# Benchmark
python benchmark_realtime.py \
    --experiment_name baseline \
    --data ../data/raw/baseline_1M.npz
```

---

## Distributed Training (DDP)

### Single Node (4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4
```

### Multi-Node (SLURM)

```bash
# Set environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Train with srun + torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4
```

### DDP Debugging

```bash
# Enable NCCL debug
export NCCL_DEBUG=INFO

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES

# Test with single GPU first
torchrun --nproc_per_node=1 train.py [args]

# Then scale up
torchrun --nproc_per_node=4 train.py [args]
```

---

## Cadence Experiments

```bash
# Simulate all cadence datasets (~5-8 min total)
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence --num_workers 200
done

# Train all with DDP (~30-40 min total)
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Evaluate all
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python evaluate.py --experiment_name cadence_${name} \
        --data ../data/raw/cadence_${name}.npz
done
```

---

## Photometric Error Experiments

```bash
# Simulate all error datasets (~5-8 min total)
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${name}.npz \
        --mag_error_std $error --num_workers 200
done

# Train all with DDP (~30-40 min total)
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/error_${name}.npz \
        --experiment_name error_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Evaluate all
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    python evaluate.py --experiment_name error_${name} \
        --data ../data/raw/error_${name}.npz
done
```

---

## Binary Topology Experiments

```bash
# Simulate all topology datasets (~5-8 min total)
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} --num_workers 200
done

# Train all with DDP (~30-40 min total)
for topo in distinct planetary stellar; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/${topo}.npz \
        --experiment_name ${topo} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Evaluate all
for topo in distinct planetary stellar; do
    python evaluate.py --experiment_name ${topo} \
        --data ../data/raw/${topo}.npz
done
```

---

## Complete Pipeline (Interactive)

### All Experiments in ~1 Hour

```bash
cd code

# ==========================================
# 1️⃣ SIMULATION (~15-20 min total)
# ==========================================

# Baseline (1M)
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline --num_workers 200

# Cadence
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence --num_workers 200
done

# Error
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${name}.npz \
        --mag_error_std $error --num_workers 200
done

# Topology
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} --num_workers 200
done

# ==========================================
# 2️⃣ TRAINING with DDP (~40-50 min total)
# ==========================================

# Baseline
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 --batch_size 128 --lr 1e-4

# Cadence
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Error
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/error_${name}.npz \
        --experiment_name error_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Topology
for topo in distinct planetary stellar; do
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/${topo}.npz \
        --experiment_name ${topo} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# ==========================================
# 3️⃣ EVALUATION (~5-10 min total)
# ==========================================

for exp in baseline cadence_* error_* distinct planetary stellar; do
    python evaluate.py --experiment_name $exp \
        --data ../data/raw/${exp}.npz 2>/dev/null || \
    python evaluate.py --experiment_name $exp \
        --data ../data/raw/baseline_1M.npz 2>/dev/null
done
```

---

## SLURM Multi-Node Training

```bash
# Set master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Baseline
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 --batch_size 128 --lr 1e-4

# Cadence
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    srun torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Error
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    srun torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        --data ../data/raw/error_${name}.npz \
        --experiment_name error_${name} \
        --epochs 50 --batch_size 128 --lr 1e-4
done

# Topology
for topo in distinct planetary stellar; do
    srun torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        --data ../data/raw/${topo}.npz \
        --experiment_name ${topo} \
        --epochs 50 --batch_size 128 --lr 1e-4
done
```

---

## Monitoring & Analysis

### Watch Training

```bash
# Monitor latest experiment
tail -f $(ls -td ../results/baseline_*/ | head -1)/training.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Quick Results Check

```bash
# Get test accuracy for one experiment
python -c "
import json
from pathlib import Path
exp = 'baseline'
runs = sorted(Path('../results').glob(f'{exp}_*'))
if runs:
    data = json.load(open(runs[-1] / 'summary.json'))
    print(f\"{exp}: {data['final_test_acc']*100:.2f}%\")
"

# Generate comparison table
python -c "
import json
from pathlib import Path

experiments = ['baseline', 'cadence_05', 'cadence_20', 'cadence_30', 'cadence_40',
               'error_05', 'error_10', 'error_20', 'distinct', 'planetary', 'stellar']

print(f'{'Experiment':<20} {'Test Acc':<12}')
print('-' * 35)

for exp in experiments:
    runs = sorted(Path('../results').glob(f'{exp}_*'))
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

# Results saved to: ../results/baseline_*/sample_plots/
```

---

## Troubleshooting

### Low Accuracy

```bash
# Check data normalization
python -c "
import numpy as np
data = np.load('../data/raw/baseline_1M.npz')
X = data['X']
print(f'Data range: [{X.min():.3f}, {X.max():.3f}]')
print(f'Data mean: {X.mean():.3f}')
"

# Validate VBMicrolensing
python test_vbm.py
```

### DDP Issues

```bash
# Enable debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Test with single GPU
torchrun --nproc_per_node=1 train.py [args]

# Check network interface (SLURM)
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
```

### GPU Memory Issues

```bash
# Reduce batch size
torchrun --nproc_per_node=4 train.py --batch_size 64

# Use gradient accumulation
torchrun --nproc_per_node=4 train.py --batch_size 32
```

### Slow Training

```bash
# Check if using all GPUs
nvidia-smi -l 1

# Ensure DDP is active
grep "World size" ../results/*/training.log

# Increase data workers
torchrun --nproc_per_node=4 train.py --num_workers 16
```

---

## Expected Performance (v4.0)

### Timing (4 GPUs)

| Task | Events | Time | Notes |
|------|--------|------|-------|
| Simulate | 1M | 10-15 min | 200 workers |
| Simulate | 200K | 2-3 min | Per experiment |
| Train DDP | 1M | 30-45 min | 4 GPUs |
| Train DDP | 200K | 8-12 min | Per experiment |
| Evaluate | Any | 1-2 min | Per experiment |
| **Complete** | **All** | **~1 hour** | Full pipeline |

### Accuracy

| Experiment | Missing/Error | Expected Acc |
|------------|---------------|--------------|
| Baseline | 20% missing | 70-75% |
| Dense | 5% missing | 75-80% |
| Sparse | 30% missing | 65-70% |
| Very Sparse | 40% missing | 60-65% |
| Low Error | 0.05 mag | 75-80% |
| High Error | 0.20 mag | 65-70% |
| Distinct | Clear caustics | 80-90% |
| Planetary | q ~ 0.001 | 70-80% |
| Stellar | q ~ 1.0 | 60-75% |

---

## Common Options

### simulate.py
- `--n_pspl`: PSPL events
- `--n_binary`: Binary events
- `--n_points`: Time series length (default: 1500)
- `--cadence_mask_prob`: Missing observations fraction
- `--mag_error_std`: Photometric error (mag)
- `--binary_params`: Parameter set (baseline/distinct/planetary/stellar)
- `--num_workers`: Parallel processes (recommend 200)

### train.py (with torchrun)
- `--data`: Path to .npz dataset
- `--experiment_name`: Experiment identifier
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-4 for DDP)
- `--num_workers`: Data loading workers per GPU (default: 4)

### evaluate.py
- `--experiment_name`: Auto-find latest run
- `--data`: Path to test data
- `--batch_size`: Inference batch size (default: 128)

---

## File Locations

```
../results/experiment_TIMESTAMP/
├── best_model.pt           # Trained model
├── config.json             # Experiment config
├── training.log            # Training logs
├── summary.json            # Final metrics
├── scaler_standard.pkl     # Normalization
├── scaler_minmax.pkl       # Normalization
└── evaluation/             # Evaluation plots
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── evaluation_summary.json
```

---

You're now set for ultra-fast experimentation! ⚡🚀