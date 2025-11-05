# Quick Reference - Command Cheatsheet (v5.3)

Fast command reference for all experiments with **DDP support**.

---

## Model Architecture Summary

**Transformer-based classifier**:
- Conv1D downsampling: 1500 → 500 timesteps (preprocessing only)
- Transformer encoder: 2 layers, 4 attention heads (the actual model)
- Output: PSPL vs Binary classification

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
    --experiment_name test --epochs 5
```

---

## Baseline Experiment (1M events)

```bash
cd code

# Generate data (~10-15 min, 200 workers)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --seed 42 \
    --num_workers 200

# Train with DDP (~30-45 min, 4 GPUs)
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
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

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
# Enable debug
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check GPUs
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Test single GPU
torchrun --nproc_per_node=1 train.py [args]

# Network interface
export NCCL_SOCKET_IFNAME=eth0
```

---

## Cadence Experiments

```bash
# Simulate all cadence datasets
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence --num_workers 200
done

# Train all with DDP
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
# Simulate all error datasets
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/error_${name}.npz \
        --mag_error_std $error --num_workers 200
done

# Train all with DDP
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
# Simulate all topology datasets
for topo in distinct planetary stellar; do
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} --num_workers 200
done

# Train all with DDP
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

## Complete Pipeline (~1 Hour)

```bash
cd code

# 1. Simulation (~15-20 min)
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/baseline_1M.npz --num_workers 200

for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    python simulate.py --n_pspl 100000 --n_binary 100000 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence --num_workers 200
done

# 2. Training with DDP (~40-50 min)
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline --epochs 50

for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    torchrun --nproc_per_node=4 train.py \
        --data ../data/raw/cadence_${name}.npz \
        --experiment_name cadence_${name} --epochs 50
done

# 3. Evaluation (~5-10 min)
for exp in baseline cadence_*; do
    python evaluate.py --experiment_name $exp \
        --data ../data/raw/${exp}.npz 2>/dev/null
done
```

---

## Monitoring

```bash
# Watch training
tail -f $(ls -td ../results/baseline_*/ | head -1)/training.log

# GPU usage
watch -n 1 nvidia-smi

# Quick results
python -c "
import json
from pathlib import Path
exp = 'baseline'
runs = sorted(Path('../results').glob(f'{exp}_*'))
if runs:
    with open(runs[-1] / 'results.json') as f:
        data = json.load(f)
    print(f\"{exp}: {data['test_acc']*100:.2f}%\")
"
```

---

## Expected Performance (v5.3)

### Timing (4 GPUs)

| Task | Time |
|------|------|
| Simulate 1M | 10-15 min |
| Simulate 200K | 2-3 min |
| Train DDP 1M | 30-45 min |
| Train DDP 200K | 8-12 min |
| Evaluate | 2-5 min |

### Accuracy

| Experiment | Expected Acc |
|------------|--------------|
| Baseline | 70-75% |
| Dense (5%) | 75-80% |
| Sparse (30%) | 65-70% |
| Low Error (0.05) | 75-80% |
| Distinct | 80-90% |

---

## Troubleshooting

### Low Accuracy

```bash
# Check data
python -c "
import numpy as np
data = np.load('../data/raw/baseline_1M.npz')
X = data['X']
print(f'Range: [{X.min():.3f}, {X.max():.3f}]')
"

# Validate VBMicrolensing
python test_vbm.py
```

### DDP Issues

```bash
# Enable debug
export NCCL_DEBUG=INFO

# Test single GPU
torchrun --nproc_per_node=1 train.py [args]

# Network interface
export NCCL_SOCKET_IFNAME=eth0
```

### GPU Memory

```bash
# Reduce batch size
torchrun --nproc_per_node=4 train.py --batch_size 64
```

---

## Common Options

### simulate.py
- `--n_pspl`: PSPL events
- `--n_binary`: Binary events
- `--cadence_mask_prob`: Missing observations
- `--mag_error_std`: Photometric error
- `--binary_params`: baseline/distinct/planetary/stellar
- `--num_workers`: Parallel processes (200 recommended)

### train.py
- `--data`: Dataset path
- `--experiment_name`: Experiment ID
- `--epochs`: Training epochs (50 default)
- `--batch_size`: Batch size (128 default)
- `--lr`: Learning rate (1e-4 default)

### evaluate.py
- `--experiment_name`: Find latest run
- `--data`: Test data path
- `--n_samples`: Sample plots (20 default)
- `--confidence_threshold`: Decision threshold (0.8 default)

---

## File Locations

```
../results/experiment_TIMESTAMP/
├── best_model.pt           # Complete checkpoint
├── config.json             # Experiment config
├── scaler_standard.pkl     # StandardScaler
├── scaler_minmax.pkl       # MinMaxScaler
├── results.json            # Final metrics
└── evaluation/             # Plots
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── samples/
```

---

Ready for ultra-fast experimentation! 🚀