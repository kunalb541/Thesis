# Quick Reference - Command Cheatsheet (v5.6.2)

Fast command reference for all experiments with **DDP support** and **verified timings**.

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

# Train with DDP (~25 min on 5 nodes, 20 GPUs)
# See multi-node section below

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

## Distributed Training (DDP) - **UPDATED v5.6.2**

### Single Node (4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4
```

### Multi-Node (SLURM) - **VERIFIED WORKING**

```bash
#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/baseline_%j.out

# Pick master from the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# Network interface selection (CRITICAL for hybrid clusters)
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*,vboxnet*,vmnet*,slirp*,br-*,veth*,wlan*"

# If InfiniBand is flaky:
# export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR

cd ~/Thesis/code

# Exactly one torchrun per node
srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
  torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
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

**Key changes in v5.6.2:**
1. ✅ Quoted `"$SLURM_NODELIST"` 
2. ✅ `NCCL_SOCKET_IFNAME` exclusion pattern
3. ✅ Explicit `srun -N ... -n ... --ntasks-per-node=1`

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

# Check network interfaces
ip link show

# Test specific interface
export NCCL_SOCKET_IFNAME=eth0

# Or use exclusion (recommended)
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*"

# For InfiniBand issues
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# Verify nodes
srun -N ${SLURM_JOB_NUM_NODES} hostname
```

---

## Overlapping Experiment (CRITICAL - Next Step)

```bash
# 1. Add to config.py first:
# BINARY_OVERLAPPING = {
#     's_min': 0.1, 's_max': 2.5,
#     'q_min': 0.001, 'q_max': 1.0,
#     'u0_min': 0.01, 'u0_max': 1.0,  # ← KEY!
#     ...
# }
# BINARY_PARAM_SETS['overlapping'] = BINARY_OVERLAPPING

# 2. Generate (~15 min)
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --binary_params overlapping \
    --output ../data/raw/overlapping_1M.npz \
    --num_workers 200 \
    --seed 42 \
    --save-params  # ← CRITICAL for u₀ analysis!

# 3. Train (5 nodes, ~25 min)
# Use same SLURM script as baseline, just change:
#   --data ../data/raw/overlapping_1M.npz
#   --experiment_name overlapping_1M

# 4. Evaluate
python evaluate.py \
    --experiment_name overlapping_1M \
    --data ../data/raw/overlapping_1M.npz

# 5. u₀ ANALYSIS (KEY FINDING!)
python analyze_u0_dependency.py \
    --experiment_name overlapping_1M \
    --data ../data/raw/overlapping_1M.npz
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

## Complete Pipeline (~1 Hour with 5 nodes)

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

# 2. Training with DDP (~40-50 min for all)
# Use your verified 5-node SLURM script for each

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

# SLURM jobs
watch squeue -u $USER

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

## Expected Performance (v5.6.2) - **UPDATED**

### Timing

| Task | Configuration | Time | Status |
|------|---------------|------|--------|
| Simulate 1M | 200 workers | 10-15 min | Typical |
| Simulate 200K | 200 workers | 2-3 min | Typical |
| Train 1M | 1 node, 4 GPUs | ~60 min | Estimated |
| Train 1M | 5 nodes, 20 GPUs | ~25 min | ✅ **VERIFIED** |
| Train 1M | 10 nodes, 40 GPUs | ~15-20 min | Estimated |
| Train 200K | 1 node, 4 GPUs | 8-12 min | Estimated |
| Evaluate | Any | 2-5 min | Typical |

### Accuracy

| Experiment | Expected Acc | Status |
|------------|--------------|--------|
| **Distinct** (s∈[0.8,1.5], q∈[0.1,0.5]) | **84%** | ✅ **VERIFIED** |
| Overlapping (includes u₀>0.3) | 55-65% | Expected |
| Baseline (mixed) | 70-75% | Expected |
| Dense (5% missing) | 75-80% | Expected |
| Sparse (30% missing) | 65-70% | Expected |
| Low Error (0.05 mag) | 75-80% | Expected |
| High Error (0.20 mag) | 65-70% | Expected |

**Verified Result (Distinct):**
- Test Accuracy: 84%
- PSPL Precision: 98% (very reliable!)
- Binary Recall: 99% (catches almost all!)
- Conservative strategy: Flags 30% PSPL as binary (safe for astronomy)

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

# Network interface (CRITICAL!)
export NCCL_SOCKET_IFNAME="^lo,docker,virbr*"

# Check connectivity
srun -N ${SLURM_JOB_NUM_NODES} hostname
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
- `--binary_params`: baseline/distinct/planetary/stellar/overlapping
- `--num_workers`: Parallel processes (200 recommended)
- `--save-params`: Save event parameters (needed for u₀ analysis!)

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

## Next Critical Experiment

**Overlapping Parameters + u₀ Analysis**

Why: THE KEY THESIS FINDING - demonstrates physical detection limit

Commands:
1. Add `BINARY_OVERLAPPING` to config.py
2. Generate with `--save-params` flag
3. Train (same SLURM script as distinct)
4. Evaluate
5. Run `analyze_u0_dependency.py` (KEY!)

Expected: 55-65% overall, but ~50-55% for u₀>0.3 (physical limit!)

Timeline: ~1 hour total

---

Ready for ultra-fast experimentation! 🚀
