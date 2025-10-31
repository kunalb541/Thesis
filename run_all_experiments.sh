#!/bin/bash
#SBATCH --job-name=thesis_all
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=logs/thesis_all_%j.out
#SBATCH --error=logs/thesis_all_%j.err

# ============================================================================
# Master Thesis Experiment Suite - Fixed Version
# ============================================================================
# This script runs ALL experiments for the thesis:
# - Data generation (simulation)
# - Distributed training (multi-node, multi-GPU)
# - Evaluation and analysis
#
# Author: Kunal Bhatia
# Date: October 2025
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Setup
# ============================================================================
echo "============================================================================"
echo "Starting Thesis Experiment Suite"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 4"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 4))"
echo "Start time: $(date)"
echo "============================================================================"

# Load modules
module load cuda/12.1
module load python/3.10

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Change to code directory
cd ~/Thesis/code

# Create logs directory
mkdir -p ../logs

# Setup distributed training variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo ""

# ============================================================================
# Pre-flight Validation
# ============================================================================
echo "============================================================================"
echo "Running Pre-flight Validation"
echo "============================================================================"

python test_vbm.py
if [ $? -ne 0 ]; then
    echo "❌ VBMicrolensing validation failed! Aborting."
    exit 1
fi
echo "✓ VBMicrolensing validation passed"
echo ""

# ============================================================================
# Phase 1: Data Generation (Simulation)
# ============================================================================
echo "============================================================================"
echo "Phase 1: Data Generation"
echo "============================================================================"
echo "This will generate all datasets (~4-6 hours)"
echo ""

# 0️⃣ Baseline (1M events) - CRITICAL
echo "--- Baseline (1M events) ---"
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --n_points 1500 \
    --output ../data/raw/baseline_1M.npz \
    --binary_params baseline \
    --seed 42 \
    --num_workers 200

# 1️⃣ Cadence Experiments
echo ""
echo "--- Cadence Experiments ---"
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    echo "Generating cadence_${name} (missing=${cadence})"
    python simulate.py \
        --n_pspl 100000 \
        --n_binary 100000 \
        --n_points 1500 \
        --output ../data/raw/cadence_${name}.npz \
        --cadence_mask_prob $cadence \
        --seed 42 \
        --num_workers 200
done

# 2️⃣ Photometric Error Experiments
echo ""
echo "--- Photometric Error Experiments ---"
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    echo "Generating error_${name} (sigma=${error} mag)"
    python simulate.py \
        --n_pspl 100000 \
        --n_binary 100000 \
        --n_points 1500 \
        --output ../data/raw/error_${name}.npz \
        --mag_error_std $error \
        --seed 42 \
        --num_workers 200
done

# 3️⃣ Binary Topology Experiments
echo ""
echo "--- Binary Topology Experiments ---"
for topo in distinct planetary stellar; do
    echo "Generating ${topo}"
    python simulate.py \
        --n_pspl 100000 \
        --n_binary 100000 \
        --n_points 1500 \
        --output ../data/raw/${topo}.npz \
        --binary_params ${topo} \
        --seed 42 \
        --num_workers 200
done

echo ""
echo "✓ All datasets generated!"
echo "Simulation complete: $(date)"
echo ""

# ============================================================================
# Phase 2: Distributed Training
# ============================================================================
echo "============================================================================"
echo "Phase 2: Distributed Training"
echo "============================================================================"
echo "This will train all models (~24-36 hours)"
echo ""

# Training function
train_experiment() {
    local exp_name=$1
    local data_path=$2
    
    echo "============================================================================"
    echo "Training: $exp_name"
    echo "Data: $data_path"
    echo "Start time: $(date)"
    echo "============================================================================"
    
    # Unique rendezvous ID per experiment
    local rdzv_id="${SLURM_JOB_ID}_${exp_name}"
    
    srun --exclusive torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=$rdzv_id \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        --data $data_path \
        --experiment_name $exp_name \
        --epochs 50 \
        --batch_size 128 \
        --lr 1e-4
    
    echo "✓ Training complete: $exp_name"
    echo "End time: $(date)"
    echo ""
}

# 0️⃣ Baseline
train_experiment "baseline" "../data/raw/baseline_1M.npz"

# 1️⃣ Cadence Experiments
for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    train_experiment "cadence_${name}" "../data/raw/cadence_${name}.npz"
done

# 2️⃣ Photometric Error Experiments
for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    train_experiment "error_${name}" "../data/raw/error_${name}.npz"
done

# 3️⃣ Binary Topology Experiments
for topo in distinct planetary stellar; do
    train_experiment "${topo}" "../data/raw/${topo}.npz"
done

echo "✓ All models trained!"
echo "Training complete: $(date)"
echo ""

# ============================================================================
# Phase 3: Evaluation
# ============================================================================
echo "============================================================================"
echo "Phase 3: Evaluation"
echo "============================================================================"
echo "This will evaluate all models (~2-4 hours)"
echo ""

# Evaluation function
evaluate_experiment() {
    local exp_name=$1
    local data_path=$2
    
    echo "--- Evaluating: $exp_name ---"
    
    python evaluate.py \
        --experiment_name $exp_name \
        --data $data_path \
        --early_detection \
        --batch_size 256
    
    python benchmark_realtime.py \
        --experiment_name $exp_name \
        --data $data_path \
        --n_samples 10000
    
    python plot_samples.py \
        --experiment_name $exp_name \
        --data $data_path \
        --n_samples 12
    
    echo "✓ Evaluation complete: $exp_name"
    echo ""
}

# Evaluate all experiments
evaluate_experiment "baseline" "../data/raw/baseline_1M.npz"

for cadence in 0.05 0.20 0.30 0.40; do
    name=$(echo $cadence | sed 's/0\.//')
    evaluate_experiment "cadence_${name}" "../data/raw/cadence_${name}.npz"
done

for error in 0.05 0.10 0.20; do
    name=$(echo $error | sed 's/0\.//')
    evaluate_experiment "error_${name}" "../data/raw/error_${name}.npz"
done

for topo in distinct planetary stellar; do
    evaluate_experiment "${topo}" "../data/raw/${topo}.npz"
done

echo "✓ All evaluations complete!"
echo "Evaluation complete: $(date)"
echo ""

# ============================================================================
# Phase 4: Generate Summary
# ============================================================================
echo "============================================================================"
echo "Phase 4: Results Summary"
echo "============================================================================"

python << 'PYTHON_SCRIPT'
import json
from pathlib import Path

experiments = [
    'baseline',
    'cadence_05', 'cadence_20', 'cadence_30', 'cadence_40',
    'error_05', 'error_10', 'error_20',
    'distinct', 'planetary', 'stellar'
]

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"\n{'Experiment':<20} {'Test Acc':<12} {'Best Epoch':<12} {'Status':<10}")
print("-" * 60)

for exp in experiments:
    runs = sorted(Path('../results').glob(f'{exp}_*'))
    if runs:
        summary_path = runs[-1] / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            acc = data.get('final_test_acc', 0) * 100
            epoch = data.get('best_epoch', 0)
            print(f"{exp:<20} {acc:>10.2f}% {epoch:>10d}    ✓")
        else:
            print(f"{exp:<20} {'---':<12} {'---':<12}  ✗")
    else:
        print(f"{exp:<20} {'NOT FOUND':<12} {'---':<12}  ✗")

print("\n" + "="*80)
PYTHON_SCRIPT

echo ""
echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved in: ~/Thesis/results/"
echo ""
echo "Next steps:"
echo "  1. Review results summary above"
echo "  2. Generate comparison plots (see RESEARCH_GUIDE.md)"
echo "  3. Start writing thesis results chapter"
echo "============================================================================"