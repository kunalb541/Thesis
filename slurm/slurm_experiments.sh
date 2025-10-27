#!/bin/bash
#SBATCH --job-name=experiments_all
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=128200mb
#SBATCH --time=72:00:00
#SBATCH --output=/u/hd_vm305/thesis-microlens/logs/experiments_%j.out
#SBATCH --error=/u/hd_vm305/thesis-microlens/logs/experiments_%j.err

# Job info
echo "=========================================="
echo "AUTOMATED EXPERIMENT SUITE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Load modules and activate environment
source ~/.bashrc
module load devel/cuda/12.1
conda activate microlens

# Set environment variables for ROCm (AMD GPU)
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3

# TensorFlow optimizations
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Go to code directory
cd /u/hd_vm305/thesis-microlens/code

# Check GPU availability
echo "=========================================="
echo "GPU Information:"
rocm-smi
echo "=========================================="

# Run baseline experiment first
echo "=========================================="
echo "EXPERIMENT 1: BASELINE (20% missing data)"
echo "=========================================="
python train.py \
    --data /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz \
    --output /u/hd_vm305/thesis-microlens/models/baseline.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline

# Evaluate baseline
python evaluate.py \
    --model /u/hd_vm305/thesis-microlens/models/baseline.keras \
    --data /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz \
    --output_dir /u/hd_vm305/thesis-microlens/results/baseline

echo "Baseline complete!"
echo ""

# Note: For other experiments, you would need to generate new datasets with different parameters
# For now, this script focuses on baseline. Uncomment and modify below for additional experiments:

# EXPERIMENT 2: Dense cadence (5% missing)
# echo "=========================================="
# echo "EXPERIMENT 2: DENSE CADENCE (5% missing)"
# echo "=========================================="
# python train.py \
#     --data /u/hd_vm305/thesis-microlens/data/raw/events_cadence_05.npz \
#     --output /u/hd_vm305/thesis-microlens/models/cadence_05.keras \
#     --epochs 50 \
#     --batch_size 128 \
#     --experiment_name cadence_05
# 
# python evaluate.py \
#     --model /u/hd_vm305/thesis-microlens/models/cadence_05.keras \
#     --data /u/hd_vm305/thesis-microlens/data/raw/events_cadence_05.npz \
#     --output_dir /u/hd_vm305/thesis-microlens/results/cadence_05
# 
# echo "Cadence 5% complete!"
# echo ""

# Compare all experiments
echo "=========================================="
echo "COMPARING ALL EXPERIMENTS"
echo "=========================================="
python experiments.py --compare_only

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "=========================================="
