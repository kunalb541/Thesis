#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=128200mb
#SBATCH --time=24:00:00
#SBATCH --output=/u/hd_vm305/thesis-microlens/logs/train_baseline_%j.out
#SBATCH --error=/u/hd_vm305/thesis-microlens/logs/train_baseline_%j.err

# Job info
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

# Run training on existing 1M dataset
python train.py \
    --data /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz \
    --output /u/hd_vm305/thesis-microlens/models/baseline_model.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
