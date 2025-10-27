#!/bin/bash
#SBATCH --job-name=ml_debug
#SBATCH --partition=dev_gpu_h100    # Development partition for debugging
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=193300mb
#SBATCH --time=00:30:00             # 30 minutes for debugging
#SBATCH --output=../logs/debug_%j.out
#SBATCH --error=../logs/debug_%j.err

echo "=========================================="
echo "DEBUG SESSION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load modules
module purge
module load compiler/gnu/12
module load devel/cuda/12.1

# Activate conda
source ~/.bashrc
conda activate microlens

# GPU check
nvidia-smi

# Change to project directory
cd ~/thesis-microlens/code

# Quick test with small dataset
echo "Testing with small dataset (1000 samples)..."
python train.py \
    --data ../data/raw/events_1M.npz \
    --output ../models/debug_model.keras \
    --epochs 2 \
    --batch_size 32 \
    --experiment_name debug_test

echo "Debug test complete!"
