#!/bin/bash
#SBATCH --job-name=train_baseline_pt
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=128200mb
#SBATCH --time=24:00:00
#SBATCH --output=../logs/train_baseline_%j.out
#SBATCH --error=../logs/train_baseline_%j.err

echo "=========================================="
echo "PYTORCH BASELINE TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module load devel/cuda/12.1

# Activate environment
source ~/.bashrc
conda activate microlens

# Check GPUs
echo "GPU Check:"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Train
cd ~/thesis-microlens/code

python train.py \
    --data ../data/raw/events_1M.npz \
    --output ../models/baseline.pt \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
