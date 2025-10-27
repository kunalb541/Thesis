#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --partition=gpu_h100        # Use H100 GPUs
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24          # 24 CPUs per GPU
#SBATCH --mem-per-gpu=193300mb      # Max memory for H100
#SBATCH --time=48:00:00             # 48 hours max
#SBATCH --output=../logs/train_%j.out
#SBATCH --error=../logs/train_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load required modules (adjust based on cluster setup)
module purge
module load compiler/gnu/12
module load devel/cuda/12.1

# Activate conda environment
source ~/.bashrc
conda activate microlens

# Verify GPU availability
nvidia-smi

# Change to project directory
cd ~/thesis-microlens/code

# Run training with your existing 1M dataset
python train.py \
    --data ../data/raw/events_1M.npz \
    --output ../models/best_model_gpu.keras \
    --epochs 50 \
    --batch_size 64 \
    --experiment_name baseline_gpu

echo "Training complete at: $(date)"
