#!/bin/bash
#SBATCH --job-name=ml_cadence_exp
#SBATCH --partition=gpu_h100        # Use H100 GPUs
#SBATCH --gres=gpu:1                # 1 GPU per experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=193300mb
#SBATCH --time=72:00:00             # 72 hours for full suite
#SBATCH --output=../logs/experiments_%j.out
#SBATCH --error=../logs/experiments_%j.err
#SBATCH --array=0-4                 # 5 cadence experiments

# Cadence experiments configuration
EXPERIMENTS=(sparse normal dense lsst roman)
CADENCE=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Running cadence experiment: $CADENCE"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load compiler/gnu/12
module load devel/cuda/12.1

# Activate conda
source ~/.bashrc
conda activate microlens

# Verify GPU
nvidia-smi

# Change to project directory
cd ~/thesis-microlens/code

# Run single cadence experiment
# Note: Using existing 1M dataset, just training with different experiment names
# If you want to generate new data with different cadences, use simulate_cadence.py

python train.py \
    --data ../data/raw/events_1M.npz \
    --output ../models/model_${CADENCE}.keras \
    --epochs 50 \
    --batch_size 64 \
    --experiment_name ${CADENCE}

# Evaluate the trained model
LATEST_DIR=$(ls -td ../results/${CADENCE}_* | head -1)
MODEL_PATH="${LATEST_DIR}/best_model.keras"
SCALER_PATH="${LATEST_DIR}/scaler.pkl"

python evaluate.py \
    --model $MODEL_PATH \
    --data ../data/raw/events_1M.npz \
    --scaler $SCALER_PATH \
    --output_dir ${LATEST_DIR}/evaluation

echo "=========================================="
echo "Experiment $CADENCE complete at: $(date)"
echo "Results in: $LATEST_DIR"
echo "=========================================="
