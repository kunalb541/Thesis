#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=128200mb
#SBATCH --time=24:00:00
#SBATCH --output=/u/hd_vm305/thesis-microlens/logs/train_baseline_%j.out
#SBATCH --error=/u/hd_vm305/thesis-microlens/logs/train_baseline_%j.err

# Exit on error
set -e

# Job info
echo "=========================================="
echo "MICROLENSING CLASSIFICATION - BASELINE TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Define paths (using environment variables with defaults)
PROJECT_DIR=${PROJECT_DIR:-/u/hd_vm305/thesis-microlens}
DATA_FILE=${DATA_FILE:-$PROJECT_DIR/data/raw/events_1M.npz}
MODEL_FILE=${MODEL_FILE:-$PROJECT_DIR/models/baseline_model.pt}
CODE_DIR=$PROJECT_DIR/code

# Check if data exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    echo "Please generate data first using:"
    echo "  python $CODE_DIR/simulate.py --output $DATA_FILE"
    exit 1
fi

# Load modules and activate environment
echo "Loading environment..."
source ~/.bashrc
module load devel/cuda/12.1
conda activate microlens

# Verify environment
echo "Verifying Python environment..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set environment variables for ROCm (AMD GPU)
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3

# PyTorch optimizations
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check GPU availability
echo "=========================================="
echo "GPU Information:"
rocm-smi || echo "rocm-smi not available"
echo "=========================================="

# Go to code directory
cd $CODE_DIR

# Run training
echo "Starting training..."
echo "Data: $DATA_FILE"
echo "Output: $MODEL_FILE"
echo ""

python train.py \
    --data "$DATA_FILE" \
    --output "$MODEL_FILE" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --experiment_name baseline \
    --grad_clip 1.0

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    
    # Find the results directory (most recent baseline_* directory)
    RESULTS_DIR=$(ls -td $PROJECT_DIR/results/baseline_* | head -1)
    
    if [ -d "$RESULTS_DIR" ]; then
        echo "Results saved to: $RESULTS_DIR"
        echo ""
        echo "Files created:"
        ls -lh "$RESULTS_DIR"
        echo ""
        
        # Run evaluation automatically
        echo "=========================================="
        echo "RUNNING EVALUATION"
        echo "=========================================="
        
        EVAL_DIR="${RESULTS_DIR}/evaluation"
        mkdir -p "$EVAL_DIR"
        
        python evaluate.py \
            --model "$RESULTS_DIR/best_model.pt" \
            --data "$DATA_FILE" \
            --scaler "$RESULTS_DIR/scaler.pkl" \
            --output_dir "$EVAL_DIR" \
            --early_detection
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "EVALUATION COMPLETED SUCCESSFULLY"
            echo "=========================================="
            echo "Evaluation results saved to: $EVAL_DIR"
            echo ""
            echo "Files created:"
            ls -lh "$EVAL_DIR"
        else
            echo "WARNING: Evaluation failed, but training succeeded"
        fi
    fi
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED"
    echo "=========================================="
    echo "Check the error log for details:"
    echo "  tail -100 $PROJECT_DIR/logs/train_baseline_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "=========================================="