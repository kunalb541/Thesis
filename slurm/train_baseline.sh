#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --partition=gpu_mi300           # Change to your partition name
#SBATCH --gres=gpu:4                    # Number of GPUs
#SBATCH --cpus-per-gpu=24               # CPUs per GPU
#SBATCH --mem-per-gpu=128G              # Memory per GPU
#SBATCH --time=24:00:00                 # Max runtime
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ============================================================================
# Microlensing Binary Classification - Baseline Training
# Works for both AMD (ROCm) and NVIDIA (CUDA) GPUs
# ============================================================================

set -e  # Exit on error

# Job info
echo "=========================================="
echo "BASELINE TRAINING - WIDE PARAMETER RANGE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start: $(date)"
echo "=========================================="

# ============================================================================
# Configuration
# ============================================================================

# Project directory (adjust if needed)
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_FILE=${DATA_FILE:-$PROJECT_DIR/data/raw/events_baseline_1M.npz}
CODE_DIR=$PROJECT_DIR/code

echo ""
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_FILE"
echo ""

# Check data exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found!"
    echo "Expected: $DATA_FILE"
    echo ""
    echo "Generate it with:"
    echo "  cd $CODE_DIR"
    echo "  python simulate.py --output $DATA_FILE"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================

echo "Loading environment..."

# Load modules (adjust for your system)
# Uncomment the appropriate line:
# module load cuda/12.1              # For NVIDIA systems
# module load rocm/6.0               # For AMD systems
# Or your cluster's module name:
module load devel/cuda/12.1 || module load cuda || true

# Activate conda/virtual environment
source ~/.bashrc
conda activate microlens || source venv/bin/activate

# Verify Python
echo "Python: $(python --version)"
echo ""

# ============================================================================
# GPU Detection and Configuration
# ============================================================================

echo "Detecting GPU backend..."
python << EOF
import sys
sys.path.insert(0, '$CODE_DIR')
from utils import detect_gpu_backend, check_gpu_availability, setup_gpu_environment

backend = detect_gpu_backend()
print(f"\nDetected: {backend.upper()}")

num_gpus = check_gpu_availability()
if num_gpus == 0:
    print("\nERROR: No GPUs detected!")
    sys.exit(1)

setup_gpu_environment()
print(f"\n✓ Ready to train on {num_gpus} GPU(s)")
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: GPU detection failed!"
    exit 1
fi

# Set environment variables for GPU
# These are set by utils.py but we ensure they're exported
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check if AMD or NVIDIA
if command -v rocm-smi &> /dev/null; then
    echo ""
    echo "AMD GPU Status:"
    rocm-smi --showuse || true
    export ROCR_VISIBLE_DEVICES=0,1,2,3
    export HIP_VISIBLE_DEVICES=0,1,2,3
elif command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "=========================================="

# ============================================================================
# Training
# ============================================================================

cd $CODE_DIR

echo ""
echo "Starting baseline training..."
echo "Configuration:"
echo "  - Dataset: Wide range (planetary to stellar)"
echo "  - Events: 1M (500K PSPL + 500K Binary)"
echo "  - Epochs: 50"
echo "  - Batch size: 128 per GPU"
echo "  - Mixed precision: Enabled"
echo ""

python train.py \
    --data "$DATA_FILE" \
    --output "$PROJECT_DIR/models/baseline.pt" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --experiment_name baseline \
    --grad_clip 1.0

TRAIN_STATUS=$?

# ============================================================================
# Post-Training
# ============================================================================

if [ $TRAIN_STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    
    # Find results directory
    RESULTS_DIR=$(ls -td $PROJECT_DIR/results/baseline_* 2>/dev/null | head -1)
    
    if [ -d "$RESULTS_DIR" ]; then
        echo ""
        echo "Results: $RESULTS_DIR"
        echo ""
        echo "Files created:"
        ls -lh "$RESULTS_DIR/" | grep -E '\.(pt|pkl|json|log)$'
        
        # ========================================
        # Evaluation
        # ========================================
        
        echo ""
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
            --batch_size 128 \
            --early_detection
        
        EVAL_STATUS=$?
        
        if [ $EVAL_STATUS -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "EVALUATION COMPLETED"
            echo "=========================================="
            echo ""
            echo "Evaluation results: $EVAL_DIR"
            echo ""
            echo "Generated files:"
            ls -lh "$EVAL_DIR/"
            echo ""
            
            # Display metrics if available
            if [ -f "$EVAL_DIR/metrics.json" ]; then
                echo "Performance Metrics:"
                python -c "import json; m=json.load(open('$EVAL_DIR/metrics.json')); print(f\"  Accuracy: {m['accuracy']:.4f}\"); print(f\"  ROC AUC:  {m['roc_auc']:.4f}\"); print(f\"  PR AUC:   {m['pr_auc']:.4f}\")"
            fi
        else
            echo ""
            echo "WARNING: Evaluation failed (training succeeded)"
        fi
    fi
    
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED"
    echo "=========================================="
    echo "Check error log: logs/baseline_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Duration: $SECONDS seconds"
echo "=========================================="