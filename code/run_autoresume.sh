#!/bin/bash
#SBATCH --job-name=microlens_auto
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:28:00
#SBATCH --output=logs/train_%A.out
#SBATCH --error=logs/train_%A.err

################################################################################
# Smart Auto-Resume Training for 30-Minute Slots
################################################################################

set -e

echo "================================================================================"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: gpu_a100_short (28 min timeout)"
echo "Nodes: $SLURM_JOB_NUM_NODES × 4 GPUs = $((SLURM_JOB_NUM_NODES * 4)) total GPUs"
echo "================================================================================"

# Setup paths
cd ~/Thesis/code
mkdir -p logs

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Environment variables
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE
export TORCH_DISTRIBUTED_ACK_TIMEOUT=1800
export TORCH_DISTRIBUTED_SEND_TIMEOUT=1200
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=5
export NCCL_MIN_NCHANNELS=16
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_ALGO=TREE

# Configuration
EXPERIMENT="production_1M_distinct"
DATA="../data/raw/train_1M_distinct.h5"
TEST_DATA="../data/test/test_300k_distinct.h5"
OUTPUT="../results/${EXPERIMENT}"
COMPLETE_FLAG="${OUTPUT}/.complete"

################################################################################
# Generate data if needed
################################################################################

if [ ! -f "$DATA" ]; then
    echo ""
    echo "Generating training data (1M samples)..."
    python simulate.py \
        --n_flat 333333 --n_pspl 333333 --n_binary 333334 \
        --binary_preset distinct --output "$DATA" \
        --num_workers 96 --seed 42
    echo "✓ Training data ready"
fi

if [ ! -f "$TEST_DATA" ]; then
    echo ""
    echo "Generating test data (300K samples)..."
    python simulate.py \
        --n_flat 100000 --n_pspl 100000 --n_binary 100000 \
        --binary_preset distinct --output "$TEST_DATA" \
        --num_workers 96 --seed 99999
    echo "✓ Test data ready"
fi

################################################################################
# Check if training complete
################################################################################

if [ -f "$COMPLETE_FLAG" ]; then
    echo ""
    echo "================================================================================"
    echo "TRAINING ALREADY COMPLETE!"
    echo "Completed: $(cat $COMPLETE_FLAG)"
    echo "Model: ${OUTPUT}/best.pt"
    echo "================================================================================"
    exit 0
fi

################################################################################
# Find latest checkpoint
################################################################################

RESUME_ARG=""
if [ -d "$OUTPUT" ]; then
    # Find latest epoch checkpoint
    LATEST=$(ls -t ${OUTPUT}/epoch_*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume $LATEST"
        EPOCH=$(basename "$LATEST" | grep -oP '\d+')
        echo ""
        echo "▶ Resuming from checkpoint: epoch_${EPOCH}.pt"
    else
        echo ""
        echo "▶ Starting fresh training"
    fi
else
    echo ""
    echo "▶ Starting fresh training"
fi

################################################################################
# Train with timeout protection
################################################################################

echo ""
echo "================================================================================"
echo "TRAINING (timeout: 26 minutes)"
echo "================================================================================"

# Run training with 26-minute timeout (2 min buffer for cleanup)
timeout 1560s srun torchrun \
    --nnodes=10 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="train-${SLURM_JOB_ID}" \
    train.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 5 \
    --clip-norm 1.0 \
    --d-model 64 \
    --n-layers 4 \
    --window-size 7 \
    --dropout 0.3 \
    --num-workers 8 \
    --prefetch-factor 12 \
    --hierarchical \
    --attention-pooling \
    --use-prefetcher \
    --save-every 5 \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "Training segment ended: $(date)"
echo "Exit code: $EXIT_CODE (124 = timeout, 0 = completed)"

################################################################################
# Check if training complete
################################################################################

# Check if we reached 100 epochs
COMPLETED=false
if [ -f "${OUTPUT}/epoch_100.pt" ] || [ -f "${OUTPUT}/final.pt" ]; then
    COMPLETED=true
    echo "✓ Training reached 100 epochs!"
fi

# If complete, mark as done and evaluate
if [ "$COMPLETED" = true ]; then
    echo "$(date)" > "$COMPLETE_FLAG"
    
    echo ""
    echo "================================================================================"
    echo "RUNNING EVALUATION"
    echo "================================================================================"
    
    # Rename checkpoint if needed
    if [ -f "${OUTPUT}/best.pt" ]; then
        cp "${OUTPUT}/best.pt" "${OUTPUT}/best_model.pt"
    fi
    
    python evaluate.py \
        --experiment-name "$EXPERIMENT" \
        --data "$TEST_DATA" \
        --batch-size 512 \
        --early-detection \
        --colorblind-safe \
        --n-evolution-per-type 5
    
    echo ""
    echo "================================================================================"
    echo "ALL DONE!"
    echo "================================================================================"
    echo "Results: $OUTPUT/"
    echo "Evaluation: ${OUTPUT}/eval_*/"
    echo "================================================================================"
    
    exit 0
fi

################################################################################
# Resubmit if not complete
################################################################################

echo ""
echo "================================================================================"
echo "Training not complete - resubmitting"
echo "================================================================================"

NEXT_JOB=$(sbatch --parsable "$0")
echo "Next job: $NEXT_JOB"
echo "Monitor: squeue -u $USER"
echo "================================================================================"

exit 0
