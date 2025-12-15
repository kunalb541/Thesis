#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_12n_%j.out
#SBATCH -e logs/train_12n_%j.err
#SBATCH --job-name=microlens_12n

set -e

# ============================================================================
# 12-NODE DISTRIBUTED TRAINING
# ============================================================================

echo "================================================================================"
echo "ROMAN MICROLENSING CLASSIFIER - 12 NODE TRAINING"
echo "================================================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE="../data/raw/train_1M_distinct.h5"
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"

# Training hyperparameters
BATCH_SIZE=16              # Per GPU (16 * 48 = 768 global batch size)
NUM_WORKERS=2              # Data loading workers per GPU
PREFETCH_FACTOR=2          # Prefetch batches per worker
ACCUMULATION_STEPS=1       # Gradient accumulation
MAX_EPOCHS=100             # Total epochs

# Model architecture
D_MODEL=128                # Hidden dimension
N_LAYERS=4                 # Number of GRU layers
DROPOUT=0.1                # Dropout rate
WINDOW_SIZE=7             # CNN kernel size

# Optimization
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=5
CLIP_NORM=1.0

# ============================================================================
# CRITICAL ENVIRONMENT VARIABLES
# ============================================================================

# Force unbuffered output (see logs immediately)
export PYTHONUNBUFFERED=1

# Memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# NCCL timeouts (CRITICAL for multi-node stability)
export NCCL_TIMEOUT=600
export NCCL_SOCKET_TIMEOUT=300
export NCCL_IB_TIMEOUT=20
export NCCL_BLOCKING_WAIT=1

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=5
export NCCL_MIN_NCHANNELS=16

# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=OFF

# CUDA optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

# Master node configuration
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

# ============================================================================
# JOB INFORMATION
# ============================================================================

echo "Job Configuration:"
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs per node:   4"
echo "  Total GPUs:      $((SLURM_NNODES * 4))"
echo "  Master node:     $MASTER_ADDR:$MASTER_PORT"
echo ""
echo "Training Parameters:"
echo "  Batch size/GPU:  $BATCH_SIZE"
echo "  Global batch:    $((SLURM_NNODES * 4 * BATCH_SIZE))"
echo "  Workers/GPU:     $NUM_WORKERS"
echo "  Prefetch:        $PREFETCH_FACTOR"
echo "  Max epochs:      $MAX_EPOCHS"
echo "  Learning rate:   $LEARNING_RATE"
echo ""
echo "Model Architecture:"
echo "  d_model:         $D_MODEL"
echo "  n_layers:        $N_LAYERS"
echo "  dropout:         $DROPOUT"
echo "  window_size:     $WINDOW_SIZE"
echo "================================================================================"
echo ""

# ============================================================================
# CHECKPOINT HANDLING
# ============================================================================

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

if [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "▶ Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
    
    # Extract current epoch
    CURRENT_EPOCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
    
    echo "  Current epoch: ${CURRENT_EPOCH}/${MAX_EPOCHS}"
else
    echo "▶ Starting fresh training"
    RESUME_FLAG=""
fi

echo ""

# ============================================================================
# VERIFY DATA FILE
# ============================================================================

if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data file not found: ${DATA_FILE}"
    exit 1
fi

echo "✓ Data file verified: ${DATA_FILE}"
echo ""

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

echo "Starting distributed training..."
echo "================================================================================"
echo ""

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id=$SLURM_JOB_ID \
    --max_restarts=3 \
    --log_dir=logs \
    train.py \
    --data "${DATA_FILE}" \
    --output "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --prefetch-factor ${PREFETCH_FACTOR} \
    --accumulation-steps ${ACCUMULATION_STEPS} \
    --epochs ${MAX_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --clip-norm ${CLIP_NORM} \
    --d-model ${D_MODEL} \
    --n-layers ${N_LAYERS} \
    --dropout ${DROPOUT} \
    --window-size ${WINDOW_SIZE} \
    --hierarchical \
    --attention-pooling \
    --use-class-weights \
    --use-prefetcher \
    --use-amp \
    --compile \
    --save-every 5 \
    ${RESUME_FLAG}

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Training finished with exit code: ${EXIT_CODE}"
echo "================================================================================"

# ============================================================================
# AUTO-RESUBMISSION FOR CONTINUED TRAINING
# ============================================================================

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully"
    
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        FINAL_EPOCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
        
        echo "  Completed epoch: ${FINAL_EPOCH}/${MAX_EPOCHS}"
        
        if [ ${FINAL_EPOCH} -lt ${MAX_EPOCHS} ]; then
            echo "  ⟳ Resubmitting to continue training..."
            sbatch "$0"
        else
            echo "  ✓ Training complete!"
        fi
    fi
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
    echo "  Check logs for errors:"
    echo "    - logs/train_12n_${SLURM_JOB_ID}.out"
    echo "    - logs/train_12n_${SLURM_JOB_ID}.err"
fi

exit ${EXIT_CODE}
