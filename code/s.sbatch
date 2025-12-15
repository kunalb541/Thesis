#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_12n_%j.out
#SBATCH -e logs/train_12n_%j.err
#SBATCH --job-name=microlens_12n

set -e

# ============================================================================
# 12-NODE DISTRIBUTED TRAINING (FIXED WITH SRUN)
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
BATCH_SIZE=128              # Per GPU 
NUM_WORKERS=4              # Data loading workers per GPU
PREFETCH_FACTOR=2          # Prefetch batches per worker
ACCUMULATION_STEPS=1       # Gradient accumulation
MAX_EPOCHS=100             # Total epochs

# Model architecture
D_MODEL=256                # Hidden dimension
N_LAYERS=4                 # Number of GRU layers
DROPOUT=0.3                # Dropout rate
WINDOW_SIZE=7              # CNN kernel size

# Optimization
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=5
CLIP_NORM=1.0

# ============================================================================
# CRITICAL ENVIRONMENT VARIABLES (WORKING CONFIG)
# ============================================================================

# Suppress warnings for cleaner logs
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_ACK_TIMEOUT=1800  # 30 min for 12-node init
export TORCH_DISTRIBUTED_SEND_TIMEOUT=1200 # 20 min send timeout

# NCCL configuration for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=5
export NCCL_MIN_NCHANNELS=16
export NCCL_ALGO=TREE  # Better for multi-node scaling

# Memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Master node configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
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
echo "  Node list:       $SLURM_NODELIST"
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
    echo "▶ Found existing checkpoint: ${RESUME_CHECKPOINT}"
    
    # Extract current epoch
    CURRENT_EPOCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(f'Error: {e}', file=__import__('sys').stderr)
    print(0)
" 2>/dev/null || echo "0")
    
    echo "  Current epoch: ${CURRENT_EPOCH}/${MAX_EPOCHS}"
    
    # Check if training is already complete
    if [ ${CURRENT_EPOCH} -ge ${MAX_EPOCHS} ]; then
        echo "✓ Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
        echo "  To restart training, delete: ${RESUME_CHECKPOINT}"
        exit 0
    fi
    
    echo "  Resuming training from epoch $((CURRENT_EPOCH + 1))"
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
else
    echo "▶ No checkpoint found - starting fresh training"
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
# LAUNCH TRAINING (CRITICAL: USE SRUN + TORCHRUN)
# ============================================================================

echo "Starting distributed training..."
echo "================================================================================"
echo ""

# CRITICAL: srun launches torchrun on ALL nodes
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="train-$SLURM_JOB_ID" \
    --max-restarts=3 \
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
    echo "✓ Training epoch(s) completed successfully"
    
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        FINAL_EPOCH=$(python3 -c "
import torch
import sys
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    print(f'{epoch}')
    print(f'{best_acc * 100:.2f}', file=sys.stderr)
except Exception as e:
    print('0')
    print('0.00', file=sys.stderr)
" 2>&1 1>&3 | { read BEST_ACC; echo "${BEST_ACC}" >&4; } 3>&1 ) 4>&2 2>&1 || echo "0")
        
        BEST_ACC=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(f\"{ckpt.get('best_acc', 0.0) * 100:.2f}\")
except:
    print('0.00')
" 2>/dev/null || echo "0.00")
        
        echo "  Completed epoch: ${FINAL_EPOCH}/${MAX_EPOCHS}"
        echo "  Best validation accuracy: ${BEST_ACC}%"
        
        if [ ${FINAL_EPOCH} -lt ${MAX_EPOCHS} ]; then
            REMAINING=$((MAX_EPOCHS - FINAL_EPOCH))
            echo "  ⟳ Resubmitting to continue training (${REMAINING} epochs remaining)..."
            NEW_JOB_ID=$(sbatch --parsable "$0")
            echo "  ✓ Resubmitted as job ${NEW_JOB_ID}"
        else
            echo "  ✓ Training complete - all ${MAX_EPOCHS} epochs finished!"
            echo "  📊 Final results:"
            echo "    - Best checkpoint: ${OUTPUT_DIR}/best.pt"
            echo "    - Final checkpoint: ${OUTPUT_DIR}/final.pt"
            echo "    - Latest checkpoint: ${RESUME_CHECKPOINT}"
        fi
    fi
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
    echo "  Check logs for errors:"
    echo "    - logs/train_12n_${SLURM_JOB_ID}.out"
    echo "    - logs/train_12n_${SLURM_JOB_ID}.err"
    echo "  Common issues:"
    echo "    1. NCCL timeout - check network between nodes"
    echo "    2. CUDA OOM - reduce batch size or model size"
    echo "    3. Data loading error - verify HDF5 file integrity"
fi

exit ${EXIT_CODE}

