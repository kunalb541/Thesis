#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_4n_local_%j.out
#SBATCH -e logs/train_4n_local_%j.err
#SBATCH --job-name=microlens_local

set -e

# ============================================================================
# FIXED: COPY HDF5 TO LOCAL STORAGE FIRST (ELIMINATES LUSTRE BOTTLENECK)
# ============================================================================

echo "================================================================================"
echo "ROMAN MICROLENSING CLASSIFIER - LOCAL STORAGE (LUSTRE BOTTLENECK FIX)"
echo "================================================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# CRITICAL FIX: Copy HDF5 to local /tmp BEFORE training
# ============================================================================

SOURCE_FILE="$HOME/Thesis/data/raw/train_1M_distinct.h5"
LOCAL_FILE="/tmp/train_1M_distinct.h5"

echo "CRITICAL FIX: Copying HDF5 to local storage to eliminate Lustre I/O bottleneck"
echo "  Source: $SOURCE_FILE (on Lustre)"
echo "  Dest:   $LOCAL_FILE (local SSD)"
echo ""

# Copy to /tmp on ALL nodes in parallel
echo "Copying to all nodes..."
srun --ntasks-per-node=1 sh -c "
  if [ ! -f /tmp/train_1M_distinct.h5 ]; then
    echo \"[\$(hostname)] Copying HDF5 to /tmp...\"
    cp $SOURCE_FILE /tmp/
    echo \"[\$(hostname)] ✓ Copy complete\"
  else
    echo \"[\$(hostname)] ✓ File already in /tmp\"
  fi
"

echo ""
echo "✓ All nodes have local copy"
echo ""

# ============================================================================
# USE LOCAL FILE FOR TRAINING
# ============================================================================

DATA_FILE="/tmp/train_1M_distinct.h5"  # ← THIS IS THE KEY FIX
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"

# Training hyperparameters
BATCH_SIZE=128
NUM_WORKERS=4              
PREFETCH_FACTOR=2
ACCUMULATION_STEPS=1
MAX_EPOCHS=100

# Model architecture
D_MODEL=256
N_LAYERS=4
DROPOUT=0.3
WINDOW_SIZE=7

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# ============================================================================
# ENVIRONMENT
# ============================================================================

export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_ACK_TIMEOUT=1800
export TORCH_DISTRIBUTED_SEND_TIMEOUT=1200
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=5
export NCCL_MIN_NCHANNELS=16
export NCCL_ALGO=TREE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# ============================================================================
# JOB INFO
# ============================================================================

echo "Job Configuration:"
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs:            $((SLURM_NNODES * 4))"
echo "  Data source:     LOCAL /tmp (Lustre bottleneck ELIMINATED)"
echo "  Batch size/GPU:  $BATCH_SIZE"
echo "  Global batch:    $((SLURM_NNODES * 4 * BATCH_SIZE))"
echo ""
echo "Expected performance:"
echo "  Before (Lustre):  6-12 s/batch  ← SLOW"
echo "  After (local):    0.3-0.5 s/batch ← FAST (20× speedup!)"
echo "================================================================================"
echo ""

# ============================================================================
# CHECKPOINT HANDLING
# ============================================================================

mkdir -p logs "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

if [ -f "${RESUME_CHECKPOINT}" ]; then
    CURRENT_EPOCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ ${CURRENT_EPOCH} -ge ${MAX_EPOCHS} ]; then
        echo "✓ Training complete"
        exit 0
    fi
    
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
else
    RESUME_FLAG=""
fi

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

echo "Starting OPTIMIZED training with LOCAL storage..."
echo "================================================================================"

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
echo "Training finished: exit code ${EXIT_CODE}"
echo "================================================================================"

# Auto-resubmit if needed
if [ ${EXIT_CODE} -eq 0 ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    FINAL_EPOCH=$(python3 -c "
import torch
try:
    ckpt = torch.load('${RESUME_CHECKPOINT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ ${FINAL_EPOCH} -lt ${MAX_EPOCHS} ]; then
        echo "  ⟳ Resubmitting..."
        sbatch --parsable "$0"
    else
        echo "  ✓ Training complete!"
    fi
fi

exit ${EXIT_CODE}

