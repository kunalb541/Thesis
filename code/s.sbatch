#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_ultrafast_%j.out
#SBATCH -e logs/train_ultrafast_%j.err
#SBATCH --job-name=microlens_ultra

set -e

# ============================================================================
# ULTRA-FAST TRAINING: Copy to /tmp → Load into RAM
# Expected: 0.2-0.5 s/batch (20× faster!)
# ============================================================================

echo "================================================================================"
echo "ROMAN MICROLENSING CLASSIFIER - ULTRA-FAST (HDF5 → /tmp → RAM)"
echo "================================================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# STEP 1: Copy HDF5 to local /tmp on each node
# ============================================================================

SOURCE_FILE="$HOME/Thesis/data/raw/train_1M_distinct.h5"
LOCAL_FILE="/tmp/train_1M_distinct_${USER}.h5"

echo "STEP 1: Copying HDF5 to local /tmp on all nodes"
echo "  Source: $SOURCE_FILE (on Lustre)"
echo "  Dest:   $LOCAL_FILE (local storage)"
echo ""

# Copy to /tmp on ALL nodes in parallel
srun --ntasks-per-node=1 sh -c "
  if [ ! -f ${LOCAL_FILE} ]; then
    echo \"[\$(hostname)] Copying to /tmp...\"
    cp $SOURCE_FILE ${LOCAL_FILE}
    echo \"[\$(hostname)] ✓ Copy complete\"
  else
    echo \"[\$(hostname)] ✓ File already in /tmp\"
  fi
"

echo ""
echo "✓ All nodes have HDF5 in /tmp"
echo ""

# ============================================================================
# STEP 2: Load from /tmp into RAM and train
# ============================================================================

DATA_FILE="${LOCAL_FILE}"  # ← Use /tmp copy
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"

# Training hyperparameters
BATCH_SIZE=128
NUM_WORKERS=0              # No workers needed with RAM loading!
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
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
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
echo "  Data source:     $DATA_FILE (local /tmp)"
echo "  Workers/GPU:     $NUM_WORKERS (RAM loading mode!)"
echo "  Batch size/GPU:  $BATCH_SIZE"
echo "  Global batch:    $((SLURM_NNODES * 4 * BATCH_SIZE))"
echo ""
echo "TWO-STEP ULTRA-FAST MODE:"
echo "  ✓ STEP 1: HDF5 copied to local /tmp (eliminates Lustre)"
echo "  ✓ STEP 2: Dataset loaded into RAM from /tmp (fast local read)"
echo "  ✓ STEP 3: Training with zero I/O (pure RAM access)"
echo ""
echo "Expected performance:"
echo "  Copy to /tmp:  30-60 seconds (one-time)"
echo "  Load to RAM:   20-30 seconds (one-time)"
echo "  Batch time:    0.2-0.5 seconds"
echo "  Epoch time:    1.5-3 minutes"
echo "  100 epochs:    2.5-5 hours"
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
        echo "✓ Training complete (${CURRENT_EPOCH} epochs)"
        exit 0
    fi
    
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
    echo "Resuming from epoch ${CURRENT_EPOCH}"
else
    RESUME_FLAG=""
    echo "Starting fresh training"
fi

echo ""

# ============================================================================
# LAUNCH ULTRA-FAST TRAINING
# ============================================================================

echo "STEP 2: Loading from /tmp into RAM and starting training..."
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
    --use-prefetcher \
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
        echo "  ⟳ Resubmitting for remaining epochs..."
        sbatch --parsable "$0"
    else
        echo "  ✓ Training complete!"
        echo ""
        echo "Best model saved to: ${CHECKPOINT_DIR}/checkpoint_best.pt"
    fi
fi

exit ${EXIT_CODE}

