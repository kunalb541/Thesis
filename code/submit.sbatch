#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_complete_%j.out
#SBATCH -e logs/train_complete_%j.err
#SBATCH --job-name=microlens_hier
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120

# =============================================================================
# ROMAN MICROLENSING CLASSIFIER - v2.9 HIERARCHICAL FIX
# =============================================================================
# This script uses the fixed hierarchical training with separate BCE losses
# for Stage 1 (Flat vs Non-Flat) and Stage 2 (PSPL vs Binary).
#
# Key improvements:
#   - Separate BCE losses prevent Stage 2 collapse
#   - Auxiliary 3-class head for gradient stability
#   - Per-class recall logging during training
#   - Temperature scaling option for Stage 2
# =============================================================================

set -e

echo "================================================================================"
echo "ROMAN MICROLENSING - v2.9 HIERARCHICAL FIX TRAINING"
echo "================================================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# CONFIGURATION
# ============================================================================
SOURCE_FILE="$HOME/Thesis/data/raw/train.h5"
TEST_FILE="$HOME/Thesis/data/test/test.h5"
LOCAL_FILE="/tmp/train_${USER}.h5"
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"
CONTINUATION_JOB_FILE="${CHECKPOINT_DIR}/.continuation_job_id"

# Training hyperparameters
BATCH_SIZE=64
NUM_WORKERS=0
ACCUMULATION_STEPS=4
MAX_EPOCHS=300

# Model architecture
D_MODEL=64
N_LAYERS=4
DROPOUT=0.3
WINDOW_SIZE=5

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# v2.9: Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=1.0
AUX_WEIGHT=0.5
STAGE2_TEMPERATURE=1.0

# ============================================================================
# CRITICAL: Set thread limits (fixes multiprocessing hang)
# ============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

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
# HELPER FUNCTION: Get current epoch from checkpoint
# ============================================================================
get_checkpoint_epoch() {
    local checkpoint_path="$1"
    if [ -f "${checkpoint_path}" ]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('${checkpoint_path}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================
mkdir -p logs "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

# ============================================================================
# CHECK IF TRAINING ALREADY COMPLETE
# ============================================================================
CURRENT_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")

if [ "${CURRENT_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
    echo "✓ Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
    echo ""
    echo "To restart training, delete: ${RESUME_CHECKPOINT}"
    echo ""
    echo "To run evaluation:"
    echo "  python evaluate.py \\"
    echo "    --experiment-name test_300k_hier \\"
    echo "    --data ${TEST_FILE} \\"
    echo "    --batch-size 512 \\"
    echo "    --early-detection"
    
    # Cancel any pending continuation job
    if [ -f "${CONTINUATION_JOB_FILE}" ]; then
        CONT_JOB=$(cat "${CONTINUATION_JOB_FILE}")
        scancel "${CONT_JOB}" 2>/dev/null || true
        rm -f "${CONTINUATION_JOB_FILE}"
    fi
    exit 0
fi

echo "Current progress: ${CURRENT_EPOCH}/${MAX_EPOCHS} epochs"
echo ""

# ============================================================================
# SUBMIT DEPENDENT CONTINUATION JOB
# ============================================================================
echo "================================================================================"
echo "SETTING UP AUTO-CONTINUATION"
echo "================================================================================"

SCRIPT_PATH="$(scontrol show job $SLURM_JOB_ID | grep -oP 'Command=\K[^ ]+')"
if [ -z "${SCRIPT_PATH}" ]; then
    SCRIPT_PATH="$HOME/Thesis/code/submit.sbatch"
fi

CONT_JOB=$(sbatch --parsable \
    --dependency=afterany:${SLURM_JOB_ID} \
    --kill-on-invalid-dep=yes \
    "${SCRIPT_PATH}")

if [ -n "${CONT_JOB}" ]; then
    echo "${CONT_JOB}" > "${CONTINUATION_JOB_FILE}"
    echo "✓ Continuation job submitted: ${CONT_JOB}"
    echo "  Dependency: afterany:${SLURM_JOB_ID}"
    echo "  Will auto-cancel if training completes all ${MAX_EPOCHS} epochs"
else
    echo "⚠ Warning: Failed to submit continuation job"
fi
echo ""

# ============================================================================
# SIGNAL HANDLING
# ============================================================================
handle_timeout() {
    echo ""
    echo "================================================================================"
    echo "⚠️  TIMEOUT WARNING: Job will be killed in ~120 seconds"
    echo "================================================================================"
    echo "Current epoch will try to finish and save checkpoint."
    echo "Continuation job ${CONT_JOB} will resume training automatically."
    echo "================================================================================"
}

trap 'handle_timeout' USR1

# ============================================================================
# STEP 0: GENERATE FRESH DATA (if needed)
# ============================================================================
echo "================================================================================"
echo "STEP 0: DATA GENERATION"
echo "================================================================================"

if [ ! -f "${SOURCE_FILE}" ]; then
    echo "Training data not found. Generating on one node (32 workers)..."
    echo ""
    
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 100000 \
        --n_pspl 500000 \
        --n_binary 500000 \
        --binary_preset baseline \
        --output ${SOURCE_FILE} \
        --num_workers 100 \
        --seed 42
    
    echo "✓ Training data generated: ${SOURCE_FILE}"
else
    echo "✓ Training data exists: ${SOURCE_FILE}"
fi
echo ""

# Generate test data if needed
if [ ! -f "${TEST_FILE}" ]; then
    echo "Test data not found. Generating..."
    mkdir -p $(dirname ${TEST_FILE})
    
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 100000 \
        --n_pspl 100000 \
        --n_binary 100000 \
        --binary_preset distinct \
        --output ${TEST_FILE} \
        --num_workers 100 \
        --seed 999
    
    echo "✓ Test data generated: ${TEST_FILE}"
else
    echo "✓ Test data exists: ${TEST_FILE}"
fi
echo ""

# Verify data contains magnifications (not magnitudes)
echo "Verifying data format..."
python << 'PYEOF'
import h5py
import numpy as np
import sys

with h5py.File("../data/raw/train.h5", 'r') as f:
    flux = f['flux'][:]
    
flux_valid = flux[flux != 0]
mean_val = np.mean(flux_valid)

print(f"  Mean value: {mean_val:.6f}")

if 0 < mean_val < 100:
    print("  ✓ Data contains MAGNIFICATIONS (correct!)")
    print("    A=1.0 is baseline, A>1 is magnified")
else:
    print(f"  ⚠ Unexpected range (mean={mean_val:.2f})")
    print("  Check simulate.py output format")
PYEOF

echo ""
echo "================================================================================"

# ============================================================================
# STEP 1: CLEAN OLD DATA FROM /tmp
# ============================================================================
echo "STEP 1: Cleaning old data from /tmp on all nodes..."
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 rm -f ${LOCAL_FILE}
echo "✓ Old data deleted"
echo ""

# ============================================================================
# STEP 2: Copy HDF5 to local /tmp on each node
# ============================================================================
echo "STEP 2: Copying data to /tmp on all nodes"
echo "  Source: $SOURCE_FILE"
echo "  Dest:   $LOCAL_FILE"
echo ""

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 sh -c "
  echo \"[\$(hostname)] Copying to /tmp...\"
  cp -f $SOURCE_FILE ${LOCAL_FILE}
  echo \"[\$(hostname)] ✓ Copy complete\"
"

echo ""
echo "✓ All nodes have data in /tmp"
echo ""

# ============================================================================
# JOB INFO
# ============================================================================
echo "================================================================================"
echo "TRAINING CONFIGURATION (v2.9 HIERARCHICAL FIX)"
echo "================================================================================"
echo "Job Configuration:"
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs:            $((SLURM_NNODES * 4))"
echo "  Batch size/GPU:  $BATCH_SIZE"
echo "  Global batch:    $((SLURM_NNODES * 4 * BATCH_SIZE))"
echo "  Max epochs:      $MAX_EPOCHS"
echo ""
echo "Hierarchical Loss Weights:"
echo "  Stage 1 (Flat vs Non-Flat):  ${STAGE1_WEIGHT}"
echo "  Stage 2 (PSPL vs Binary):    ${STAGE2_WEIGHT}"
echo "  Auxiliary (3-class CE):      ${AUX_WEIGHT}"
echo "  Stage 2 Temperature:         ${STAGE2_TEMPERATURE}"
echo ""
echo "Performance expectations:"
echo "  Batch time:    0.2-0.5 seconds"
echo "  Epoch time:    ~2 minutes"
echo "  30 epochs:     ~60-90 minutes"
echo "================================================================================"
echo ""

# ============================================================================
# CHECKPOINT HANDLING
# ============================================================================
if [ -f "${RESUME_CHECKPOINT}" ]; then
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
    echo "Resuming from epoch ${CURRENT_EPOCH}"
else
    RESUME_FLAG=""
    echo "Starting fresh training"
fi

echo ""

# ============================================================================
# STEP 3: LAUNCH TRAINING WITH HIERARCHICAL FIX
# ============================================================================
echo "================================================================================"
echo "STEP 3: TRAINING WITH v2.9 HIERARCHICAL FIX"
echo "================================================================================"
echo ""

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="train-$SLURM_JOB_ID" \
    --max-restarts=3 \
    train.py \
    --data "${LOCAL_FILE}" \
    --output "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
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
    --stage1-weight ${STAGE1_WEIGHT} \
    --stage2-weight ${STAGE2_WEIGHT} \
    --aux-weight ${AUX_WEIGHT} \
    --stage2-temperature ${STAGE2_TEMPERATURE} \
    --hierarchical \
    --use-aux-head \
    --attention-pooling \
    --save-every 2 \
    ${RESUME_FLAG}

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Training finished: exit code ${EXIT_CODE}"
echo "================================================================================"

# ============================================================================
# POST-TRAINING: Check if complete and cancel continuation job
# ============================================================================
FINAL_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")

echo ""
echo "Checkpoint status: ${FINAL_EPOCH}/${MAX_EPOCHS} epochs"

if [ "${FINAL_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
    echo ""
    echo "================================================================================"
    echo "✓ TRAINING COMPLETE!"
    echo "================================================================================"
    
    # CANCEL the continuation job since we're done
    if [ -f "${CONTINUATION_JOB_FILE}" ]; then
        CONT_JOB=$(cat "${CONTINUATION_JOB_FILE}")
        echo "Cancelling continuation job ${CONT_JOB}..."
        scancel "${CONT_JOB}" 2>/dev/null && echo "✓ Continuation job cancelled" || echo "  (already cancelled or completed)"
        rm -f "${CONTINUATION_JOB_FILE}"
    fi
    
    echo ""
    echo "Best model saved to: ${OUTPUT_DIR}/best.pt"
    echo ""
    echo "================================================================================"
    echo "NEXT STEP: EVALUATE ON TEST SET"
    echo "================================================================================"

else

    echo ""
    echo "================================================================================"
    echo "Training incomplete (${FINAL_EPOCH}/${MAX_EPOCHS} epochs)"
    echo "================================================================================"
    if [ -f "${CONTINUATION_JOB_FILE}" ]; then
        CONT_JOB=$(cat "${CONTINUATION_JOB_FILE}")
        echo "Continuation job ${CONT_JOB} will resume training automatically."
        echo ""
        echo "Monitor with: squeue -u $USER"
    fi
    echo "================================================================================"
fi

exit ${EXIT_CODE}
