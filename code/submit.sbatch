#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_complete_%j.out
#SBATCH -e logs/train_complete_%j.err
#SBATCH --job-name=microlens_flux
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# =============================================================================
# AUTO-RESUMPTION LOGIC FIX
# =============================================================================
# Problem: When SLURM kills job due to walltime, EXIT_CODE != 0, so resubmit
#          logic was never triggered.
# 
# Solution:
# 1. Trap SIGUSR1 (sent 120s before timeout via --signal=B:USR1@120)
# 2. Resubmit based on checkpoint state, NOT exit code
# 3. Handle both successful completion and timeout scenarios
# =============================================================================

set -e

# =============================================================================
# SIGNAL HANDLING FOR GRACEFUL TIMEOUT
# =============================================================================
TIMEOUT_TRIGGERED=0

handle_timeout() {
    echo ""
    echo "================================================================================"
    echo "⚠️  TIMEOUT WARNING: Job will be killed in ~120 seconds"
    echo "================================================================================"
    echo "Waiting for current epoch to finish and checkpoint to save..."
    TIMEOUT_TRIGGERED=1
    # Don't exit - let training finish current epoch and save checkpoint
}

handle_term() {
    echo ""
    echo "================================================================================"
    echo "⚠️  SIGTERM received - job is being terminated"
    echo "================================================================================"
}

# Trap SIGUSR1 (sent by SLURM 120s before timeout)
trap 'handle_timeout' USR1
# Trap SIGTERM (sent when job is actually killed)
trap 'handle_term' TERM

echo "================================================================================"
echo "ROMAN MICROLENSING - COMPLETE PIPELINE (DATA GEN + TRAINING)"
echo "================================================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# CRITICAL: Set thread limits (fixes multiprocessing hang)
# ============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# ============================================================================
# CONFIGURATION
# ============================================================================
SOURCE_FILE="$HOME/Thesis/data/raw/train_1M_distinct.h5"
TEST_FILE="$HOME/Thesis/data/test/test_300k_distinct.h5"
LOCAL_FILE="/tmp/train_1M_distinct_${USER}.h5"
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"

# Training hyperparameters
BATCH_SIZE=64
NUM_WORKERS=0
ACCUMULATION_STEPS=2
MAX_EPOCHS=30

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
# CHECK IF TRAINING ALREADY COMPLETE
# ============================================================================
mkdir -p logs "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

CURRENT_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")

if [ "${CURRENT_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
    echo "✓ Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
    echo ""
    echo "To restart training, delete: ${RESUME_CHECKPOINT}"
    echo ""
    echo "To run evaluation:"
    echo "  python evaluate.py \\"
    echo "    --experiment-name test_300k_flux \\"
    echo "    --data ${TEST_FILE} \\"
    echo "    --batch-size 512 \\"
    echo "    --early-detection"
    exit 0
fi

echo "Current progress: ${CURRENT_EPOCH}/${MAX_EPOCHS} epochs"
echo ""

# ============================================================================
# STEP 0: GENERATE FRESH DATA (if needed)
# ============================================================================
echo "================================================================================"
echo "STEP 0: DATA GENERATION"
echo "================================================================================"

if [ ! -f "${SOURCE_FILE}" ]; then
    echo "Training data not found. Generating on one node (32 workers)..."
    echo ""
    
    # Use first node for data generation
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 333334 \
        --n_pspl 333333 \
        --n_binary 333333 \
        --binary_preset distinct \
        --output ${SOURCE_FILE} \
        --num_workers 32 \
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
        --num_workers 32 \
        --seed 999
    
    echo "✓ Test data generated: ${TEST_FILE}"
else
    echo "✓ Test data exists: ${TEST_FILE}"
fi
echo ""

# Verify data contains flux (not magnitudes)
echo "Verifying data contains FLUX (not magnitudes)..."
python << 'PYEOF'
import h5py
import numpy as np
import sys

with h5py.File("../data/raw/train_1M_distinct.h5", 'r') as f:
    flux = f['flux'][:]
    
flux_valid = flux[flux != 0]
mean_val = np.mean(flux_valid)

print(f"  Mean flux: {mean_val:.6f}")

if 0 < mean_val < 10:
    print("  ✓ Data contains FLUX (correct!)")
else:
    print(f"  ❌ Data contains MAGNITUDES (mean={mean_val:.2f})")
    print("  ERROR: Regenerate data with fixed simulate.py!")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Data verification failed!"
    exit 1
fi

echo ""
echo "================================================================================"

# ============================================================================
# STEP 1: CLEAN OLD DATA FROM /tmp
# ============================================================================
echo "STEP 1: Cleaning old data from /tmp on all nodes..."
# Use --ntasks=$SLURM_NNODES to ensure exactly 1 task per node
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 rm -f ${LOCAL_FILE}
echo "✓ Old data deleted"
echo ""

# ============================================================================
# STEP 2: Copy FRESH HDF5 to local /tmp on each node
# ============================================================================
echo "STEP 2: Copying FRESH data to /tmp on all nodes"
echo "  Source: $SOURCE_FILE"
echo "  Dest:   $LOCAL_FILE"
echo ""

# Use --ntasks=$SLURM_NNODES to ensure exactly 1 task per node
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 sh -c "
  echo \"[\$(hostname)] Copying to /tmp...\"
  cp -f $SOURCE_FILE ${LOCAL_FILE}
  echo \"[\$(hostname)] ✓ Copy complete\"
"

echo ""
echo "✓ All nodes have FRESH FLUX data in /tmp"
echo ""

# ============================================================================
# JOB INFO
# ============================================================================
echo "================================================================================"
echo "TRAINING CONFIGURATION"
echo "================================================================================"
echo "Job Configuration:"
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs:            $((SLURM_NNODES * 4))"
echo "  Batch size/GPU:  $BATCH_SIZE"
echo "  Global batch:    $((SLURM_NNODES * 4 * BATCH_SIZE))"
echo "  Max epochs:      $MAX_EPOCHS"
echo ""
echo "Performance expectations:"
echo "  Batch time:    0.2-0.5 seconds"
echo "  Epoch time:    ~2 minutes"
echo "  100 epochs:    ~3-4 hours"
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
# STEP 3: LAUNCH TRAINING
# ============================================================================
echo "================================================================================"
echo "STEP 3: TRAINING WITH FLUX DATA"
echo "================================================================================"
echo ""

# Run training in background so we can handle signals
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
    ${RESUME_FLAG} --hierarchical --attention-pooling --save-every 2 &

# Store PID and wait
TRAIN_PID=$!
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Training process finished: exit code ${EXIT_CODE}"
echo "================================================================================"

# ============================================================================
# AUTO-RESUBMIT LOGIC (FIXED)
# ============================================================================
# Key fix: Check checkpoint state, NOT exit code
# Training may exit with non-zero code due to SLURM timeout, but checkpoint
# should still be valid for resumption.
# ============================================================================

FINAL_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")

echo ""
echo "Checkpoint status:"
echo "  Epochs completed: ${FINAL_EPOCH}/${MAX_EPOCHS}"

if [ "${FINAL_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
    # Training complete!
    echo ""
    echo "================================================================================"
    echo "✓ TRAINING COMPLETE! (${FINAL_EPOCH} epochs)"
    echo "================================================================================"
    echo ""
    echo "Best model saved to: ${OUTPUT_DIR}/best.pt"
    echo ""
    echo "================================================================================"
    echo "NEXT STEP: EVALUATE ON TEST SET"
    echo "================================================================================"
    echo ""
    echo "Run evaluation with:"
    echo ""
    echo "  cd ~/Thesis/code"
    echo "  python evaluate.py \\"
    echo "    --experiment-name test_300k_flux \\"
    echo "    --data ${TEST_FILE} \\"
    echo "    --batch-size 512 \\"
    echo "    --early-detection"
    echo ""
    echo "Expected results with FLUX normalization:"
    echo "  Overall accuracy: 92-96% (vs 33.33% before!)"
    echo "  Flat   → Flat:   95%+ (vs 0.075% before!)"
    echo "  PSPL   → PSPL:   93%+ (vs 17.5% before!)"
    echo "  Binary → Binary: 99%+ (maintained)"
    echo ""
    echo "================================================================================"
    exit 0

elif [ -f "${RESUME_CHECKPOINT}" ]; then
    # Training incomplete but checkpoint exists - RESUBMIT
    echo ""
    echo "================================================================================"
    echo "⟳ RESUBMITTING FOR REMAINING EPOCHS"
    echo "================================================================================"
    echo "  Progress: ${FINAL_EPOCH}/${MAX_EPOCHS} epochs"
    echo "  Remaining: $((MAX_EPOCHS - FINAL_EPOCH)) epochs"
    echo ""
    
    # Get the script path (handles both direct run and sbatch submission)
    SCRIPT_PATH="${SLURM_JOB_SCRIPT:-$0}"
    
    # Resubmit the job
    NEW_JOB=$(sbatch --parsable "${SCRIPT_PATH}")
    SUBMIT_STATUS=$?
    
    if [ ${SUBMIT_STATUS} -eq 0 ]; then
        echo "  ✓ New job submitted: ${NEW_JOB}"
        echo ""
        echo "Monitor with: squeue -u $USER"
        echo "View logs:    tail -f logs/train_complete_${NEW_JOB}.out"
    else
        echo "  ❌ Failed to submit new job (exit code: ${SUBMIT_STATUS})"
        echo ""
        echo "Manually resubmit with:"
        echo "  sbatch ${SCRIPT_PATH}"
    fi
    echo "================================================================================"
    
    # Exit with the original training exit code
    exit ${EXIT_CODE}

else
    # No checkpoint exists and training failed - don't resubmit
    echo ""
    echo "================================================================================"
    echo "❌ TRAINING FAILED - No checkpoint found"
    echo "================================================================================"
    echo "Exit code: ${EXIT_CODE}"
    echo ""
    echo "Check error logs: logs/train_complete_${SLURM_JOB_ID}.err"
    echo "================================================================================"
    exit ${EXIT_CODE}
fi
