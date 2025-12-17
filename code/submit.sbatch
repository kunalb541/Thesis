#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_complete_%j.out
#SBATCH -e logs/train_complete_%j.err
#SBATCH --job-name=microlens_v3
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120

# =============================================================================
# ROMAN MICROLENSING - v3.0.0 ROBUST PIPELINE
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# ============================================================================
# CONFIGURATION
# ============================================================================

# DATA PATHS
SOURCE_FILE="$HOME/Thesis/data/raw/train.h5"
TEST_FILE="$HOME/Thesis/data/test/test.h5"
LOCAL_FILE="/tmp/train_${USER}.h5"

# OUTPUT PATHS (Fixed for evaluate.py compatibility)
OUTPUT_DIR="../results/checkpoints"
mkdir -p logs "${OUTPUT_DIR}"

# TRACKING FILES
LATEST_EXP_FILE="${OUTPUT_DIR}/.current_experiment"
CONTINUATION_JOB_FILE="${OUTPUT_DIR}/.continuation_job_id"

# Training hyperparameters
BATCH_SIZE=256
NUM_WORKERS=0 # Must be 0 for RAMLensingDataset
ACCUMULATION_STEPS=2
MAX_EPOCHS=300

# Model architecture
D_MODEL=128
N_LAYERS=6
DROPOUT=0.3
WINDOW_SIZE=7

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# v3.0.0: Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=1.0
AUX_WEIGHT=1.0
STAGE2_TEMPERATURE=1.0

# Evaluation settings
EVAL_BATCH_SIZE=512
EVAL_DEVICE="cuda"

# ============================================================================
# CRITICAL: THREAD LIMITS (Prevent Freezing)
# ============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# ============================================================================
# ENVIRONMENT & NETWORK
# ============================================================================
# PYTHONUNBUFFERED=1 ensures you see simulation progress bars in real-time
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# ============================================================================
# HELPER: Get Checkpoint Epoch
# ============================================================================
get_checkpoint_epoch() {
    local checkpoint_path="$1"
    if [ -f "${checkpoint_path}" ]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('${checkpoint_path}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ============================================================================
# LOGIC: DETERMINE RESUME STATE
# ============================================================================

RESUME_FLAG=""
CURRENT_EPOCH=0

# Check if we have a record of a running experiment
if [ -f "${LATEST_EXP_FILE}" ]; then
    EXP_NAME=$(cat "${LATEST_EXP_FILE}")
    RESUME_CHECKPOINT="${OUTPUT_DIR}/${EXP_NAME}/checkpoints/checkpoint_latest.pt"
    
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        echo "Found previous experiment: ${EXP_NAME}"
        CURRENT_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")
        
        if [ "${CURRENT_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
            echo "✓ Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
            SKIP_TRAINING=true
        else
            echo "Resuming from epoch ${CURRENT_EPOCH}"
            RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
        fi
    else
        echo "Previous experiment record found but checkpoint missing. Starting fresh."
    fi
else
    echo "No previous experiment found. Starting fresh."
fi

# ============================================================================
# AUTO-CONTINUATION SETUP
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    echo "================================================================================"
    echo "SETTING UP AUTO-CONTINUATION"
    echo "================================================================================"

    SCRIPT_PATH="$(scontrol show job $SLURM_JOB_ID | grep -oP 'Command=\K[^ ]+')"
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --kill-on-invalid-dep=yes \
        "${SCRIPT_PATH}")

    if [ -n "${CONT_JOB}" ]; then
        echo "${CONT_JOB}" > "${CONTINUATION_JOB_FILE}"
        echo "✓ Continuation job submitted: ${CONT_JOB}"
    else
        echo "⚠ Warning: Failed to submit continuation job"
    fi
fi

# Signal Handler
handle_timeout() {
    echo ""
    echo "⚠️ TIMEOUT WARNING: Job will be killed soon."
    echo "Continuation job ${CONT_JOB} will take over."
}
trap 'handle_timeout' USR1

# ============================================================================
# STEP 0: DATA GENERATION
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    # Generate Train Data (Using SAFE worker count 32)
    if [ ! -f "${SOURCE_FILE}" ]; then
        echo "Generating training data..."
        FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
        srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
            --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
            --output ${SOURCE_FILE} --num_workers 100 --seed 42 --oversample 1.3
    fi

    # Copy to Local /tmp
    echo "Copying data to local SSD /tmp..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f $SOURCE_FILE ${LOCAL_FILE}
fi

# Always check Test Data (Needed for evaluation even if training is skipped)
if [ ! -f "${TEST_FILE}" ]; then
    echo "Generating test data..."
    mkdir -p $(dirname ${TEST_FILE})
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --binary_preset distinct \
        --output ${TEST_FILE} --num_workers 100 --seed 999 --oversample 1.3
fi

# ============================================================================
# STEP 1: TRAINING (Updated Flags for v3.0.0)
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    echo "Starting training..."
    
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=4 \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv-id="train-$SLURM_JOB_ID" \
        train.py \
        --data "${LOCAL_FILE}" \
        --output "${OUTPUT_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --accumulation-steps ${ACCUMULATION_STEPS} \
        --epochs ${MAX_EPOCHS} \
        --lr ${LEARNING_RATE} \
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
        --save-every 5 \
        ${RESUME_FLAG}
        
    EXIT_CODE=$?
else
    EXIT_CODE=0
fi

# ============================================================================
# STEP 2: POST-TRAINING CHECKS & EVALUATION
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    # Re-verify completion by checking the actual checkpoint
    if [ -f "${LATEST_EXP_FILE}" ]; then
        EXP_NAME=$(cat "${LATEST_EXP_FILE}")
        CHECKPOINT_PATH="${OUTPUT_DIR}/${EXP_NAME}/best.pt"
        LATEST_PATH="${OUTPUT_DIR}/${EXP_NAME}/checkpoints/checkpoint_latest.pt"
        
        # Check if we actually reached the target epoch
        FINAL_EPOCH=$(get_checkpoint_epoch "${LATEST_PATH}")
        
        if [ "${FINAL_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
            echo ""
            echo "================================================================================"
            echo "✓ TRAINING COMPLETE (${FINAL_EPOCH}/${MAX_EPOCHS})"
            echo "================================================================================"
            
            # 1. Cancel continuation job (so it doesn't restart us)
            if [ -f "${CONTINUATION_JOB_FILE}" ]; then
                scancel $(cat "${CONTINUATION_JOB_FILE}") 2>/dev/null || true
                rm "${CONTINUATION_JOB_FILE}"
            fi

            # 2. AUTO-SUBMIT EVALUATION JOB
            echo "Submitting evaluation job..."
            sbatch << EVALEOF
#!/bin/bash
#SBATCH -p gpu_h100_il
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o logs/eval_${EXP_NAME}.out
#SBATCH --job-name=eval_v3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

python evaluate.py \\
    --experiment-name  "${EXP_NAME}" \\
    --data "${TEST_FILE}" \\
    --batch-size ${EVAL_BATCH_SIZE} \\
    --n-evolution-per-type 10 \\
    --early-detection  \\
    --save-formats png
EVALEOF
            echo "✓ Evaluation job submitted for ${EXP_NAME}"
        else
            echo "Training incomplete (${FINAL_EPOCH}/${MAX_EPOCHS}). Continuation job will resume."
        fi
    fi
else
    echo "Training failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE
