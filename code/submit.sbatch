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
# ROMAN MICROLENSING CLASSIFIER - v3.0.0 TRAINING PIPELINE
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

# OUTPUT PATHS (CRITICAL FIX FOR EVALUATE.PY)
# evaluate.py looks in ../results/checkpoints, so we must save there
OUTPUT_DIR="../results/checkpoints"
mkdir -p logs "${OUTPUT_DIR}"

# TRACKING FILES
# train.py writes the specific experiment name to this file
LATEST_EXP_FILE="${OUTPUT_DIR}/.current_experiment"
CONTINUATION_JOB_FILE="${OUTPUT_DIR}/.continuation_job_id"

# Training hyperparameters
BATCH_SIZE=64
NUM_WORKERS=0 # Must be 0 for RAMLensingDataset
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

# v3.0.0: Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=1.0
AUX_WEIGHT=0.5
STAGE2_TEMPERATURE=1.0

# Evaluation settings
EVAL_BATCH_SIZE=512
EVAL_DEVICE="cuda"

# ============================================================================
# ENVIRONMENT
# ============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1

# ============================================================================
# LOGIC: DETERMINE RESUME STATE
# ============================================================================

# Check if there is a previous experiment to resume
if [ -f "${LATEST_EXP_FILE}" ]; then
    EXP_NAME=$(cat "${LATEST_EXP_FILE}")
    RESUME_CHECKPOINT="${OUTPUT_DIR}/${EXP_NAME}/checkpoints/checkpoint_latest.pt"
    
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        echo "Found previous experiment: ${EXP_NAME}"
        echo "Resuming from: ${RESUME_CHECKPOINT}"
        RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
    else
        echo "Previous experiment record found but checkpoint missing. Starting fresh."
        RESUME_FLAG=""
    fi
else
    echo "No previous experiment found. Starting fresh."
    RESUME_FLAG=""
fi

# ============================================================================
# CHECK IF TRAINING ALREADY COMPLETE
# ============================================================================

get_checkpoint_epoch() {
    local path="$1"
    if [ -f "${path}" ]; then
        python3 -c "import torch; print(torch.load('${path}', map_location='cpu', weights_only=False).get('epoch', 0))" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

if [ -n "${RESUME_FLAG}" ]; then
    CURRENT_EPOCH=$(get_checkpoint_epoch "${RESUME_CHECKPOINT}")
    if [ "${CURRENT_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
        echo "✓ Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
        # Proceed to submission logic below...
        SKIP_TRAINING=true
    fi
fi

# ============================================================================
# AUTO-CONTINUATION SETUP
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    SCRIPT_PATH="$(scontrol show job $SLURM_JOB_ID | grep -oP 'Command=\K[^ ]+')"
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --kill-on-invalid-dep=yes \
        "${SCRIPT_PATH}")
    
    if [ -n "${CONT_JOB}" ]; then
        echo "${CONT_JOB}" > "${CONTINUATION_JOB_FILE}"
        echo "✓ Continuation job submitted: ${CONT_JOB}"
    fi
fi

# ============================================================================
# STEP 0: DATA PREP
# ============================================================================

# Generate Train Data
if [ ! -f "${SOURCE_FILE}" ]; then
    echo "Generating training data..."
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 100000 --n_pspl 500000 --n_binary 500000 \
        --output ${SOURCE_FILE} --num_workers 100 --seed 42 --oversample 1.3
fi

# Generate Test Data (Critical for evaluate.py)
if [ ! -f "${TEST_FILE}" ]; then
    echo "Generating test data..."
    mkdir -p $(dirname ${TEST_FILE})
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --binary_preset distinct \
        --output ${TEST_FILE} --num_workers 100 --seed 999 --oversample 1.3
fi

# Copy to Local /tmp
echo "Copying data to local SSD /tmp..."
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f $SOURCE_FILE ${LOCAL_FILE}

# ============================================================================
# STEP 1: TRAINING
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
        --save-every 2 \
        ${RESUME_FLAG}
        
    EXIT_CODE=$?
else
    EXIT_CODE=0
fi

# ============================================================================
# STEP 2: EVALUATION SUBMISSION
# ============================================================================

# Re-read experiment name in case this was a fresh run
if [ -f "${LATEST_EXP_FILE}" ]; then
    EXP_NAME=$(cat "${LATEST_EXP_FILE}")
    CHECKPOINT_PATH="${OUTPUT_DIR}/${EXP_NAME}/best.pt"
    
    # Check if training actually finished (hit max epochs)
    FINAL_EPOCH=$(get_checkpoint_epoch "${OUTPUT_DIR}/${EXP_NAME}/checkpoints/checkpoint_latest.pt")
    
    if [ "${FINAL_EPOCH}" -ge "${MAX_EPOCHS}" ]; then
        echo "Training complete. Submitting evaluation..."
        
        # Cancel continuation job
        if [ -f "${CONTINUATION_JOB_FILE}" ]; then
            scancel $(cat "${CONTINUATION_JOB_FILE}") 2>/dev/null || true
            rm "${CONTINUATION_JOB_FILE}"
        fi

        # Submit Evaluation
        sbatch << EVALEOF
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 45:00
#SBATCH -o logs/eval_${EXP_NAME}.out
#SBATCH --job-name=eval_v3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

python evaluate.py \\
    --checkpoint "${CHECKPOINT_PATH}" \\
    --data "${TEST_FILE}" \\
    --batch-size ${EVAL_BATCH_SIZE} \\
    --early-detection \\
    --n-evolution-per-type 5 \\
    --save-formats png
EVALEOF
        echo "✓ Evaluation job submitted for ${EXP_NAME}"
    else
        echo "Training incomplete. Continuation job will handle next steps."
    fi
fi

exit $EXIT_CODE
