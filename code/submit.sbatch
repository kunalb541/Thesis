#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 30:00
#SBATCH -o logs/train_complete_%j.out
#SBATCH -e logs/train_complete_%j.err
#SBATCH --job-name=microlens_v31
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL


# =============================================================================
# ROMAN MICROLENSING - v3.1.0 TRAINING PIPELINE
# =============================================================================
# v3.1.0 UPDATES:
#   - 72-day Roman season (was 200 days)
#   - 15-minute cadence = 6912 observations (was 2400)
#   - Added --no-class-weights option for realistic distribution
#   - Global m_base array in HDF5 for correct magnitude display
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
BATCH_SIZE=1024
NUM_WORKERS=0 # Must be 0 for RAMLensingDataset
ACCUMULATION_STEPS=3
MAX_EPOCHS=50

# Model architecture
D_MODEL=64
N_LAYERS=4
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

# v3.1.0: Class weights option
# Set to "true" to use inverse-frequency class weights (balanced training)
# Set to "false" for no class weights (realistic distribution: more flats)
USE_CLASS_WEIGHTS="false"

# Evaluation settings
EVAL_BATCH_SIZE=1024
EVAL_DEVICE="cuda"

# ============================================================================
# v3.1.0: SIMULATION PARAMETERS (72-day Roman season)
# ============================================================================
# Single Roman observing season parameters:
#   - 72 days (Sun angle constraints limit Galactic bulge visibility)
#   - 15-minute cadence = 6912 observations per season
#   - Events constrained to complete rise-peak-fall within window
# ============================================================================

# Dataset sizes (adjust based on memory/compute)
# With 6912 points per event (3x more than before), reduce counts if needed
N_FLAT_TRAIN=1000000
N_PSPL_TRAIN=500000
N_BINARY_TRAIN=500000

N_FLAT_TEST=50000
N_PSPL_TEST=50000
N_BINARY_TEST=50000

# Binary preset (distinct/baseline/planetary/stellar)
BINARY_PRESET="distinct"

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
            echo "Training already complete (${CURRENT_EPOCH}/${MAX_EPOCHS} epochs)"
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
        echo "Continuation job submitted: ${CONT_JOB}"
    else
        echo "Warning: Failed to submit continuation job"
    fi
fi

# Signal Handler
handle_timeout() {
    echo ""
    echo "TIMEOUT WARNING: Job will be killed soon."
    echo "Continuation job ${CONT_JOB} will take over."
}
trap 'handle_timeout' USR1

# ============================================================================
# STEP 0: DATA GENERATION (v3.1.0 - 72-day Roman season)
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    # Generate Train Data (Using 100 workers)
    if [ ! -f "${SOURCE_FILE}" ]; then
        echo "================================================================================"
        echo "GENERATING TRAINING DATA (v3.1.0 - 72-day season, 15-min cadence)"
        echo "================================================================================"
        FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
        srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
            --n_flat ${N_FLAT_TRAIN} \
            --n_pspl ${N_PSPL_TRAIN} \
            --n_binary ${N_BINARY_TRAIN} \
            --binary_preset ${BINARY_PRESET} \
            --output ${SOURCE_FILE} \
            --num_workers 100 \
            --seed 42 \
            --oversample 1.3
    fi

    # Copy to Local /tmp
    echo "Copying data to local SSD /tmp..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f $SOURCE_FILE ${LOCAL_FILE}
fi

# Always check Test Data (Needed for evaluation even if training is skipped)
if [ ! -f "${TEST_FILE}" ]; then
    echo "================================================================================"
    echo "GENERATING TEST DATA (v3.1.0 - 72-day season, 15-min cadence)"
    echo "================================================================================"
    mkdir -p $(dirname ${TEST_FILE})
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat ${N_FLAT_TEST} \
        --n_pspl ${N_PSPL_TEST} \
        --n_binary ${N_BINARY_TEST} \
        --binary_preset ${BINARY_PRESET} \
        --output ${TEST_FILE} \
        --num_workers 100 \
        --seed 999 \
        --oversample 1.3
fi

# ============================================================================
# STEP 1: TRAINING (v3.1.0 - with optional class weights)
# ============================================================================

if [ "$SKIP_TRAINING" != "true" ]; then
    echo "================================================================================"
    echo "STARTING TRAINING (v3.1.0)"
    echo "  Class weights: ${USE_CLASS_WEIGHTS}"
    echo "  Binary preset: ${BINARY_PRESET}"
    echo "================================================================================"
    
    # Build class weights flag
    CLASS_WEIGHTS_FLAG=""
    if [ "${USE_CLASS_WEIGHTS}" = "false" ]; then
        CLASS_WEIGHTS_FLAG="--no-class-weights"
    fi
    
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
        ${CLASS_WEIGHTS_FLAG} \
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
            echo "TRAINING COMPLETE (${FINAL_EPOCH}/${MAX_EPOCHS})"
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
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/eval_${EXP_NAME}.out
#SBATCH --job-name=eval_v31
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

python evaluate.py \\
    --experiment-name  "${EXP_NAME}" \\
    --data "${TEST_FILE}" \\
    --batch-size ${EVAL_BATCH_SIZE} \\
    --n-evolution-per-type 10 \\
    --save-formats png
EVALEOF
            echo "Evaluation job submitted for ${EXP_NAME}"
        else
            echo "Training incomplete (${FINAL_EPOCH}/${MAX_EPOCHS}). Continuation job will resume."
        fi
    fi
else
    echo "Training failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE
