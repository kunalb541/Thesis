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
# Compatible with:
#   - train.py v3.0.0
#   - model.py v3.0.0
#   - simulate.py v3.0.0
#   - evaluate.py v3.0.0
#
# Key features:
#   - Hierarchical classification with separate BCE losses
#   - Auxiliary 3-class head for gradient stability
#   - Auto-continuation on timeout
#   - Automatic evaluation job submission after training completes
# =============================================================================

set -e

echo "================================================================================"
echo "ROMAN MICROLENSING - v3.0.0 TRAINING PIPELINE"
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
EXPERIMENT_NAME_FILE="${CHECKPOINT_DIR}/.experiment_name"

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

# v3.0.0: Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=1.0
AUX_WEIGHT=0.5
STAGE2_TEMPERATURE=1.0

# Evaluation settings
EVAL_BATCH_SIZE=512
EVAL_DEVICE="cuda"

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
# HELPER FUNCTIONS
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

# Find the most recent experiment directory
find_experiment_name() {
    local output_dir="$1"
    python3 -c "
import os
from pathlib import Path

results_dir = Path('${output_dir}')
if not results_dir.exists():
    print('')
    exit(0)

# Find all experiment directories (exclude 'checkpoints')
exp_dirs = [d for d in results_dir.iterdir() 
            if d.is_dir() and d.name != 'checkpoints']

if not exp_dirs:
    print('')
    exit(0)

# Sort by modification time (most recent first)
exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

# Return the name of the most recent experiment
print(exp_dirs[0].name)
" 2>/dev/null || echo ""
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
    
    # Find experiment name for evaluation
    EXPERIMENT_NAME=$(find_experiment_name "${OUTPUT_DIR}")
    
    if [ -n "${EXPERIMENT_NAME}" ]; then
        echo "Experiment found: ${EXPERIMENT_NAME}"
        echo ""
        echo "Submitting evaluation job..."
        
        # Submit evaluation job
        EVAL_JOB=$(sbatch --parsable << EVALEOF
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -o logs/eval_${EXPERIMENT_NAME}_%j.out
#SBATCH -e logs/eval_${EXPERIMENT_NAME}_%j.err
#SBATCH --job-name=eval_v3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

echo "================================================================================"
echo "EVALUATION: ${EXPERIMENT_NAME}"
echo "================================================================================"

python evaluate.py \\
    --experiment-name "${EXPERIMENT_NAME}" \\
    --data "${TEST_FILE}" \\
    --batch-size ${EVAL_BATCH_SIZE} \\
    --device ${EVAL_DEVICE} \\
    --early-detection \\
    --n-evolution-per-type 5 \\
    --save-formats png pdf \\
    --colorblind-safe

echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
EVALEOF
)
        
        if [ -n "${EVAL_JOB}" ]; then
            echo "✓ Evaluation job submitted: ${EVAL_JOB}"
        else
            echo "⚠ Failed to submit evaluation job"
            echo ""
            echo "Run manually:"
            echo "  python evaluate.py \\"
            echo "    --experiment-name ${EXPERIMENT_NAME} \\"
            echo "    --data ${TEST_FILE} \\"
            echo "    --batch-size ${EVAL_BATCH_SIZE} \\"
            echo "    --early-detection"
        fi
    else
        echo "⚠ Could not find experiment directory"
        echo "Check ${OUTPUT_DIR} for results"
    fi
    
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
echo "STEP 0: DATA GENERATION (simulate.py v3.0.0)"
echo "================================================================================"

if [ ! -f "${SOURCE_FILE}" ]; then
    echo "Training data not found. Generating on one node (100 workers)..."
    echo ""
    
    FIRST_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    
    srun --nodes=1 --ntasks=1 --nodelist=${FIRST_NODE} python simulate.py \
        --n_flat 100000 \
        --n_pspl 500000 \
        --n_binary 500000 \
        --binary_preset baseline \
        --output ${SOURCE_FILE} \
        --num_workers 100 \
        --seed 42 \
        --oversample 1.3
    
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
        --seed 999 \
        --oversample 1.3
    
    echo "✓ Test data generated: ${TEST_FILE}"
else
    echo "✓ Test data exists: ${TEST_FILE}"
fi
echo ""

# Verify data contains magnifications (not magnitudes)
echo "Verifying data format (v3.0.0 expects MAGNIFICATIONS)..."
python << 'PYEOF'
import h5py
import numpy as np
import sys

train_file = "../data/raw/train.h5"

with h5py.File(train_file, 'r') as f:
    # v3.0.0: Check for both 'flux' and 'mag' keys
    if 'flux' in f:
        flux = f['flux'][:]
        print("  Data key: 'flux'")
    elif 'mag' in f:
        flux = f['mag'][:]
        print("  Data key: 'mag'")
    else:
        print("  ERROR: No flux/mag key found!")
        sys.exit(1)
    
flux_valid = flux[flux != 0]
mean_val = np.mean(flux_valid)
print(f"  Mean value: {mean_val:.6f}")

if 0 < mean_val < 100:
    print("  ✓ Data contains MAGNIFICATIONS (correct for v3.0.0!)")
    print("    A=1.0 is baseline, A>1 is magnified")
else:
    print(f"  ⚠ Unexpected range (mean={mean_val:.2f})")
    print("  Check simulate.py v3.0.0 output format")
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
echo "TRAINING CONFIGURATION (v3.0.0)"
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
echo "Model Architecture:"
echo "  d_model:         ${D_MODEL}"
echo "  n_layers:        ${N_LAYERS}"
echo "  dropout:         ${DROPOUT}"
echo "  window_size:     ${WINDOW_SIZE}"
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
# STEP 3: LAUNCH TRAINING (train.py v3.0.0)
# ============================================================================

echo "================================================================================"
echo "STEP 3: TRAINING WITH train.py v3.0.0"
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
# POST-TRAINING: Check completion and handle evaluation
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
    
    # Find experiment name
    EXPERIMENT_NAME=$(find_experiment_name "${OUTPUT_DIR}")
    
    if [ -z "${EXPERIMENT_NAME}" ]; then
        echo "⚠ Could not determine experiment name"
        echo "Check ${OUTPUT_DIR} for results"
    else
        # Save experiment name for reference
        echo "${EXPERIMENT_NAME}" > "${EXPERIMENT_NAME_FILE}"
        
        echo ""
        echo "Experiment: ${EXPERIMENT_NAME}"
        echo "Best model: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/best_model.pt"
        echo ""
        
        # ====================================================================
        # SUBMIT EVALUATION JOB
        # ====================================================================
        echo "================================================================================"
        echo "SUBMITTING EVALUATION JOB"
        echo "================================================================================"
        
        EVAL_JOB=$(sbatch --parsable << EVALSCRIPT
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -o logs/eval_${SLURM_JOB_ID}_%j.out
#SBATCH -e logs/eval_${SLURM_JOB_ID}_%j.err
#SBATCH --job-name=eval_microlens

# =============================================================================
# EVALUATION JOB (evaluate.py v3.0.0)
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "================================================================================"
echo "ROMAN MICROLENSING - EVALUATION (v3.0.0)"
echo "================================================================================"
echo ""
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Test data:   ${TEST_FILE}"
echo "Device:      ${EVAL_DEVICE}"
echo "Batch size:  ${EVAL_BATCH_SIZE}"
echo ""
echo "================================================================================"

python evaluate.py \\
    --experiment-name "${EXPERIMENT_NAME}" \\
    --data "${TEST_FILE}" \\
    --batch-size ${EVAL_BATCH_SIZE} \\
    --device ${EVAL_DEVICE} \\
    --early-detection \\
    --n-evolution-per-type 5 \\
    --n-example-grid-per-type 4 \\
    --save-formats png pdf \\
    --colorblind-safe \\
    --verbose

EVAL_EXIT=\$?

echo ""
echo "================================================================================"
if [ \${EVAL_EXIT} -eq 0 ]; then
    echo "✓ EVALUATION COMPLETE"
    
    # Find the evaluation output directory
    EVAL_DIR=\$(ls -td ${OUTPUT_DIR}/${EXPERIMENT_NAME}/eval_* 2>/dev/null | head -1)
    
    if [ -n "\${EVAL_DIR}" ]; then
        echo ""
        echo "Results saved to: \${EVAL_DIR}"
        echo ""
        echo "Key files:"
        echo "  - evaluation_summary.json"
        echo "  - classification_report.txt"
        echo "  - confusion_matrix.png"
        echo "  - roc_curves.png"
        echo "  - early_detection_curve.png"
    fi
else
    echo "⚠ EVALUATION FAILED (exit code \${EVAL_EXIT})"
fi
echo "================================================================================"

exit \${EVAL_EXIT}
EVALSCRIPT
)
        
        if [ -n "${EVAL_JOB}" ]; then
            echo "✓ Evaluation job submitted: ${EVAL_JOB}"
            echo ""
            echo "Monitor with: squeue -u $USER"
            echo "View logs:    tail -f logs/eval_${SLURM_JOB_ID}_${EVAL_JOB}.out"
        else
            echo "⚠ Failed to submit evaluation job automatically"
            echo ""
            echo "Run manually with:"
            echo "  python evaluate.py \\"
            echo "    --experiment-name \"${EXPERIMENT_NAME}\" \\"
            echo "    --data \"${TEST_FILE}\" \\"
            echo "    --batch-size ${EVAL_BATCH_SIZE} \\"
            echo "    --device ${EVAL_DEVICE} \\"
            echo "    --early-detection \\"
            echo "    --colorblind-safe"
        fi
    fi
    
    echo ""
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
