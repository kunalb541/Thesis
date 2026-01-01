#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err
#SBATCH --job-name=ml_pipeline_v5
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

# =============================================================================
# ML PIPELINE v5.0 - COMPLETE REWRITE WITH WORKING RESUME
# =============================================================================
#
# FIXES:
# - Proper resume logic using .exp_baseline and .exp_distinct tracking files
# - Experiment directories named with preset: baseline_model, distinct_model
# - Correct batch size (64 not 512)
# - max_seq_len not hardcoded (uses data length)
#
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR="$HOME/Thesis/data"
BASELINE_TRAIN="${DATA_DIR}/raw/baseline_train.h5"
BASELINE_TEST="${DATA_DIR}/test/baseline_test.h5"
DISTINCT_TRAIN="${DATA_DIR}/raw/distinct_train.h5"
DISTINCT_TEST="${DATA_DIR}/test/distinct_test.h5"

OUTPUT_BASE="../results/checkpoints"
mkdir -p logs "${OUTPUT_BASE}"

PROGRESS_FILE="${OUTPUT_BASE}/.pipeline_progress"

# Training hyperparameters
BATCH_SIZE=64                # Per-GPU batch (NOT 512!)
NUM_WORKERS=0                # Pure RAM loading via /dev/shm
ACCUMULATION_STEPS=2         # Effective batch = 128 per GPU
MAX_EPOCHS=300

# Model architecture
D_MODEL=32
N_LAYERS=4
DROPOUT=0.3
WINDOW_SIZE=5

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=1.0
AUX_WEIGHT=0.5
STAGE2_TEMPERATURE=1.0

USE_CLASS_WEIGHTS="false"

# =============================================================================
# ENVIRONMENT
# =============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

export LOCAL_WORLD_SIZE=4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

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

mark_stage_complete() {
    local stage="$1"
    echo "${stage}:complete:$(date +%s)" >> "${PROGRESS_FILE}"
    log_info "Stage complete: ${stage}"
}

is_stage_complete() {
    local stage="$1"
    if [ -f "${PROGRESS_FILE}" ]; then
        grep -q "^${stage}:complete:" "${PROGRESS_FILE}" && return 0
    fi
    return 1
}

# =============================================================================
# TRAINING FUNCTION v5.0 - FIXED RESUME LOGIC
# =============================================================================

train_model() {
    local train_data="$1"
    local preset_name="$2"
    
    log_info "========================================"
    log_info "Training: ${preset_name}"
    log_info "========================================"
    
    # =========================================================================
    # DATA CACHING TO /tmp
    # =========================================================================
    
    local local_train="/tmp/train_${preset_name}_${USER}.h5"
    
    if [ -f "${local_train}" ]; then
        log_info "Found cached data in /tmp"
        local source_size=$(stat -c%s "${train_data}" 2>/dev/null || echo "0")
        local cached_size=$(stat -c%s "${local_train}" 2>/dev/null || echo "0")
        
        if [ "$source_size" = "$cached_size" ] && [ "$source_size" != "0" ]; then
            log_info "Cache valid, skipping copy"
        else
            log_info "Cache invalid, re-copying..."
            srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
        fi
    else
        log_info "Copying training data to /tmp..."
        srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
    fi
    
    # =========================================================================
    # RESUME LOGIC - CHECK .exp_${preset_name} FILE
    # =========================================================================
    
    local resume_flag=""
    local current_epoch=0
    local exp_file="${OUTPUT_BASE}/.exp_${preset_name}"
    local exp_dir="${OUTPUT_BASE}/${preset_name}_model"
    
    # Check if we have a tracked experiment
    if [ -f "${exp_file}" ]; then
        local tracked_dir=$(cat "${exp_file}")
        exp_dir="${OUTPUT_BASE}/${tracked_dir}"
        log_info "Found tracked experiment: ${tracked_dir}"
    fi
    
    # Check for checkpoint in experiment directory
    local latest_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"
    
    if [ -f "${latest_ckpt}" ]; then
        current_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
        log_info "Found checkpoint at epoch ${current_epoch}"
        
        if [ ${current_epoch} -ge ${MAX_EPOCHS} ]; then
            log_info "Training already complete (${current_epoch} >= ${MAX_EPOCHS})"
            echo "$(basename ${exp_dir})" > "${exp_file}"
            return 0
        else
            log_info "RESUMING from epoch ${current_epoch}"
            resume_flag="--resume ${latest_ckpt}"
        fi
    else
        log_info "No checkpoint found, starting fresh"
    fi
    
    # =========================================================================
    # BUILD COMMAND FLAGS
    # =========================================================================
    
    local class_weights_flag=""
    if [ "${USE_CLASS_WEIGHTS}" = "false" ]; then
        class_weights_flag="--no-class-weights"
    fi
    
    # =========================================================================
    # LAUNCH DISTRIBUTED TRAINING
    # =========================================================================
    
    log_info "Launching training..."
    log_info "  GPUs: $((SLURM_NNODES * 4))"
    log_info "  Batch size: ${BATCH_SIZE}"
    log_info "  Resume flag: ${resume_flag:-'(none)'}"
    log_info "  Output: ${exp_dir}"
    
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --rdzv_id="train-${preset_name}-${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        train.py \
        --data "${local_train}" \
        --output "${exp_dir}" \
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
        --save-every 3 \
        ${class_weights_flag} \
        ${resume_flag}
    
    local exit_code=$?
    
    # Save experiment name for tracking
    echo "$(basename ${exp_dir})" > "${exp_file}"
    log_info "Saved experiment name to ${exp_file}"
    
    if [ $exit_code -eq 0 ]; then
        # Check if training is complete
        local final_epoch=$(get_checkpoint_epoch "${exp_dir}/checkpoints/checkpoint_latest.pt")
        
        if [ ${final_epoch} -ge ${MAX_EPOCHS} ]; then
            log_info "Training COMPLETE: ${preset_name} (${final_epoch} epochs)"
            return 0
        else
            log_info "Training incomplete: ${final_epoch}/${MAX_EPOCHS} epochs"
            return 1
        fi
    else
        log_error "Training failed with exit code ${exit_code}"
        return 1
    fi
}

# =============================================================================
# EVALUATION SUBMISSION
# =============================================================================

submit_evaluations() {
    local baseline_exp="$1"
    local distinct_exp="$2"
    
    log_info "Submitting evaluation jobs..."
    
    # Create evaluation script
    cat > "${OUTPUT_BASE}/eval_job.sh" << 'EVALSCRIPT'
#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

EXPERIMENT_NAME="$1"
TEST_DATA="$2"
EVAL_NAME="$3"

echo "=========================================="
echo "Evaluation: ${EVAL_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="

# Check if already done
EVAL_DIR="../results/checkpoints/${EXPERIMENT_NAME}/eval_$(basename ${TEST_DATA} .h5)"
if [ -d "${EVAL_DIR}" ] && [ -f "${EVAL_DIR}/evaluation_summary.json" ]; then
    echo "Already complete: ${EVAL_DIR}"
    exit 0
fi

timeout 28m python evaluate.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --data "${TEST_DATA}" \
    --batch-size 512 \
    --n-evolution-per-type 10 \
    --save-formats png \
    --colorblind-safe \
    --device cuda

echo "Evaluation complete: ${EVAL_NAME}"
EVALSCRIPT

    chmod +x "${OUTPUT_BASE}/eval_job.sh"
    
    # Submit 4 evaluation jobs
    JOB1=$(sbatch --parsable --job-name=eval_b2b --output=logs/eval_b2b_%j.out --error=logs/eval_b2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_exp}" "${BASELINE_TEST}" "baseline_on_baseline")
    log_info "Submitted eval 1/4: baseline->baseline (${JOB1})"
    
    JOB2=$(sbatch --parsable --job-name=eval_b2d --output=logs/eval_b2d_%j.out --error=logs/eval_b2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_exp}" "${DISTINCT_TEST}" "baseline_on_distinct")
    log_info "Submitted eval 2/4: baseline->distinct (${JOB2})"
    
    JOB3=$(sbatch --parsable --job-name=eval_d2d --output=logs/eval_d2d_%j.out --error=logs/eval_d2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_exp}" "${DISTINCT_TEST}" "distinct_on_distinct")
    log_info "Submitted eval 3/4: distinct->distinct (${JOB3})"
    
    JOB4=$(sbatch --parsable --job-name=eval_d2b --output=logs/eval_d2b_%j.out --error=logs/eval_d2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_exp}" "${BASELINE_TEST}" "distinct_on_baseline")
    log_info "Submitted eval 4/4: distinct->baseline (${JOB4})"
    
    # Submit summary job after all evals complete
    sbatch --dependency=afterok:${JOB1}:${JOB2}:${JOB3}:${JOB4} \
        --job-name=summary \
        --output=logs/summary_%j.out --error=logs/summary_%j.err \
        --time=00:10:00 -p gpu_a100_short --nodes=1 --ntasks=1 \
        --wrap="cd ~/Thesis/code && python3 << 'PYEOF'
import json
from pathlib import Path

output_base = Path('../results/checkpoints')
baseline_exp = (output_base / '.exp_baseline').read_text().strip()
distinct_exp = (output_base / '.exp_distinct').read_text().strip()

evaluations = [
    (baseline_exp, 'baseline_test', 'Baseline -> Baseline'),
    (baseline_exp, 'distinct_test', 'Baseline -> Distinct'),
    (distinct_exp, 'distinct_test', 'Distinct -> Distinct'),
    (distinct_exp, 'baseline_test', 'Distinct -> Baseline'),
]

print()
print('='*80)
print('PIPELINE COMPLETE - FINAL RESULTS')
print('='*80)

for exp, test_set, description in evaluations:
    eval_dir = output_base / exp / f'eval_{test_set}'
    summary_file = eval_dir / 'evaluation_summary.json'
    
    if not summary_file.exists():
        print(f'[X] {description}: NOT FOUND')
        continue
    
    with open(summary_file) as f:
        data = json.load(f)
    
    metrics = data.get('overall_metrics', data.get('metrics', {}))
    
    print(f'[OK] {description}')
    print(f'     Accuracy: {metrics.get(\"accuracy\", 0)*100:.2f}%')
    print(f'     Flat: {metrics.get(\"recall_Flat\", 0)*100:.1f}%  PSPL: {metrics.get(\"recall_PSPL\", 0)*100:.1f}%  Binary: {metrics.get(\"recall_Binary\", 0)*100:.1f}%')
    print()

print('='*80)
PYEOF"
    
    log_info "Summary job submitted (will run after evaluations)"
}

# =============================================================================
# AUTO-CONTINUATION
# =============================================================================

setup_continuation() {
    log_info "Setting up auto-continuation..."
    
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --kill-on-invalid-dep=yes \
        "$0")
    
    if [ -n "${CONT_JOB}" ]; then
        echo "${CONT_JOB}" > "${OUTPUT_BASE}/.continuation_job_id"
        log_info "Continuation job: ${CONT_JOB}"
    fi
}

cancel_continuation() {
    if [ -f "${OUTPUT_BASE}/.continuation_job_id" ]; then
        local cont_job=$(cat "${OUTPUT_BASE}/.continuation_job_id")
        scancel ${cont_job} 2>/dev/null || true
        rm -f "${OUTPUT_BASE}/.continuation_job_id"
        log_info "Cancelled continuation job: ${cont_job}"
    fi
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

log_info "========================================"
log_info "ML PIPELINE v5.0"
log_info "========================================"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
log_info "Batch: ${BATCH_SIZE}, Epochs: ${MAX_EPOCHS}"
log_info "========================================"

# Determine current phase
PHASE="unknown"
if ! is_stage_complete "train_baseline"; then
    PHASE="train_baseline"
elif ! is_stage_complete "train_distinct"; then
    PHASE="train_distinct"
elif ! is_stage_complete "evaluations"; then
    PHASE="evaluations"
else
    PHASE="complete"
fi

log_info "Current phase: ${PHASE}"

# =============================================================================
# PHASE 1: BASELINE TRAINING
# =============================================================================

if [ "${PHASE}" = "train_baseline" ]; then
    log_info "========================================"
    log_info "PHASE 1: BASELINE TRAINING"
    log_info "========================================"
    
    setup_continuation
    
    if train_model "${BASELINE_TRAIN}" "baseline"; then
        mark_stage_complete "train_baseline"
        cancel_continuation
        
        log_info "Baseline complete, submitting for distinct..."
        sbatch "$0"
        exit 0
    else
        log_info "Baseline incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 2: DISTINCT TRAINING
# =============================================================================

if [ "${PHASE}" = "train_distinct" ]; then
    log_info "========================================"
    log_info "PHASE 2: DISTINCT TRAINING"
    log_info "========================================"
    
    setup_continuation
    
    if train_model "${DISTINCT_TRAIN}" "distinct"; then
        mark_stage_complete "train_distinct"
        cancel_continuation
        PHASE="evaluations"
    else
        log_info "Distinct incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 3: EVALUATIONS
# =============================================================================

if [ "${PHASE}" = "evaluations" ]; then
    log_info "========================================"
    log_info "PHASE 3: EVALUATIONS"
    log_info "========================================"
    
    BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline")
    DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct")
    
    log_info "Baseline: ${BASELINE_EXP}"
    log_info "Distinct: ${DISTINCT_EXP}"
    
    submit_evaluations "${BASELINE_EXP}" "${DISTINCT_EXP}"
    mark_stage_complete "evaluations"
    
    log_info "========================================"
    log_info "PIPELINE SUBMITTED - MONITOR WITH: squeue -u \$USER"
    log_info "========================================"
    exit 0
fi

# =============================================================================
# PHASE 4: COMPLETE
# =============================================================================

if [ "${PHASE}" = "complete" ]; then
    log_info "========================================"
    log_info "PIPELINE ALREADY COMPLETE"
    log_info "========================================"
    exit 0
fi

exit 0

