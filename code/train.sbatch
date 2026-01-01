#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err
#SBATCH --job-name=ml_pipeline
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

# =============================================================================
# ML TRAINING PIPELINE v4.0
# =============================================================================
# 
# Fixed resume logic:
# - Correctly finds timestamped experiment directories created by train.py
# - Tracks full experiment path (e.g., baseline/d32_l4_hier_20250101_143022)
# - Properly passes checkpoint path for resumption
#
# Directory structure:
#   ../results/checkpoints/
#   ├── baseline/
#   │   └── d32_l4_hier_TIMESTAMP/
#   │       ├── checkpoints/
#   │       │   └── checkpoint_latest.pt
#   │       ├── best.pt
#   │       └── config.json
#   ├── distinct/
#   │   └── d32_l4_hier_TIMESTAMP/
#   │       └── ...
#   ├── .exp_baseline    (contains: d32_l4_hier_TIMESTAMP)
#   ├── .exp_distinct    (contains: d32_l4_hier_TIMESTAMP)
#   └── .pipeline_progress
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
BATCH_SIZE=256
NUM_WORKERS=0
ACCUMULATION_STEPS=2
MAX_EPOCHS=200

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
STAGE2_WEIGHT=2.0
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
# FIND LATEST EXPERIMENT DIRECTORY
# =============================================================================
# train.py creates timestamped directories like d32_l4_hier_20250101_143022
# This function finds the most recent one in a given base directory

find_latest_experiment() {
    local base_dir="$1"
    
    if [ ! -d "${base_dir}" ]; then
        echo ""
        return
    fi
    
    # Find most recently modified directory matching pattern d*_l*_*
    local latest=$(ls -td "${base_dir}"/d*_l*_* 2>/dev/null | head -1)
    
    if [ -n "${latest}" ] && [ -d "${latest}" ]; then
        basename "${latest}"
    else
        echo ""
    fi
}

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

train_model() {
    local train_data="$1"
    local preset_name="$2"
    
    log_info "========================================"
    log_info "Training: ${preset_name}"
    log_info "========================================"
    
    # Output directory for this preset
    local preset_output="${OUTPUT_BASE}/${preset_name}"
    mkdir -p "${preset_output}"
    
    # Tracking file for this preset's experiment
    local exp_file="${OUTPUT_BASE}/.exp_${preset_name}"
    
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
    # RESUME LOGIC - Find existing experiment directory
    # =========================================================================
    
    local resume_flag=""
    local current_epoch=0
    local exp_name=""
    
    # Check if we have a tracked experiment from previous run
    if [ -f "${exp_file}" ]; then
        exp_name=$(cat "${exp_file}" | tr -d '[:space:]')
        log_info "Found tracked experiment: ${exp_name}"
    fi
    
    # If no tracked experiment, look for latest directory
    if [ -z "${exp_name}" ]; then
        exp_name=$(find_latest_experiment "${preset_output}")
        if [ -n "${exp_name}" ]; then
            log_info "Found existing experiment directory: ${exp_name}"
        fi
    fi
    
    # Check for checkpoint in experiment directory
    if [ -n "${exp_name}" ]; then
        local exp_dir="${preset_output}/${exp_name}"
        local latest_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"
        
        if [ -f "${latest_ckpt}" ]; then
            current_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
            log_info "Found checkpoint at epoch ${current_epoch}"
            
            if [ ${current_epoch} -ge ${MAX_EPOCHS} ]; then
                log_info "Training already complete (${current_epoch} >= ${MAX_EPOCHS})"
                echo "${exp_name}" > "${exp_file}"
                return 0
            else
                log_info "RESUMING from epoch ${current_epoch}"
                resume_flag="--resume ${latest_ckpt}"
            fi
        else
            log_info "No checkpoint found in ${exp_dir}"
        fi
    else
        log_info "No existing experiment found, starting fresh"
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
    log_info "  Output base: ${preset_output}"
    log_info "  Resume: ${resume_flag:-'(none)'}"
    
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --rdzv_id="train-${preset_name}-${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        train.py \
        --data "${local_train}" \
        --output "${preset_output}" \
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
    
    # =========================================================================
    # UPDATE TRACKING FILE
    # =========================================================================
    # train.py writes .current_experiment in the output directory
    # Read it to get the actual experiment name
    
    local current_exp_file="${preset_output}/.current_experiment"
    if [ -f "${current_exp_file}" ]; then
        exp_name=$(cat "${current_exp_file}" | tr -d '[:space:]')
        echo "${exp_name}" > "${exp_file}"
        log_info "Updated tracking file: ${exp_name}"
    else
        # Fallback: find latest directory
        exp_name=$(find_latest_experiment "${preset_output}")
        if [ -n "${exp_name}" ]; then
            echo "${exp_name}" > "${exp_file}"
            log_info "Updated tracking file (fallback): ${exp_name}"
        fi
    fi
    
    # =========================================================================
    # CHECK COMPLETION STATUS
    # =========================================================================
    
    if [ $exit_code -eq 0 ]; then
        local exp_dir="${preset_output}/${exp_name}"
        local final_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"
        
        if [ -f "${final_ckpt}" ]; then
            local final_epoch=$(get_checkpoint_epoch "${final_ckpt}")
            
            if [ ${final_epoch} -ge ${MAX_EPOCHS} ]; then
                log_info "Training COMPLETE: ${preset_name} (${final_epoch} epochs)"
                return 0
            else
                log_info "Training incomplete: ${final_epoch}/${MAX_EPOCHS} epochs"
                return 1
            fi
        fi
    fi
    
    log_error "Training exited with code ${exit_code}"
    return 1
}

# =============================================================================
# EVALUATION SUBMISSION
# =============================================================================

submit_evaluations() {
    local baseline_exp="$1"
    local distinct_exp="$2"
    
    log_info "Submitting evaluation jobs..."
    log_info "  Baseline experiment: ${baseline_exp}"
    log_info "  Distinct experiment: ${distinct_exp}"
    
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

EXPERIMENT_DIR="$1"
TEST_DATA="$2"
EVAL_NAME="$3"

echo "=========================================="
echo "Evaluation: ${EVAL_NAME}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="

# Check if experiment directory exists
if [ ! -d "${EXPERIMENT_DIR}" ]; then
    echo "ERROR: Experiment directory not found: ${EXPERIMENT_DIR}"
    exit 1
fi

# Check for best.pt
if [ ! -f "${EXPERIMENT_DIR}/best.pt" ]; then
    echo "ERROR: best.pt not found in ${EXPERIMENT_DIR}"
    exit 1
fi

# Check if evaluation already done
EVAL_DIR="${EXPERIMENT_DIR}/eval_$(basename ${TEST_DATA} .h5)"
if [ -d "${EVAL_DIR}" ] && [ -f "${EVAL_DIR}/evaluation_summary.json" ]; then
    echo "Already complete: ${EVAL_DIR}"
    exit 0
fi

timeout 28m python evaluate.py \
    --experiment-name "${EXPERIMENT_DIR}/best.pt" \
    --data "${TEST_DATA}" \
    --batch-size 512 \
    --n-evolution-per-type 10 \
    --save-formats png \
    --colorblind-safe \
    --device cuda

echo "Evaluation complete: ${EVAL_NAME}"
EVALSCRIPT

    chmod +x "${OUTPUT_BASE}/eval_job.sh"
    
    # Full paths to experiment directories
    local baseline_dir="${OUTPUT_BASE}/baseline/${baseline_exp}"
    local distinct_dir="${OUTPUT_BASE}/distinct/${distinct_exp}"
    
    # Submit 4 evaluation jobs
    JOB1=$(sbatch --parsable --job-name=eval_b2b \
        --output=logs/eval_b2b_%j.out --error=logs/eval_b2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_dir}" "${BASELINE_TEST}" "baseline_on_baseline")
    log_info "Submitted eval 1/4: baseline->baseline (${JOB1})"
    
    JOB2=$(sbatch --parsable --job-name=eval_b2d \
        --output=logs/eval_b2d_%j.out --error=logs/eval_b2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_dir}" "${DISTINCT_TEST}" "baseline_on_distinct")
    log_info "Submitted eval 2/4: baseline->distinct (${JOB2})"
    
    JOB3=$(sbatch --parsable --job-name=eval_d2d \
        --output=logs/eval_d2d_%j.out --error=logs/eval_d2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_dir}" "${DISTINCT_TEST}" "distinct_on_distinct")
    log_info "Submitted eval 3/4: distinct->distinct (${JOB3})"
    
    JOB4=$(sbatch --parsable --job-name=eval_d2b \
        --output=logs/eval_d2b_%j.out --error=logs/eval_d2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_dir}" "${BASELINE_TEST}" "distinct_on_baseline")
    log_info "Submitted eval 4/4: distinct->baseline (${JOB4})"
    
    # Submit summary job
    sbatch --dependency=afterok:${JOB1}:${JOB2}:${JOB3}:${JOB4} \
        --job-name=summary \
        --output=logs/summary_%j.out --error=logs/summary_%j.err \
        --time=00:10:00 -p gpu_a100_short --nodes=1 --ntasks=1 \
        --wrap="cd ~/Thesis/code && python3 << 'PYEOF'
import json
from pathlib import Path

output_base = Path('../results/checkpoints')

# Read experiment names
baseline_exp = (output_base / '.exp_baseline').read_text().strip()
distinct_exp = (output_base / '.exp_distinct').read_text().strip()

baseline_dir = output_base / 'baseline' / baseline_exp
distinct_dir = output_base / 'distinct' / distinct_exp

evaluations = [
    (baseline_dir, 'baseline_test', 'Baseline -> Baseline'),
    (baseline_dir, 'distinct_test', 'Baseline -> Distinct'),
    (distinct_dir, 'distinct_test', 'Distinct -> Distinct'),
    (distinct_dir, 'baseline_test', 'Distinct -> Baseline'),
]

print()
print('='*80)
print('PIPELINE COMPLETE - FINAL RESULTS')
print('='*80)

for exp_dir, test_set, description in evaluations:
    # Find eval directory
    eval_dirs = list(exp_dir.glob(f'eval_{test_set}*'))
    if not eval_dirs:
        print(f'[X] {description}: NOT FOUND')
        continue
    
    eval_dir = sorted(eval_dirs)[-1]
    summary_file = eval_dir / 'evaluation_summary.json'
    
    if not summary_file.exists():
        print(f'[X] {description}: NO SUMMARY')
        continue
    
    with open(summary_file) as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    acc = metrics.get('accuracy', 0) * 100
    flat_r = metrics.get('recall_Flat', 0) * 100
    pspl_r = metrics.get('recall_PSPL', 0) * 100
    binary_r = metrics.get('recall_Binary', 0) * 100
    
    print(f'[OK] {description}')
    print(f'     Accuracy: {acc:.2f}%')
    print(f'     Recall - Flat: {flat_r:.1f}%  PSPL: {pspl_r:.1f}%  Binary: {binary_r:.1f}%')
    print()

print('='*80)
PYEOF"
    
    log_info "Summary job submitted"
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
# SIGNAL HANDLER FOR PREEMPTION
# =============================================================================

handle_preemption() {
    log_info "Received preemption signal, saving state..."
    # Training script handles its own checkpointing
    # Just ensure continuation is set up
    exit 0
}

trap handle_preemption USR1

# =============================================================================
# MAIN PIPELINE
# =============================================================================

log_info "========================================"
log_info "ML TRAINING PIPELINE v6.0"
log_info "========================================"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Nodes: ${SLURM_NNODES}"
log_info "GPUs: $((SLURM_NNODES * 4))"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Max epochs: ${MAX_EPOCHS}"
log_info "Model: d${D_MODEL}_l${N_LAYERS}"
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
        
        log_info "Baseline complete, submitting next phase..."
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
    
    # Read experiment names from tracking files
    if [ ! -f "${OUTPUT_BASE}/.exp_baseline" ] || [ ! -f "${OUTPUT_BASE}/.exp_distinct" ]; then
        log_error "Missing tracking files"
        exit 1
    fi
    
    BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline" | tr -d '[:space:]')
    DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct" | tr -d '[:space:]')
    
    log_info "Baseline experiment: ${BASELINE_EXP}"
    log_info "Distinct experiment: ${DISTINCT_EXP}"
    
    submit_evaluations "${BASELINE_EXP}" "${DISTINCT_EXP}"
    mark_stage_complete "evaluations"
    
    log_info "========================================"
    log_info "EVALUATION JOBS SUBMITTED"
    log_info "Monitor with: squeue -u \$USER"
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
    
    # Show final summary
    if [ -f "${OUTPUT_BASE}/.exp_baseline" ] && [ -f "${OUTPUT_BASE}/.exp_distinct" ]; then
        BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline" | tr -d '[:space:]')
        DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct" | tr -d '[:space:]')
        log_info "Baseline: ${OUTPUT_BASE}/baseline/${BASELINE_EXP}"
        log_info "Distinct: ${OUTPUT_BASE}/distinct/${DISTINCT_EXP}"
    fi
    
    exit 0
fi

exit 0
