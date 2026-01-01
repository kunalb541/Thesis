#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err
#SBATCH --job-name=ml_pipeline_v4.1
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

# =============================================================================
# OPTIMIZED ML PIPELINE v4.1.0 - CRITICAL FIXES + BROADCAST OPTIMIZATION
# =============================================================================
# 
# v4.1.0 FIXES FROM v3.2.0:
# -------------------------
# - Fixed /dev/shm cleanup race condition (barrier before delete)
# - Fixed stats/indices broadcast (rank0 computes, others receive)
# - Fixed /dev/shm path collision (was using PID, now uses SLURM_JOB_ID)
# - Fixed file suffix preservation (.npz/.h5 now preserved)
# - Fixed model API calls (lengths=keyword, tensor on device)
# - Fixed torch.load compatibility for older PyTorch versions
# 
# WHAT /dev/shm ACTUALLY PROVIDES (corrected from v3.2.0):
# --------------------------------------------------------
# - Fast RAM-backed filesystem reads (no disk I/O after copy)
# - Linux page cache sharing at OS level (reduces disk reads)
# - Each process still allocates own numpy arrays after load
# - NOT true shared memory (would need multiprocessing.shared_memory)
# 
# RESOURCE ALLOCATION:
# --------------------
# - 10 nodes x 4 GPUs = 40 GPUs total
# - RAM per node: ~14 GB per process (page cache shared, arrays duplicated)
# - Batch size: 512 per GPU
# - Accumulation: 2 (effective batch = 1024 per GPU)
# 
# EXPECTED PERFORMANCE:
# --------------------
# - Data loading: ~3-4 min (fast RAM reads from /dev/shm)
# - Throughput: 4x faster than single-GPU baseline
# - ~8-12 epochs per 30-min job
# - ~25-30 jobs to 300 epochs
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
CONTINUATION_JOB_FILE="${OUTPUT_BASE}/.continuation_job_id"

# Training hyperparameters (OPTIMIZED for 40 GPUs)
BATCH_SIZE=64                # Per-GPU batch
NUM_WORKERS=0                # Pure RAM loading via /dev/shm
ACCUMULATION_STEPS=2         # Effective batch = 1024 per GPU
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

# Evaluation settings
EVAL_BATCH_SIZE=64
N_EVOLUTION_SAMPLES=10
SAVE_FORMATS="png"

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

# v4.1.0: Export LOCAL_WORLD_SIZE for accurate node_id calculation
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
except Exception:
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
# v4.1.0 TRAINING FUNCTION WITH CRITICAL FIXES
# =============================================================================

train_model() {
    local train_data="$1"
    local preset_name="$2"
    
    log_info "Starting training: ${preset_name} (v4.1.0 - cleanup race fix + broadcast optimization)"
    
    # =========================================================================
    # SMART DATA CACHING (same as before)
    # =========================================================================
    
    local local_train="/tmp/train_${preset_name}_${USER}.h5"
    
    if [ -f "${local_train}" ]; then
        log_info "Found cached data in /tmp, verifying..."
        
        local source_size=$(stat -c%s "${train_data}" 2>/dev/null || stat -f%z "${train_data}" 2>/dev/null || echo "0")
        local cached_size=$(stat -c%s "${local_train}" 2>/dev/null || stat -f%z "${local_train}" 2>/dev/null || echo "0")
        
        if [ "$source_size" = "$cached_size" ] && [ "$source_size" != "0" ]; then
            log_info "Cache valid (${cached_size} bytes), skipping copy! [SAVED ~3 min]"
        else
            log_info "Cache size mismatch, re-copying..."
            srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
        fi
    else
        log_info "No cache found, copying training data to /tmp..."
        srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
    fi
    
    # Check for existing experiment
    local resume_flag=""
    local current_epoch=0
    
    for dir in "${OUTPUT_BASE}"/*${preset_name}*; do
        if [ -d "$dir" ]; then
            local latest_ckpt="${dir}/checkpoints/checkpoint_latest.pt"
            if [ -f "${latest_ckpt}" ]; then
                current_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
                if [ ${current_epoch} -ge ${MAX_EPOCHS} ]; then
                    log_info "Training already complete at epoch ${current_epoch}"
                    local exp_name=$(basename "$dir")
                    echo "${exp_name}" > "${OUTPUT_BASE}/.exp_${preset_name}"
                    return 0
                else
                    log_info "Resuming from epoch ${current_epoch}"
                    resume_flag="--resume ${latest_ckpt}"
                    local exp_name=$(basename "$dir")
                    echo "${exp_name}" > "${OUTPUT_BASE}/.exp_${preset_name}"
                    break
                fi
            fi
        fi
    done
    
    # Build class weights flag
    local class_weights_flag=""
    if [ "${USE_CLASS_WEIGHTS}" = "false" ]; then
        class_weights_flag="--no-class-weights"
    fi
    
    # =========================================================================
    # LAUNCH TRAINING with v4.1.0
    # CRITICAL: --nproc-per-node=4 (4 GPUs per node)
    # train.py v4.1.0 includes:
    #   - Cleanup race fix (barrier before /dev/shm delete)
    #   - Broadcast optimization (rank0 computes stats, broadcasts to others)
    #   - Accurate /dev/shm documentation
    # =========================================================================
    
    log_info "Launching training with train.py v4.1.0..."
    log_info "  40 GPUs total (10 nodes x 4 GPUs)"
    log_info "  /dev/shm benefit: Fast RAM reads, page cache shared at OS level"
    log_info "  v4.1.0 fixes: cleanup race, broadcast optimization, accurate docs"
    
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=4 \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv-id="train-${preset_name}-$SLURM_JOB_ID" \
        train.py \
        --data "${local_train}" \
        --output "${OUTPUT_BASE}" \
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
        --save-every 3 \
        ${class_weights_flag} \
        ${resume_flag}
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local exp_name=""
        if [ -f "${OUTPUT_BASE}/.current_experiment" ]; then
            exp_name=$(cat "${OUTPUT_BASE}/.current_experiment")
        fi
        
        if [ -n "${exp_name}" ]; then
            local final_ckpt="${OUTPUT_BASE}/${exp_name}/checkpoints/checkpoint_latest.pt"
            local final_epoch=$(get_checkpoint_epoch "${final_ckpt}")
            
            if [ ${final_epoch} -ge ${MAX_EPOCHS} ]; then
                log_info "Training complete: ${exp_name} (${final_epoch} epochs)"
                echo "${exp_name}" > "${OUTPUT_BASE}/.exp_${preset_name}"
                return 0
            else
                log_info "Training in progress: ${final_epoch}/${MAX_EPOCHS} epochs"
                return 1
            fi
        else
            log_error "Could not find experiment name"
            return 1
        fi
    else
        log_error "Training failed with exit code ${exit_code}"
        return ${exit_code}
    fi
}

# =============================================================================
# SUBMIT EVALUATION JOBS (same as before)
# =============================================================================

submit_evaluations() {
    local baseline_exp="$1"
    local distinct_exp="$2"
    
    log_info "Submitting evaluation jobs..."
    
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

echo "Evaluation: ${EVAL_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Test data: ${TEST_DATA}"

EVAL_DIR="../results/checkpoints/${EXPERIMENT_NAME}/eval_$(basename ${TEST_DATA} .h5)_"
EXISTING=$(ls -d ${EVAL_DIR}* 2>/dev/null | head -1)

if [ -n "${EXISTING}" ] && [ -f "${EXISTING}/evaluation_summary.json" ]; then
    echo "Already complete: ${EXISTING}"
    exit 0
fi

timeout 28m python evaluate.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --data "${TEST_DATA}" \
    --batch-size 1024 \
    --n-evolution-per-type 10 \
    --save-formats png \
    --colorblind-safe \
    --device cuda

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Evaluation complete: ${EVAL_NAME}"
elif [ ${EXIT_CODE} -eq 124 ]; then
    echo "TIMEOUT"
    exit 1
else
    echo "Failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
EVALSCRIPT

    chmod +x "${OUTPUT_BASE}/eval_job.sh"
    
    local eval_jobs=()
    
    JOB1=$(sbatch --parsable --job-name=eval_b2b --output=logs/eval_b2b_%j.out --error=logs/eval_b2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_exp}" "${BASELINE_TEST}" "baseline_on_baseline")
    eval_jobs+=("${JOB1}")
    log_info "Submitted eval 1/4: ${JOB1}"
    
    JOB2=$(sbatch --parsable --job-name=eval_b2d --output=logs/eval_b2d_%j.out --error=logs/eval_b2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${baseline_exp}" "${DISTINCT_TEST}" "baseline_on_distinct")
    eval_jobs+=("${JOB2}")
    log_info "Submitted eval 2/4: ${JOB2}"
    
    JOB3=$(sbatch --parsable --job-name=eval_d2d --output=logs/eval_d2d_%j.out --error=logs/eval_d2d_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_exp}" "${DISTINCT_TEST}" "distinct_on_distinct")
    eval_jobs+=("${JOB3}")
    log_info "Submitted eval 3/4: ${JOB3}"
    
    JOB4=$(sbatch --parsable --job-name=eval_d2b --output=logs/eval_d2b_%j.out --error=logs/eval_d2b_%j.err \
        "${OUTPUT_BASE}/eval_job.sh" "${distinct_exp}" "${BASELINE_TEST}" "distinct_on_baseline")
    eval_jobs+=("${JOB4}")
    log_info "Submitted eval 4/4: ${JOB4}"
    
    local dependency=""
    for job in "${eval_jobs[@]}"; do
        if [ -z "${dependency}" ]; then
            dependency="afterok:${job}"
        else
            dependency="${dependency}:${job}"
        fi
    done
    
    sbatch --dependency=${dependency} --job-name=gen_summary \
        --output=logs/summary_%j.out --error=logs/summary_%j.err \
        --time=00:05:00 --nodes=1 --ntasks=1 \
        --wrap="cd ~/Thesis/code && bash -c '$(cat << 'SUMSCRIPT'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

echo "Generating final summary..."
python3 << 'PYEOF'
import json
from pathlib import Path

output_base = Path("../results/checkpoints")
baseline_exp = (output_base / ".exp_baseline").read_text().strip()
distinct_exp = (output_base / ".exp_distinct").read_text().strip()

evaluations = [
    (baseline_exp, "baseline_test", "Baseline -> Baseline"),
    (baseline_exp, "distinct_test", "Baseline -> Distinct"),
    (distinct_exp, "distinct_test", "Distinct -> Distinct"),
    (distinct_exp, "baseline_test", "Distinct -> Baseline"),
]

print("\n" + "="*80)
print("PIPELINE COMPLETE - FINAL RESULTS (v4.1.0)")
print("="*80)

for exp, test_set, description in evaluations:
    pattern = f"eval_{test_set}_*"
    eval_dirs = list((output_base / exp).glob(pattern))
    
    if not eval_dirs:
        print(f"\n[X] {description}: NOT FOUND")
        continue
    
    eval_dir = sorted(eval_dirs)[-1]
    summary_file = eval_dir / "evaluation_summary.json"
    
    if not summary_file.exists():
        print(f"\n[X] {description}: INCOMPLETE")
        continue
    
    with open(summary_file) as f:
        data = json.load(f)
    
    metrics = data.get("overall_metrics", data.get("metrics", {}))
    
    print(f"\n[OK] {description}")
    print(f"  Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Recall - Flat: {metrics.get('recall_Flat', 0)*100:.1f}%")
    print(f"  Recall - PSPL: {metrics.get('recall_PSPL', 0)*100:.1f}%")
    print(f"  Recall - Binary: {metrics.get('recall_Binary', 0)*100:.1f}%")
    print(f"  Location: {eval_dir}")

print("\n" + "="*80)
print("v4.1.0: Critical fixes applied (cleanup race, broadcast optimization)")
print("="*80)
PYEOF

echo "Summary generation complete!"
SUMSCRIPT
)'"
    
    log_info "Summary generator will run after evaluations complete"
}

# =============================================================================
# AUTO-CONTINUATION
# =============================================================================

setup_continuation() {
    log_info "Setting up auto-continuation..."
    
    SCRIPT_PATH="$0"
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --kill-on-invalid-dep=yes \
        "${SCRIPT_PATH}")
    
    if [ -n "${CONT_JOB}" ]; then
        echo "${CONT_JOB}" > "${CONTINUATION_JOB_FILE}"
        log_info "Continuation job submitted: ${CONT_JOB}"
    else
        log_error "Failed to submit continuation job"
    fi
}

handle_timeout() {
    log_info "TIMEOUT WARNING: Job will be killed soon"
}

trap 'handle_timeout' USR1

# =============================================================================
# MAIN PIPELINE LOGIC
# =============================================================================

log_info "================================================================================"
log_info "OPTIMIZED ML PIPELINE v4.1.0 - CRITICAL FIXES APPLIED"
log_info "================================================================================"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Configuration:"
log_info "  Nodes: ${SLURM_NNODES}"
log_info "  GPUs per node: 4"
log_info "  Total GPUs: $((SLURM_NNODES * 4))"
log_info "  Batch size per GPU: ${BATCH_SIZE}"
log_info "  Accumulation steps: ${ACCUMULATION_STEPS}"
log_info "  Effective batch per GPU: $((BATCH_SIZE * ACCUMULATION_STEPS))"
log_info "v4.1.0 fixes:"
log_info "  - Cleanup race condition (barrier before /dev/shm delete)"
log_info "  - Broadcast optimization (rank0 computes stats, broadcasts)"
log_info "  - Deterministic /dev/shm path (uses SLURM_JOB_ID, not PID)"
log_info "  - Model API compatibility (lengths=keyword, tensor on device)"
log_info "================================================================================"

PHASE="unknown"

if ! is_stage_complete "train_baseline"; then
    PHASE="train_baseline"
elif ! is_stage_complete "train_distinct"; then
    PHASE="train_distinct"
elif ! is_stage_complete "evaluations_submitted"; then
    PHASE="submit_evals"
else
    PHASE="complete"
fi

log_info "Current phase: ${PHASE}"

# =============================================================================
# PHASE 1: TRAIN BASELINE
# =============================================================================

if [ "${PHASE}" = "train_baseline" ]; then
    log_info "================================================================================"
    log_info "PHASE 1: Training BASELINE model (v4.1.0)"
    log_info "================================================================================"
    
    setup_continuation
    
    if train_model "${BASELINE_TRAIN}" "baseline"; then
        mark_stage_complete "train_baseline"
        BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline")
        log_info "Baseline training complete: ${BASELINE_EXP}"
        
        if [ -f "${CONTINUATION_JOB_FILE}" ]; then
            scancel $(cat "${CONTINUATION_JOB_FILE}") 2>/dev/null || true
            rm "${CONTINUATION_JOB_FILE}"
        fi
        
        log_info "Re-submitting for distinct training..."
        sbatch "$0"
        exit 0
    else
        log_info "Baseline training incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 2: TRAIN DISTINCT
# =============================================================================

if [ "${PHASE}" = "train_distinct" ]; then
    log_info "================================================================================"
    log_info "PHASE 2: Training DISTINCT model (v4.1.0)"
    log_info "================================================================================"
    
    setup_continuation
    
    if train_model "${DISTINCT_TRAIN}" "distinct"; then
        mark_stage_complete "train_distinct"
        DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct")
        log_info "Distinct training complete: ${DISTINCT_EXP}"
        
        if [ -f "${CONTINUATION_JOB_FILE}" ]; then
            scancel $(cat "${CONTINUATION_JOB_FILE}") 2>/dev/null || true
            rm "${CONTINUATION_JOB_FILE}"
        fi
        
        PHASE="submit_evals"
    else
        log_info "Distinct training incomplete, continuation will resume"
        exit 0
    fi
fi

# =============================================================================
# PHASE 3: SUBMIT EVALUATIONS
# =============================================================================

if [ "${PHASE}" = "submit_evals" ]; then
    log_info "================================================================================"
    log_info "PHASE 3: Submitting evaluation jobs"
    log_info "================================================================================"
    
    BASELINE_EXP=$(cat "${OUTPUT_BASE}/.exp_baseline")
    DISTINCT_EXP=$(cat "${OUTPUT_BASE}/.exp_distinct")
    
    log_info "Baseline experiment: ${BASELINE_EXP}"
    log_info "Distinct experiment: ${DISTINCT_EXP}"
    
    submit_evaluations "${BASELINE_EXP}" "${DISTINCT_EXP}"
    
    mark_stage_complete "evaluations_submitted"
    
    log_info "================================================================================"
    log_info "TRAINING COMPLETE - EVALUATIONS SUBMITTED"
    log_info "================================================================================"
    log_info "Monitor with: squeue -u \$USER"
    log_info "================================================================================"
    exit 0
fi

# =============================================================================
# PHASE 4: COMPLETE
# =============================================================================

if [ "${PHASE}" = "complete" ]; then
    log_info "================================================================================"
    log_info "PIPELINE ALREADY COMPLETE"
    log_info "================================================================================"
    log_info "All training and evaluations finished"
    log_info "Check results in: ${OUTPUT_BASE}"
    log_info "================================================================================"
    exit 0
fi

exit 0
