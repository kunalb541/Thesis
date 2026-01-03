#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o logs/train_only_%j.out
#SBATCH -e logs/train_only_%j.err
#SBATCH --job-name=train
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR="$HOME/Thesis/data"
BASELINE_TRAIN="${DATA_DIR}/raw/general_train.h5"
DISTINCT_TRAIN="${DATA_DIR}/raw/distinct_train.h5"

OUTPUT_BASE="../results/checkpoints"
mkdir -p logs "${OUTPUT_BASE}"

PROGRESS_FILE="${OUTPUT_BASE}/.train_progress"

# Training hyperparameters
BATCH_SIZE=256
NUM_WORKERS=0
ACCUMULATION_STEPS=2
MAX_EPOCHS=50

# Model architecture
D_MODEL=32
N_LAYERS=4
DROPOUT=0.3
WINDOW_SIZE=5
NUM_ATTENTION_HEADS=4
NUM_GROUPS=8

# Optimization
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=3
CLIP_NORM=1.0

# Hierarchical loss weights
STAGE1_WEIGHT=1.0
STAGE2_WEIGHT=0.8
AUX_WEIGHT=1
NLL_WEIGHT=2
STAGE2_TEMPERATURE=1.0

USE_CLASS_WEIGHTS="true"   # set "true" to enable class weights

# =============================================================================
# ENV
# =============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# Suppress PyTorch/NCCL warnings
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN                          # Keep WARN, not INFO

# FIX: Use new TORCH_ prefixed variables instead of deprecated NCCL_ ones
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1        # Replaces NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_BLOCKING_WAIT=1               # Replaces NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING                 # Remove deprecated
unset NCCL_BLOCKING_WAIT                        # Remove deprecated

# Remove the expandable_segments warning
unset PYTORCH_CUDA_ALLOC_CONF                   # Remove unsupported option

# Keep these
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

export LOCAL_WORLD_SIZE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Suppress Python warnings at runtime
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# =============================================================================
# UTIL
# =============================================================================
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

mark_stage_complete() {
  local stage="$1"
  echo "${stage}:complete:$(date +%s)" >> "${PROGRESS_FILE}"
  log_info "Stage complete: ${stage}"
}

is_stage_complete() {
  local stage="$1"
  [[ -f "${PROGRESS_FILE}" ]] && grep -q "^${stage}:complete:" "${PROGRESS_FILE}"
}

get_checkpoint_epoch() {
  local checkpoint_path="$1"
  if [[ -f "${checkpoint_path}" ]]; then
    python3 -c "
import torch
try:
  ckpt = torch.load('${checkpoint_path}', map_location='cpu', weights_only=False)
  print(int(ckpt.get('epoch', 0)))
except Exception:
  print(0)
" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

find_latest_experiment() {
  local base_dir="$1"
  [[ -d "${base_dir}" ]] || { echo ""; return; }
  local latest
  latest=$(ls -td "${base_dir}"/d*_l*_* 2>/dev/null | head -1 || true)
  [[ -n "${latest}" && -d "${latest}" ]] && basename "${latest}" || echo ""
}

# =============================================================================
# CLEANUP
# =============================================================================
cleanup_tmp() {
  log_info "Cleaning /tmp..."
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    rm -f /tmp/train_*_'${USER}'.h5 2>/dev/null || true
    echo "Node $(hostname): /tmp cleaned"
  '
}
cleanup_shm() {
  log_info "Cleaning /dev/shm..."
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    rm -f /dev/shm/*_job'${SLURM_JOB_ID}'_* 2>/dev/null || true
    echo "Node $(hostname): /dev/shm cleaned"
  '
}
cleanup_all() {
  log_info "Running cleanup..."
  cleanup_tmp
  cleanup_shm
  log_info "Cleanup complete"
}
trap cleanup_all EXIT

# =============================================================================
# AUTO-CONTINUATION
# =============================================================================
setup_continuation() {
  log_info "Setting up continuation job..."
  CONT_JOB=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --kill-on-invalid-dep=yes "$0")
  [[ -n "${CONT_JOB}" ]] && echo "${CONT_JOB}" > "${OUTPUT_BASE}/.continuation_job_id" && log_info "Continuation job: ${CONT_JOB}"
}
cancel_continuation() {
  if [[ -f "${OUTPUT_BASE}/.continuation_job_id" ]]; then
    local cont_job
    cont_job=$(cat "${OUTPUT_BASE}/.continuation_job_id")
    scancel "${cont_job}" 2>/dev/null || true
    rm -f "${OUTPUT_BASE}/.continuation_job_id"
    log_info "Cancelled continuation job: ${cont_job}"
  fi
}
handle_preemption() {
  log_info "Received USR1 preemption signal; exiting so continuation can resume."
  exit 0
}
trap handle_preemption USR1

# =============================================================================
# TRAIN FUNCTION
# =============================================================================
train_model() {
  local train_data="$1"
  local preset_name="$2"

  log_info "========================================"
  log_info "TRAINING: ${preset_name}"
  log_info "========================================"

  local preset_output="${OUTPUT_BASE}/${preset_name}"
  mkdir -p "${preset_output}"

  local exp_file="${OUTPUT_BASE}/.exp_${preset_name}"
  local local_train="/tmp/train_${preset_name}_${USER}.h5"

  # Copy train data to /tmp on each node
  if [[ -f "${local_train}" ]]; then
    log_info "Found cached ${local_train}"
  else
    log_info "Copying ${train_data} -> ${local_train} on all nodes..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -f "${train_data}" "${local_train}"
  fi

  # Resume logic
  local resume_flag=""
  local output_dir_flag=""
  local exp_name=""

  if [[ -f "${exp_file}" ]]; then
    exp_name=$(cat "${exp_file}" | tr -d '[:space:]')
    [[ -n "${exp_name}" ]] && log_info "Tracked experiment: ${exp_name}"
  fi

  if [[ -z "${exp_name}" ]]; then
    exp_name=$(find_latest_experiment "${preset_output}")
    [[ -n "${exp_name}" ]] && log_info "Found existing experiment: ${exp_name}" && echo "${exp_name}" > "${exp_file}"
  fi

  if [[ -n "${exp_name}" ]]; then
    local exp_dir="${preset_output}/${exp_name}"
    local latest_ckpt="${exp_dir}/checkpoints/checkpoint_latest.pt"

    if [[ -f "${latest_ckpt}" ]]; then
      local current_epoch
      current_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
      log_info "Found checkpoint at epoch ${current_epoch}"

      if [[ "${current_epoch}" -ge "${MAX_EPOCHS}" ]]; then
        log_info "Already complete (${current_epoch} >= ${MAX_EPOCHS})"
        return 0
      fi

      log_info "RESUMING from epoch ${current_epoch}"
      resume_flag="--resume ${latest_ckpt}"
      output_dir_flag="--output-dir ${exp_dir}"
    else
      log_info "No checkpoint in ${exp_dir}; continuing in same dir"
      output_dir_flag="--output-dir ${exp_dir}"
    fi
  else
    log_info "No existing experiment found; starting fresh"
  fi

  local class_weights_flag=""
  [[ "${USE_CLASS_WEIGHTS}" == "false" ]] && class_weights_flag="--no-class-weights"

  log_info "Launching torchrun..."
  log_info "  Nodes: ${SLURM_NNODES}, GPUs total: $((SLURM_NNODES*4))"
  log_info "  Output: ${preset_output}"
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
        ${output_dir_flag} \
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
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --num-groups ${NUM_GROUPS} \
        --stage1-weight ${STAGE1_WEIGHT} \
        --stage2-weight ${STAGE2_WEIGHT} \
        --aux-weight ${AUX_WEIGHT} \
        --nll-weight ${NLL_WEIGHT} \
        --stage2-temperature ${STAGE2_TEMPERATURE} \
        --hierarchical \
        --use-aux-head \
        --use-attention-pooling \
        --save-every 3 \
        ${class_weights_flag} \
        ${resume_flag}

  # update tracking file from .current_experiment if present
  local current_exp_file="${preset_output}/.current_experiment"
  if [[ -f "${current_exp_file}" ]]; then
    exp_name=$(cat "${current_exp_file}" | tr -d '[:space:]')
    echo "${exp_name}" > "${exp_file}"
    log_info "Updated tracking: ${preset_name} -> ${exp_name}"
  fi

  return 0
}

# =============================================================================
# MAIN
# =============================================================================
log_info "========================================"
log_info "TRAIN-ONLY PIPELINE"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Nodes: ${SLURM_NNODES}"
log_info "========================================"

PHASE="unknown"
if ! is_stage_complete "train_baseline"; then
  PHASE="train_baseline"
elif ! is_stage_complete "train_distinct"; then
  PHASE="train_distinct"
else
  PHASE="complete"
fi

log_info "Phase: ${PHASE}"

if [[ "${PHASE}" == "train_baseline" ]]; then
  setup_continuation
  train_model "${BASELINE_TRAIN}" "baseline"
  mark_stage_complete "train_baseline"
  cancel_continuation
  sbatch "$0"
  exit 0
fi

if [[ "${PHASE}" == "train_distinct" ]]; then
  setup_continuation
  train_model "${DISTINCT_TRAIN}" "distinct"
  mark_stage_complete "train_distinct"
  cancel_continuation
  exit 0
fi

if [[ "${PHASE}" == "complete" ]]; then
  log_info "========================================"
  log_info "TRAINING COMPLETE (baseline + distinct)"
  log_info "========================================"
  if [[ -f "${OUTPUT_BASE}/.exp_baseline" ]]; then
    log_info "Baseline exp:  $(cat "${OUTPUT_BASE}/.exp_baseline")"
  fi
  if [[ -f "${OUTPUT_BASE}/.exp_distinct" ]]; then
    log_info "Distinct exp:  $(cat "${OUTPUT_BASE}/.exp_distinct")"
  fi
  exit 0
fi
