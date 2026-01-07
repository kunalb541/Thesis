#!/bin/bash
# =============================================================================
# EVALUATE.SBATCH v4.1.0 - Cross-Evaluation Pipeline
# =============================================================================
# Runs 4 cross-evaluations as a SLURM array job:
#   - g2g: general model on general test
#   - g2d: general model on distinct test
#   - d2d: distinct model on distinct test
#   - d2g: distinct model on general test
#
# Usage:
#   sbatch evaluate.sbatch
# =============================================================================
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH --job-name=eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

mkdir -p logs

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR="$HOME/Thesis/data"
GENERAL_TEST="${DATA_DIR}/test/general_test.h5"
DISTINCT_TEST="${DATA_DIR}/test/distinct_test.h5"

OUTPUT_BASE="../results/checkpoints"

BATCH_SIZE=128
N_EVOLUTION=5
N_EXAMPLES=4
SAVE_FORMATS="png"

# =============================================================================
# UTILITIES
# =============================================================================
log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

find_latest_experiment() {
    local base_dir="$1"
    [[ -d "${base_dir}" ]] || { echo ""; return; }
    local latest
    latest=$(ls -td "${base_dir}"/d*_l*_* 2>/dev/null | head -1 || true)
    [[ -n "${latest}" && -d "${latest}" ]] && basename "${latest}" || echo ""
}

get_experiment_dir() {
    local preset="$1"
    local preset_dir="${OUTPUT_BASE}/${preset}"
    local exp_file="${OUTPUT_BASE}/.exp_${preset}"

    local exp_name=""
    if [[ -f "${exp_file}" ]]; then
        exp_name=$(tr -d '[:space:]' < "${exp_file}")
    fi

    if [[ -z "${exp_name}" ]]; then
        exp_name=$(find_latest_experiment "${preset_dir}")
    fi

    [[ -n "${exp_name}" ]] && echo "${preset_dir}/${exp_name}" || echo ""
}

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
run_evaluation() {
    local exp_dir="$1"
    local test_data="$2"
    local eval_name="$3"

    log_info "========================================"
    log_info "Evaluation: ${eval_name}"
    log_info "========================================"
    log_info "  Experiment: ${exp_dir}"
    log_info "  Test data:  ${test_data}"

    [[ -d "${exp_dir}" ]] || { log_error "Experiment directory not found: ${exp_dir}"; return 1; }

    local checkpoint="${exp_dir}/best.pt"
    [[ -f "${checkpoint}" ]] || checkpoint="${exp_dir}/checkpoints/checkpoint_latest.pt"
    [[ -f "${checkpoint}" ]] || { log_error "No checkpoint found in ${exp_dir}"; return 1; }

    log_info "  Checkpoint: ${checkpoint}"

    timeout 28m python evaluate.py \
        --experiment-name "${checkpoint}" \
        --data "${test_data}" \
        --batch-size ${BATCH_SIZE} \
        --n-evolution-per-type ${N_EVOLUTION} \
        --n-example-grid-per-type ${N_EXAMPLES} \
        --save-formats ${SAVE_FORMATS} \
        --colorblind-safe \
        --device cuda
}

# =============================================================================
# MAIN
# =============================================================================
# Array mapping: 0=g2g, 1=g2d, 2=d2d, 3=d2g
modes=(g2g g2d d2d d2g)
MODE="${modes[${SLURM_ARRAY_TASK_ID}]?}"

log_info "========================================"
log_info "EVALUATION PIPELINE v4.1.0"
log_info "Job: ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_TASK_ID}  Mode: ${MODE}"
log_info "========================================"

GENERAL_DIR=$(get_experiment_dir "general")
DISTINCT_DIR=$(get_experiment_dir "distinct")

case "${MODE}" in
    g2g)
        [[ -n "${GENERAL_DIR}" ]] || { log_error "General experiment not found"; exit 1; }
        run_evaluation "${GENERAL_DIR}" "${GENERAL_TEST}" "general_on_general"
        ;;
    g2d)
        [[ -n "${GENERAL_DIR}" ]] || { log_error "General experiment not found"; exit 1; }
        run_evaluation "${GENERAL_DIR}" "${DISTINCT_TEST}" "general_on_distinct"
        ;;
    d2d)
        [[ -n "${DISTINCT_DIR}" ]] || { log_error "Distinct experiment not found"; exit 1; }
        run_evaluation "${DISTINCT_DIR}" "${DISTINCT_TEST}" "distinct_on_distinct"
        ;;
    d2g)
        [[ -n "${DISTINCT_DIR}" ]] || { log_error "Distinct experiment not found"; exit 1; }
        run_evaluation "${DISTINCT_DIR}" "${GENERAL_TEST}" "distinct_on_general"
        ;;
esac

log_info "========================================"
log_info "DONE: ${MODE}"
log_info "========================================"
