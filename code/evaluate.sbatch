#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH --job-name=eval
#SBATCH --output=logs/eval_%x_%A_%a.out
#SBATCH --error=logs/eval_%x_%A_%a.err
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

# IMPORTANT:
# The logs/ directory MUST exist at submit time (before sbatch),
# otherwise SLURM won't write there.
# Create it where you run `sbatch` from:
#   mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# (You can keep this too; it helps for any runtime-created logs)
mkdir -p logs

DATA_DIR="$HOME/Thesis/data"
BASELINE_TEST="${DATA_DIR}/test/general_test.h5"
DISTINCT_TEST="${DATA_DIR}/test/distinct_test.h5"

OUTPUT_BASE="../results/checkpoints"

BATCH_SIZE=128
N_EVOLUTION=10
N_EXAMPLES=4
SAVE_FORMATS="png"

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

run_evaluation() {
    local exp_dir="$1"
    local test_data="$2"
    local eval_name="$3"

    log_info "========================================"
    log_info "Evaluation: ${eval_name}"
    log_info "========================================"
    log_info "  Experiment: ${exp_dir}"
    log_info "  Test data: ${test_data}"

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

modes=(b2b b2d d2d d2b)
MODE="${modes[${SLURM_ARRAY_TASK_ID}]?}"

log_info "Job: ${SLURM_JOB_ID}  ArrayTask: ${SLURM_ARRAY_TASK_ID}  Mode: ${MODE}"

BASELINE_DIR=$(get_experiment_dir "baseline")
DISTINCT_DIR=$(get_experiment_dir "distinct")

case "${MODE}" in
    b2b) [[ -n "${BASELINE_DIR}" ]] || { log_error "Baseline experiment not found"; exit 1; }
         run_evaluation "${BASELINE_DIR}" "${BASELINE_TEST}" "baseline_on_baseline" ;;
    b2d) [[ -n "${BASELINE_DIR}" ]] || { log_error "Baseline experiment not found"; exit 1; }
         run_evaluation "${BASELINE_DIR}" "${DISTINCT_TEST}" "baseline_on_distinct" ;;
    d2d) [[ -n "${DISTINCT_DIR}" ]] || { log_error "Distinct experiment not found"; exit 1; }
         run_evaluation "${DISTINCT_DIR}" "${DISTINCT_TEST}" "distinct_on_distinct" ;;
    d2b) [[ -n "${DISTINCT_DIR}" ]] || { log_error "Distinct experiment not found"; exit 1; }
         run_evaluation "${DISTINCT_DIR}" "${BASELINE_TEST}" "distinct_on_baseline" ;;
esac

log_info "DONE (${MODE})"
