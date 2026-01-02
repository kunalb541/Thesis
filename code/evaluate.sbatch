#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH --job-name=eval
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR="$HOME/Thesis/data"
BASELINE_TEST="${DATA_DIR}/test/general_test.h5"
DISTINCT_TEST="${DATA_DIR}/test/distinct_test.h5"

OUTPUT_BASE="../results/checkpoints"
mkdir -p logs

# Evaluation settings
BATCH_SIZE=128
N_EVOLUTION=10
N_EXAMPLES=4
SAVE_FORMATS="png"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"; }
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
        exp_name=$(cat "${exp_file}" | tr -d '[:space:]')
    fi
    
    if [[ -z "${exp_name}" ]]; then
        exp_name=$(find_latest_experiment "${preset_dir}")
    fi
    
    if [[ -n "${exp_name}" ]]; then
        echo "${preset_dir}/${exp_name}"
    else
        echo ""
    fi
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
    
    if [[ ! -d "${exp_dir}" ]]; then
        log_error "Experiment directory not found: ${exp_dir}"
        return 1
    fi
    
    local checkpoint="${exp_dir}/best.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        checkpoint="${exp_dir}/checkpoints/checkpoint_latest.pt"
    fi
    
    if [[ ! -f "${checkpoint}" ]]; then
        log_error "No checkpoint found in ${exp_dir}"
        return 1
    fi
    
    log_info "  Checkpoint: ${checkpoint}"
    
    timeout 55m python evaluate.py \
        --experiment-name "${checkpoint}" \
        --data "${test_data}" \
        --batch-size ${BATCH_SIZE} \
        --n-evolution-per-type ${N_EVOLUTION} \
        --n-example-grid-per-type ${N_EXAMPLES} \
        --save-formats ${SAVE_FORMATS} \
        --colorblind-safe \
        --device cuda
    
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_info "Evaluation complete: ${eval_name}"
    else
        log_error "Evaluation failed: ${eval_name} (exit code ${exit_code})"
    fi
    
    return ${exit_code}
}

# =============================================================================
# MAIN
# =============================================================================

log_info "========================================"
log_info "EVALUATION PIPELINE v7.0"
log_info "========================================"
log_info "Job ID: ${SLURM_JOB_ID:-local}"

# Get experiment directories
BASELINE_DIR=$(get_experiment_dir "baseline")
DISTINCT_DIR=$(get_experiment_dir "distinct")

log_info "Baseline experiment: ${BASELINE_DIR:-NOT FOUND}"
log_info "Distinct experiment: ${DISTINCT_DIR:-NOT FOUND}"

# Determine evaluation mode from arguments or run all
EVAL_MODE="${1:-all}"

case "${EVAL_MODE}" in
    baseline_on_baseline|b2b)
        if [[ -n "${BASELINE_DIR}" ]]; then
            run_evaluation "${BASELINE_DIR}" "${BASELINE_TEST}" "baseline_on_baseline"
        else
            log_error "Baseline experiment not found"
            exit 1
        fi
        ;;
    
    baseline_on_distinct|b2d)
        if [[ -n "${BASELINE_DIR}" ]]; then
            run_evaluation "${BASELINE_DIR}" "${DISTINCT_TEST}" "baseline_on_distinct"
        else
            log_error "Baseline experiment not found"
            exit 1
        fi
        ;;
    
    distinct_on_distinct|d2d)
        if [[ -n "${DISTINCT_DIR}" ]]; then
            run_evaluation "${DISTINCT_DIR}" "${DISTINCT_TEST}" "distinct_on_distinct"
        else
            log_error "Distinct experiment not found"
            exit 1
        fi
        ;;
    
    distinct_on_baseline|d2b)
        if [[ -n "${DISTINCT_DIR}" ]]; then
            run_evaluation "${DISTINCT_DIR}" "${BASELINE_TEST}" "distinct_on_baseline"
        else
            log_error "Distinct experiment not found"
            exit 1
        fi
        ;;
    
    all)
        log_info "Running all four evaluations"
        
        FAILED=0
        
        if [[ -n "${BASELINE_DIR}" ]]; then
            run_evaluation "${BASELINE_DIR}" "${BASELINE_TEST}" "baseline_on_baseline" || ((FAILED++))
            run_evaluation "${BASELINE_DIR}" "${DISTINCT_TEST}" "baseline_on_distinct" || ((FAILED++))
        else
            log_error "Baseline experiment not found, skipping baseline evaluations"
            ((FAILED+=2))
        fi
        
        if [[ -n "${DISTINCT_DIR}" ]]; then
            run_evaluation "${DISTINCT_DIR}" "${DISTINCT_TEST}" "distinct_on_distinct" || ((FAILED++))
            run_evaluation "${DISTINCT_DIR}" "${BASELINE_TEST}" "distinct_on_baseline" || ((FAILED++))
        else
            log_error "Distinct experiment not found, skipping distinct evaluations"
            ((FAILED+=2))
        fi
        
        log_info "========================================"
        log_info "EVALUATION SUMMARY"
        log_info "========================================"
        log_info "Completed: $((4 - FAILED))/4"
        log_info "Failed: ${FAILED}/4"
        
        if [[ ${FAILED} -gt 0 ]]; then
            exit 1
        fi
        ;;
    
    summary)
        log_info "Generating summary from existing evaluations"
        
        python3 << 'PYEOF'
import json
from pathlib import Path

output_base = Path('../results/checkpoints')

def get_exp_name(preset):
    exp_file = output_base / f'.exp_{preset}'
    if exp_file.exists():
        return exp_file.read_text().strip()
    
    preset_dir = output_base / preset
    if preset_dir.exists():
        dirs = sorted(preset_dir.glob('d*_l*_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            return dirs[0].name
    return None

baseline_exp = get_exp_name('baseline')
distinct_exp = get_exp_name('distinct')

if not baseline_exp or not distinct_exp:
    print("Could not find experiment directories")
    exit(1)

baseline_dir = output_base / 'baseline' / baseline_exp
distinct_dir = output_base / 'distinct' / distinct_exp

evaluations = [
    (baseline_dir, 'baseline_test', 'Baseline -> Baseline'),
    (baseline_dir, 'distinct_test', 'Baseline -> Distinct'),
    (distinct_dir, 'distinct_test', 'Distinct -> Distinct'),
    (distinct_dir, 'baseline_test', 'Distinct -> Baseline'),
]

print()
print('=' * 70)
print('FINAL RESULTS')
print('=' * 70)

results = []

for exp_dir, test_set, description in evaluations:
    eval_dirs = sorted(exp_dir.glob(f'eval_{test_set}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not eval_dirs:
        print(f'[X] {description}: NOT FOUND')
        continue
    
    eval_dir = eval_dirs[0]
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
    
    results.append({
        'evaluation': description,
        'accuracy': acc,
        'recall_flat': flat_r,
        'recall_pspl': pspl_r,
        'recall_binary': binary_r
    })

print('=' * 70)

combined = output_base / 'final_results.json'
with open(combined, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to: {combined}')
PYEOF
        ;;
    
    *)
        log_error "Unknown evaluation mode: ${EVAL_MODE}"
        log_info "Usage: $0 [all|b2b|b2d|d2d|d2b|summary]"
        exit 1
        ;;
esac

log_info "========================================"
log_info "DONE"
log_info "========================================"

