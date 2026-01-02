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
