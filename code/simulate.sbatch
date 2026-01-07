#!/bin/bash
# =============================================================================
# SIMULATE.SBATCH v4.1.0 - Parallel Data Generation
# =============================================================================
# Generates training and test datasets for microlensing classification.
# 
# Datasets:
#   - general: Full parameter space (baseline preset)
#   - distinct: Forced caustic crossings (distinct preset)
#
# Usage:
#   sbatch simulate.sbatch
# =============================================================================
#SBATCH -p dev_cpu_il
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --exclusive
#SBATCH -t 30:00
#SBATCH --job-name=datagen
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/hd/hd_hd/hd_vm305/Thesis/code
#SBATCH -o logs/datagen_%j.out
#SBATCH -e logs/datagen_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# =============================================================================
# DIRECTORIES
# =============================================================================
DATA_DIR="$HOME/Thesis/data"
TRAIN_DIR="$DATA_DIR/train"
TEST_DIR="$DATA_DIR/test"
LOG_DIR="$HOME/Thesis/code/logs"
mkdir -p "$TRAIN_DIR" "$TEST_DIR" "$LOG_DIR"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Training: 100K per class = 300K total per dataset
N_TRAIN_FLAT=100000
N_TRAIN_PSPL=100000
N_TRAIN_BINARY=100000

# Test: 100K per class = 300K total per dataset
N_TEST_FLAT=100000
N_TEST_PSPL=100000
N_TEST_BINARY=100000

# Generation settings
OVERSAMPLE=1.2
N_WORKERS=100

# Seeds for reproducibility
SEED_GENERAL_TRAIN=42
SEED_GENERAL_TEST=999
SEED_DISTINCT_TRAIN=1337
SEED_DISTINCT_TEST=7331

# =============================================================================
# START
# =============================================================================
echo "============================================================"
echo "PARALLEL DATA GENERATION"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Started: $(date)"
echo "============================================================"

# Print simulate.py version
python - <<'PY' || true
try:
    import simulate
    print(f"simulate.py version: {getattr(simulate, '__version__', 'UNKNOWN')}")
except Exception as e:
    print(f"simulate.py version: UNKNOWN ({e})")
PY
echo ""

# =============================================================================
# PARALLEL EXECUTION
# =============================================================================
run_step() {
    local name="$1"
    local out="$2"
    local n_flat="$3"
    local n_pspl="$4"
    local n_bin="$5"
    local preset="$6"
    local seed="$7"

    echo "[$(date '+%H:%M:%S')] Launching: $name -> $out"

    srun --exclusive -N 1 -n 1 -c 32 --mem=128G \
        --job-name="$name" \
        -o "logs/${name}_%j.out" -e "logs/${name}_%j.err" \
        python -u simulate.py \
            --output "$out" \
            --n_flat "$n_flat" \
            --n_pspl "$n_pspl" \
            --n_binary "$n_bin" \
            --binary_preset "$preset" \
            --oversample "$OVERSAMPLE" \
            --num_workers "$N_WORKERS" \
            --seed "$seed" &
}

# Launch 4 parallel jobs (1 per node)
run_step datagen_general_train  "$TRAIN_DIR/general_train.h5"  "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" baseline "$SEED_GENERAL_TRAIN"
run_step datagen_distinct_train "$TRAIN_DIR/distinct_train.h5" "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" distinct "$SEED_DISTINCT_TRAIN"
run_step datagen_general_test   "$TEST_DIR/general_test.h5"    "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  baseline "$SEED_GENERAL_TEST"
run_step datagen_distinct_test  "$TEST_DIR/distinct_test.h5"   "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  distinct "$SEED_DISTINCT_TEST"

# Wait for all jobs
wait

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "DATA GENERATION COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Training datasets:"
ls -lh "$TRAIN_DIR"/*.h5 2>/dev/null || echo "  (none found)"
echo ""
echo "Test datasets:"
ls -lh "$TEST_DIR"/*.h5 2>/dev/null || echo "  (none found)"
