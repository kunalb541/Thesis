#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --job-name=datagen_parallel
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/hd/hd_hd/hd_vm305/Thesis/code
#SBATCH -o logs/datagen_parallel_%j.out
#SBATCH -e logs/datagen_parallel_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Prevent thread oversubscription (important when you already parallelize at SLURM level)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# Directory setup
DATA_DIR="$HOME/Thesis/data"
RAW_DIR="$DATA_DIR/raw"
TEST_DIR="$DATA_DIR/test"
LOG_DIR="$HOME/Thesis/code/logs"
mkdir -p "$RAW_DIR" "$TEST_DIR" "$LOG_DIR"

# =============================================================================
# Dataset sizes
# =============================================================================
N_TRAIN_FLAT=200000
N_TRAIN_PSPL=200000
N_TRAIN_BINARY=125000

N_TEST_FLAT=100000
N_TEST_PSPL=100000
N_TEST_BINARY=100000

OVERSAMPLE=1.5
N_WORKERS=32

# Seeds
SEED_GENERAL_TRAIN=42
SEED_GENERAL_TEST=999
SEED_DISTINCT_TRAIN=1337
SEED_DISTINCT_TEST=7331

echo "Starting PARALLEL data generation at $(date)"
python - <<'PY' || true
try:
    import simulate
    print("simulate.py version:", getattr(simulate, "__version__", "UNKNOWN"))
except Exception as e:
    print("simulate.py version: UNKNOWN (import failed)", e)
PY
echo ""

run_step () {
  local name="$1"
  local out="$2"
  local n_flat="$3"
  local n_pspl="$4"
  local n_bin="$5"
  local preset="$6"
  local seed="$7"

  echo "Launching step: $name -> $out"

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

# 4 parallel steps (each grabs 1 node exclusively)
run_step datagen_general_train   "$RAW_DIR/general_train.h5"   "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" general  "$SEED_GENERAL_TRAIN"
run_step datagen_distinct_train  "$RAW_DIR/distinct_train.h5"  "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" distinct "$SEED_DISTINCT_TRAIN"
run_step datagen_general_test    "$TEST_DIR/general_test.h5"   "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  general  "$SEED_GENERAL_TEST"
run_step datagen_distinct_test   "$TEST_DIR/distinct_test.h5"  "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  distinct "$SEED_DISTINCT_TEST"

# Wait for all background srun steps to finish
wait

echo "============================================================"
echo "All 4 datasets generated successfully at $(date)"
echo "============================================================"
echo "Output files:"
ls -lh "$RAW_DIR"/*.h5 "$TEST_DIR"/*.h5 2>/dev/null || true
