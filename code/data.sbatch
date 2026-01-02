#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/datagen_%j.out
#SBATCH -e logs/datagen_%j.err
#SBATCH --job-name=datagen
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

DATA_DIR="$HOME/Thesis/data"
RAW_DIR="$DATA_DIR/raw"
TEST_DIR="$DATA_DIR/test"
LOG_DIR="$HOME/Thesis/code/logs"
mkdir -p "$RAW_DIR" "$TEST_DIR" "$LOG_DIR"

N_TRAIN_FLAT=250000
N_TRAIN_PSPL=125000
N_TRAIN_BINARY=125000

N_TEST_FLAT=100000
N_TEST_PSPL=100000
N_TEST_BINARY=100000

OVERSAMPLE=1.5
N_WORKERS=32

SEED_GENERAL_TRAIN=42
SEED_GENERAL_TEST=999
SEED_DISTINCT_TRAIN=1337
SEED_DISTINCT_TEST=7331

run_one() {
  local out="$1"
  local n_flat="$2"
  local n_pspl="$3"
  local n_bin="$4"
  local preset="$5"
  local seed="$6"

  python -u simulate.py \
    --output "$out" \
    --n_flat "$n_flat" \
    --n_pspl "$n_pspl" \
    --n_binary "$n_bin" \
    --binary_preset "$preset" \
    --oversample "$OVERSAMPLE" \
    --num_workers "$N_WORKERS" \
    --seed "$seed"
}

run_one "$RAW_DIR/general_train.h5"   "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" "general"  "$SEED_GENERAL_TRAIN"
run_one "$RAW_DIR/distinct_train.h5"  "$N_TRAIN_FLAT" "$N_TRAIN_PSPL" "$N_TRAIN_BINARY" "distinct" "$SEED_DISTINCT_TRAIN"
run_one "$TEST_DIR/general_test.h5"   "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  "general"  "$SEED_GENERAL_TEST"
run_one "$TEST_DIR/distinct_test.h5"  "$N_TEST_FLAT"  "$N_TEST_PSPL"  "$N_TEST_BINARY"  "distinct" "$SEED_DISTINCT_TEST"
