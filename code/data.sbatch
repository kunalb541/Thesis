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

# =============================================================================
# ROMAN MICROLENSING - DATA GENERATION v4.0
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR="$HOME/Thesis/data"
RAW_DIR="${DATA_DIR}/raw"
TEST_DIR="${DATA_DIR}/test"
mkdir -p "${RAW_DIR}" "${TEST_DIR}" logs

# Dataset sizes
N_TRAIN_FLAT=250000
N_TRAIN_PSPL=125000
N_TRAIN_BINARY=125000

N_TEST_FLAT=50000
N_TEST_PSPL=50000
N_TEST_BINARY=50000

# Simulation parameters
OVERSAMPLE=1.3
N_WORKERS=10  

# Seeds
SEED_BASELINE_TRAIN=42
SEED_BASELINE_TEST=999
SEED_DISTINCT_TRAIN=1337
SEED_DISTINCT_TEST=7331

# ============================================================================
# HEADER
# ============================================================================

echo "================================================================================"
echo "Roman Microlensing Data Generation v4.0"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Workers: ${N_WORKERS}"
echo "================================================================================"

# ============================================================================
# GENERATE DATASETS (SEQUENTIAL)
# ============================================================================

generate_dataset() {
    local name="$1"
    local output="$2"
    local n_flat="$3"
    local n_pspl="$4"
    local n_binary="$5"
    local preset="$6"
    local seed="$7"
    
    echo ""
    echo "================================================================================"
    echo "GENERATING: ${name}"
    echo "================================================================================"
    echo "  Output: ${output}"
    echo "  Events: Flat=${n_flat}, PSPL=${n_pspl}, Binary=${n_binary}"
    echo "  Total: $((n_flat + n_pspl + n_binary)) events"
    echo "  Preset: ${preset}, Seed: ${seed}, Workers: ${N_WORKERS}"
    echo "  Started: $(date +%H:%M:%S)"
    echo "================================================================================"
    
    local start_time=$(date +%s)
    
    python -u simulate.py \
        --output "${output}" \
        --n_flat ${n_flat} \
        --n_pspl ${n_pspl} \
        --n_binary ${n_binary} \
        --binary_preset ${preset} \
        --oversample ${OVERSAMPLE} \
        --num_workers ${N_WORKERS} \
        --seed ${seed}
    
    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_h=$((elapsed / 3600))
    local elapsed_m=$(((elapsed % 3600) / 60))
    local elapsed_s=$((elapsed % 60))
    
    if [ ${exit_code} -eq 0 ]; then
        local filesize=$(du -h "${output}" | cut -f1)
        echo ""
        echo "✓ SUCCESS"
        echo "  Time: ${elapsed_h}h ${elapsed_m}m ${elapsed_s}s"
        echo "  Size: ${filesize}"
        echo "  Finished: $(date +%H:%M:%S)"
        
        # Quick validation
        python -c "
import h5py, numpy as np
with h5py.File('${output}', 'r') as f:
    n = len(f['labels'])
    labels = f['labels'][:]
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Flat', 'PSPL', 'Binary']
    print('  Validation:')
    print(f'    Total events: {n:,}')
    for cls_idx, count in zip(unique, counts):
        print(f'      {class_names[cls_idx]}: {count:,}')
    if 'm_base' in f:
        m_base = f['m_base'][:]
        print(f'    Baseline mags: [{m_base.min():.1f}, {m_base.max():.1f}] mag')
"
        echo "================================================================================"
        return 0
    else
        echo ""
        echo "✗ FAILED (exit code: ${exit_code})"
        echo "  Time: ${elapsed_h}h ${elapsed_m}m ${elapsed_s}s"
        echo "================================================================================"
        return ${exit_code}
    fi
}

# ============================================================================
# TRAINING DATASETS
# ============================================================================

echo ""
echo "PHASE 1: TRAINING DATASETS"
echo "================================================================================"

generate_dataset \
    "baseline_train" \
    "${RAW_DIR}/baseline_train.h5" \
    ${N_TRAIN_FLAT} ${N_TRAIN_PSPL} ${N_TRAIN_BINARY} \
    "baseline" ${SEED_BASELINE_TRAIN} || exit 1

generate_dataset \
    "distinct_train" \
    "${RAW_DIR}/distinct_train.h5" \
    ${N_TRAIN_FLAT} ${N_TRAIN_PSPL} ${N_TRAIN_BINARY} \
    "distinct" ${SEED_DISTINCT_TRAIN} || exit 1

# ============================================================================
# TEST DATASETS
# ============================================================================

echo ""
echo "PHASE 2: TEST DATASETS"
echo "================================================================================"

generate_dataset \
    "baseline_test" \
    "${TEST_DIR}/baseline_test.h5" \
    ${N_TEST_FLAT} ${N_TEST_PSPL} ${N_TEST_BINARY} \
    "baseline" ${SEED_BASELINE_TEST} || exit 1

generate_dataset \
    "distinct_test" \
    "${TEST_DIR}/distinct_test.h5" \
    ${N_TEST_FLAT} ${N_TEST_PSPL} ${N_TEST_BINARY} \
    "distinct" ${SEED_DISTINCT_TEST} || exit 1

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "DATA GENERATION COMPLETE"
echo "================================================================================"
echo "Finished: $(date)"
echo ""
echo "Training datasets:"
for file in "${RAW_DIR}"/*.h5; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        n=$(python -c "import h5py; print(len(h5py.File('$file','r')['labels']))" 2>/dev/null || echo "?")
        echo "  $(basename $file): ${size} (${n} events)"
    fi
done

echo ""
echo "Test datasets:"
for file in "${TEST_DIR}"/*.h5; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        n=$(python -c "import h5py; print(len(h5py.File('$file','r')['labels']))" 2>/dev/null || echo "?")
        echo "  $(basename $file): ${size} (${n} events)"
    fi
done

echo ""
echo "Total disk usage: $(du -sh "${DATA_DIR}" | cut -f1)"
echo ""
echo "================================================================================"
echo "READY FOR TRAINING!"
echo "================================================================================"

exit 0
