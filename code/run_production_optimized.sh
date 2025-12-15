#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH -t 00:30:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens
cd ~/Thesis/code

DATA_FILE="../data/raw/train_1M_distinct.h5"
OUTPUT_DIR="../results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_latest.pt"

# OPTIMIZED SETTINGS
BATCH_SIZE=16            
NUM_WORKERS=2            
PREFETCH_FACTOR=2       
ACCUMULATION_STEPS=1       
MAX_EPOCHS=100

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

if [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "▶ Resuming from checkpoint"
    RESUME_FLAG="--resume ${RESUME_CHECKPOINT}"
else
    echo "▶ Starting fresh training"
    RESUME_FLAG=""
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
fi

mkdir -p logs

echo "================================================================================"
echo "TRAINING - OPTIMIZED"
echo "  Output: ${OUTPUT_DIR}"
echo "  Batch: ${BATCH_SIZE}/GPU (768 total), Workers: ${NUM_WORKERS}, Prefetch: ${PREFETCH_FACTOR}"
echo "================================================================================"

srun torchrun \
    --nnodes=12 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostname $SLURM_NODELIST | head -n1):29500 \
    --rdzv_id=$SLURM_JOB_ID \
    train.py \
    --data "${DATA_FILE}" \
    --output "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --prefetch-factor ${PREFETCH_FACTOR} \
    --accumulation-steps ${ACCUMULATION_STEPS} \
    --epochs ${MAX_EPOCHS} \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --warmup-epochs 5 \
    --use-class-weights \
    --use-prefetcher \
    --compile \
    --save-every 5 \
    ${RESUME_FLAG}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ] || [ -f "${RESUME_CHECKPOINT}" ]; then
    CURRENT_EPOCH=$(python3 -c "import torch; print(torch.load('${RESUME_CHECKPOINT}', map_location='cpu')['epoch'])" 2>/dev/null || echo "0")
    if [ ${CURRENT_EPOCH} -lt ${MAX_EPOCHS} ]; then
        echo "⟳ Resubmitting (epoch ${CURRENT_EPOCH}/${MAX_EPOCHS})"
        sbatch "$0"
    fi
fi

exit ${EXIT_CODE}
