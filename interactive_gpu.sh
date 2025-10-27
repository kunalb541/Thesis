#!/bin/bash
# Interactive GPU session for testing and debugging
# Usage: ./interactive_gpu.sh

echo "Requesting interactive GPU session..."
echo "This will give you a shell on a GPU node for testing"
echo ""

# Request interactive session on development H100 node
srun --partition=dev_gpu_h100 \
     --gres=gpu:1 \
     --ntasks=1 \
     --cpus-per-task=24 \
     --mem-per-gpu=193300mb \
     --time=00:30:00 \
     --pty bash -l

# After getting the session, you can:
# - conda activate microlens
# - nvidia-smi
# - cd ~/thesis-microlens/code
# - python train.py --data ../data/raw/events_1M.npz --epochs 1 --batch_size 16
