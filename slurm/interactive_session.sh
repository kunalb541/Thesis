#!/bin/bash
# Quick script to start interactive GPU session and test code

echo "=========================================="
echo "Starting Interactive GPU Session"
echo "=========================================="

# Cancel any existing jobs
scancel -u hd_vm305

# Request interactive session
echo "Requesting 4x AMD MI300 GPUs for 8 hours..."
salloc --partition=gpu_mi300 --gres=gpu:4 --cpus-per-gpu=24 --mem-per-gpu=128200mb --time=8:00:00
