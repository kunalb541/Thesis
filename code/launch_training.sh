#!/bin/bash
# launch_training.sh

cd ~/Thesis/code
mkdir -p logs

echo "================================================================================"
echo "Microlensing Auto-Resume Training"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  • Partition: gpu_a100_short (30 min slots)"
echo "  • Nodes: 10 (40 GPUs)"
echo "  • Auto-resume: Yes"
echo "  • Target: 100 epochs"
echo ""
echo "This will automatically:"
echo "  1. Generate 1M training + 300K test data (once)"
echo "  2. Train for ~15 epochs per 30-min job"
echo "  3. Auto-resume from latest checkpoint"
echo "  4. Resubmit until 100 epochs complete"
echo "  5. Run evaluation when done"
echo ""
echo "Monitor with:"
echo "  watch -n 30 'squeue -u \$USER'"
echo "  tail -f ~/Thesis/code/logs/train_*.out"
echo ""
echo "Stop training:"
echo "  scancel -u \$USER"
echo ""
echo "================================================================================"

read -p "Launch training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Submit first job
JOB_ID=$(sbatch --parsable run_autoresume.sh)

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ Training launched!"
echo "================================================================================"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/train_${JOB_ID}.out"
echo ""
echo "Progress:"
echo "  ls -lh ../results/production_1M_distinct/epoch_*.pt"
echo "================================================================================"
