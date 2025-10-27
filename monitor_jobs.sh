#!/bin/bash
# Monitor your SLURM jobs with useful information

USER="hd_vm305"  # Change if needed

echo "========================================"
echo "SLURM JOB MONITOR"
echo "========================================"
echo ""

# Check running jobs
echo "📊 Your Running Jobs:"
squeue -u $USER --format="%.10i %.12j %.8T %.10M %.12l %.6D %.20R"
echo ""

# Check recent jobs
echo "📜 Recently Completed Jobs (last 10):"
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode -S $(date -d '7 days ago' +%Y-%m-%d) | tail -n 11
echo ""

# Check GPU availability
echo "🖥️  Available GPUs:"
echo ""
echo "Development Partitions (30 min max):"
sinfo -p dev_gpu_h100 -o "  %-20P %-10a %.10l %.6D %.10T"
sinfo -p dev_gpu_a100_il -o "  %-20P %-10a %.10l %.6D %.10T"
echo ""
echo "Regular GPU Partitions:"
sinfo -p gpu_h100 -o "  %-20P %-10a %.10l %.6D %.10T"
sinfo -p gpu_a100_il -o "  %-20P %-10a %.10l %.6D %.10T"
sinfo -p gpu_mi300 -o "  %-20P %-10a %.10l %.6D %.10T"
echo ""

# Check logs
echo "📁 Recent Log Files:"
ls -lht ~/thesis-microlens/logs/*.out 2>/dev/null | head -5
echo ""

# Quick stats
RUNNING=$(squeue -u $USER -t RUNNING | wc -l)
PENDING=$(squeue -u $USER -t PENDING | wc -l)

echo "========================================"
echo "Summary: $((RUNNING-1)) running, $((PENDING-1)) pending"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  scancel <job_id>         - Cancel a job"
echo "  tail -f logs/train_*.out - Watch training progress"
echo "  scontrol show job <id>   - Detailed job info"
echo ""
