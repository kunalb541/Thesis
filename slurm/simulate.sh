#!/bin/bash
#SBATCH --job-name=ml_simulate
#SBATCH --partition=cpu_il          # CPU partition for simulation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # Use multiple CPUs
#SBATCH --mem=128G                  # 128GB memory
#SBATCH --time=24:00:00             # 24 hours
#SBATCH --output=../logs/simulate_%j.out
#SBATCH --error=../logs/simulate_%j.err

echo "=========================================="
echo "SIMULATION JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load compiler/gnu/12

# Activate conda
source ~/.bashrc
conda activate microlens

# Change to project directory
cd ~/thesis-microlens/code

# Generate new dataset with specific cadence
# Adjust n_pspl and n_binary as needed

echo "Generating dataset..."
python simulate_cadence.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --cadence_prob 0.2 \
    --output ../data/raw/events_1M_new.npz

echo "Simulation complete at: $(date)"
