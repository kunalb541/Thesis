#!/bin/bash
# Quick setup script for bwUniCluster 3.0
# Run this after cloning the repository

echo "=========================================="
echo "Setting up Microlensing Classifier"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please load/install conda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'microlens'..."
conda create -n microlens python=3.10 -y

# Activate environment
echo "Activating environment..."
source ~/.bashrc
conda activate microlens

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install tensorflow>=2.15.0
pip install numpy scikit-learn matplotlib seaborn tqdm joblib pandas

# Install VBMicrolensing (might need special handling)
echo "Installing VBMicrolensing..."
pip install VBMicrolensing || echo "WARNING: VBMicrolensing installation failed. You may need to install manually."

# Create directories
echo "Creating directory structure..."
mkdir -p data/raw data/processed models results logs

# Make SLURM scripts executable
echo "Making scripts executable..."
chmod +x slurm/*.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate microlens"
echo "2. Copy your events_1M.npz to data/raw/"
echo "3. Submit training job: cd slurm && sbatch train_gpu.sh"
echo ""
echo "For testing: sbatch debug_gpu.sh"
echo "For cadence experiments: sbatch train_cadence_array.sh"
echo ""
