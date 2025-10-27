# Microlensing Event Classification with Deep Learning

Real-time classification of gravitational microlensing events (PSPL vs Binary) using 1D CNNs with TimeDistributed layers for early detection capability.

## 🎯 Project Overview

This project implements a deep learning pipeline to classify microlensing light curves in real-time, enabling early detection of binary lens events. The model uses TimeDistributed layers to make predictions at each timestep, simulating real-time observational scenarios.

**Key Features:**
- Real-time classification with TimeDistributed CNN architecture
- GPU-optimized training with mixed precision
- Cadence experiments (sparse, normal, dense, LSST, Roman)
- Early detection analysis
- Automated experiment tracking
- bwUniCluster 3.0 ready with SLURM scripts

## 📁 Project Structure

```
thesis-microlens/
├── code/
│   ├── config.py                    # Configuration parameters
│   ├── simulate_cadence.py          # Generate datasets with different cadences
│   ├── train.py                     # GPU-optimized training script
│   ├── evaluate.py                  # Evaluation and early detection analysis
│   └── run_experiments.py           # Automated experiment suite
├── slurm/
│   ├── train_gpu.sh                 # Single GPU training job
│   ├── train_cadence_array.sh       # Array job for cadence experiments
│   ├── debug_gpu.sh                 # Quick debugging on dev partition
│   └── simulate.sh                  # CPU-based simulation job
├── data/
│   ├── raw/                         # Raw simulated datasets
│   └── processed/                   # Preprocessed data
├── models/                          # Saved models
├── results/                         # Experiment results and plots
├── logs/                           # SLURM output logs
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start on bwUniCluster 3.0

### 1. Initial Setup on Your Laptop

```bash
# Clone or create the repository
mkdir thesis-microlens
cd thesis-microlens

# Initialize git
git init
git add .
git commit -m "Initial commit: microlensing classifier"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/thesis-microlens.git
git branch -M main
git push -u origin main
```

### 2. Setup on bwUniCluster

```bash
# SSH into the cluster
ssh hd_vm305@uc3.scc.kit.edu

# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# Create conda environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install dependencies
pip install tensorflow numpy scikit-learn matplotlib seaborn tqdm joblib
pip install VBMicrolensing  # Or build from source if needed

# Create necessary directories
mkdir -p data/raw data/processed models results logs
```

### 3. Using Your Existing 1M Dataset

You already have `events_1M.npz` - great! You can start training immediately.

```bash
# Make scripts executable
chmod +x slurm/*.sh

# Test on development partition (30 min, quick check)
cd slurm
sbatch debug_gpu.sh

# Check job status
squeue -u hd_vm305

# View output
tail -f ../logs/debug_*.out
```

### 4. Full GPU Training

```bash
# Submit full training job to H100 GPU
sbatch train_gpu.sh

# Monitor job
squeue -u hd_vm305
watch -n 5 'squeue -u hd_vm305'  # Auto-refresh

# View live progress
tail -f ../logs/train_*.out
```

### 5. Run Cadence Experiments (Recommended for Thesis)

This runs 5 experiments in parallel (one per GPU):

```bash
# Submit array job for all cadence experiments
sbatch train_cadence_array.sh

# This will run:
# - sparse (50% missing)
# - normal (20% missing) 
# - dense (5% missing)
# - lsst (30% missing)
# - roman (15% missing)

# Check array job status
squeue -u hd_vm305
```

## 🔬 Generating New Datasets (Optional)

If you want to generate new datasets with specific cadences:

```bash
# Generate dataset with sparse cadence
cd code
python simulate_cadence.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --cadence_prob 0.5 \
    --output ../data/raw/events_sparse.npz

# Or submit simulation job to SLURM
cd ../slurm
sbatch simulate.sh
```

## 📊 Analyzing Results

After training completes, results are organized by experiment:

```
results/
├── baseline_gpu_20250127_143022/
│   ├── best_model.keras
│   ├── scaler.pkl
│   ├── experiment_config.json
│   ├── results.json
│   ├── evaluation/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── early_detection_curve.png
│   │   ├── detection_time_distribution.png
│   │   └── evaluation_results.json
│   └── tensorboard/
```

### View Results

```bash
# View evaluation results
cat results/baseline_gpu_*/evaluation/evaluation_results.json

# Download results to your laptop
# On your laptop:
scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./local_results
```

### TensorBoard (Optional)

```bash
# On cluster, start TensorBoard
tensorboard --logdir results/baseline_gpu_*/tensorboard --port 6006

# On your laptop, create SSH tunnel
ssh -L 6006:localhost:6006 hd_vm305@uc3.scc.kit.edu

# Open browser: http://localhost:6006
```

## 🎯 Key Experiments for Your Thesis

### 1. Baseline Performance
```bash
sbatch slurm/train_gpu.sh
```
**Objective:** Establish baseline accuracy with standard cadence

### 2. Cadence Study
```bash
sbatch slurm/train_cadence_array.sh
```
**Objective:** How does observation cadence affect early detection?

### 3. Early Detection Analysis
Automatically run after each training - analyzes:
- Classification accuracy at 10%, 20%, ..., 100% of observations
- Detection time distribution
- Confidence thresholds

### 4. Survey Simulation
Compare LSST vs Roman cadences:
```bash
# Already included in cadence experiments
# Results will show which survey enables earlier detection
```

## 🔧 Troubleshooting

### GPU Not Available
```bash
# Check available GPUs
sinfo_t_idle

# Use A100 instead of H100 if needed
# Edit slurm scripts: change partition to gpu_a100_il
```

### Out of Memory
```bash
# Reduce batch size in train.py
python train.py --batch_size 16  # Instead of 64
```

### Module Not Found
```bash
# Ensure conda environment is activated
conda activate microlens

# Reinstall if needed
pip install -r requirements.txt
```

### Job Failed
```bash
# Check error log
cat logs/train_*.err

# Check output log
cat logs/train_*.out

# Debug on dev partition
sbatch slurm/debug_gpu.sh
```

## 📈 Expected Runtime

- **Debug test**: 5-10 minutes
- **Full training (1M events, 50 epochs)**: 2-4 hours on H100
- **Single cadence experiment**: 2-4 hours
- **Full cadence suite (5 experiments)**: 2-4 hours (parallel)

## 🎓 Thesis Results Checklist

- [ ] Baseline model trained on 1M events
- [ ] Early detection analysis complete
- [ ] Cadence experiments (sparse, normal, dense, LSST, Roman)
- [ ] Confusion matrices generated
- [ ] ROC curves computed
- [ ] Detection time distributions analyzed
- [ ] Results compared across cadences
- [ ] Figures exported for thesis

## 📝 Important Notes

### TimeDistributed Architecture
The model uses `TimeDistributed(Dense(...))` layers to make predictions at each timestep. This is **essential** for real-time classification and should **NOT be removed**. It simulates receiving observations sequentially and making predictions as data arrives.

### Mixed Precision Training
Enabled by default for faster GPU training. Speeds up training by ~2x on H100 GPUs.

### Data Format
- Input shape: `(batch, 1500, 1)` - 1500 timepoints, 1 feature (flux/magnitude)
- Output shape: `(batch, 1500, 2)` - classification at each timepoint
- Missing data: Replaced with 0 (configurable in `config.py`)

## 🔗 Useful Commands

```bash
# Check job status
squeue -u hd_vm305

# Cancel job
scancel <job_id>

# Check GPU usage
srun --partition=dev_gpu_h100 --gres=gpu:1 --pty bash
nvidia-smi

# Monitor disk usage
du -sh data/ models/ results/

# Clean old logs
rm logs/*.out logs/*.err

# Git workflow
git add .
git commit -m "Add results from cadence experiments"
git push
```

## 📧 Support

For cluster-specific issues:
- bwUniCluster documentation: https://wiki.bwhpc.de/
- Support: bwunicluster@lists.kit.edu

For code issues:
- Check logs in `logs/`
- Review configuration in `code/config.py`
- Test with debug script first

## 🎉 Ready to Start!

```bash
# Quick start command sequence
cd ~/thesis-microlens
conda activate microlens
cd slurm
sbatch train_gpu.sh  # Start training!
watch squeue -u hd_vm305  # Monitor
```

Good luck with your thesis! 🚀
