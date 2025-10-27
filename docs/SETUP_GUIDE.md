# Complete Setup and Workflow Guide

## Initial Setup on Your Laptop

### 1. Create GitHub Repository
```bash
# On your laptop
cd ~/Desktop  # or wherever you want
mkdir thesis-microlens
cd thesis-microlens

# Initialize git
git init
git branch -M main

# Create directory structure
mkdir -p code slurm data/raw data/processed models results logs

# Copy all files from Claude (you'll receive these)
# - Place all .py files in code/
# - Place all .sh files in slurm/
# - Place README.md and .gitignore in root

# Create placeholder files
touch data/raw/.gitkeep data/processed/.gitkeep
touch models/.gitkeep results/.gitkeep logs/.gitkeep

# Stage and commit
git add .
git commit -m "Initial commit: Complete microlensing classification framework"

# Create GitHub repo and push
# Go to github.com and create new repository
git remote add origin https://github.com/YOUR_USERNAME/thesis-microlens.git
git push -u origin main
```

## Setup on bwUniCluster 3.0

### 2. Clone and Install

```bash
# SSH to cluster
ssh hd_vm305@uc3.scc.kit.edu

# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# Load CUDA module
module load devel/cuda/12.1

# Create conda environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install dependencies
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn
pip install tensorflow[and-cuda] scikit-learn tqdm
pip install VBMicrolensing

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import VBMicrolensing; print('VBMicrolensing imported successfully')"

# Make shell scripts executable
chmod +x slurm/*.sh
```

### 3. Copy Existing Data

```bash
# You mentioned you already have 1M light curves
# Make sure they are at:
ls -lh /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz

# If they're elsewhere, copy them:
# cp /path/to/your/events_1M.npz data/raw/
```

## Quick Test (Interactive Session)

### 4. Debug and Test

```bash
# Request interactive GPU session
salloc --partition=gpu_mi300 --gres=gpu:4 --cpus-per-gpu=24 --mem-per-gpu=128200mb --time=4:00:00

# Once allocated, you'll see something like:
# salloc: Nodes uc3n083 are ready for job

# Activate environment
source ~/.bashrc
conda activate microlens
module load devel/cuda/12.1

# Navigate to project
cd ~/thesis-microlens/code

# Run quick test (tests GPU, data loading, and small training)
python test_quick.py

# If all tests pass, exit
exit
```

## Running Experiments

### 5. Baseline Training (Batch Job)

```bash
# From login node, submit baseline training job
cd ~/thesis-microlens
sbatch slurm/slurm_train_baseline.sh

# Check job status
squeue -u hd_vm305

# Monitor output
tail -f logs/train_baseline_*.out

# Check GPU usage (if you ssh to the compute node)
# ssh uc3nXXX  # replace XXX with your node number
# rocm-smi
```

### 6. Evaluation

```bash
# After training completes, evaluate the model
# Either submit as job or run interactively

# Interactive evaluation
salloc --partition=gpu_mi300 --gres=gpu:1 --cpus-per-gpu=24 --mem-per-gpu=128200mb --time=2:00:00

source ~/.bashrc
conda activate microlens
cd ~/thesis-microlens/code

python evaluate.py \
    --model ../models/baseline_model.keras \
    --data ../data/raw/events_1M.npz \
    --output_dir ../results/baseline

exit
```

### 7. View Results

```bash
# Check results
cd ~/thesis-microlens/results/baseline
ls -lh

# Files you'll find:
# - metrics.json (numerical results)
# - confusion_matrix.png
# - roc_curve.png
# - precision_recall_curve.png
# - early_detection.png
# - classification_report.txt

# Copy results to your laptop for analysis
# On your laptop:
scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./local_results
```

## Systematic Experiments

### 8. Running Multiple Experiments

For your thesis benchmarking, you'll want to:

1. **Test different cadences**
   - Generate new datasets with different cadence_mask_prob values
   - Train separate models
   - Compare results

2. **Test early detection**
   - Truncate time series at different points
   - Measure accuracy vs. observation time

3. **Test different binary parameters**
   - If needed, generate data with different s, q, rho ranges

```bash
# Example workflow for cadence experiment:

# 1. Generate new dataset (modify simulate.py parameters)
python code/simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output data/raw/events_cadence_05.npz \
    --cadence_mask_prob 0.05

# 2. Train on new data
sbatch slurm/slurm_train_cadence_05.sh  # create this from template

# 3. Evaluate
python code/evaluate.py \
    --model models/cadence_05.keras \
    --data data/raw/events_cadence_05.npz \
    --output_dir results/cadence_05

# 4. Compare experiments
python code/experiments.py --compare_only
```

## Monitoring and Troubleshooting

### Check Job Status
```bash
# View your jobs
squeue -u hd_vm305

# Cancel all your jobs
scancel -u hd_vm305

# Cancel specific job
scancel JOBID

# Check job details
scontrol show job JOBID

# Check past jobs
sacct -u hd_vm305 --format=JobID,JobName,Partition,State,Elapsed,MaxRSS
```

### Check GPU Usage
```bash
# On compute node (after ssh to node or in interactive session)
rocm-smi

# Expected output: Shows 4 AMD MI300 GPUs with memory usage
```

### Common Issues and Fixes

**Issue: TensorFlow not finding GPUs**
```bash
# Make sure you loaded CUDA module
module load devel/cuda/12.1

# Set environment variables
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
```

**Issue: Out of memory**
```bash
# Reduce batch size in train.py
# Edit slurm script: --batch_size 64  # instead of 128
```

**Issue: Training too slow**
```bash
# Verify GPUs are being used:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Should show 4 GPUs
# If not, check CUDA installation
```

## Workflow Summary

```
1. Setup (once)
   ├─ Create GitHub repo on laptop
   ├─ Clone to cluster
   └─ Install dependencies

2. Quick Test (once)
   ├─ Interactive session
   ├─ Run test_quick.py
   └─ Verify everything works

3. Baseline Experiment
   ├─ Submit batch job
   ├─ Wait for completion
   ├─ Evaluate model
   └─ Review results

4. Systematic Experiments (repeat)
   ├─ Generate data with new parameters
   ├─ Train model
   ├─ Evaluate
   └─ Compare with baseline

5. Thesis Writing
   ├─ Collect all results
   ├─ Generate comparison plots
   └─ Write conclusions
```

## Expected Timeline

- **Quick test**: 15 minutes
- **Baseline training**: 6-12 hours (1M samples, 50 epochs, 4 GPUs)
- **Evaluation**: 30 minutes
- **Per experiment**: 2-4 hours (smaller datasets)

## Key Points

✓ **TimeDistributed preserved**: Architecture supports real-time classification
✓ **Binary parameters unchanged**: Original ranges kept for distinct caustic crossings
✓ **GPU optimized**: Using all 4 AMD MI300 GPUs with mixed precision
✓ **Reproducible**: Fixed random seeds throughout
✓ **Automated**: Minimal manual intervention after initial setup

## Next Steps After Baseline

1. **Generate results**: Run baseline first, get metrics
2. **Cadence experiments**: Test 5%, 10%, 30%, 40% missing data
3. **Early detection**: Test classification at 500, 750, 1000 points
4. **Comparison**: Create figures showing performance vs. parameters
5. **Discussion**: Relate to LSST and Roman survey strategies

Good luck with your thesis! 🚀
