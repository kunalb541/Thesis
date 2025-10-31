# Setup Guide - Installation & Configuration

Complete installation guide for local workstations and HPC clusters.

---

## Quick Links

- **First time?** → [Quick Start](#quick-start)
- **On a cluster?** → [HPC Setup](#hpc-cluster-setup)
- **Having issues?** → [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 16 GB (32 GB+ recommended)
- **Storage**: 50 GB free space
- **OS**: Linux (Ubuntu 20.04+) or macOS

### Recommended for Training
- **GPU**: NVIDIA RTX 3090 / AMD MI200 or better
- **RAM**: 64 GB+
- **Storage**: 200 GB SSD
- **Multi-GPU**: 2-4 GPUs for faster training

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis
```

### 2. Create Environment

**Option A: Conda (Recommended)**

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens
```

**Option B: venv**

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Install PyTorch

Choose based on your hardware:

**NVIDIA GPUs (CUDA 12.1)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**NVIDIA GPUs (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPUs (ROCm 6.0)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

**CPU only**:
```bash
pip install torch torchvision
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Installs:
- NumPy, SciPy (scientific computing)
- scikit-learn (preprocessing, metrics)
- matplotlib, seaborn (visualization)
- VBMicrolensing (simulation)
- tqdm, joblib (utilities)

### 5. Verify Installation

```bash
python code/utils.py
```

Expected output:
```
============================================================
GPU Check:
============================================================
✓ CUDA available: 1 GPU(s)
  GPU 0: NVIDIA RTX 4090
    Memory: 24.0 GB
============================================================

Testing Scaler Functions:
✓ Normalization test passed!
============================================================
```

### 6. Quick Test

```bash
cd code

# Generate small dataset (5 min)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --output ../data/raw/test.npz

# Quick training test (10 min)
python train.py \
    --data ../data/raw/test.npz \
    --experiment_name test \
    --epochs 5 \
    --batch_size 32
```

If training completes successfully, you're ready!

---

## HPC Cluster Setup

### SLURM Clusters

**1. Create conda environment on login node:**

```bash
# Load modules
module load cuda/12.1
module load python/3.10

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Install PyTorch and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**2. Create batch script:**

```bash
#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# Load modules
module load cuda/12.1

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Run training
cd ~/Thesis/code
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128
```

**3. Submit job:**

```bash
mkdir -p logs
sbatch train_baseline.sh
```

**4. Monitor:**

```bash
# Check queue
squeue -u $USER

# Follow logs
tail -f logs/baseline_*.out

# Check GPU usage
ssh compute-node
nvidia-smi
```

### PBS/Torque Clusters

```bash
#!/bin/bash
#PBS -N baseline
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/baseline.log

cd $PBS_O_WORKDIR
module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

cd code
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50
```

Submit: `qsub train_baseline.pbs`

---

## Configuration

### GPU Setup

The code automatically detects and uses available GPUs:

```python
# In code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Multi-GPU (automatic)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

Check GPU detection:
```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Memory Management

For large datasets:

```bash
# Reduce batch size if OOM
python train.py --batch_size 64  # instead of 128

# Reduce sequence length
python simulate.py --n_points 1000  # instead of 1500

# Use gradient checkpointing (in model.py if needed)
```

### Performance Tuning

```bash
# More data loading workers
python train.py --num_workers 16

# Adjust based on CPU cores
export OMP_NUM_THREADS=8
```

---

## Directory Structure

After setup:

```
Thesis/
├── code/              # Python scripts
├── data/
│   └── raw/           # Generated datasets
├── results/           # Training outputs
├── models/            # (unused, results has models)
├── docs/              # Documentation
├── requirements.txt
└── README.md
```

Create directories if missing:
```bash
mkdir -p data/raw results models logs
```

---

## Environment Variables

Optional optimizations:

```bash
# Add to ~/.bashrc or submit script

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Weights & Biases (if using)
export WANDB_MODE=offline
```

---

## Data Management

### Storage Locations

```bash
# Raw datasets: 1-10 GB each
data/raw/*.npz

# Trained models: ~50 MB each
results/experiment_*/best_model.pt

# Scalers: ~1 KB each (important!)
results/experiment_*/scaler_*.pkl
```

### Cleaning Old Results

```bash
# List all experiments
ls -lh results/

# Remove specific experiment
rm -rf results/baseline_20251027_143022/

# Archive important results
tar -czf results_archive.tar.gz results/baseline_*
```

---

## Troubleshooting

### Installation Issues

**VBMicrolensing won't install**:
```bash
# Try with conda
conda install -c conda-forge vbmicrolensing

# Or build from source
git clone https://github.com/valboz/VBBinaryLensing
cd VBBinaryLensing/VBBinaryLensingLibrary/lib
python setup.py install
```

**PyTorch CUDA mismatch**:
```bash
# Check CUDA version
nvidia-smi

# Reinstall matching PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Runtime Issues

**CUDA out of memory**:
```bash
# Solution 1: Reduce batch size
python train.py --batch_size 64

# Solution 2: Use single GPU
export CUDA_VISIBLE_DEVICES=0

# Solution 3: Enable memory optimization
# Add to train.py:
torch.cuda.empty_cache()
```

**Slow data loading**:
```bash
# Check disk I/O
iostat -x 1

# Use faster storage (SSD)
# Copy data to /tmp or local SSD
cp data/raw/baseline.npz /tmp/
python train.py --data /tmp/baseline.npz
```

**Training divergence (NaN loss)**:
```bash
# Reduce learning rate
python train.py --lr 1e-4  # instead of 1e-3

# Check data quality
python -c "import numpy as np; data = np.load('data/raw/baseline.npz')['X']; print(f'Min: {data.min()}, Max: {data.max()}, NaN: {np.isnan(data).any()}')"
```

### Multi-GPU Issues

**Only using 1 GPU**:
```bash
# Check GPU visibility
python -c "import torch; print(torch.cuda.device_count())"

# Verify DataParallel
grep "DataParallel" code/train.py

# Force multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**Unbalanced GPU usage**:
```bash
# Normal - GPU 0 typically has higher load (model synchronization)
# To balance, ensure batch_size is multiple of GPU count
python train.py --batch_size 128  # for 4 GPUs (32 per GPU)
```

---

## Performance Expectations

### Training Times

Hardware configurations (baseline 1M):

| Hardware | Batch Size | Time |
|----------|------------|------|
| 4× A100 80GB | 128 | 4-5 hours |
| 4× V100 32GB | 128 | 6-8 hours |
| 4× RTX 3090 | 64 | 8-10 hours |
| 1× RTX 4090 | 128 | 18-20 hours |
| CPU only | 64 | 5-7 days |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Model | ~200 MB | Per GPU |
| Batch (128) | ~2 GB | Per GPU |
| Optimizer | ~400 MB | Per GPU |
| Overhead | ~1 GB | CUDA kernels |
| **Total** | **~4 GB** | Per GPU |

Minimum GPU: 8 GB (with batch_size=64)  
Recommended: 16 GB+ (with batch_size=128)

---

## Verification Checklist

Before starting experiments:

- [ ] Python 3.10+ installed
- [ ] PyTorch with GPU support installed
- [ ] All requirements installed
- [ ] GPU detected (`nvidia-smi` or `rocm-smi`)
- [ ] Code runs without errors (`python code/utils.py`)
- [ ] Quick test completes successfully
- [ ] Results directory created
- [ ] Sufficient disk space (100+ GB)

Run verification:
```bash
python code/utils.py
```

---

## Next Steps

After successful setup:

1. **Generate baseline dataset**:
   ```bash
   python simulate.py --n_pspl 500000 --n_binary 500000 \
       --output ../data/raw/baseline_1M.npz
   ```

2. **Train baseline model**:
   ```bash
   python train.py --data ../data/raw/baseline_1M.npz \
       --experiment_name baseline
   ```

3. **Evaluate**:
   ```bash
   python evaluate.py --experiment_name baseline \
       --data ../data/raw/baseline_1M.npz --early_detection
   ```

4. **Run systematic experiments** (see [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md))

---

## Getting Help

### Resources

- **Installation**: This guide
- **Usage**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Research**: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)
- **Code Issues**: GitHub Issues

### Support Channels

- **Email**: kunal29bhatia@gmail.com
- **GPU Issues**: 
  - NVIDIA: https://forums.developer.nvidia.com/
  - AMD: https://community.amd.com/
- **PyTorch**: https://discuss.pytorch.org/

### Before Asking for Help

Include:
1. Full error message
2. Output of `python code/utils.py`
3. GPU info (`nvidia-smi` output)
4. Command that failed
5. Python/PyTorch versions

---

You're ready to start training! 🚀