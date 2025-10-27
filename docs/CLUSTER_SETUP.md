# Complete Setup Guide - From Scratch

Step-by-step instructions for setting up the microlensing classification project on **any system** (local workstation or HPC cluster).

---

## 🎯 Overview

This guide covers:
- ✅ Local setup (your laptop/workstation)
- ✅ HPC cluster setup (SLURM-based systems)
- ✅ AMD GPU support (ROCm)
- ✅ NVIDIA GPU support (CUDA)
- ✅ CPU-only fallback

---

## 📋 Prerequisites

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 16 GB (32+ GB recommended)
- **Storage**: 50 GB free space
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+) or macOS

### Recommended for Training
- **GPU**: NVIDIA RTX 3090 / AMD MI200 series or better
- **RAM**: 64 GB+
- **Storage**: 200 GB SSD
- **Multi-GPU**: 2-4 GPUs for faster training

---

## 🚀 Part 1: Local Workstation Setup

### Step 1: Check Your Hardware

```bash
# Check GPU (NVIDIA)
nvidia-smi

# Check GPU (AMD)
rocm-smi

# Check CPU and memory
lscpu
free -h

# Check disk space
df -h
```

---

### Step 2: Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    curl \
    wget
```

#### CentOS/RHEL:
```bash
sudo yum install -y \
    python3 \
    python3-devel \
    git \
    gcc \
    gcc-c++ \
    make \
    curl \
    wget
```

#### macOS:
```bash
brew install python@3.10 git
```

---

### Step 3: Install GPU Drivers

#### For NVIDIA GPUs:

**Check CUDA version**:
```bash
nvidia-smi
# Look for "CUDA Version: 12.x"
```

**Install CUDA Toolkit** (if needed):
```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Or use package manager
sudo apt install nvidia-cuda-toolkit
```

**Add to PATH**:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### For AMD GPUs:

**Install ROCm**:
```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Add to PATH
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verify**:
```bash
rocm-smi
```

---

### Step 4: Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens
```

---

### Step 5: Create Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Install Miniconda if you don't have it
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens
```

#### Option B: Using venv

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

---

### Step 6: Install PyTorch

#### For NVIDIA GPUs (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For NVIDIA GPUs (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For AMD GPUs (ROCm 6.0):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

#### For AMD GPUs (ROCm 5.7):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

#### For CPU only:
```bash
pip install torch torchvision
```

---

### Step 7: Install Project Dependencies

```bash
pip install -r requirements.txt
```

**What this installs**:
- NumPy, SciPy (scientific computing)
- scikit-learn (preprocessing, metrics)
- matplotlib, seaborn (visualization)
- VBMicrolensing (light curve simulation)
- tqdm (progress bars)
- joblib (model persistence)

---

### Step 8: Verify Installation

```bash
# Comprehensive check
python code/utils.py

# Quick GPU test
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
EOF
```

**Expected output** (with GPU):
```
PyTorch version: 2.1.0+cu121
CUDA available: True
GPU count: 1
GPU 0: NVIDIA GeForce RTX 4090
```

---

### Step 9: Create Directory Structure

```bash
cd ~/thesis-microlens

# Create required directories
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p results
mkdir -p logs

# Verify structure
tree -L 2
```

---

## 🖥️ Part 2: HPC Cluster Setup

### Step 1: SSH to Cluster

```bash
ssh username@cluster.domain.edu
```

---

### Step 2: Check Available Resources

```bash
# Check partitions
sinfo

# Check your account
sacctmgr show user $USER

# Check loaded modules
module list

# Check available modules
module avail
```

---

### Step 3: Load Modules

```bash
# Example for a typical SLURM cluster
module load python/3.10
module load cuda/12.1        # For NVIDIA
# OR
module load rocm/6.0         # For AMD

# Make permanent (optional)
echo "module load python/3.10" >> ~/.bashrc
echo "module load cuda/12.1" >> ~/.bashrc  # or rocm/6.0
source ~/.bashrc
```

---

### Step 4: Setup Conda (if not available)

```bash
# Install Miniconda to your home
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens
```

---

### Step 5: Clone and Install

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# Install PyTorch (adjust for your cluster's GPUs)
# Check with: nvidia-smi or rocm-smi

# For NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For AMD:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install -r requirements.txt
```

---

### Step 6: Test Interactive Session

```bash
# Request interactive GPU node (adjust parameters for your cluster)
salloc --partition=gpu \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=1:00:00

# Once allocated, you'll be on a compute node
# Activate environment
conda activate microlens

# Test GPU
python code/utils.py

# Exit when done
exit
```

---

### Step 7: Configure SLURM Scripts

**Edit `slurm/train_baseline.sh`**:

```bash
cd ~/thesis-microlens
nano slurm/train_baseline.sh
```

**Update these lines**:
```bash
#SBATCH --partition=YOUR_PARTITION_NAME    # Change to your partition
#SBATCH --account=YOUR_ACCOUNT            # If required
#SBATCH --gres=gpu:4                      # Adjust GPU count
#SBATCH --mem-per-gpu=32G                 # Adjust memory

# Update module loading
module load cuda/12.1  # or your cluster's module name
```

**Make executable**:
```bash
chmod +x slurm/*.sh
```

---

## 🧪 Part 3: Generate Baseline Dataset

### Small Test Dataset (5 minutes)

```bash
cd ~/thesis-microlens/code

python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --output ../data/raw/test_2k.npz \
    --binary_difficulty baseline

# Verify
python -c "import numpy as np; d=np.load('../data/raw/test_2k.npz'); print(f'Shape: {d[\"X\"].shape}')"
```

---

### Full Baseline Dataset (2-3 hours on 24 cores)

```bash
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --output ../data/raw/events_baseline_1M.npz \
    --binary_difficulty baseline \
    --n_processes 24  # Adjust to your CPU count
```

**Monitor progress**:
- Watch the progress bars
- PSPL generation: ~30-40 min
- Binary generation: ~90-120 min

---

## 🏋️ Part 4: Train Baseline Model

### Test Training (10 minutes)

```bash
python train.py \
    --data ../data/raw/test_2k.npz \
    --output ../models/test_model.pt \
    --epochs 5 \
    --batch_size 32 \
    --experiment_name test
```

---

### Full Baseline Training

#### Local (single GPU):
```bash
python train.py \
    --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline
```

#### HPC Cluster (batch job):
```bash
sbatch slurm/train_baseline.sh
```

**Monitor**:
```bash
# Check queue
squeue -u $USER

# Watch log (when running)
tail -f logs/baseline_*.out

# Check GPU usage (ssh to compute node)
watch -n 2 nvidia-smi  # or rocm-smi
```

---

## 📊 Part 5: Evaluate Model

```bash
cd ~/thesis-microlens/code

# Find your trained model
ls -lh ../results/baseline_*/

# Evaluate
python evaluate.py \
    --model ../results/baseline_TIMESTAMP/best_model.pt \
    --data ../data/raw/events_baseline_1M.npz \
    --output_dir ../results/baseline_eval \
    --early_detection
```

**Check results**:
```bash
cd ../results/baseline_eval
ls -lh

# View metrics
cat metrics.json

# View plots
# (Download to local machine to view):
# scp username@cluster:~/thesis-microlens/results/baseline_eval/*.png ./
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'torch'"

```bash
# Ensure environment is activated
conda activate microlens

# Reinstall PyTorch
pip install torch torchvision --index-url <appropriate-url>
```

---

### Issue: "CUDA out of memory"

```bash
# Reduce batch size
python train.py --batch_size 64  # or 32
```

---

### Issue: "No GPUs detected"

```bash
# Check drivers
nvidia-smi  # or rocm-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct version
```

---

### Issue: "Permission denied" for SLURM scripts

```bash
chmod +x slurm/*.sh
```

---

### Issue: VBMicrolensing installation fails

```bash
# Install dependencies first
pip install numpy cython

# Then install VBMicrolensing
pip install VBMicrolensing

# If still fails, try from source
git clone https://github.com/valboz/VBMicrolensing.git
cd VBMicrolensing/VBMicrolensingLibrary/lib
python setup.py install
```

---

## ✅ Verification Checklist

Before starting baseline training, verify:

- [ ] Python 3.8+ installed
- [ ] PyTorch installed and GPU detected
- [ ] All required packages installed (`pip list`)
- [ ] Repository cloned and directories created
- [ ] GPU drivers working (nvidia-smi or rocm-smi)
- [ ] Test dataset (2K events) generated successfully
- [ ] Test training (5 epochs) completed without errors
- [ ] SLURM scripts configured (if on cluster)
- [ ] Storage space available (50+ GB)

---

## 📞 Getting Help

### For GPU/CUDA Issues:
- NVIDIA: https://forums.developer.nvidia.com/
- AMD: https://community.amd.com/

### For PyTorch Issues:
- Forum: https://discuss.pytorch.org/
- GitHub: https://github.com/pytorch/pytorch

### For Cluster Issues:
- Contact your local HPC support
- Check cluster documentation

### For This Project:
- GitHub Issues: (your repo)
- Email: your.email@university.edu

---

## 🎯 Next Steps

After setup is complete:

1. **Generate baseline dataset** (if not done)
   ```bash
   python code/simulate.py --output data/raw/events_baseline_1M.npz
   ```

2. **Start baseline training**
   ```bash
   sbatch slurm/train_baseline.sh  # or run locally
   ```

3. **Monitor progress**
   ```bash
   tail -f logs/baseline_*.out
   ```

4. **Evaluate when done**
   ```bash
   python code/evaluate.py --model results/baseline_*/best_model.pt ...
   ```

5. **Read thesis guide**
   - See `docs/THESIS_GUIDE.md` for next experiments

---

**You're now ready to start your baseline training!** 🚀