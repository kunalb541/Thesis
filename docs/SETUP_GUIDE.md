# Setup Guide: From Zero to Training

Complete installation guide for local workstations and HPC clusters.

---

## 🎯 Quick Links

- **Just want to get started?** → Jump to [Quick Start](#quick-start)
- **Having issues?** → See [Troubleshooting](#troubleshooting)
- **On a cluster?** → See [HPC Cluster Setup](#hpc-cluster-setup)

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

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis
```

---

### 2. Create Python Environment

**Option A: Using Conda (Recommended)**

```bash
# Install Miniconda if you don't have it
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n microlens python=3.10 -y
conda activate microlens
```

**Option B: Using venv**

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
```

---

### 3. Install PyTorch

Choose based on your hardware:

**For NVIDIA GPUs (CUDA 12.1)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For NVIDIA GPUs (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPUs (ROCm 6.0)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

**For AMD GPUs (ROCm 5.7)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

**For CPU only**:
```bash
pip install torch torchvision
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- NumPy, SciPy (scientific computing)
- scikit-learn (preprocessing, metrics)
- matplotlib, seaborn (visualization)
- VBMicrolensing (light curve simulation)
- tqdm, joblib (utilities)

---

### 5. Verify Installation

```bash
python code/preflight_check.py
```

This comprehensive check will:
- ✅ Verify directory structure
- ✅ Check Python version and packages
- ✅ Detect GPUs and test computation
- ✅ Check disk space and memory
- ✅ Validate code files

**Expected output** (with GPU):
```
✅ ALL CRITICAL CHECKS PASSED!

✓ System is ready for baseline training
```

---

### 6. Quick Test

Generate a small test dataset and train briefly:

```bash
cd code

# Generate 2K events (5 minutes)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --output ../data/raw/test_2k.npz

# Quick training test (10 minutes)
python train.py \
    --data ../data/raw/test_2k.npz \
    --output ../models/test.pt \
    --epochs 5 \
    --batch_size 32 \
    --experiment_name test
```

If this completes successfully, you're ready for full training!

---

## 🖥️ HPC Cluster Setup

### Prerequisites

Most HPC clusters use SLURM for job scheduling. You'll need:
- SSH access to the cluster
- Allocated compute hours
- Basic SLURM knowledge

---

### 1. Connect to Cluster

```bash
ssh username@cluster.domain.edu
```

---

### 2. Check Resources

```bash
# Check available partitions
sinfo

# Check your account
sacctmgr show user $USER

# Check available modules
module avail
```

---

### 3. Load Modules

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

### 4. Setup Environment

```bash
# Install Miniconda to your home directory
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

### 5. Install Project

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/Thesis.git
cd Thesis

# Install PyTorch (adjust for your cluster's GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

---

### 6. Configure SLURM Scripts

Edit `slurm/train_baseline.sh`:

```bash
nano slurm/train_baseline.sh
```

**Update these lines**:
```bash
#SBATCH --partition=YOUR_PARTITION_NAME    # Change to your partition
#SBATCH --account=YOUR_ACCOUNT             # If required
#SBATCH --gres=gpu:4                       # Adjust GPU count
#SBATCH --mem-per-gpu=32G                  # Adjust memory

# Update module loading
module load cuda/12.1  # or your cluster's module name
```

**Make executable**:
```bash
chmod +x slurm/*.sh
```

---

### 7. Test Interactive Session

```bash
# Request interactive GPU node (adjust parameters)
salloc --partition=gpu \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=1:00:00

# Once allocated, test
conda activate microlens
python code/utils.py

# Exit when done
exit
```

---

### 8. Submit Batch Job

```bash
# Generate baseline dataset first (can be done in batch or interactive)
cd code
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --output ../data/raw/events_baseline_1M.npz

# Submit training job
cd ..
sbatch slurm/train_baseline.sh

# Monitor
squeue -u $USER
tail -f logs/baseline_*.out
```

---

## 🔧 GPU Setup Details

### NVIDIA GPUs

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

**Verify**:
```bash
nvidia-smi
nvcc --version
```

---

### AMD GPUs

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

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"

```bash
# Ensure environment is activated
conda activate microlens

# Reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue: "No GPUs detected"

```bash
# Check drivers
nvidia-smi  # or rocm-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA/ROCm version
```

---

### Issue: "CUDA out of memory"

```bash
# Reduce batch size
python train.py --batch_size 64  # or 32

# Or use gradient checkpointing (add to train.py)
# torch.utils.checkpoint.checkpoint(...)
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

### Issue: Slow data loading

```bash
# Copy data to fast storage (if on cluster)
cp data/raw/events_baseline_1M.npz /tmp/

# Update training command
python train.py --data /tmp/events_baseline_1M.npz ...
```

---

### Issue: "Out of disk space"

```bash
# Check usage
df -h

# Clean up
rm -rf data/raw/*.npz  # After backing up to cluster storage
rm -rf results/old_*/
conda clean --all
```

---

## ✅ Pre-Training Checklist

Before starting full baseline training:

- [ ] Python 3.8+ installed
- [ ] PyTorch installed and GPU detected
- [ ] All required packages installed
- [ ] Directory structure created
- [ ] GPU drivers working (nvidia-smi or rocm-smi)
- [ ] Test dataset (2K events) generated successfully
- [ ] Test training (5 epochs) completed without errors
- [ ] SLURM scripts configured (if on cluster)
- [ ] Storage space available (50+ GB)
- [ ] `preflight_check.py` passes all tests

**Run the comprehensive check**:
```bash
python code/preflight_check.py
```

---

## 📊 Expected Timeline

### Dataset Generation
- **Test dataset (2K events)**: 5 minutes
- **Small dataset (100K events)**: 30 minutes
- **Baseline dataset (1M events)**: 2-3 hours (24 cores)

### Training
- **4× AMD MI300A**: 6-8 hours
- **4× NVIDIA A100**: 6-8 hours
- **1× NVIDIA RTX 4090**: 24-30 hours
- **CPU only**: Not recommended (5-7 days)

### Evaluation
- **Per experiment**: 10-20 minutes

---

## 🎯 Next Steps

After successful setup:

1. **Generate baseline dataset** (if not done):
   ```bash
   python code/simulate.py --output data/raw/events_baseline_1M.npz
   ```

2. **Start baseline training**:
   ```bash
   sbatch slurm/train_baseline.sh  # or run locally
   ```

3. **Monitor progress**:
   ```bash
   tail -f logs/baseline_*.out
   ```

4. **Evaluate when done**:
   ```bash
   python code/evaluate.py --model results/baseline_*/best_model.pt ...
   ```

5. **Read research guide**:
   - See `docs/RESEARCH_GUIDE.md` for thesis workflow

---

## 📞 Getting Help

### For GPU/CUDA Issues:
- NVIDIA: https://forums.developer.nvidia.com/
- AMD: https://community.amd.com/

### For PyTorch Issues:
- Forum: https://discuss.pytorch.org/
- GitHub: https://github.com/pytorch/pytorch

### For This Project:
- Email: kunal29bhatia@gmail.com
- GitHub Issues: (your repo)

---

**You're now ready to start your research!** 🚀