# Setup Guide v3.0: From Zero to Training

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
python code/utils.py
```

This will check:
- ✅ PyTorch installed
- ✅ GPU detection
- ✅ All packages available

**Expected output** (with GPU):
```
✓ CUDA available: 1 GPU(s)
  GPU 0: NVIDIA RTX 4090
    Memory: 24.0 GB
    Compute: 8.9
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
    --epochs 5 \
    --batch_size 32 \
    --experiment_name test
```

If this completes successfully, you're ready for full training!

**Check your results**:
```bash
# List created directories
ls -ltr ../results/

# Should see: test_20251027_HHMMSS/
```

---

## 🆕 Understanding v3.0 Directory Structure

### Results Organization

Every training run creates a unique timestamped directory:

```
results/
├── baseline_20251027_143022/      # First training run
│   ├── best_model.pt              # Best checkpoint (auto-saved)
│   ├── config.json                # Exact configuration used
│   ├── training.log               # Full training logs
│   ├── summary.json               # Final metrics
│   ├── evaluation/                # Created when you run evaluate.py
│   │   ├── evaluation_summary.json
│   │   ├── roc.png
│   │   ├── pr.png
│   │   └── confusion_matrix.png
│   └── benchmark/                 # Created when you run benchmark_realtime.py
│       ├── benchmark_results.json
│       └── throughput_vs_batch_size.png
│
└── baseline_20251028_091234/      # Second run (for comparison)
    └── ...
```

### Benefits
- **No overwriting**: Each run preserved separately
- **Easy comparison**: Compare multiple runs side-by-side
- **Full reproducibility**: Config saved with each experiment
- **Auto-detection**: Scripts find latest run automatically

---

## 💡 Using v3.0 Features

### Auto-Detection of Models

**Old workflow (v2.0)**:
```bash
# Had to manually specify paths
python evaluate.py \
    --model results/baseline_20251027_143022/best_model.pt \
    --data data/raw/baseline_1M.npz \
    --output_dir results/baseline_eval
```

**New workflow (v3.0)**:
```bash
# Auto-detects latest model
python evaluate.py \
    --experiment_name baseline \
    --data data/raw/baseline_1M.npz
```

The script automatically:
1. Finds all directories matching `baseline_*`
2. Selects the most recent one
3. Uses `best_model.pt` from that directory
4. Saves evaluation to `{experiment_dir}/evaluation/`

### Multiple Runs for Statistics

```bash
# Train same experiment with different seeds
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 42
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 123
python train.py --data data/raw/baseline_1M.npz --experiment_name baseline --seed 456

# Results in:
# results/baseline_20251027_143022/  (seed 42)
# results/baseline_20251027_150315/  (seed 123)
# results/baseline_20251027_152748/  (seed 456)

# Evaluate latest automatically
python evaluate.py --experiment_name baseline --data data/raw/baseline_1M.npz

# Or evaluate specific run
python evaluate.py \
    --model results/baseline_20251027_143022/best_model.pt \
    --data data/raw/baseline_1M.npz \
    --output_dir results/baseline_20251027_143022/evaluation_v2
```

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

### 6. Test Interactive Session

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

### 7. Submit Batch Jobs (v3.0 style)

Create a SLURM script `train_baseline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# Load modules
module load cuda/12.1

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate microlens

# Change to code directory
cd ~/Thesis/code

# Run training (v3.0 - auto-creates timestamped directory)
python train.py \
    --data ../data/raw/baseline_1M.npz \
    --experiment_name baseline \
    --epochs 50 \
    --batch_size 128

echo "Training complete! Results in: $(ls -td ../results/baseline_* | head -1)"
```

Submit:
```bash
# Make executable
chmod +x train_baseline.sh

# Submit
sbatch train_baseline.sh

# Monitor
squeue -u $USER
tail -f logs/baseline_*.out
```

---

## 🔧 GPU Setup Details

### NVIDIA GPUs

**Check CUDA version**:
```bash
nvcc --version
nvidia-smi
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

# Or use single GPU
export CUDA_VISIBLE_DEVICES=0
```

---

### Issue: Can't find latest results directory

```bash
# Manually list
ls -ltr results/baseline_*/

# Check if experiment name matches
ls results/  # See all experiments

# Specify exact model path
python evaluate.py \
    --model results/baseline_20251027_143022/best_model.pt \
    --data data/raw/baseline_1M.npz \
    --output_dir results/baseline_20251027_143022/evaluation
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

# Clean up old results (keep last 3)
for exp in baseline cadence_05; do
    ls -td results/${exp}_*/ | tail -n +4 | xargs -r rm -rf
done

# Clean conda cache
conda clean --all
```

---

## ✅ Pre-Training Checklist

Before starting full baseline training:

- [ ] Python 3.8+ installed
- [ ] PyTorch installed and GPU detected (`python code/utils.py`)
- [ ] All required packages installed
- [ ] Directory structure created (automatic in v3.0)
- [ ] GPU drivers working (`nvidia-smi` or `rocm-smi`)
- [ ] Test dataset (2K events) generated successfully
- [ ] Test training (5 epochs) completed without errors
- [ ] Storage space available (50+ GB)
- [ ] Understand v3.0 directory structure

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
- **Per experiment**: 5-10 minutes (auto-detection makes it faster!)

---

## 🎯 Next Steps

After successful setup:

1. **Generate baseline dataset** (if not done):
   ```bash
   cd code
   python simulate.py \
       --n_pspl 500000 --n_binary 500000 \
       --output ../data/raw/baseline_1M.npz \
       --binary_params baseline
   ```

2. **Start baseline training**:
   ```bash
   python train.py \
       --data ../data/raw/baseline_1M.npz \
       --experiment_name baseline
   ```

3. **Monitor progress**:
   ```bash
   # Find your results directory
   ls -ltr ../results/baseline_*/
   
   # Watch logs
   tail -f $(ls -td ../results/baseline_*/ | head -1)/training.log
   ```

4. **Evaluate when done**:
   ```bash
   # Auto-detection makes this easy!
   python evaluate.py \
       --experiment_name baseline \
       --data ../data/raw/baseline_1M.npz \
       --early_detection
   ```

5. **Benchmark**:
   ```bash
   python benchmark_realtime.py \
       --experiment_name baseline \
       --data ../data/raw/baseline_1M.npz
   ```

6. **Read research guide**:
   - See `docs/RESEARCH_GUIDE.md` for thesis workflow

---

## 💡 v3.0 Pro Tips

### Finding Your Results

```bash
# List all runs for an experiment
ls -ltr results/baseline_*/

# Get most recent
LATEST=$(ls -td results/baseline_*/ | head -1)
echo "Latest results: $LATEST"

# View summary
cat $LATEST/summary.json
```

### Comparing Multiple Runs

```bash
# Train with different seeds
for seed in 42 123 456; do
    python train.py \
        --data data/raw/baseline_1M.npz \
        --experiment_name baseline \
        --seed $seed
done

# Compare results
python -c "
import json
from pathlib import Path

for run_dir in sorted(Path('results').glob('baseline_*')):
    summary = run_dir / 'summary.json'
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
        print(f'{run_dir.name}: Acc={data[\"final_test_acc\"]:.4f}')
"
```

### Archiving Completed Runs

```bash
# Archive a specific run
RUN=baseline_20251027_143022
tar -czf ${RUN}.tar.gz results/${RUN}/

# Or archive latest
LATEST=$(ls -td results/baseline_*/ | head -1 | xargs basename)
tar -czf ${LATEST}.tar.gz results/${LATEST}/
```

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
- Check `docs/QUICK_REFERENCE.md` for command examples

---

**You're now ready to start your research with v3.0!** 🚀

The new auto-detection and timestamped directories make everything easier and more organized!