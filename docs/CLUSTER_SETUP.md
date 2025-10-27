# bwUniCluster 3.0 Setup Guide

Complete setup instructions for running your thesis project on the cluster with AMD MI300 GPUs.

---

## 🖥️ Cluster Specifications

**Hardware**:
- **GPUs**: 4× AMD MI300 (128GB HBM3 each, 512GB total)
- **GPU Architecture**: CDNA 3 (optimized for HPC/AI workloads)
- **CPUs**: 24 cores per GPU (96 cores total per node)
- **CPU Architecture**: AMD EPYC (Zen 4)
- **Memory**: 128GB RAM per GPU
- **Interconnect**: High-speed Infiniband for multi-node jobs
- **Software**: ROCm 6.0, PyTorch 2.1+, TensorFlow 2.15+

**Your allocation**:
- **Partition**: `gpu_mi300`
- **Max time**: 72 hours per job
- **Max GPUs**: 4 simultaneous
- **Priority**: Standard (shared partition)

---

## 🌐 Access

### SSH Connection
```bash
# From your laptop
ssh hd_vm305@uc3.scc.kit.edu

# If you have SSH keys configured
ssh -i ~/.ssh/id_rsa hd_vm305@uc3.scc.kit.edu
```

### File Transfer

**Using scp (small files)**:
```bash
# Upload to cluster
scp local_file.py hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/code/

# Download from cluster
scp hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results/plot.png ./
```

**Using rsync (large files/directories)**:
```bash
# Upload directory
rsync -avz --progress ./data/ hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/data/

# Download results
rsync -avz --progress hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results/ ./local_results/
```

---

## 🚀 Initial Setup (One-Time)

### Step 1: Clone Repository
```bash
# SSH to cluster
ssh hd_vm305@uc3.scc.kit.edu

# Clone your repo
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# Check directory structure
tree -L 2
```

### Step 2: Environment Setup
```bash
# Load CUDA/ROCm module
module load devel/cuda/12.1

# Check available modules
module avail

# Create conda environment
conda create -n microlens python=3.10 -y
conda activate microlens

# Verify Python
python --version  # Should show Python 3.10.x

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install scientific computing stack
pip install numpy scipy pandas matplotlib seaborn
pip install scikit-learn tqdm joblib

# Install VBMicrolensing (microlensing simulation library)
pip install VBMicrolensing

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
python -c "import VBMicrolensing; print('VBMicrolensing: OK')"
```

### Step 4: Configure Environment
```bash
# Add to ~/.bashrc for automatic loading
echo "module load devel/cuda/12.1" >> ~/.bashrc
echo "conda activate microlens" >> ~/.bashrc

# Source it
source ~/.bashrc

# Set ROCm environment variables (add to ~/.bashrc)
echo "export ROCR_VISIBLE_DEVICES=0,1,2,3" >> ~/.bashrc
echo "export HIP_VISIBLE_DEVICES=0,1,2,3" >> ~/.bashrc
```

### Step 5: Make Scripts Executable
```bash
cd ~/thesis-microlens
chmod +x slurm/*.sh
chmod +x docs/*.sh

# Verify
ls -lh slurm/*.sh
```

---

## 🧪 Testing Installation

### Quick Test (Interactive Session)
```bash
# Request interactive GPU node for testing
salloc --partition=gpu_mi300 \
       --gres=gpu:1 \
       --cpus-per-gpu=24 \
       --mem-per-gpu=128200mb \
       --time=1:00:00

# Once allocated (you'll see: "salloc: Nodes uc3nXXX are ready")

# Check GPU
rocm-smi

# Test PyTorch GPU
python -c "import torch; print(f'GPUs detected: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU 0: {torch.cuda.get_device_name(0)}')"

# Test simple computation
python << EOF
import torch
import time

# Create large tensors on GPU
x = torch.randn(10000, 10000, device='cuda')
y = torch.randn(10000, 10000, device='cuda')

# Time matrix multiplication
start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"Matrix multiplication on GPU: {end-start:.4f} seconds")
print(f"Result shape: {z.shape}")
print("✓ GPU computation working!")
EOF

# Exit interactive session
exit
```

### Test Simulation
```bash
# Test data generation with multiprocessing
cd ~/thesis-microlens/code

# Small test dataset (should take ~1-2 minutes)
python simulate.py \
    --n_pspl 1000 \
    --n_binary 1000 \
    --output ../data/raw/test_small.npz \
    --n_processes 4

# Check output
ls -lh ../data/raw/test_small.npz
python -c "import numpy as np; d=np.load('../data/raw/test_small.npz'); print(f'Shape: {d[\"X\"].shape}')"
```

### Test Training
```bash
# Quick training test (small dataset, 2 epochs)
cd ~/thesis-microlens/code

python train.py \
    --data ../data/raw/test_small.npz \
    --output ../models/test_model.pt \
    --epochs 2 \
    --batch_size 32 \
    --experiment_name test

# Should complete in ~2-3 minutes
# If it works, you're ready for full-scale experiments!
```

---

## 📋 SLURM Job System

### Basic Commands
```bash
# Submit a job
sbatch slurm/train_baseline.sh

# Check your jobs
squeue -u hd_vm305

# Check detailed job status
scontrol show job JOBID

# Cancel a job
scancel JOBID

# Cancel all your jobs
scancel -u hd_vm305

# View past jobs
sacct -u hd_vm305 --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES%30

# Check partition availability
sinfo -p gpu_mi300
```

### Interactive Sessions

**For debugging** (1 GPU, 4 hours):
```bash
salloc --partition=gpu_mi300 \
       --gres=gpu:1 \
       --cpus-per-gpu=24 \
       --mem-per-gpu=128200mb \
       --time=4:00:00
```

**For testing** (1 GPU, 1 hour):
```bash
salloc --partition=gpu_mi300 \
       --gres=gpu:1 \
       --cpus-per-gpu=24 \
       --mem-per-gpu=128200mb \
       --time=1:00:00
```

**For full training** (4 GPUs, 8 hours):
```bash
salloc --partition=gpu_mi300 \
       --gres=gpu:4 \
       --cpus-per-gpu=24 \
       --mem-per-gpu=128200mb \
       --time=8:00:00
```

### Job Script Template
```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --partition=gpu_mi300
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=128200mb
#SBATCH --time=24:00:00
#SBATCH --output=/u/hd_vm305/thesis-microlens/logs/my_experiment_%j.out
#SBATCH --error=/u/hd_vm305/thesis-microlens/logs/my_experiment_%j.err

# Load environment
source ~/.bashrc
module load devel/cuda/12.1
conda activate microlens

# Set ROCm variables
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3

# Go to work directory
cd /u/hd_vm305/thesis-microlens/code

# Run your command
python train.py --data ../data/raw/events_1M.npz --output ../models/model.pt
```

---

## 🔧 Monitoring

### Check GPU Usage (on compute node)
```bash
# Basic GPU info
rocm-smi

# Continuous monitoring (update every 2 seconds)
watch -n 2 rocm-smi

# GPU memory usage
rocm-smi --showmeminfo vram

# GPU utilization
rocm-smi --showuse
```

### Expected Output
```
========================= ROCm System Management Interface =========================
GPU  Temp   AvgPwr  Power  GPU%   GFX%  MEM%  GPU Memory
0    35.0c  75W     500W   95%    95%   85%   45GB / 128GB
1    36.0c  78W     500W   94%    94%   86%   46GB / 128GB
2    35.5c  76W     500W   95%    95%   84%   44GB / 128GB
3    37.0c  80W     500W   96%    96%   87%   47GB / 128GB
====================================================================================
```

### Check Job Logs
```bash
# View output (real-time)
tail -f ~/thesis-microlens/logs/train_baseline_JOBID.out

# View errors
tail -f ~/thesis-microlens/logs/train_baseline_JOBID.err

# View last 50 lines
tail -n 50 ~/thesis-microlens/logs/train_baseline_JOBID.out

# Search for specific info
grep "Epoch" ~/thesis-microlens/logs/train_baseline_JOBID.out
grep "accuracy" ~/thesis-microlens/logs/train_baseline_JOBID.out
```

---

## 🐛 Troubleshooting

### Issue: PyTorch Not Finding GPUs
```bash
# Check ROCm module
module list

# Should see: devel/cuda/12.1

# If missing, load it
module load devel/cuda/12.1

# Check environment variables
echo $ROCR_VISIBLE_DEVICES
echo $HIP_VISIBLE_DEVICES

# Should both show: 0,1,2,3

# Test GPU detection
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Issue: Out of Memory (OOM)
```bash
# Reduce batch size in training script
python train.py --batch_size 64  # instead of 128

# Or reduce model size (edit architecture in train.py)

# Check memory usage during training
# (in another terminal, ssh to compute node)
watch -n 1 rocm-smi --showmeminfo vram
```

### Issue: Job Stuck in Queue
```bash
# Check partition status
sinfo -p gpu_mi300

# Check your job priority
sprio -u hd_vm305

# If urgent, request smaller allocation
# (more likely to start quickly)
sbatch --gres=gpu:1 slurm/train_baseline.sh  # instead of gpu:4
```

### Issue: Slow Data Loading
```bash
# Use more workers in DataLoader
# Edit train.py:
# DataLoader(..., num_workers=8, pin_memory=True)

# Or: Copy data to local SSD (if available)
cp /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz /tmp/
# Then train from /tmp/ (much faster I/O)
```

### Issue: SLURM Commands Not Found
```bash
# Load SLURM module (if needed)
module load slurm

# Or add to ~/.bashrc
echo "module load slurm" >> ~/.bashrc
source ~/.bashrc
```

---

## 💾 Storage and Quotas

### Check Your Quota
```bash
# Home directory quota
quota -s

# Typical output:
# Filesystem   space   quota   limit   grace   files   quota   limit
# /home        50G     100G    110G            25000   30000   35000
```

### Storage Locations
```bash
# Home directory (backed up, limited space)
/u/hd_vm305/                        # Your home
├── thesis-microlens/               # Code repository
│   ├── code/                       # Scripts (small)
│   ├── slurm/                      # Job scripts (small)
│   └── docs/                       # Documentation (small)

# Project/scratch space (not backed up, large space)
# Ask your advisor about project storage if you need >100GB

# Temporary (local to compute node, very fast but deleted after job)
/tmp/                               # Use for training data during job
```

### Best Practices
1. **Keep code in home**: Small, version controlled
2. **Store large data in scratch**: If you have project space
3. **Use /tmp for training**: Copy data to /tmp at start of job
4. **Clean up regularly**: Delete old models and logs
5. **Compress results**: Use `tar -czf` for completed experiments

### Example: Using /tmp for Fast I/O
```bash
#!/bin/bash
#SBATCH ...

# Copy data to local SSD
echo "Copying data to /tmp..."
cp /u/hd_vm305/thesis-microlens/data/raw/events_1M.npz /tmp/

# Train from /tmp (much faster)
python train.py --data /tmp/events_1M.npz --output ../models/model.pt

# Copy results back
cp ../models/model.pt /u/hd_vm305/thesis-microlens/models/

# Cleanup /tmp (automatic, but good practice)
rm /tmp/events_1M.npz
```

---

## 📊 Performance Optimization

### PyTorch Optimizations
```python
# In train.py, add these for better performance:

# Use mixed precision training (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():  # Automatic mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Use DataLoader optimizations
train_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# Compile model (PyTorch 2.0+)
model = torch.compile(model)  # Faster inference/training
```

### Data Generation Optimization
```bash
# Use all available CPU cores
python simulate.py --n_processes 96  # All cores on the node

# Or let it auto-detect
python simulate.py  # Automatically uses cpu_count()
```

### Batch Job Array (for many experiments)
```bash
# Submit multiple jobs at once
sbatch --array=1-5 slurm/train_array.sh

# In train_array.sh:
#!/bin/bash
#SBATCH --array=1-5  # 5 different experiments

# Map array index to experiment
case $SLURM_ARRAY_TASK_ID in
    1) CADENCE=0.05 ;;
    2) CADENCE=0.10 ;;
    3) CADENCE=0.20 ;;
    4) CADENCE=0.30 ;;
    5) CADENCE=0.40 ;;
esac

python train.py --data ../data/raw/events_cadence_${CADENCE}.npz ...
```

---

## 🔐 Security Best Practices

1. **Don't store passwords in scripts**: Use SSH keys instead
2. **Don't commit sensitive data**: Add to .gitignore
3. **Use private repos**: For your thesis work
4. **Regular backups**: Copy important results to your laptop
5. **Document everything**: Future you will thank present you

---

## 📞 Support

### Cluster Issues
- **Email**: bwunicluster@lists.kit.edu
- **Ticket system**: https://www.scc.kit.edu/dienste/14906.php
- **Documentation**: https://wiki.bwhpc.de/e/BwUniCluster_3.0

### Common Questions
```bash
# How much time is left on my job?
squeue -u hd_vm305 -o "%.18i %.9P %.8j %.8u %.2t %.10M %.10L %.6D %R"

# What's the maximum time limit?
sinfo -p gpu_mi300 -o "%P %l"  # Shows 72:00:00 for gpu_mi300

# How do I get email notifications?
# Add to job script:
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@kit.edu

# How do I check past job efficiency?
seff JOBID
```

---

## ✅ Pre-Flight Checklist

Before submitting your first production job:

- [ ] Environment installed and tested
- [ ] Small test job completed successfully
- [ ] Data paths correct in scripts
- [ ] Output directories exist
- [ ] Logs directory exists
- [ ] Batch scripts are executable
- [ ] You know how to check job status
- [ ] You know how to cancel jobs if needed
- [ ] You've estimated job runtime
- [ ] You've checked storage quota

---

## 🎓 Example Workflow

### Day 1: Setup
```bash
# Morning: Get access, clone repo, setup environment
# Afternoon: Test installation with small dataset
# Evening: Submit first test job
```

### Day 2-3: Baseline Training
```bash
# Submit 1M event baseline training
sbatch slurm/train_baseline.sh

# Monitor progress
watch -n 60 "tail -n 20 logs/train_baseline_*.out"

# Check GPU usage (ssh to compute node)
watch -n 5 rocm-smi
```

### Week 2: Experiments
```bash
# Generate experiment datasets (parallel)
for config in cadence_05 cadence_30 binary_easy binary_hard; do
    sbatch slurm/simulate_${config}.sh
done

# Submit training jobs
for config in cadence_05 cadence_30 binary_easy binary_hard; do
    sbatch slurm/train_${config}.sh
done
```

### Week 3: Analysis
```bash
# Download results to laptop
rsync -avz hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results/ ./local_results/

# Analyze on laptop (Jupyter notebook)
# Generate thesis plots
```

---

**You're now ready to run your thesis experiments on bwUniCluster 3.0!** 🚀

For questions specific to this project, consult [THESIS_GUIDE.md](THESIS_GUIDE.md).