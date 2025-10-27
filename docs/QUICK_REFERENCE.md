# Quick Reference Card

**Thesis Project**: Microlensing Binary Classification  
**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)

Keep this handy for common commands!

---

## 🏃 Quick Commands

### Check System Status
```bash
# Full system check
python code/preflight_check.py

# GPU status
python code/utils.py

# Quick GPU check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

### Generate Data

```bash
cd code

# Small test (5 min)
python simulate.py --n_pspl 1000 --n_binary 1000 \
    --output ../data/raw/test_2k.npz

# Medium size (30 min)
python simulate.py --n_pspl 50000 --n_binary 50000 \
    --output ../data/raw/events_100k.npz

# Baseline (2-3 hours)
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/events_baseline_1M.npz
```

---

### Train Model

```bash
# Quick test (10 min)
python train.py \
    --data ../data/raw/test_2k.npz \
    --output ../models/test.pt \
    --epochs 5 \
    --experiment_name test

# Full baseline (6-8 hours on 4 GPUs)
python train.py \
    --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt \
    --epochs 50 \
    --experiment_name baseline

# Resume training
python train.py \
    --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt \
    --resume results/baseline_*/checkpoint_epoch_10.pt
```

---

### Evaluate Model

```bash
# Basic evaluation
python evaluate.py \
    --model ../models/baseline.pt \
    --data ../data/raw/events_baseline_1M.npz \
    --output_dir ../results/baseline_eval

# With early detection analysis
python evaluate.py \
    --model results/baseline_*/best_model.pt \
    --data ../data/raw/events_baseline_1M.npz \
    --output_dir ../results/baseline_eval \
    --early_detection
```

---

### Monitor Training

```bash
# Watch training log
tail -f results/baseline_*/training.log

# Watch SLURM output
tail -f logs/baseline_*.out

# Check GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Check GPU usage (AMD)
watch -n 1 rocm-smi

# Check job status
squeue -u $USER
```

---

## 📁 Important Paths

```bash
# Data
~/Thesis/data/raw/                     # Input datasets
~/Thesis/data/processed/               # Preprocessed data

# Models
~/Thesis/models/                       # Saved models
~/Thesis/results/[experiment]_*/       # Training results

# Logs
~/Thesis/logs/                         # SLURM outputs
~/Thesis/results/[experiment]_*/training.log  # Training logs
```

---

## 🔧 Common Fixes

### Out of Memory
```bash
# Reduce batch size
python train.py ... --batch_size 64  # or 32

# Or use CPU
CUDA_VISIBLE_DEVICES="" python train.py ...
```

---

### Can't Find Data
```bash
# Check if file exists
ls -lh data/raw/events_baseline_1M.npz

# Check data directory
ls -lh data/raw/

# Generate if missing
cd code
python simulate.py --output ../data/raw/events_baseline_1M.npz
```

---

### GPU Not Detected
```bash
# Check drivers
nvidia-smi  # or rocm-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### SLURM Job Failed
```bash
# Check error log
cat logs/baseline_*.err

# Check output log
cat logs/baseline_*.out

# Check job status
scontrol show job JOBID

# Cancel job
scancel JOBID
```

---

## 🎯 Experiment Workflow

### 1. Generate Dataset
```bash
cd code
python simulate.py --output ../data/raw/my_experiment.npz [options]
```

### 2. Train Model
```bash
python train.py \
    --data ../data/raw/my_experiment.npz \
    --output ../models/my_experiment.pt \
    --experiment_name my_experiment
```

### 3. Evaluate
```bash
# Find your results directory
ls -td ../results/my_experiment_* | head -1

# Evaluate
python evaluate.py \
    --model ../results/my_experiment_*/best_model.pt \
    --data ../data/raw/my_experiment.npz \
    --output_dir ../results/my_experiment_eval \
    --early_detection
```

### 4. Analyze
```bash
# View metrics
cat ../results/my_experiment_eval/metrics.json

# View plots (download to local machine)
scp user@cluster:~/Thesis/results/my_experiment_eval/*.png ./
```

---

## 📊 Standard Experiments

### Baseline
```bash
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/events_baseline_1M.npz \
    --binary_params baseline
```

### Dense Cadence
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_cadence_05.npz \
    --cadence 0.05 --binary_params baseline
```

### Sparse Cadence
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_cadence_40.npz \
    --cadence 0.40 --binary_params baseline
```

### Low Error
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_error_low.npz \
    --error 0.05 --binary_params baseline
```

### High Error
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_error_high.npz \
    --error 0.20 --binary_params baseline
```

### Distinct Events
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_distinct.npz \
    --binary_params distinct
```

### Planetary
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_planetary.npz \
    --binary_params planetary
```

### Stellar
```bash
python simulate.py --n_pspl 100000 --n_binary 100000 \
    --output ../data/raw/events_stellar.npz \
    --binary_params stellar
```

---

## 🔄 Git Workflow

### Daily Work
```bash
# See what changed
git status

# Add changes
git add file1.py file2.md

# Commit
git commit -m "feat: Add feature X"

# Push
git push origin main
```

---

### Before Major Changes
```bash
# Create backup branch
git checkout -b backup-YYYYMMDD
git push origin backup-YYYYMMDD

# Go back to main
git checkout main
```

---

### Good Commit Messages
```bash
git commit -m "feat: Add early detection analysis"
git commit -m "fix: Correct GPU detection for AMD"
git commit -m "docs: Update README with new examples"
git commit -m "refactor: Simplify data loading pipeline"
```

---

## 📧 Getting Help

### Documentation
```bash
# Main docs
cat README.md
cat docs/RESEARCH_GUIDE.md
cat docs/SETUP_GUIDE.md

# Code help
python simulate.py --help
python train.py --help
python evaluate.py --help
```

---

### Contacts
- **You**: kunal29bhatia@gmail.com
- **Advisor**: [Your advisor]
- **HPC Support**: [Your cluster support]

---

## 💾 Backup Important Files

### What to Backup Regularly
```bash
# Models
rsync -avz ~/Thesis/models/ /backup/location/models/

# Results
rsync -avz ~/Thesis/results/ /backup/location/results/

# Code (or use git)
rsync -avz ~/Thesis/code/ /backup/location/code/
```

---

### What NOT to Backup
```bash
# Don't backup (too large or regenerable):
- data/raw/*.npz        # Can regenerate
- logs/*.out            # Not critical
- __pycache__/          # Auto-generated
```

---

## 🎓 Thesis Reminders

### Key Findings to Emphasize
- u₀ > 0.3: Fundamental detection limit (physics, not ML!)
- Cadence matters more than photometric error
- Early detection possible at ~50% completion
- Planetary vs stellar: both have same u₀ limit

### Figures to Make
- [ ] Accuracy vs cadence
- [ ] Accuracy vs photometric error
- [ ] Early detection curve
- [ ] u₀ distribution analysis
- [ ] Confusion matrices for each experiment
- [ ] ROC curves comparison

### Tables to Make
- [ ] Experiment comparison (all metrics)
- [ ] Training time comparison
- [ ] Parameter ranges used
- [ ] Performance by binary type

---

## ⚡ Emergency Commands

### Stop Everything
```bash
# Cancel all your jobs
scancel -u $USER

# Kill local processes
pkill -9 -u $USER python
```

---

### Free Up Space
```bash
# Check usage
df -h

# Clean conda
conda clean --all -y

# Remove old results (CAREFUL!)
rm -rf results/old_experiment_*/
rm -rf data/raw/old_*.npz
```

---

### Reset Environment
```bash
# Deactivate
conda deactivate

# Remove environment
conda env remove -n microlens

# Recreate
conda create -n microlens python=3.10 -y
conda activate microlens
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 📱 One-Liners

```bash
# Count events in dataset
python -c "import numpy as np; print(len(np.load('data/raw/events_baseline_1M.npz')['X']))"

# Check dataset balance
python -c "import numpy as np; d=np.load('data/raw/events_baseline_1M.npz'); print(dict(zip(*np.unique(d['y'], return_counts=True))))"

# Get latest results directory
ls -td results/baseline_* | head -1

# Get best accuracy from metrics
python -c "import json; m=json.load(open('results/baseline_eval/metrics.json')); print(f\"{m['accuracy']:.4f}\")"

# Count parameters in model
python -c "import torch; m=torch.load('models/baseline.pt'); print(sum(p.numel() for p in m['model_state_dict'].values()))"
```

---

**Save this file!**  
Refer to it whenever you need a quick command.

---

**Last Updated**: January 2025