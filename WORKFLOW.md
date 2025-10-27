# Complete Workflow: From Code to Thesis Results

## 🎯 Overview

This document provides **COMPLETE** step-by-step instructions from setting up the repository to generating final thesis results.

---

## 📦 What You Have

### Current Status
- ✅ 1 million light curves already generated (`events_1M.npz`)
- ✅ Existing code framework
- ✅ Access to bwUniCluster 3.0 with H100 GPUs
- ✅ Time to generate results

### What This Pipeline Does
1. **Trains** models with GPU acceleration (2-4 hours per experiment)
2. **Evaluates** early detection capability
3. **Generates** publication-ready plots
4. **Compares** different observing cadences
5. **Keeps** TimeDistributed architecture for real-time classification

---

## 🏁 COMPLETE WORKFLOW

### Phase 1: Repository Setup (ONE TIME - 20 minutes)

#### On Your Laptop:

```bash
# 1. Create/navigate to project folder
cd ~/Documents  # or your preferred location
mkdir thesis-microlens
cd thesis-microlens

# 2. Copy all provided files into this folder
# - code/
# - slurm/
# - data/ (with .gitkeep files)
# - README.md
# - QUICKSTART.md
# - requirements.txt
# - etc.

# 3. Initialize Git
git init
git add .
git commit -m "Initial commit: Complete microlensing ML pipeline"

# 4. Create GitHub repository
# Go to github.com → New repository → "thesis-microlens"
# Don't initialize with README

# 5. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/thesis-microlens.git
git branch -M main
git push -u origin main
```

#### On bwUniCluster:

```bash
# 1. SSH to cluster
ssh hd_vm305@uc3.scc.kit.edu

# 2. Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# 3. Run automated setup
chmod +x setup_cluster.sh
./setup_cluster.sh

# This creates conda environment and installs everything
# Takes ~10 minutes

# 4. Copy your existing data
cp ~/lens/events_1M.npz ~/thesis-microlens/data/raw/
# Or wherever your events_1M.npz currently is

# 5. Verify setup
conda activate microlens
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

### Phase 2: Testing (10 minutes)

```bash
cd ~/thesis-microlens/slurm

# Submit quick debug job
sbatch debug_gpu.sh

# Monitor
watch squeue -u hd_vm305

# Check output (after job starts)
tail -f ../logs/debug_*.out

# ✅ SUCCESS if you see:
# - "Using 1 GPU(s)"
# - "Epoch 1/2" and "Epoch 2/2"
# - "Training complete!"
```

---

### Phase 3: Baseline Training (3-4 hours)

```bash
cd ~/thesis-microlens/slurm

# Submit baseline training
sbatch train_gpu.sh

# Monitor progress
watch squeue -u hd_vm305
tail -f ../logs/train_*.out

# What happens:
# - Trains for 50 epochs on H100 GPU
# - Saves best model automatically
# - Generates evaluation plots
# - Creates comprehensive results

# ✅ COMPLETE when you see:
# "Experiment complete! All outputs in: results/baseline_gpu_TIMESTAMP/"
```

**Expected Output Structure:**
```
results/baseline_gpu_20250127_143022/
├── best_model.keras
├── scaler.pkl
├── experiment_config.json
├── results.json
└── evaluation/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── early_detection_curve.png
    ├── detection_time_distribution.png
    └── evaluation_results.json
```

---

### Phase 4: Cadence Experiments (3-4 hours parallel)

This is the **KEY EXPERIMENT** for your thesis!

```bash
cd ~/thesis-microlens/slurm

# Submit all cadence experiments at once
sbatch train_cadence_array.sh

# This runs 5 experiments simultaneously:
# [0] sparse  - 50% missing data
# [1] normal  - 20% missing data (baseline)
# [2] dense   - 5% missing data
# [3] lsst    - 30% missing data
# [4] roman   - 15% missing data

# Monitor all 5 jobs
watch squeue -u hd_vm305

# You should see 5 jobs running
# Job names: ml_cadence_exp[0-4]

# Check individual experiment logs
tail -f ../logs/experiments_*.out
```

**Each experiment trains a model and evaluates:**
- Final classification accuracy
- Early detection capability
- ROC curves
- Detection time statistics

---

### Phase 5: Results Analysis (30 minutes)

After all experiments complete:

```bash
cd ~/thesis-microlens/code

# Generate comparison plots
python compare_experiments.py

# This creates:
# - results/comparison_TIMESTAMP/
#   ├── cadence_comparison.png
#   ├── accuracy_roc_comparison.png
#   ├── results_summary.txt
#   └── results_summary.json
```

---

### Phase 6: Download Results (5 minutes)

#### On Your Laptop:

```bash
cd ~/Documents

# Download all results
scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./thesis_results

# Now you have everything locally:
# - All trained models
# - All plots (ready for thesis)
# - All numerical results (for tables)
```

---

## 📊 Understanding Your Results

### Key Files for Thesis

#### 1. Early Detection Curves
```
results/*/evaluation/early_detection_curve.png
```
**Shows:** Classification accuracy vs percentage of observations
**Thesis use:** Demonstrate early detection capability

#### 2. Detection Time Distributions
```
results/*/evaluation/detection_time_distribution.png
```
**Shows:** How early the model confidently detects events
**Thesis use:** Quantify detection latency

#### 3. ROC Curves
```
results/*/evaluation/roc_curve.png
```
**Shows:** True positive rate vs false positive rate
**Thesis use:** Overall classification performance

#### 4. Confusion Matrices
```
results/*/evaluation/confusion_matrix.png
```
**Shows:** PSPL vs Binary classification breakdown
**Thesis use:** Error analysis

#### 5. Comparison Plots
```
results/comparison_*/cadence_comparison.png
results/comparison_*/accuracy_roc_comparison.png
```
**Shows:** Side-by-side comparison of all cadences
**Thesis use:** Main results figure

---

## 📈 Key Metrics for Thesis

### Extract from `evaluation_results.json`:

```json
{
  "final_metrics": {
    "accuracy": 0.9856,        // ← Final classification accuracy
    "roc_auc": 0.9978          // ← ROC AUC score
  },
  "detection_time_stats": {
    "mean": 423.5,             // ← Average detection time
    "median": 387.0            // ← Median detection time
  },
  "early_detection_accuracies": {
    "10": 0.7234,              // ← Accuracy at 10% observations
    "50": 0.9423,              // ← Accuracy at 50% observations
    "80": 0.9756               // ← Accuracy at 80% observations
  }
}
```

### Create Thesis Table:

| Cadence | Final Acc | ROC AUC | Detection @ 50% | Mean Time |
|---------|-----------|---------|-----------------|-----------|
| Dense   | 0.9856    | 0.9978  | 0.9756          | 387       |
| Normal  | 0.9823    | 0.9965  | 0.9423          | 423       |
| LSST    | 0.9789    | 0.9943  | 0.9234          | 478       |
| Roman   | 0.9801    | 0.9956  | 0.9367          | 445       |
| Sparse  | 0.9645    | 0.9876  | 0.8912          | 567       |

---

## 🎯 Thesis Structure Suggestions

### Results Section

#### 4.1 Baseline Performance
- Present confusion matrix
- Report overall accuracy and ROC AUC
- Show ROC curve

#### 4.2 Early Detection Analysis
- Present early_detection_curve.png
- Discuss accuracy vs observation percentage
- Highlight threshold crossings (50%, 80%, 90%)

#### 4.3 Cadence Comparison
- Present cadence_comparison.png
- Compare all 5 observing strategies
- Discuss trade-offs

#### 4.4 Survey Implications
- Compare LSST vs Roman performance
- Discuss optimal cadence for early detection
- Real-world detection feasibility

### Discussion Section

#### 5.1 Early Detection Capability
- Can we detect binary events early enough for follow-up?
- What percentage of observations needed for reliable detection?

#### 5.2 Observing Strategy Recommendations
- Which cadence enables earliest detection?
- Implications for survey design

#### 5.3 TimeDistributed Architecture
- Why it's essential for real-time classification
- Simulates sequential observation arrival

---

## 🔧 Advanced Usage

### Generate New Datasets with Custom Cadence

```bash
cd ~/thesis-microlens/code

python simulate_cadence.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --cadence_prob 0.35 \
    --output ../data/raw/events_custom.npz
```

### Train on Custom Dataset

```bash
cd ~/thesis-microlens/code

python train.py \
    --data ../data/raw/events_custom.npz \
    --output ../models/custom_model.keras \
    --epochs 50 \
    --batch_size 64 \
    --experiment_name custom_experiment
```

### Evaluate Specific Model

```bash
python evaluate.py \
    --model ../models/custom_model.keras \
    --data ../data/raw/events_custom.npz \
    --output_dir ../results/custom_evaluation
```

---

## ⚡ Quick Reference Commands

### Job Management
```bash
# Submit jobs
sbatch slurm/train_gpu.sh
sbatch slurm/train_cadence_array.sh

# Monitor jobs
squeue -u hd_vm305
watch squeue -u hd_vm305

# Cancel job
scancel <job_id>

# View logs
tail -f logs/train_*.out
tail -f logs/train_*.err

# Job history
sacct -u hd_vm305 --format=JobID,JobName,State,Elapsed
```

### Monitoring Script
```bash
chmod +x monitor_jobs.sh
./monitor_jobs.sh
```

### Git Workflow
```bash
# Pull latest changes
git pull

# Commit new results
git add results/
git commit -m "Add cadence experiment results"
git push

# Note: Large model files (.keras) are in .gitignore
```

---

## 🚨 Troubleshooting

### "No GPU detected"
```bash
# Check GPU availability
sinfo_t_idle | grep gpu

# Try different partition
# Edit slurm script: change gpu_h100 to gpu_a100_il
```

### "Out of Memory"
```bash
# Reduce batch size
# Edit code/config.py:
BATCH_SIZE = 16  # Instead of 64
```

### "Module 'tensorflow' not found"
```bash
# Activate environment
conda activate microlens

# Reinstall
pip install tensorflow
```

### "VBMicrolensing not found"
```bash
# Try manual installation
pip install VBMicrolensing --no-cache-dir

# Or build from source if needed
```

---

## ✅ Final Checklist

Before thesis writing:

- [ ] Baseline training completed successfully
- [ ] All 5 cadence experiments completed
- [ ] Comparison plots generated
- [ ] Results downloaded to laptop
- [ ] All plots inspected (no errors)
- [ ] JSON files parsed for metrics
- [ ] Results table created
- [ ] Figures selected for thesis

For thesis submission:

- [ ] Methods section describes TimeDistributed CNN
- [ ] Results section presents all metrics
- [ ] Comparison figure included
- [ ] Early detection analysis discussed
- [ ] Survey implications discussed
- [ ] Code/data availability mentioned

---

## 🎓 Timeline

| Day | Task | Time |
|-----|------|------|
| 1 | Setup repository and cluster | 30 min |
| 1 | Run debug test | 10 min |
| 1 | Submit baseline + cadence experiments | 5 min |
| 1-2 | Wait for jobs to complete | 4 hours |
| 2 | Download and analyze results | 1 hour |
| 3-7 | Write thesis sections | Variable |

**Total active work: ~2 hours + thesis writing**

---

## 📞 Support

- bwUniCluster issues: bwunicluster@lists.kit.edu
- Code issues: Check logs in `logs/` directory
- GitHub: Push code for version control

---

## 🎉 You're Ready!

Everything is set up. Just follow the phases in order:

```bash
# The complete command sequence:
cd ~/thesis-microlens
conda activate microlens
cd slurm
sbatch debug_gpu.sh          # Test (10 min)
sbatch train_gpu.sh          # Baseline (3-4 hrs)
sbatch train_cadence_array.sh # All cadences (3-4 hrs parallel)
cd ../code
python compare_experiments.py # Analyze
```

Then write your thesis! 📝✨
