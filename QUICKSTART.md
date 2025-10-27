# QUICKSTART GUIDE
## Microlensing Classifier on bwUniCluster 3.0

### 📋 Prerequisites
- Access to bwUniCluster 3.0 (you have: hd_vm305@uc3.scc.kit.edu)
- Your existing `events_1M.npz` dataset
- GitHub account (for version control)

---

## 🚀 PART 1: Setup on Your Laptop (5 minutes)

### Step 1: Create GitHub Repository

```bash
# On your laptop, navigate to where you want to work
cd ~/Documents  # or wherever you prefer

# Download/clone the provided code
# (Assuming you have the microlensing-classifier folder)
cd microlensing-classifier

# Initialize git
git init
git add .
git commit -m "Initial commit: Complete microlensing classifier pipeline"

# Create repo on GitHub (via browser):
# 1. Go to github.com
# 2. Click "New repository"
# 3. Name it: thesis-microlens
# 4. Don't initialize with README (we have one)
# 5. Copy the repository URL

# Connect and push
git remote add origin https://github.com/YOUR_USERNAME/thesis-microlens.git
git branch -M main
git push -u origin main
```

---

## 🖥️ PART 2: Setup on bwUniCluster (10 minutes)

### Step 2: SSH into Cluster

```bash
ssh hd_vm305@uc3.scc.kit.edu
```

### Step 3: Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens
```

### Step 4: Run Setup Script

```bash
chmod +x setup_cluster.sh
./setup_cluster.sh
```

This will:
- Create conda environment
- Install all dependencies
- Create directory structure
- Make scripts executable

### Step 5: Copy Your Existing Dataset

```bash
# If your events_1M.npz is in ~/lens/ or ~/sl/
cp ~/lens/events_1M.npz ~/thesis-microlens/data/raw/

# Verify it's there
ls -lh ~/thesis-microlens/data/raw/events_1M.npz
```

---

## 🧪 PART 3: Test Everything (5 minutes)

### Step 6: Quick Test on Development GPU

```bash
cd ~/thesis-microlens/slurm
sbatch debug_gpu.sh
```

**Monitor the job:**
```bash
# Check if it's running
squeue -u hd_vm305

# Watch it in real-time
watch -n 5 'squeue -u hd_vm305'

# View the output (wait ~5 min for job to start)
tail -f ../logs/debug_*.out
```

**What to expect:**
- Job should start within 1-5 minutes
- Should complete in ~5-10 minutes
- Will train for 2 epochs on small sample
- Success = no errors in debug_*.err file

---

## 🏃 PART 4: Full Training (3-4 hours)

### Step 7: Start Full Training

```bash
cd ~/thesis-microlens/slurm
sbatch train_gpu.sh
```

**This will:**
- Use 1 H100 GPU
- Train for 50 epochs
- Save best model automatically
- Generate evaluation plots
- Take ~3-4 hours

**Monitor progress:**
```bash
# Check job status
squeue -u hd_vm305

# View live training output
tail -f ../logs/train_*.out

# View errors (should be empty)
tail -f ../logs/train_*.err
```

**Training complete when you see:**
```
Experiment complete! All outputs in: results/baseline_gpu_TIMESTAMP/
```

---

## 🔬 PART 5: Cadence Experiments (Thesis Results)

### Step 8: Run All Cadence Experiments

This is the key experiment suite for your thesis!

```bash
cd ~/thesis-microlens/slurm
sbatch train_cadence_array.sh
```

**This submits 5 jobs simultaneously:**
1. **sparse** - 50% missing data (poor cadence)
2. **normal** - 20% missing data (baseline)
3. **dense** - 5% missing data (excellent cadence)
4. **lsst** - 30% missing data (LSST-like)
5. **roman** - 15% missing data (Roman-like)

**Monitor all jobs:**
```bash
squeue -u hd_vm305

# You should see 5 jobs running:
# - ml_cadence_exp[0-4]
```

**Each job takes ~3-4 hours, but they run in PARALLEL**

---

## 📊 PART 6: Get Your Results

### Step 9: Check Results

```bash
cd ~/thesis-microlens/results
ls -lt  # List experiments by date

# View results from baseline experiment
cat baseline_gpu_*/evaluation/evaluation_results.json

# See all the plots generated
ls baseline_gpu_*/evaluation/*.png
```

### Step 10: Download Results to Your Laptop

```bash
# On your laptop (new terminal):
cd ~/Documents
scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./thesis_results

# Now you have all plots and results locally!
```

---

## 📈 Understanding Your Results

After training completes, each experiment folder contains:

```
results/baseline_gpu_20250127_143022/
├── best_model.keras              # Your trained model
├── scaler.pkl                    # Data preprocessing scaler
├── experiment_config.json        # All hyperparameters
├── results.json                  # Training history
└── evaluation/
    ├── confusion_matrix.png      # Classification accuracy
    ├── roc_curve.png            # ROC curve with AUC
    ├── early_detection_curve.png # KEY: Accuracy vs time
    ├── detection_time_distribution.png
    └── evaluation_results.json   # Numerical results
```

### Key Metrics to Report in Thesis:

1. **Final Accuracy** (from evaluation_results.json)
2. **ROC AUC** (from roc_curve.png)
3. **Early Detection Performance** (from early_detection_curve.png)
   - At what % of observations do you reach 80% accuracy?
   - At what % do you reach 90% accuracy?
4. **Mean Detection Time** (from evaluation_results.json)
5. **Comparison Across Cadences** (compare all 5 experiments)

---

## 🎯 What to Do for Your Thesis

### Recommended Workflow:

1. ✅ **Run baseline training** (Step 7) - 1 job
2. ✅ **Run cadence experiments** (Step 8) - 5 jobs in parallel
3. 📊 **Analyze early_detection_curve.png** for each cadence
4. 📊 **Compare all 5 cadences** - which enables earliest detection?
5. 📝 **Write results section** showing:
   - Baseline performance
   - How cadence affects early detection
   - Comparison to LSST/Roman observing strategies
6. 💡 **Discussion**: 
   - Can we detect binary events early enough for follow-up?
   - What's the optimal cadence for early detection?
   - Implications for LSST and Roman missions

---

## 🔧 Troubleshooting

### Job Not Starting?
```bash
# Check queue status
squeue -p gpu_h100

# If all GPUs busy, try A100:
# Edit slurm/train_gpu.sh: change partition to gpu_a100_il
```

### Out of Memory?
```bash
# Edit code/config.py
# Change: BATCH_SIZE = 32  (from 64)
```

### Want to Generate New Data with Different Cadences?
```bash
cd ~/thesis-microlens/code
python simulate_cadence.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --cadence_prob 0.4 \
    --output ../data/raw/events_custom.npz
```

### Need Help?
```bash
# Check logs
cat ~/thesis-microlens/logs/*.err

# Or email: bwunicluster@lists.kit.edu
```

---

## ⏱️ Timeline Summary

| Task | Time | Notes |
|------|------|-------|
| Setup (laptop + cluster) | 20 min | One-time only |
| Debug test | 10 min | Verify everything works |
| Baseline training | 3-4 hours | Single GPU |
| Cadence experiments | 3-4 hours | 5 GPUs in parallel |
| Download & analyze | 30 min | Get your plots |
| **TOTAL** | **~5 hours** | Mostly automated! |

---

## 🎉 Success Checklist

After running everything, you should have:

- [ ] `results/baseline_gpu_*/` with full results
- [ ] `results/sparse_*/evaluation/` with early detection analysis
- [ ] `results/normal_*/evaluation/` with early detection analysis
- [ ] `results/dense_*/evaluation/` with early detection analysis
- [ ] `results/lsst_*/evaluation/` with early detection analysis
- [ ] `results/roman_*/evaluation/` with early detection analysis
- [ ] All confusion matrices
- [ ] All ROC curves
- [ ] All early detection curves
- [ ] Detection time distributions
- [ ] JSON files with all numerical results

---

## 📝 Next Steps for Thesis Writing

1. **Methods Section**: Describe the TimeDistributed CNN architecture
2. **Results Section**: Present accuracy, early detection curves
3. **Comparison Table**: Create table comparing all 5 cadences
4. **Figures**: Use the generated PNG files
5. **Discussion**: Interpret implications for real surveys

---

## 🚀 Ready to Start?

```bash
# Complete command sequence from scratch:

# 1. SSH to cluster
ssh hd_vm305@uc3.scc.kit.edu

# 2. Clone repo (if not done)
cd ~
git clone https://github.com/YOUR_USERNAME/thesis-microlens.git
cd thesis-microlens

# 3. Setup
./setup_cluster.sh

# 4. Copy your data
cp ~/lens/events_1M.npz data/raw/

# 5. Test
cd slurm
sbatch debug_gpu.sh

# 6. Run full experiments
sbatch train_gpu.sh           # Baseline
sbatch train_cadence_array.sh # All cadences

# 7. Monitor
watch squeue -u hd_vm305

# Done! Come back in 4 hours for results.
```

Good luck! 🎓✨
