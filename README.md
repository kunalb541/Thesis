# Microlensing Binary Classification with Deep Learning

**Real-time classification of gravitational microlensing events (PSPL vs Binary) using 1D CNNs**

Master's Thesis Project | [Your Name] | [Your University] | 2025

---

## 🎯 Quick Start

Currently running baseline training on 1M events. See [THESIS_GUIDE.md](docs/THESIS_GUIDE.md) for complete workflow.

### Prerequisites
- Access to bwUniCluster 3.0 (AMD MI300 GPUs)
- Existing dataset: `events_1M.npz` (1 million light curves)

### Current Status
✅ Data: 1M events (500K PSPL + 500K Binary)  
🔄 Baseline training: In progress  
⏳ Systematic benchmarking: Pending

---

## 📊 Project Overview

This project benchmarks deep learning performance for classifying gravitational microlensing events under different observational conditions:

### Research Questions
1. **What's the best achievable performance?** (ideal conditions)
2. **How does observing cadence affect classification?** (5-40% missing data)
3. **How early can we detect binary events?** (partial light curves)
4. **What's the physical detection limit?** (which binaries are intrinsically hard?)

### Key Innovation
**TimeDistributed CNN architecture** enables real-time classification as observations arrive sequentially, critical for triggering follow-up observations.

---

## 🗂️ Repository Structure
```
thesis-microlens/
├── code/          # Python scripts
├── slurm/         # Batch job scripts
├── docs/          # Documentation
├── data/          # Datasets (not in git)
├── models/        # Trained models (not in git)
├── results/       # Experiment outputs (not in git)
└── logs/          # SLURM logs (not in git)
```

---

## 🚀 Workflow

### 1. Baseline Training (Current)
```bash
sbatch slurm/train_baseline.sh
```
Expected: 95-96% accuracy, ROC AUC ~0.98, Training time: 6-12 hours

### 2. Systematic Benchmarking (Next)
After baseline completes, run experiments varying:
- **Cadence**: 5%, 20%, 30% missing observations
- **Binary difficulty**: Easy (caustic-crossing) vs Hard (PSPL-like)
- **Photometric error**: 0.05-0.20 magnitudes

### 3. Analysis & Thesis Writing
Generate comparison plots and write results section.

See [docs/THESIS_GUIDE.md](docs/THESIS_GUIDE.md) for detailed instructions.

---

## 📈 Expected Results

| Scenario | Configuration | Expected Accuracy |
|----------|--------------|-------------------|
| **Best case** | Dense cadence + low error + easy binaries | 98-99% |
| **Baseline** | 20% missing + 0.1 mag + standard binaries | 95-96% |
| **Worst case** | Sparse cadence + high error + hard binaries | 82-88% |

**Key Finding**: ~15-20% of binary events are intrinsically PSPL-like (high u₀) and fundamentally hard to distinguish.

---

## 🛠️ Installation

See [docs/CLUSTER_SETUP.md](docs/CLUSTER_SETUP.md) for complete setup instructions.

**Quick version:**
```bash
conda create -n microlens python=3.10 -y
conda activate microlens
pip install -r requirements.txt
```

---

## 📝 Key Files

- **`code/config.py`**: All experimental parameters
- **`code/simulate.py`**: Generate light curves with configurable parameters
- **`code/train.py`**: GPU-optimized training (PyTorch + AMD ROCm)
- **`code/evaluate.py`**: Comprehensive evaluation + early detection analysis
- **`slurm/train_baseline.sh`**: Main training job
- **`docs/THESIS_GUIDE.md`**: Complete thesis workflow

---

## 📖 Documentation

- **[THESIS_GUIDE.md](docs/THESIS_GUIDE.md)**: Complete workflow from setup to thesis writing
- **[PARAMETERS.md](docs/PARAMETERS.md)**: Binary parameters and why they matter
- **[CLUSTER_SETUP.md](docs/CLUSTER_SETUP.md)**: bwUniCluster 3.0 setup guide

---

## 🎓 Thesis Contributions

1. **Systematic benchmarking** of CNN performance across observational conditions
2. **Real-time classification** capability via TimeDistributed architecture
3. **Physical interpretation** connecting performance to caustic crossing physics
4. **Survey optimization** guidance for LSST/Roman telescope strategies

---

## 📧 Contact

For questions about this project: [Your Email]  
For cluster issues: bwunicluster@lists.kit.edu

---

## 📄 License

This project is part of a Master's thesis at [Your University].

---

**Status**: Baseline training in progress ⏳  
**Last updated**: January 2025
