# Microlensing Binary Classification with Deep Learning

**Real-time classification of gravitational microlensing events (PSPL vs Binary) using 1D CNNs**

Master's Thesis Project | University of Heidelberg  
**Author**: Kunal Bhatia (kunal29bhatia@gmail.com)  
**Last Updated**: January 2025

---

## 🎯 Project Overview

This project uses deep learning to classify gravitational microlensing light curves into two categories:
- **PSPL (Point Source Point Lens)**: Single lens events
- **Binary**: Two-body lens systems (planetary or stellar)

### Research Goals

1. **Baseline Performance**: Establish achievable accuracy across diverse binary systems
2. **Observational Effects**: Quantify impact of missing data, photometric errors, and cadence
3. **Real-time Classification**: Enable early detection for triggering follow-up observations
4. **Physical Limits**: Identify which binary configurations are fundamentally hard to distinguish from PSPL

### Key Innovation

**TimeDistributed CNN architecture** enables classification at each timestep, allowing real-time detection as observations arrive—critical for triggering follow-up observations before events complete.

---

## 📁 Repository Structure

```
Thesis/
├── code/
│   ├── config.py              # All experiment configurations
│   ├── simulate.py            # Generate light curves (multiprocessing)
│   ├── train.py               # PyTorch training (GPU-optimized)
│   ├── evaluate.py            # Model evaluation + early detection
│   ├── utils.py               # GPU detection, plotting, helpers
│   └── preflight_check.py     # Pre-submission validation
├── slurm/                     # HPC batch job scripts
│   ├── train_baseline.sh      # Main training job
│   └── interactive.sh         # Interactive GPU session
├── docs/
│   ├── RESEARCH_GUIDE.md      # Complete thesis workflow
│   └── SETUP_GUIDE.md         # Installation and setup
├── data/
│   ├── raw/                   # Simulated datasets (.npz)
│   └── processed/             # Preprocessed data (if needed)
├── models/                    # Trained models (.pt files)
├── results/                   # Experiment outputs
│   └── [experiment]_*/        # Results for each run
│       ├── best_model.pt
│       ├── scaler.pkl
│       ├── metrics.json
│       └── *.png
└── logs/                      # Training logs (SLURM outputs)
```

---

## 🚀 Quick Start

### Prerequisites

**Hardware** (recommended):
- Multi-GPU system (AMD MI300, NVIDIA A100, or similar)
- 64+ GB RAM for large datasets
- Fast storage (SSD)

**Or use CPU** (functional but slower for testing)

**Software**:
- Python 3.8+
- PyTorch 2.0+ (supports both AMD ROCm and NVIDIA CUDA)
- VBMicrolensing for light curve simulation

---

### Installation

#### 1. Clone Repository
```bash
git clone <your-repo-url> Thesis
cd Thesis
```

#### 2. Create Environment
```bash
# Using conda (recommended)
conda create -n microlens python=3.10 -y
conda activate microlens

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

#### 3. Install Dependencies

**For NVIDIA GPUs (CUDA)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For AMD GPUs (ROCm)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

**For CPU only**:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python code/utils.py
```

This will detect your hardware and confirm GPU availability.

---

### Baseline Training Workflow

#### Step 1: Generate Baseline Dataset

The baseline uses **wide parameter ranges** covering:
- **Planetary systems**: q ~ 0.001 (Jupiter-mass planets)
- **Stellar binaries**: q ~ 0.5-1.0 (equal-mass stars)
- **All separations**: s = 0.1 to 10.0 Einstein radii
- **All impact parameters**: u₀ = 0.001 to 1.0

```bash
cd code

# Generate 1M events (500K PSPL + 500K Binary)
# Takes ~2-3 hours on 24-core CPU
python simulate.py \
    --n_pspl 500000 \
    --n_binary 500000 \
    --output ../data/raw/events_baseline_1M.npz \
    --binary_params baseline

# Verify dataset
python -c "import numpy as np; d=np.load('../data/raw/events_baseline_1M.npz'); print(f'Shape: {d[\"X\"].shape}, Labels: {set(d[\"y\"])}')"
```

**What this creates**:
- 1 million light curves
- Each with 1500 time points
- 20% random missing observations (realistic cadence)
- 0.1 magnitude photometric errors (ground-based quality)

---

#### Step 2: Train Baseline Model

```bash
# Single GPU training
python train.py \
    --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline

# Multi-GPU training (automatic detection)
# Will use all available GPUs with DataParallel
python train.py \
    --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt \
    --epochs 50 \
    --batch_size 512 \
    --experiment_name baseline
```

**Training time estimates**:
- 4× AMD MI300A: ~6-8 hours
- 4× NVIDIA A100: ~6-8 hours  
- 1× NVIDIA RTX 4090: ~24-30 hours
- CPU only: ~5-7 days (not recommended)

---

#### Step 3: Evaluate Model

```bash
python evaluate.py \
    --model ../models/baseline.pt \
    --data ../data/raw/events_baseline_1M.npz \
    --output_dir ../results/baseline_eval \
    --early_detection
```

**Outputs**:
- Classification report
- Confusion matrix
- ROC and Precision-Recall curves
- Early detection analysis (performance at 10%, 25%, 33%, 50%, 67%, 83%, 100% observed)

---

## 🔬 Advanced Experiments

After baseline, systematically test:

### 1. Distinct Binary Events
Train on events guaranteed to have caustic crossings:
```bash
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --output ../data/raw/events_distinct.npz \
    --binary_params distinct
```

### 2. Cadence Studies
Test with different observation frequencies:
```bash
# Dense (5% missing - LSST-like)
python simulate.py --cadence 0.05 --output ../data/raw/events_cadence_05.npz

# Sparse (40% missing - poor coverage)
python simulate.py --cadence 0.40 --output ../data/raw/events_cadence_40.npz
```

### 3. Photometric Quality
```bash
# Space-based quality (0.01 mag)
python simulate.py --error 0.01 --output ../data/raw/events_error_low.npz

# Poor ground conditions (0.20 mag)
python simulate.py --error 0.20 --output ../data/raw/events_error_high.npz
```

### 4. Planetary vs Stellar
```bash
# Planetary systems only
python simulate.py --binary_params planetary --output ../data/raw/events_planetary.npz

# Stellar binaries only
python simulate.py --binary_params stellar --output ../data/raw/events_stellar.npz
```

---

## 💻 Hardware Support

### GPU Support

**Automatic Detection**: The code automatically detects and configures:
- **NVIDIA GPUs**: Via CUDA
- **AMD GPUs**: Via ROCm
- **CPU fallback**: If no GPU available

**Multi-GPU**: Automatically uses all available GPUs with `torch.nn.DataParallel`

### Tested Configurations

✅ **AMD**:
- MI300A (128GB HBM3)
- MI250X (128GB HBM2e)
- RX 7900 XTX (24GB GDDR6)

✅ **NVIDIA**:
- H100 (80GB HBM3)
- A100 (40GB/80GB HBM2e)
- RTX 4090 (24GB GDDR6X)
- RTX 3090 (24GB GDDR6X)

✅ **CPU**: Works but ~100× slower

---

## 🔧 Configuration

All parameters in `code/config.py`:

### Binary Parameter Sets

```python
# BASELINE: Wide range (planetary to stellar)
'baseline': {
    's': (0.1, 10.0),      # Separation: close to very wide
    'q': (0.001, 1.0),     # Mass ratio: planetary to equal-mass
    'u₀': (0.001, 1.0),    # Impact: all values
}

# DISTINCT: Guaranteed caustic crossings
'distinct': {
    's': (0.8, 1.5),       # Wide binary (largest caustics)
    'q': (0.01, 0.5),      # Asymmetric
    'u₀': (0.001, 0.15),   # Small impact (must cross)
}

# PLANETARY: Planet-hosting systems
'planetary': {
    'q': (0.0001, 0.01),   # Jupiter/Sun ~ 0.001
}

# STELLAR: Binary stars
'stellar': {
    'q': (0.3, 1.0),       # Near-equal to equal mass
}
```

---

## 📈 Monitoring Training

### Check Progress
```bash
# View training log (updates in real-time)
tail -f results/baseline_*/training.log

# Or if using SLURM
tail -f logs/train_baseline_*.out
```

### GPU Monitoring

**NVIDIA**:
```bash
watch -n 1 nvidia-smi
```

**AMD**:
```bash
watch -n 1 rocm-smi
```

### TensorBoard (optional)
```bash
# If you add TensorBoard logging to train.py
tensorboard --logdir results/
```

---

## 🐛 Troubleshooting

### "No GPUs detected"

**Check installation**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**NVIDIA**: Install CUDA toolkit  
**AMD**: Install ROCm 6.0+

### Out of Memory

**Reduce batch size**:
```bash
python train.py --batch_size 64  # or 32
```

### Slow data loading

**Increase workers**:
```python
# In train.py DataLoader
num_workers=8  # or more
```

**Or copy data to faster storage** (e.g., SSD, /tmp)

---

## 📚 Documentation

- **[Research Guide](docs/RESEARCH_GUIDE.md)**: Complete thesis workflow, experiment design, analysis plan
- **[Setup Guide](docs/SETUP_GUIDE.md)**: Detailed installation for local and HPC systems
- **[Pre-flight Check](code/preflight_check.py)**: Run before starting experiments

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@mastersthesis{bhatia2025,
  title={Real-time Classification of Gravitational Microlensing Events using Deep Learning},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg}
}
```

---

## 📧 Contact

**Author**: Kunal Bhatia  
**Email**: kunal29bhatia@gmail.com  
**Institution**: University of Heidelberg  
**Project**: Master's Thesis in Astrophysics

---

## 🎓 Acknowledgments

- **VBMicrolensing**: Valerio Bozza for the ray-tracing library
- **University of Heidelberg**: Compute resources and supervision
- **Advisor**: [Your advisor's name]

---

## 📄 License

This project is part of a Master's thesis. Code is provided for research and educational purposes.

---

**Status**: Active development  
**Last updated**: January 2025

---

## 🔗 Quick Links

- [Complete Research Workflow](docs/RESEARCH_GUIDE.md)
- [Installation Guide](docs/SETUP_GUIDE.md)
- [Pre-flight Checklist](code/preflight_check.py)