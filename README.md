# Gravitational Microlensing Event Classification

**MSc Thesis** | University of Heidelberg  
Supervisor: Prof. Dr. Joachim Wambsganß | Co-supervisor: Dr. Yiannis Tsapras

---

## Overview

Classification of gravitational microlensing events into three categories:

- **Flat**: No lensing signal
- **PSPL**: Point Source Point Lens events
- **Binary**: Binary lens systems (planetary or stellar companions)

The classifier uses a CNN-LSTM architecture with hierarchical two-stage classification, designed for variable-length photometric time series from the Nancy Grace Roman Space Telescope.

---

## Pipeline

The workflow consists of three stages: data simulation, model training, and evaluation.

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  simulate   │  →   │    train    │  →   │  evaluate   │
│             │      │             │      │             │
│  Generate   │      │  Train CNN- │      │  Compute    │
│  synthetic  │      │  LSTM with  │      │  metrics &  │
│  light      │      │  hierarchi- │      │  generate   │
│  curves     │      │  cal loss   │      │  figures    │
└─────────────┘      └─────────────┘      └─────────────┘
     ↓                     ↓                    ↓
   .h5 file           checkpoints         plots & JSON
```

### Stage 1: Data Simulation (`simulate.py`)

Generates synthetic microlensing light curves using the VBBinaryLensing library.

```bash
python simulate.py \
    --n_flat 100000 \
    --n_pspl 100000 \
    --n_binary 100000 \
    --binary_preset baseline \
    --output ../data/raw/train.h5 \
    --num_workers 32 \
    --seed 42
```

**Binary presets** control the parameter space for binary lens events:

| Preset | Mass Ratio (q) | Separation (s) | Impact (u₀) | Description |
|--------|----------------|----------------|-------------|-------------|
| `baseline` | 10⁻⁴ – 1.0 | 0.3 – 3.0 | 0.001 – 1.0 | Full parameter space |
| `distinct` | 0.1 – 1.0 | 0.8 – 1.2 | 0.001 – 0.3 | Caustic-crossing events |
| `planetary` | 10⁻⁴ – 10⁻² | 0.6 – 1.6 | 0.001 – 0.3 | Exoplanet regime |
| `stellar` | 0.1 – 1.0 | 0.3 – 3.0 | 0.001 – 0.5 | Binary stars |

**Output format** (HDF5):
- `flux`: Magnification values (A=1.0 baseline, A>1 magnified, A=0 masked)
- `delta_t`: Time intervals between observations
- `labels`: Class labels (0=Flat, 1=PSPL, 2=Binary)
- `timestamps`: Observation times in days
- `params_{class}`: Physical parameters per class

For HPC job submission:
```bash
sbatch simulate.sbatch
```

### Stage 2: Training (`train.py`)

Trains the classifier using distributed data-parallel training.

```bash
# Single GPU
python train.py \
    --data ../data/raw/train.h5 \
    --output ../results \
    --epochs 150 \
    --batch-size 256 \
    --lr 5e-4 \
    --d-model 64 \
    --n-layers 4 \
    --hierarchical \
    --use-aux-head
```

**Hierarchical classification** uses two binary decisions:

```
Stage 1: P(deviation) = σ(logit₁)     →  Flat vs Non-Flat
Stage 2: P(PSPL|dev) = σ(logit₂)      →  PSPL vs Binary

Final probabilities:
  P(Flat)   = 1 - P(deviation)
  P(PSPL)   = P(deviation) × P(PSPL|deviation)
  P(Binary) = P(deviation) × (1 - P(PSPL|deviation))
```

**Loss function**:
```
L = λ₁·BCE(stage1) + λ₂·BCE(stage2) + λ_aux·NLL(3-class)
```

**Distributed training** on HPC:
```bash
sbatch train.sbatch
```

The sbatch script handles:
- Multi-node distributed training with `torchrun`
- Automatic checkpoint resumption on job timeout
- Data caching to `/tmp` for faster I/O

**Checkpoints** are saved to `../results/checkpoints/<preset>/<experiment>/`:
- `best.pt`: Checkpoint with highest validation accuracy
- `checkpoints/checkpoint_latest.pt`: Most recent (for resumption)
- `config.json`: Training configuration

### Stage 3: Evaluation (`evaluate.py`)

Computes metrics and generates visualizations.

```bash
python evaluate.py \
    --experiment-name ../results/checkpoints/baseline/d64_l4_hier_*/best.pt \
    --data ../data/test/test.h5 \
    --batch-size 128 \
    --colorblind-safe
```

For HPC:
```bash
sbatch eval.sbatch all  # Run all four cross-evaluations
```

**Generated outputs**:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Normalized confusion matrix |
| `roc_curves.png` | ROC curves with AUC and bootstrap CI |
| `calibration.png` | Reliability diagram and confidence distribution |
| `per_class_metrics.png` | Precision, recall, F1 per class |
| `u0_dependency.png` | Binary accuracy vs impact parameter |
| `temporal_bias_check.png` | KS test for temporal selection bias |
| `evolution_*.png` | Probability evolution over time |
| `evaluation_summary.json` | All metrics in JSON format |

---

## Installation

```bash
# Clone and create environment
git clone https://github.com/kunalb541/Thesis.git
cd Thesis
conda env create -f environment.yml
conda activate microlens

# Optional: Flash Attention (requires CUDA, provides ~2x speedup for attention)
pip install flash-attn --no-build-isolation

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import VBBinaryLensing; print('VBBinaryLensing OK')"
```

**GPU configuration**: The default `environment.yml` uses CUDA 12.1. For other setups:

- **CUDA 11.8**: Change `pytorch-cuda=12.1` to `pytorch-cuda=11.8`
- **AMD ROCm**: Install PyTorch via pip after creating the environment
- **CPU only**: Replace PyTorch lines with `pytorch::cpuonly`

---

## Model Architecture

```
Input: [flux, Δt, mask] → Linear(3, d_model)
           ↓
Causal Conv Blocks (depthwise separable, dilation 1 & 2)
           ↓
Bidirectional LSTM (n_layers)
           ↓
Layer Norm + Attention Pooling
           ↓
Classification Head:
  ├── Stage 1: Linear → σ → P(deviation)
  ├── Stage 2: Linear → σ → P(PSPL|deviation)  
  └── Auxiliary: Linear → softmax → 3-class
           ↓
Output: log P(Flat), log P(PSPL), log P(Binary)
```

**Configuration**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Hidden dimension |
| `n_layers` | 4 | LSTM layers |
| `dropout` | 0.3 | Dropout rate |
| `window_size` | 5 | Convolution kernel size |

---

## Physical Parameters

**Observational setup** (Roman-like):

| Parameter | Value |
|-----------|-------|
| Cadence | 15 minutes |
| Season duration | 72 days |
| Missing observations | 5% (random) |
| Source magnitude | 18–24 AB mag |

**Microlensing parameters**:

| Parameter | Range | Unit |
|-----------|-------|------|
| Einstein timescale (t_E) | 5–30 | days |
| Peak time (t₀) | 10–90% of season | days |
| Impact parameter (u₀) | 0.001–1.0 | Einstein radii |
| Binary separation (s) | 0.1–3.0 | Einstein radii |
| Mass ratio (q) | 10⁻⁴–1.0 | — |

---

## Project Structure

```
Thesis/
├── code/
│   ├── simulate.py       # Data generation
│   ├── train.py          # Model training
│   ├── model.py          # Network architecture
│   ├── evaluate.py       # Evaluation suite
│   ├── simulate.sbatch   # HPC job: simulation
│   ├── train.sbatch      # HPC job: training
│   └── eval.sbatch       # HPC job: evaluation
├── data/
│   ├── raw/              # Training data
│   └── test/             # Test data
├── results/
│   └── checkpoints/      # Model outputs
├── environment.yml
├── README.md
└── LICENSE
```

---

## Version

v7.0.0 — January 2025

Component versions:
- `model.py` v7.1.0
- `simulate.py` v7.0.0  
- `train.py` v7.0.0
- `evaluate.py` v7.0.0

---

## Citation

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={From Light Curves to Labels: Machine Learning in Microlensing},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis}
}
```

---

## References

- OGLE Survey: http://ogle.astrouw.edu.pl/
- MOA Collaboration: https://www.massey.ac.nz/~iabond/moa/
- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- VBBinaryLensing: Bozza (2010), MNRAS 408, 2188

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgments

- Prof. Dr. Joachim Wambsganß (Supervisor)
- Dr. Yiannis Tsapras (Co-supervisor)
- bwForCluster HPC facility
