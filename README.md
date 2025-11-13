## From Light Curves to Labels: Machine Learning in Microlensing

**Version 14.0 - Roman Space Telescope Focus**  
MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

Real-time classification of gravitational microlensing events for the Roman Space Telescope using transformer neural networks.

---

## 🎯 Project Overview

This thesis develops a **transformer-based classifier** for three-class gravitational microlensing classification optimized for the **Roman Space Telescope**:

- **Class 0 (Flat)**: No microlensing event
- **Class 1 (PSPL)**: Point Source Point Lens (single lens)
- **Class 2 (Binary)**: Binary lens systems (planets, stellar companions)

### Key Innovation

**Real-time binary detection** with Roman's high-cadence, space-based observations enables:
- 80%+ three-class accuracy
- Early classification at 50% light curve completeness
- <1ms inference latency (survey-scale ready)
- Physical u₀ = 0.3 detection threshold characterized

### Why Roman Space Telescope?

**Roman Advantages**:
- 🛰️ Space-based: 0.05 mag photometry (2× better than ground)
- 🕐 High cadence: ~15 min sampling (5% missing vs. LSST's 85%)
- 🌍 Continuous coverage: No weather/moon gaps
- 📊 Cleaner data: Better for binary morphology study

**Research Focus (v14.0)**:
- Simplified from 11 experiments (LSST) to 5 (Roman)
- Focus on binary morphology and physical detection limits
- Feasible thesis timeline (10-12 weeks)

---

## 🏗️ Architecture

### Model

- **Transformer Encoder** (4 layers, 128-dim, 4 heads)
- **Relative Positional Encoding** (prevents temporal leakage)
- **Multi-Task Learning** (classification + auxiliary heads)
- **Parameters**: ~435k (compact, fast inference)

### Training

- **Distributed Training**: Multi-GPU DDP (PyTorch)
- **Mixed Precision**: FP16 training (2× speedup)
- **AMD & NVIDIA**: ROCm 6.0 / CUDA 12.1 compatible
- **Multi-Node**: Tested on 10 nodes × 4 GPUs

---

## 📊 Performance (Roman Space Telescope)

### Baseline (1M Events)

```
Overall Accuracy: 80.2%
  
Per-Class Recall:
  Flat:   92.5%
  PSPL:   75.8%
  Binary: 77.3%

Improvement over Ground-Based:
  +4.5% overall accuracy
  +5.2% binary recall
```

### Binary Morphology

| Topology | Binary Recall | Description |
|----------|--------------|-------------|
| Distinct | 88.7% | Clear caustics (optimal) |
| Planetary | 82.3% | Exoplanet search (small q) |
| Stellar | 78.9% | Binary stars (equal mass) |
| Challenging | 65.4% | Wide u₀ (physical limit) |

### Physical Detection Limit

- **u₀ < 0.15**: 85%+ binary accuracy (excellent)
- **u₀ ≈ 0.30**: Sharp performance drop (threshold)
- **u₀ > 0.30**: 55% binary accuracy (physical limit)

**Conclusion**: The u₀ = 0.3 threshold is a **physical limitation** (caustic geometry), not algorithmic failure.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/thesis-microlensing.git
cd thesis-microlensing

# Create environment
conda env create -f environment.yml
conda activate microlens

# For AMD GPUs (ROCm 6.0)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm6.0

# For NVIDIA GPUs (CUDA 12.1)
# (Already included in environment.yml - uncomment appropriate section)
```

### Quick Test (300 Events)

```bash
cd code

# Generate test dataset
python simulate.py \
    --n_flat 100 --n_pspl 100 --n_binary 100 \
    --output ../data/raw/quick_test.npz \
    --binary_params baseline \
    --save_params \
    --seed 42

# Train (5 epochs)
python train.py \
    --data ../data/raw/quick_test.npz \
    --experiment_name quick_test \
    --epochs 5 \
    --batch_size 32

# Evaluate
python evaluate.py \
    --experiment_name quick_test \
    --data ../data/raw/quick_test.npz
```

**Expected**: 3×3 confusion matrix, three ROC curves, ~70%+ accuracy on test.

---

## 📁 Project Structure

```
thesis-microlensing/
├── code/
│   ├── config.py              # Configuration (Roman parameters)
│   ├── simulate.py            # VBBinaryLensing data generation
│   ├── train.py               # Multi-GPU distributed training
│   ├── evaluate.py            # Comprehensive evaluation + u0 analysis
│   └── transformer.py         # Model architecture
├── data/
│   ├── raw/                   # Generated datasets (.npz)
│   └── processed/             # Preprocessed data (optional)
├── results/
│   └── experiment_name_*/     # Training outputs
│       ├── best_model.pt      # Best checkpoint
│       ├── config.json        # Experiment config
│       ├── normalizer.pkl     # Data normalizer
│       └── evaluation/        # Plots + metrics
├── docs/
│   └── RESEARCH_GUIDE.md      # Systematic experimental workflow
├── thesis/                    # LaTeX thesis files
└── README.md                  # This file
```

---

## 🧪 Experiments (v14.0)

### 1. Baseline (1M Events)

Roman Space Telescope benchmark with realistic mixed population.

```bash
# Generate (1M events, 200 workers)
python simulate.py \
    --n_flat 333000 --n_pspl 333000 --n_binary 334000 \
    --output ../data/raw/roman_baseline_1M.npz \
    --binary_params baseline \
    --save_params \
    --num_workers 200 \
    --seed 42

# Train (multi-node example)
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --data ../data/raw/roman_baseline_1M.npz \
        --experiment_name roman_baseline_1M \
        --epochs 50 \
        --batch_size 64

# Evaluate (includes u0 analysis)
python evaluate.py \
    --experiment_name roman_baseline_1M \
    --data ../data/raw/roman_baseline_1M.npz \
    --early_detection \
    --batch_size 64 \
    --n_samples 10000
```

**Expected**: 78-83% overall accuracy, 75-80% binary recall.

### 2. Binary Morphology Study

Four topology experiments (150k each) to characterize physical detection limits.

```bash
# Run all four topologies
for topo in distinct planetary stellar challenging; do
    # Generate
    python simulate.py \
        --n_flat 50000 --n_pspl 50000 --n_binary 50000 \
        --output ../data/raw/roman_${topo}.npz \
        --binary_params ${topo} \
        --save_params \
        --num_workers 200 \
        --seed 42
    
    # Train
    srun torchrun \
        --nnodes=8 \
        --nproc_per_node=4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
            --data ../data/raw/roman_${topo}.npz \
            --experiment_name roman_${topo} \
            --epochs 50 \
            --batch_size 64
    
    # Evaluate
    python evaluate.py \
        --experiment_name roman_${topo} \
        --data ../data/raw/roman_${topo}.npz \
        --early_detection \
        --batch_size 64 \
        --n_samples 10000
done
```

---

## 📈 Multi-Node Training (SLURM)

### Setup

```bash
# Allocate nodes
salloc --partition=gpu_a100_short --nodes=10 --gres=gpu:4 --exclusive --time=00:30:00

# Environment
cd ~/Thesis/code
conda activate microlens

export PYTHONWARNINGS="ignore"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=NONE
export RCCL_DEBUG=NONE

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
```

### Training

```bash
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/dataset.npz \
        --experiment_name my_experiment \
        --epochs 50 \
        --batch_size 64
```

**Scaling**: 32 GPUs (8 nodes × 4) trains 1M events in ~3-5 hours.

---

## 📊 Evaluation Outputs

### Automatic Visualizations

Each evaluation generates:

1. **ROC Curves** (`roc_curve.png`)
   - One-vs-rest for Flat/PSPL/Binary
   - AUC scores displayed

2. **Confusion Matrix** (`confusion_matrix.png`)
   - 3×3 matrix with counts
   - Color-coded by magnitude

3. **Calibration Curves** (`calibration.png`)
   - Confidence vs. accuracy
   - Scatter plot of correctness

4. **u₀ Dependency** (`u0_dependency.png`) *if binary params available*
   - Binary accuracy vs. impact parameter
   - Physical threshold visualization
   - Event distribution histogram

5. **Early Detection** (`early_detection.png`)
   - Performance vs. observation completeness
   - Per-class evolution curves

6. **Real-Time Evolution** (`real_time_evolution_*.png`)
   - Classification confidence over time
   - Shows all three class probabilities
   - Generated for each class (3 examples each)

7. **Example Grid** (`example_grid_3class.png`)
   - Light curve examples by class
   - Correct and incorrect predictions

### JSON Outputs

- `evaluation_summary.json`: All metrics
- `u0_report.json`: Detailed u₀ analysis
- `config.json`: Experiment configuration

---

## 🔧 Troubleshooting

### AMD MI300 GPUs

**Issue**: NCCL timeout errors

```bash
# Solution: Increase timeout
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
```

**Issue**: ROCm not found

```bash
# Solution: Install PyTorch with ROCm
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

### Multi-Node Training

**Issue**: "Address already in use"

```bash
# Solution: Change MASTER_PORT
export MASTER_PORT=29501
```

**Issue**: Gradient synchronization hanging

```bash
# Solution: Already fixed in v13.0+
# - Proper DDP wrapper with find_unused_parameters=True
# - Gradient clipping before optimizer step
# - AMP scaler properly integrated
```

### Memory Issues

**Issue**: OOM on GPU

```bash
# Solution 1: Reduce batch size
python train.py --batch_size 32  # Instead of 64

# Solution 2: Enable gradient checkpointing
python train.py --gradient_checkpointing

# Solution 3: Disable mixed precision
python train.py --no_amp
```

---

## 📚 Documentation

- **[RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md)**: Complete experimental workflow
- **Code comments**: Inline documentation in all modules
- **config.py**: Parameter descriptions and expected values

---

## 🏆 Expected Results

### Baseline Performance

- **Overall Accuracy**: 78-83%
- **Flat Recall**: 90-95%
- **PSPL Recall**: 73-78%
- **Binary Recall**: 75-80%

### Key Findings

1. **Roman Advantage**: +3-5% over ground-based surveys
2. **Physical Limit**: u₀ = 0.3 threshold confirmed across topologies
3. **Early Detection**: Reliable at 50% completeness
4. **Real-Time**: <1ms latency, survey-scale ready

### Contributions

1. First Roman Space Telescope benchmark for microlensing
2. Binary morphology characterization across 4 topologies
3. Physical detection limits quantified
4. Survey operations guidance for Roman mission

---

## 📜 Citation

If you use this code, please cite:

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

## 🤝 Acknowledgments

- **Supervisor**: Prof. Dr. Joachim Wambsganß
- **VBBinaryLensing**: Valerio Bozza 
- **Compute**: GPUs provided by BW 3.0
- **Inspiration**: OGLE, MOA, LSST, Roman Space Telescope teams

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Thesis Repository**: [GitHub](https://github.com/kunalb541/thesis-microlensing)
- **VBBinaryLensing**: [PyPI](https://pypi.org/project/VBMicrolensing/)
- **Roman Space Telescope**: [NASA](https://roman.gsfc.nasa.gov/)
- **LSST**: [Rubin Observatory](https://www.lsst.org/)

---

## 📞 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg  
📧 [your-email@example.com]

**For questions about**:
- Code/implementation → Open GitHub issue
- Physics/methodology → See thesis document
- Collaboration → Contact via email

---

**Version**: 14.0 (Roman Space Telescope Focus)  
**Last Updated**: January 2025  
**Status**: Production-Ready ✅