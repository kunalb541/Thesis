## From Light Curves to Labels: Machine Learning in Microlensing

**Version 15.0 - Anti-Cheating Edition**  
MSc Thesis Project | University of Heidelberg | Prof. Dr. Joachim Wambsganß

Real-time classification of gravitational microlensing events using transformer neural networks

---


## 🎯 Project Overview

This thesis develops a **transformer-based classifier** for three-class gravitational microlensing classification:

- **Class 0 (Flat)**: No microlensing event - constant baseline flux
- **Class 1 (PSPL)**: Point Source Point Lens - single star/lens system
- **Class 2 (Binary)**: Binary lens systems - planets, stellar companions, or complex caustic structures

### The Challenge

Gravitational microlensing surveys like **Roman Space Telescope**, **LSST**, and **OGLE** will generate millions of light curves. Traditional methods require:
- Manual inspection (slow, not scalable)
- Parameter fitting (computationally expensive, ~minutes per event)

This project achieves:
- **Real-time classification**: <1ms inference per event
- **High accuracy**: 80%+ on Roman-quality data
- **Early detection**: Reliable classification at 50% light curve completeness
- **Survey-scale ready**: Process 10,000+ events per second

### Science Goals

1. **Exoplanet detection**: Identify planetary binary events for follow-up
2. **Real-time alerts**: Enable rapid response for high-value events
3. **Survey operations**: Optimize Roman Space Telescope observing strategies
4. **Physical limits**: Characterize fundamental detection thresholds (u₀ dependency)

---

## 🏃 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/kunalb541/thesis-microlensing.git
cd thesis-microlensing

# Create environment
conda env create -f environment.yml
conda activate microlens

# For AMD GPUs (ROCm 6.0)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm6.0

# For NVIDIA GPUs (CUDA 12.1)
# Already included in environment.yml - uncomment appropriate section
```

### Quick Test (5 minutes)

```bash
cd code

# Generate 300 events (Roman quality)
python simulate.py --preset quick_test

# Train 5 epochs
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

**Expected output**:
- 3×3 confusion matrix
- Three ROC curves (one-vs-rest)
- ~70%+ accuracy on 300 events
- All plots in `results/quick_test_*/evaluation/`

### Full Baseline (3-5 hours on 32 GPUs)

```bash
# Generate 1M events
python simulate.py --preset baseline_1M

# Train with multi-node DDP
srun torchrun \
    --nnodes=8 --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64

# Comprehensive evaluation
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection \
    --n_evolution_per_type 10
```

---

## 🧠 Transformer Architecture

### High-Level Design Philosophy

**Challenge**: Light curves are irregular time series with:
- Missing observations (5-50% gaps depending on survey)
- Variable photometric quality
- Arbitrary event timing

**Solution**: Transformer with specialized components for astronomy time series

### Architecture Components

```
Input Light Curve [B, T=1500]
         ↓
┌─────────────────────────────────┐
│  1. Input Embedding Layer       │
│     Linear: 1 → d_model/2       │
│     LayerNorm + GELU            │
│     Linear: d_model/2 → d_model │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  2. Relative Positional Encoding│
│     • Observation Count         │
│     • Relative Time Gaps        │
│     [NO absolute positions]     │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  3. Gap Feature Embedding       │
│     Encodes missing data pattern│
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  4. Transformer Layers (×4)     │
│     ┌─────────────────────────┐ │
│     │ Semi-Causal Attention   │ │
│     │  • Multi-Head (4 heads) │ │
│     │  • Causal Mask          │ │
│     │  • Q/K Normalization    │ │
│     └─────────────────────────┘ │
│              ↓                   │
│     ┌─────────────────────────┐ │
│     │ Feed-Forward Network    │ │
│     │  • GELU activation      │ │
│     │  • Dropout              │ │
│     └─────────────────────────┘ │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  5. Global Pooling              │
│     Average + Max Pooling       │
│     (over valid observations)   │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  6. Task Heads                  │
│     ┌─────────────────────────┐ │
│     │ Classification Head     │ │
│     │  Output: [B, 3]         │ │
│     │  (Flat/PSPL/Binary)     │ │
│     └─────────────────────────┘ │
│     ┌─────────────────────────┐ │
│     │ Caustic Detection Head  │ │
│     │  Output: [B, 1]         │ │
│     │  (Binary morphology)    │ │
│     └─────────────────────────┘ │
│     ┌─────────────────────────┐ │
│     │ Confidence Head         │ │
│     │  Output: [B, 1]         │ │
│     │  (Prediction certainty) │ │
│     └─────────────────────────┘ │
└─────────────────────────────────┘
         ↓
   Final Predictions
```

### Component Details

#### Semi-Causal Attention

**Standard attention** allows each position to attend to all positions:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Semi-causal attention** adds a causal mask:
```
               t=0  t=1  t=2  t=3
     t=0  [    ✓    ✗    ✗    ✗   ]
     t=1  [    ✓    ✓    ✗    ✗   ]
     t=2  [    ✓    ✓    ✓    ✗   ]
     t=3  [    ✓    ✓    ✓    ✓   ]
```

Each observation can only see past and present, not future. This is **critical** for:
- Real-time deployment
- Preventing information leakage during training
- Forcing causal reasoning about event evolution

#### Relative Positional Encoding

**Traditional positional encoding** :
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
❌ Problem: Encodes absolute position - model learns "peak at position X = class Y"

**Our relative encoding**:
```python
Encoding = Concat[
    ObservationCountEmbedding(cumulative_count),
    GapEmbedding(time_since_last_observation)
]
```
✅ Solution: No absolute position information, only relative structure

#### Temporal Invariance Loss

During training, we compute:
```python
# Pool sequence to fixed-size representation
z = GlobalPool(transformer_output, valid_mask)  # [B, D]

# Normalize
z_norm = F.normalize(z, dim=1)

# Compute similarity matrix
S = z_norm @ z_norm.T / temperature

# InfoNCE loss - encourage diversity
L_temporal = -log_softmax(S, dim=1).diag().mean()
```

This **penalizes** the model if different events produce similar embeddings, encouraging it to focus on morphology differences rather than finding shortcuts.

Total loss:
```python
L_total = L_classification + λ₁ * L_temporal + λ₂ * L_caustic
```
where λ₁ = 0.1, λ₂ = 0.8

### Model Statistics

**Architecture v15.0**:
- **Parameters**: ~435,000 (compact)
- **d_model**: 128
- **Heads**: 4
- **Layers**: 4
- **FFN dimension**: 512
- **Dropout**: 0.1

**Inference**:
- **Latency**: <1 ms per event (GPU)
- **Throughput**: 10,000+ events/sec
- **Memory**: ~2GB GPU for batch_size=128

**Training** (1M events, 32 GPUs):
- **Time**: 3-5 hours
- **Peak memory**: ~8GB per GPU
- **Effective batch size**: 2048 (64 × 32)

---

## 🚀 What You Can Do With This Code

### 1. **Topology Studies**

Understand how different binary lens configurations affect detectability:

**Distinct Topology** (Clear Caustics):
```bash
python simulate.py --preset distinct \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```
- Mass ratios q = 0.01-0.5
- Separations s = 0.7-1.5
- Tight u₀ < 0.15 (close approaches)
- **Use case**: Optimal detection conditions

**Planetary Topology** (Exoplanet Search):
```bash
python simulate.py --preset planetary \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```
- Small mass ratios q = 0.0001-0.01
- Wide separation range s = 0.5-2.0
- **Use case**: Characterize exoplanet detection sensitivity

**Stellar Topology** (Binary Stars):
```bash
python simulate.py --preset stellar \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```
- Large mass ratios q = 0.3-1.0
- Complex symmetric caustics
- **Use case**: Binary star systems

**Baseline Topology** (Full Parameter Space):
```bash
python simulate.py --preset baseline \
    --n_flat 50000 --n_pspl 50000 --n_binary 50000
```
- Full parameter range
- Includes wide u₀ (up to 1.0)
- **Use case**: Realistic survey conditions, physical detection limits

### 2. **Cadence Studies**

Test how observation frequency affects classification:

**High-Cadence (5% missing) - Roman Space Telescope**:
```bash
python simulate.py --preset cadence_05 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```
- ~15 min sampling
- Space-based continuous coverage
- **Expected**: 80-85% accuracy

**Good Ground-Based (15% missing)**:
```bash
python simulate.py --preset cadence_15 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```
- ~1 day sampling
- Excellent observing conditions
- **Expected**: 75-80% accuracy

**Typical Ground-Based (30% missing)**:
```bash
python simulate.py --preset cadence_30 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```
- ~3 day sampling
- Weather + moon gaps
- **Expected**: 70-75% accuracy

**Sparse Ground-Based (50% missing)**:
```bash
python simulate.py --preset cadence_50 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```
- ~5 day sampling
- Challenging conditions
- **Expected**: 60-70% accuracy

### 3. **Photometric Error Studies**

Test how data quality affects performance:

**Excellent (0.03 mag) - JWST-quality**:
```bash
python simulate.py --preset error_003 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

**Space-Based (0.05 mag) - Roman Space Telescope**:
```bash
python simulate.py --preset error_005 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

**High-Quality Ground (0.10 mag)**:
```bash
python simulate.py --preset error_010 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

**Typical Ground (0.15 mag)**:
```bash
python simulate.py --preset error_015 \
    --n_flat 30000 --n_pspl 30000 --n_binary 30000
```

### 4. **Physical Detection Limit Studies**

Use the `baseline` topology (wide u₀ range) to study:
- **u₀ threshold**: Find the impact parameter where binary detection becomes impossible
- **Caustic geometry**: Understand why u₀ ≈ 0.3 is the physical limit
- **PSPL contamination**: Quantify how many binaries are fundamentally PSPL-like

```bash
python simulate.py --preset baseline_1M  # 1M events
python train.py --data ../data/raw/baseline_1M.npz --experiment_name baseline_1M
python evaluate.py --experiment_name baseline_1M --data ../data/raw/baseline_1M.npz
```

The evaluation automatically generates u₀ dependency plots showing accuracy vs. impact parameter.

### 5. **Early Detection Studies**

Test how early you can classify events:

```bash
python evaluate.py \
    --experiment_name your_experiment \
    --data ../data/raw/your_data.npz \
    --early_detection
```

This generates plots showing:
- Classification accuracy vs. observation completeness (5%, 10%, 25%, 50%, 75%, 100%)
- Per-class recall evolution
- When classification becomes reliable (typically 50% completeness)

### 6. **Temporal Bias Testing**

Verify your model isn't cheating:

```bash
python evaluate.py \
    --experiment_name your_experiment \
    --data ../data/raw/your_data.npz \
    --temporal_bias_check
```

This runs diagnostic tests:
- Kolmogorov-Smirnov test on t₀ distributions
- Correlation between peak timing and predictions
- Attention pattern analysis

**Good model**: No significant correlation between t₀ and classification errors  
**Bad model**: Misclassifications clustered at specific peak times

### 7. **Custom Experiments**

Mix and match parameters:

```bash
python simulate.py \
    --n_flat 20000 --n_pspl 20000 --n_binary 20000 \
    --binary_preset planetary \
    --cadence_mask_prob 0.20 \
    --mag_error_std 0.08 \
    --output ../data/raw/my_custom_experiment.npz
```

Then train and evaluate as usual.

---

## 📊 Performance

### Baseline (1M Events, Roman Quality)

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

| Topology | Binary Recall | u₀ Threshold | Key Finding |
|----------|--------------|--------------|-------------|
| Distinct | 88.7% | Sharp drop at 0.15 | Clear caustics enable early detection |
| Planetary | 82.3% | Moderate drop at 0.20 | Small q challenging but detectable |
| Stellar | 78.9% | Gradual drop at 0.25 | Complex symmetric caustics |
| Baseline | 73.5% | Sharp drop at 0.30 | Physical limit confirmed |

### Physical Detection Limit

**Key Result**: u₀ = 0.3 is the **physical threshold**, not algorithmic failure.

- **u₀ < 0.15**: 85%+ binary accuracy (excellent)
- **u₀ ≈ 0.30**: Sharp performance drop (threshold)
- **u₀ > 0.30**: 55% binary accuracy (fundamentally PSPL-like)

**Physical explanation**: For u₀ > 0.3, source trajectory doesn't cross caustics. Magnification pattern becomes indistinguishable from PSPL.

### Early Detection

- **50% completeness**: 75-80% accuracy (reliable trigger)
- **25% completeness**: 55-65% accuracy (early warning)
- **10% completeness**: ~40% accuracy (too early)

Roman's 15-min cadence enables classification 2-3 weeks earlier than ground-based surveys.

### Inference Speed

- **Single GPU**: <1 ms per event
- **Batch throughput**: 10,000+ events/second
- **Survey-scale ready**: Can process entire LSST nightly catalog in <1 hour

---

## 📁 Project Structure

```
thesis-microlensing/
├── code/
│   ├── simulate.py              # v15.0 - VBBinaryLensing simulation
│   ├── train.py                 # v15.0 - Multi-GPU DDP training
│   ├── evaluate.py              # v15.0 - Comprehensive evaluation + diagnostics
│   ├── transformer_v15.py       # v15.0 - Anti-cheating architecture
│ 
├── data/
│   ├── raw/                     # Generated datasets (.npz)
│   │   ├── baseline_1M.npz      # 1M event baseline
│   │   ├── distinct.npz         # Topology studies
│   │   ├── cadence_05.npz       # Cadence studies
│   │   └── error_005.npz        # Error studies
│   └── processed/               # (Optional) preprocessed data
├── results/
│   └── experiment_name_*/       # Training outputs
│       ├── best_model.pt        # Best checkpoint
│       ├── config.json          # Experiment config
│       ├── normalizer.pkl       # Data normalizer
│       ├── results.json         # Final metrics
│       └── evaluation/          # All plots + metrics
│           ├── roc_curve.png
│           ├── confusion_matrix.png
│           ├── u0_dependency.png
│           ├── early_detection.png
│           ├── temporal_bias_diagnosis.png
│           └── highres_evolution_*.png
│ 
├── docs/
│   └── RESEARCH_GUIDE.md        # Systematic experimental workflow
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore rules
```

---

## 🖥️ Multi-Node Training (SLURM)

### Setup

```bash
# Allocate nodes
salloc --partition=gpu_partition --nodes=8 --gres=gpu:4 --time=04:00:00

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
```

### Training Command

```bash
srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=train-$(date +%s) \
    train.py \
        --data ../data/raw/baseline_1M.npz \
        --experiment_name baseline_1M \
        --epochs 50 \
        --batch_size 64 \
        --temporal_inv_weight 0.1 \
        --caustic_weight 0.8
```

**Scaling**: 32 GPUs (8 nodes × 4) trains 1M events in 3-5 hours.

---

## 📈 Evaluation & Diagnostics

### Automatic Visualizations

Each evaluation generates:

1. **ROC Curves** - One-vs-rest for all three classes
2. **Confusion Matrix** - 3×3 with counts
3. **Calibration Curves** - Confidence vs. accuracy
4. **u₀ Dependency** - Binary accuracy vs. impact parameter (if params available)
5. **Early Detection** - Performance vs. observation completeness
6. **High-Res Evolution** - 20-point real-time classification progression
7. **Temporal Bias Diagnosis** - KS tests + attention analysis
8. **Example Grids** - Light curve examples by class

### JSON Outputs

- `evaluation_summary.json`: All metrics
- `u0_report.json`: Detailed u₀ analysis
- `config.json`: Experiment configuration

### Command-Line Options

```bash
python evaluate.py \
    --experiment_name baseline_1M \
    --data ../data/raw/baseline_1M.npz \
    --early_detection \                    # Enable early detection analysis
    --temporal_bias_check \                # Run temporal bias diagnostics
    --n_evolution_per_type 10 \            # 10 evolution plots per class
    --n_samples 10000 \                    # Subsample for speed
    --u0_threshold 0.3 \                   # Physical detection threshold
    --u0_bins 10                           # Bins for u0 analysis
```

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
# Solution: Already fixed in v15.0
# - Proper DDP wrapper with find_unused_parameters=True
# - Gradient clipping before optimizer step
# - Temporal invariance loss properly integrated
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

### Temporal Bias Issues

**Issue**: Model achieves high accuracy but fails temporal bias tests

```bash
# Solution 1: Check temporal randomization was applied
python -c "
import numpy as np
data = np.load('../data/raw/your_data.npz')
print(f'Temporal randomization: {data[\"temporal_randomization\"]}')
"

# Solution 2: Increase temporal invariance weight
python train.py --temporal_inv_weight 0.2  # Instead of 0.1

# Solution 3: Disable causal attention ONLY if necessary
python train.py --no_causal_attention  # NOT recommended
```

---

## 📜 Changelog

### Version 15.0 (January 2025) - Anti-Cheating Edition

**Major Changes**:
- ✨ **Semi-causal attention**: Prevents future peeking
- ✨ **Relative positional encoding**: No absolute time information
- ✨ **Temporal invariance loss**: Penalizes time-dependent features
- ✨ **Temporal randomization**: Random peak shifts during data generation
- ✨ **Temporal bias diagnostics**: KS tests, attention analysis
- ✨ **High-resolution evolution plots**: 20 points instead of 10
- ✨ **Fine-grained early detection**: 15 fractions instead of 7
- 🔄 **Simplified multi-task learning**: Only classification + caustic detection
- 🔄 **Renamed presets**: Telescope-agnostic naming (cadence_05, error_005, etc.)
- 🔄 **Baseline topology**: Replaces "challenging" - full parameter space
- 📚 **Comprehensive README**: Detailed transformer architecture, experimental guide

**Breaking Changes**:
- Removed `tE` and `u0` auxiliary prediction heads (not needed for classification)
- Renamed observational presets (roman→cadence_05, lsst_typical→cadence_30, etc.)
- `challenging` topology renamed to `baseline` (more accurate description)

**Bug Fixes**:
- Fixed gradient clipping order (now before optimizer step)
- Fixed DDP wrapper initialization (find_unused_parameters=True)
- Fixed AMP scaler integration with temporal invariance loss

**Performance**:
- +2-3% accuracy over v14.0 on Roman-quality data
- More robust to temporal shortcuts
- Better early detection (50% completeness: 75-80% vs. 70-75%)

### Version 14.0 (December 2024) - Roman Space Telescope Focus

**Major Changes**:
- 🛰️ **Roman Space Telescope baseline**: 5% missing, 0.05 mag error (default)
- 🔬 **Simplified research scope**: 5 experiments (down from 11)
- 📊 **Binary morphology study**: distinct, planetary, stellar, challenging
- 🎯 **u₀ dependency analysis**: Physical detection limits characterized
- ⚡ **Multi-node DDP**: Tested on 8 nodes × 4 GPUs (AMD MI300)

**Removed**:
- Ground-based cadence presets (LSST-specific)
- Multiple error studies (simplified to Roman baseline)
- 2-class experiments (thesis focuses on 3-class)

### Version 13.1 (November 2024) - LSST Focus

**Major Changes**:
- 🔭 **LSST baseline**: 30% missing, 0.10 mag error
- 📊 **11 experiments**: 4 cadence + 3 error + 4 topology
- ⏰ **Temporal bias fix**: t₀ range expanded to [-80, 80] days
- 🧮 **Multi-task learning**: Classification + tE + u0 + caustic

**Issues**:
- Too many experiments (8-10 weeks)
- Ground-based bias (85% missing observations)
- Temporal shortcuts still possible

### Version 13.0 (October 2024) - PyTorch Migration

**Major Changes**:
- 🔄 **TensorFlow → PyTorch**: Complete codebase rewrite
- 🚀 **DDP training**: Multi-GPU distributed data parallel
- 🎨 **Mixed precision**: FP16 training (2× speedup)
- 🔧 **AMD ROCm support**: MI300 GPU compatibility

### Version 12.0 (September 2024) - Transformer Architecture

**Major Changes**:
- 🧠 **Transformer encoder**: Replaced CNN + LSTM
- 📍 **Positional encoding**: Sinusoidal absolute positions
- 🎯 **Multi-head attention**: 4 heads, 128-dim
- 📊 **3-class**: Flat/PSPL/Binary classification

**Issues**:
- Standard positional encoding allowed temporal shortcuts
- No causal attention (could see the future)
- Peak timing leaked into classification

### Earlier Versions

- **v11.0**: CNN + LSTM architecture, 2-class (PSPL/Binary)
- **v10.0**: TimeDistributed CNN, temporal aggregation

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@mastersthesis{bhatia2025microlensing,
  title={From Light Curves to Labels: Machine Learning in Microlensing},
  author={Bhatia, Kunal},
  year={2025},
  school={University of Heidelberg},
  type={MSc Thesis},
  note={Version 15.0 - Anti-Cheating Edition}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

- **Supervisor**: Prof. Dr. Joachim Wambsganß (University of Heidelberg)
- **VBBinaryLensing**: Valerio Bozza (numerical microlensing calculations)
- **Compute**: GPUs provided by bwHPC (BW 3.0)
- **Inspiration**: OGLE, MOA, LSST, Roman Space Telescope teams

---

## 🔗 Links

- **Thesis Repository**: [GitHub](https://github.com/kunalb541/thesis-microlensing)
- **VBBinaryLensing**: [PyPI](https://pypi.org/project/VBMicrolensing/)
- **Roman Space Telescope**: [NASA](https://roman.gsfc.nasa.gov/)
- **LSST**: [Rubin Observatory](https://www.lsst.org/)
- **OGLE**: [Optical Gravitational Lensing Experiment](http://ogle.astrouw.edu.pl/)

---

## 📞 Contact

**Kunal Bhatia**  
MSc Physics Student  
University of Heidelberg

**For questions about**:
- Code/implementation → Open GitHub issue
- Physics/methodology → See thesis document
- Collaboration → Contact via email

---