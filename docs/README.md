# Microlensing Binary Classification Thesis Project

## Project Structure
```
thesis-microlens/
├── code/
│   ├── config.py              # Configuration for all experiments
│   ├── simulate.py            # Generate light curves
│   ├── train.py               # Train CNN classifier
│   ├── evaluate.py            # Evaluate models
│   ├── experiments.py         # Run systematic experiments
│   └── utils.py               # Helper functions
├── slurm/
│   ├── simulate.sh            # SLURM job for simulation
│   ├── train.sh               # SLURM job for training
│   └── experiments.sh         # SLURM job for experiments
├── data/
│   ├── raw/                   # Raw simulated data
│   └── processed/             # Preprocessed data
├── models/                    # Saved models
├── results/                   # Experiment results
└── logs/                      # SLURM logs
```

## Quick Start on bwUniCluster 3.0

### 1. Initial Setup
```bash
# Clone repo
cd ~
git clone <your-repo-url> thesis-microlens
cd thesis-microlens

# Create conda environment
module load devel/cuda/12.1
conda create -n microlens python=3.10 -y
conda activate microlens

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn
pip install tensorflow[and-cuda] scikit-learn tqdm
pip install VBMicrolensing

# Create directories
mkdir -p data/raw data/processed models results logs
```

### 2. Interactive GPU Session (for debugging)
```bash
# Get GPU node
salloc --partition=gpu_mi300 --gres=gpu:4 --cpus-per-gpu=24 --mem-per-gpu=128200mb --time=8:00:00

# Once allocated
conda activate microlens
cd ~/thesis-microlens/code

# Test training on existing 1M dataset
python train.py --data ../data/raw/events_1M.npz \
                --output ../models/baseline.keras \
                --epochs 50 \
                --batch_size 128

# Exit when done
exit
```

### 3. Run Full Experiment Suite (Batch Jobs)
```bash
# Submit experiment automation
cd ~/thesis-microlens
sbatch slurm/experiments.sh
```

## Workflow

### Phase 1: Baseline Model (Your Current 1M Dataset)
Train on existing data to establish baseline performance.

### Phase 2: Systematic Experiments
Test different configurations:
1. **Cadence variations** (5%, 10%, 20%, 30%, 40% missing)
2. **Binary parameter ranges** (distinct caustic crossings)
3. **Early detection** (truncated time series)
4. **Observation windows** (different survey strategies)

### Phase 3: Analysis
Compare results and identify optimal configurations for LSST/Roman.

## Key Files

- `code/config.py` - All experimental parameters
- `code/experiments.py` - Automated experiment runner
- `slurm/experiments.sh` - SLURM job for full experiment suite

## Monitoring

```bash
# Check job status
squeue -u hd_vm305

# Check GPU usage (on allocated node)
rocm-smi

# View logs
tail -f logs/experiment_*.out
```

## Results Structure

Results are saved in `results/` with naming convention:
```
experiment_{cadence}_{binary_params}_{timestamp}/
├── model.keras
├── metrics.json
├── confusion_matrix.png
├── roc_curve.png
└── training_history.png
```

## Notes

- **TimeDistributed preserved**: All architectures use TimeDistributed for real-time classification
- **Binary parameters unchanged**: Original ranges kept for distinct caustic crossings
- **GPU optimized**: Batch size and memory usage tuned for AMD MI300 GPUs
- **Reproducible**: All random seeds fixed for consistent results
