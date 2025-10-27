#!/bin/bash
# Automated benchmark suite runner
# Generates all datasets and trains all models systematically

echo "=========================================="
echo "SYSTEMATIC BENCHMARK SUITE"
echo "=========================================="
echo ""
echo "This will run:"
echo "1. Baseline (1M events)"
echo "2. Cadence experiments (5%, 20%, 30%)"
echo "3. Error experiments (0.05, 0.20 mag)"
echo "4. Binary difficulty (easy, hard)"
echo ""
echo "Estimated total time: 24-48 hours on 4x MI300 GPUs"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Activate environment
source ~/.bashrc
conda activate microlens
cd ~/thesis-microlens/code

# ============================================================================
# EXPERIMENT 1: BASELINE (uses existing 1M dataset)
# ============================================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 1: BASELINE"
echo "=========================================="

if [ -f ../data/raw/events_1M.npz ]; then
    echo "Using existing 1M dataset"
else
    echo "ERROR: baseline dataset not found at ../data/raw/events_1M.npz"
    echo "Please ensure you have the 1M events dataset"
    exit 1
fi

python train.py \
    --data ../data/raw/events_1M.npz \
    --output ../models/baseline.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name baseline

python evaluate.py \
    --model ../models/baseline.keras \
    --data ../data/raw/events_1M.npz \
    --output_dir ../results/baseline

echo "✓ Baseline complete"

# ============================================================================
# EXPERIMENT 2: CADENCE EXPERIMENTS
# ============================================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 2: CADENCE EXPERIMENTS"
echo "=========================================="

# Dense cadence (5% missing)
echo ""
echo "--- Dense cadence (95% coverage) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --cadence 0.05 \
    --output ../data/raw/events_cadence_05.npz

python train.py \
    --data ../data/raw/events_cadence_05.npz \
    --output ../models/cadence_05.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name cadence_05

python evaluate.py \
    --model ../models/cadence_05.keras \
    --data ../data/raw/events_cadence_05.npz \
    --output_dir ../results/cadence_05

echo "✓ Dense cadence complete"

# Sparse cadence (30% missing)
echo ""
echo "--- Sparse cadence (70% coverage) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --cadence 0.30 \
    --output ../data/raw/events_cadence_30.npz

python train.py \
    --data ../data/raw/events_cadence_30.npz \
    --output ../models/cadence_30.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name cadence_30

python evaluate.py \
    --model ../models/cadence_30.keras \
    --data ../data/raw/events_cadence_30.npz \
    --output_dir ../results/cadence_30

echo "✓ Sparse cadence complete"

# ============================================================================
# EXPERIMENT 3: PHOTOMETRIC ERROR EXPERIMENTS
# ============================================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 3: PHOTOMETRIC ERROR EXPERIMENTS"
echo "=========================================="

# Low error (space-based)
echo ""
echo "--- Low photometric error (0.05 mag) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --error 0.05 \
    --output ../data/raw/events_error_low.npz

python train.py \
    --data ../data/raw/events_error_low.npz \
    --output ../models/error_low.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name error_low

python evaluate.py \
    --model ../models/error_low.keras \
    --data ../data/raw/events_error_low.npz \
    --output_dir ../results/error_low

echo "✓ Low error complete"

# High error
echo ""
echo "--- High photometric error (0.20 mag) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --error 0.20 \
    --output ../data/raw/events_error_high.npz

python train.py \
    --data ../data/raw/events_error_high.npz \
    --output ../models/error_high.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name error_high

python evaluate.py \
    --model ../models/error_high.keras \
    --data ../data/raw/events_error_high.npz \
    --output_dir ../results/error_high

echo "✓ High error complete"

# ============================================================================
# EXPERIMENT 4: BINARY DIFFICULTY EXPERIMENTS
# ============================================================================
echo ""
echo "=========================================="
echo "EXPERIMENT 4: BINARY DIFFICULTY EXPERIMENTS"
echo "=========================================="

# Easy binaries
echo ""
echo "--- Easy binaries (clear caustic crossings) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --binary_difficulty easy \
    --output ../data/raw/events_binary_easy.npz

python train.py \
    --data ../data/raw/events_binary_easy.npz \
    --output ../models/binary_easy.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name binary_easy

python evaluate.py \
    --model ../models/binary_easy.keras \
    --data ../data/raw/events_binary_easy.npz \
    --output_dir ../results/binary_easy

echo "✓ Easy binaries complete"

# Hard binaries
echo ""
echo "--- Hard binaries (PSPL-like) ---"
python simulate.py \
    --n_pspl 100000 \
    --n_binary 100000 \
    --binary_difficulty hard \
    --output ../data/raw/events_binary_hard.npz

python train.py \
    --data ../data/raw/events_binary_hard.npz \
    --output ../models/binary_hard.keras \
    --epochs 50 \
    --batch_size 128 \
    --experiment_name binary_hard

python evaluate.py \
    --model ../models/binary_hard.keras \
    --data ../data/raw/events_binary_hard.npz \
    --output_dir ../results/binary_hard

echo "✓ Hard binaries complete"

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Generating comparison plots..."

python -c "
import json
import os
import glob

# Collect all results
results_dir = '../results'
experiments = {}

for exp_dir in glob.glob(os.path.join(results_dir, '*')):
    if not os.path.isdir(exp_dir):
        continue
    
    exp_name = os.path.basename(exp_dir)
    metrics_file = os.path.join(exp_dir, 'metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        experiments[exp_name] = {
            'roc_auc': metrics.get('roc_auc', 0),
            'pr_auc': metrics.get('pr_auc', 0),
        }

# Print summary table
print('\n' + '='*80)
print('BENCHMARK RESULTS SUMMARY')
print('='*80)
print(f'{'Experiment':<30} {'ROC AUC':<12} {'PR AUC':<12}')
print('-'*80)

for exp_name in sorted(experiments.keys()):
    metrics = experiments[exp_name]
    print(f'{exp_name:<30} {metrics[\"roc_auc\"]:<12.4f} {metrics[\"pr_auc\"]:<12.4f}')

print('='*80)

# Save summary
with open('../results/benchmark_summary.json', 'w') as f:
    json.dump(experiments, f, indent=2)

print('\nSummary saved to ../results/benchmark_summary.json')
"

echo ""
echo "Results are in: ~/thesis-microlens/results/"
echo ""
echo "Download to your laptop with:"
echo "  scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./thesis_results"
echo ""
echo "Next steps:"
echo "1. Review all metrics.json files"
echo "2. Create comparison figures"
echo "3. Write thesis discussion!"
echo ""
