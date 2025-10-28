#!/usr/bin/env python3
"""
Summarize all experiment results
"""
import json
import glob
import numpy as np

experiments = {
    'baseline': 'Baseline (wide u₀)',
    'distinct': 'Distinct (u₀<0.15)',
    'cadence_05': 'Dense Cadence (5%)',
    'cadence_40': 'Sparse Cadence (40%)',
    'error_low': 'Low Error (0.05)',
    'error_high': 'High Error (0.20)',
}

print("="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"\n{'Experiment':<30} {'Accuracy':<12} {'ROC AUC':<12} {'Early (50%)':<12}")
print("-"*80)

results_dict = {}

for exp_name, exp_desc in experiments.items():
    pattern = f'../results/{exp_name}_*/evaluation/metrics.json'
    files = glob.glob(pattern)
    
    if not files:
        print(f"{exp_desc:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        continue
    
    # Load metrics
    with open(files[0]) as f:
        metrics = json.load(f)
    
    accuracy = metrics.get('accuracy', 0)
    roc_auc = metrics.get('roc_auc', 0)
    
    # Early detection at 50%
    early_50 = 'N/A'
    if 'early_detection' in metrics:
        early_50 = metrics['early_detection'].get('50pct', 0)
    
    print(f"{exp_desc:<30} {accuracy:<12.4f} {roc_auc:<12.4f} {early_50 if early_50 == 'N/A' else f'{early_50:.4f}':<12}")
    
    results_dict[exp_name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'early_50': early_50
    }

print("="*80)

# Save summary
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n✓ Summary saved to analysis/results_summary.json")
