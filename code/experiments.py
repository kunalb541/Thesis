"""
Automated experiment runner for systematic benchmarking
Runs multiple configurations and compares results
"""

import numpy as np
import os
import sys
import json
import subprocess
from datetime import datetime
from config import EXPERIMENTS, BASE_DIR, DATA_DIR, MODEL_DIR, RESULTS_DIR

def run_experiment(exp_name, exp_config):
    """Run a single experiment"""
    print("=" * 80)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"Description: {exp_config['description']}")
    print(f"Configuration: {exp_config}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Determine data file
    if exp_name == 'baseline':
        data_file = os.path.join(DATA_DIR, 'events_1M.npz')
        if not os.path.exists(data_file):
            print(f"ERROR: Baseline data not found at {data_file}")
            print("Please ensure you have the 1M events dataset")
            return False
    else:
        # For other experiments, generate new data
        data_file = os.path.join(DATA_DIR, f'events_{exp_name}.npz')
        
        if not os.path.exists(data_file):
            print(f"Generating data for {exp_name}...")
            # TODO: Call simulate.py with custom parameters
            # For now, assume data exists or skip
            print(f"WARNING: Data file {data_file} not found. Skipping data generation.")
            print("You can generate it manually with simulate.py")
            return False
    
    # Train model
    model_path = os.path.join(MODEL_DIR, f'model_{exp_name}_{timestamp}.keras')
    
    print(f"\nTraining model...")
    train_cmd = [
        'python', 'train.py',
        '--data', data_file,
        '--output', model_path,
        '--epochs', '50',
        '--batch_size', '128',
        '--experiment_name', exp_name
    ]
    
    result = subprocess.run(train_cmd, cwd=os.path.join(BASE_DIR, 'code'))
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {exp_name}")
        return False
    
    # Evaluate model
    print(f"\nEvaluating model...")
    eval_cmd = [
        'python', 'evaluate.py',
        '--model', model_path,
        '--data', data_file,
        '--output_dir', exp_dir
    ]
    
    result = subprocess.run(eval_cmd, cwd=os.path.join(BASE_DIR, 'code'))
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {exp_name}")
        return False
    
    print(f"\nExperiment {exp_name} completed successfully!")
    print(f"Results saved to {exp_dir}")
    
    return True

def compare_experiments():
    """Compare results from all experiments"""
    print("=" * 80)
    print("COMPARING ALL EXPERIMENTS")
    print("=" * 80)
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(RESULTS_DIR) 
                if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    if not exp_dirs:
        print("No experiment results found.")
        return
    
    # Collect results
    results_summary = []
    
    for exp_dir in sorted(exp_dirs):
        metrics_file = os.path.join(RESULTS_DIR, exp_dir, 'metrics.json')
        config_file = os.path.join(RESULTS_DIR, exp_dir, 'config.json')
        
        if os.path.exists(metrics_file) and os.path.exists(config_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            results_summary.append({
                'experiment': exp_dir,
                'description': config.get('description', 'N/A'),
                'roc_auc': metrics.get('roc_auc', 0),
                'pr_auc': metrics.get('pr_auc', 0),
                'cadence_prob': config.get('cadence_mask_prob', 'N/A')
            })
    
    # Print summary table
    print("\nEXPERIMENT SUMMARY")
    print("-" * 120)
    print(f"{'Experiment':<40} {'Description':<40} {'ROC AUC':<10} {'PR AUC':<10} {'Cadence':<10}")
    print("-" * 120)
    
    for result in sorted(results_summary, key=lambda x: x['roc_auc'], reverse=True):
        print(f"{result['experiment']:<40} {result['description']:<40} "
              f"{result['roc_auc']:<10.4f} {result['pr_auc']:<10.4f} {str(result['cadence_prob']):<10}")
    
    print("-" * 120)
    
    # Save summary
    summary_file = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

def main():
    """Run all experiments"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=['baseline'],
                       help='List of experiments to run (default: baseline)')
    parser.add_argument('--compare_only', action='store_true',
                       help='Only compare existing results without running new experiments')
    args = parser.parse_args()
    
    if args.compare_only:
        compare_experiments()
        return
    
    # Run requested experiments
    success_count = 0
    fail_count = 0
    
    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment '{exp_name}'. Skipping.")
            continue
        
        exp_config = EXPERIMENTS[exp_name]
        
        if run_experiment(exp_name, exp_config):
            success_count += 1
        else:
            fail_count += 1
    
    # Compare all results
    print("\n")
    compare_experiments()
    
    # Final summary
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results directory: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
