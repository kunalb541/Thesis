"""
Master experiment orchestrator for thesis benchmarking
Runs systematic experiments across all parameter combinations
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from config_experiments import EXPERIMENT_MATRIX, get_binary_params, BINARY_REGIMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CODE_DIR = os.path.join(BASE_DIR, 'code')

def generate_dataset(exp_config, n_pspl=100_000, n_binary=100_000):
    """Generate dataset for an experiment"""
    exp_name = exp_config['name']
    data_file = os.path.join(DATA_DIR, f'events_{exp_name}.npz')
    
    if os.path.exists(data_file):
        print(f"  Dataset already exists: {data_file}")
        return data_file
    
    print(f"  Generating dataset...")
    
    cmd = [
        'python', os.path.join(CODE_DIR, 'simulate_flexible.py'),
        '--n_pspl', str(n_pspl),
        '--n_binary', str(n_binary),
        '--binary_regime', exp_config['binary_regime'],
        '--mag_error', str(exp_config['error']),
        '--cadence', str(exp_config['cadence']),
        '--output', data_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR generating dataset:")
        print(result.stderr)
        return None
    
    print(f"  Dataset generated: {data_file}")
    return data_file

def train_model(exp_name, data_file, epochs=50, batch_size=128):
    """Train model on dataset"""
    model_file = os.path.join(MODEL_DIR, f'model_{exp_name}.keras')
    
    if os.path.exists(model_file):
        print(f"  Model already exists: {model_file}")
        return model_file
    
    print(f"  Training model...")
    
    cmd = [
        'python', os.path.join(CODE_DIR, 'train.py'),
        '--data', data_file,
        '--output', model_file,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--experiment_name', exp_name
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"  ERROR training model")
        return None
    
    print(f"  Model trained: {model_file}")
    return model_file

def evaluate_model(exp_name, model_file, data_file):
    """Evaluate trained model"""
    results_dir = os.path.join(RESULTS_DIR, exp_name)
    
    if os.path.exists(os.path.join(results_dir, 'metrics.json')):
        print(f"  Results already exist: {results_dir}")
        return results_dir
    
    print(f"  Evaluating model...")
    
    os.makedirs(results_dir, exist_ok=True)
    
    cmd = [
        'python', os.path.join(CODE_DIR, 'evaluate.py'),
        '--model', model_file,
        '--data', data_file,
        '--output_dir', results_dir
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"  ERROR evaluating model")
        return None
    
    print(f"  Results saved: {results_dir}")
    return results_dir

def run_experiment(exp_config, n_pspl=100_000, n_binary=100_000, 
                  epochs=50, batch_size=128, force=False):
    """Run complete experiment pipeline"""
    exp_name = exp_config['name']
    
    print("=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"Binary regime: {exp_config['binary_regime']}")
    regime = BINARY_REGIMES[exp_config['binary_regime']]
    print(f"  Description: {regime['description']}")
    print(f"  s: [{regime['s_min']:.2f}, {regime['s_max']:.2f}]")
    print(f"  q: [{regime['q_min']:.2f}, {regime['q_max']:.2f}]")
    print(f"Cadence: {exp_config['cadence']*100:.1f}% missing")
    print(f"Error: {exp_config['error']:.3f} mag")
    print()
    
    # Step 1: Generate dataset
    print("Step 1: Generate dataset")
    data_file = generate_dataset(exp_config, n_pspl, n_binary)
    if data_file is None:
        return False
    
    # Step 2: Train model
    print("\nStep 2: Train model")
    model_file = train_model(exp_name, data_file, epochs, batch_size)
    if model_file is None:
        return False
    
    # Step 3: Evaluate model
    print("\nStep 3: Evaluate model")
    results_dir = evaluate_model(exp_name, model_file, data_file)
    if results_dir is None:
        return False
    
    print(f"\n✓ Experiment {exp_name} completed successfully!")
    print("=" * 80)
    print()
    
    return True

def compare_all_results():
    """Generate comparison of all experiments"""
    print("=" * 80)
    print("COMPARING ALL EXPERIMENTS")
    print("=" * 80)
    
    results = []
    
    for exp_config in EXPERIMENT_MATRIX:
        exp_name = exp_config['name']
        results_dir = os.path.join(RESULTS_DIR, exp_name)
        metrics_file = os.path.join(results_dir, 'metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            results.append({
                'experiment': exp_name,
                'binary_regime': exp_config['binary_regime'],
                'cadence': exp_config['cadence'],
                'error': exp_config['error'],
                'roc_auc': metrics.get('roc_auc', 0),
                'pr_auc': metrics.get('pr_auc', 0),
            })
    
    if not results:
        print("No results found yet.")
        return
    
    # Print comparison table
    print()
    print("RESULTS SUMMARY")
    print("-" * 120)
    print(f"{'Experiment':<30} {'Binary Regime':<20} {'Missing%':<10} {'Error':<10} {'ROC AUC':<10} {'PR AUC':<10}")
    print("-" * 120)
    
    # Sort by ROC AUC
    for r in sorted(results, key=lambda x: x['roc_auc'], reverse=True):
        print(f"{r['experiment']:<30} {r['binary_regime']:<20} "
              f"{r['cadence']*100:<10.1f} {r['error']:<10.3f} "
              f"{r['roc_auc']:<10.4f} {r['pr_auc']:<10.4f}")
    
    print("-" * 120)
    
    # Save summary
    summary_file = os.path.join(RESULTS_DIR, 'comparison_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {summary_file}")
    
    # Analyze by category
    print("\nANALYSIS BY CATEGORY:")
    print()
    
    # By binary regime
    print("By Binary Regime:")
    by_regime = {}
    for r in results:
        regime = r['binary_regime']
        if regime not in by_regime:
            by_regime[regime] = []
        by_regime[regime].append(r['roc_auc'])
    
    for regime in sorted(by_regime.keys()):
        avg_auc = sum(by_regime[regime]) / len(by_regime[regime])
        print(f"  {regime:<20}: {avg_auc:.4f} (n={len(by_regime[regime])})")
    
    # By cadence
    print("\nBy Cadence:")
    by_cadence = {}
    for r in results:
        cad = f"{r['cadence']*100:.0f}%"
        if cad not in by_cadence:
            by_cadence[cad] = []
        by_cadence[cad].append(r['roc_auc'])
    
    for cad in sorted(by_cadence.keys()):
        avg_auc = sum(by_cadence[cad]) / len(by_cadence[cad])
        print(f"  {cad:<10}: {avg_auc:.4f} (n={len(by_cadence[cad])})")
    
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run systematic benchmarking experiments'
    )
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--n_pspl', type=int, default=100_000,
                       help='Number of PSPL events per experiment')
    parser.add_argument('--n_binary', type=int, default=100_000,
                       help='Number of Binary events per experiment')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--compare_only', action='store_true',
                       help='Only compare existing results')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAVAILABLE EXPERIMENTS:")
        print("=" * 80)
        for i, exp in enumerate(EXPERIMENT_MATRIX, 1):
            print(f"{i:2d}. {exp['name']:<30} | {exp['binary_regime']:<20} | "
                  f"Missing: {exp['cadence']*100:4.1f}% | Error: {exp['error']:.3f}")
        print("=" * 80)
        return
    
    if args.compare_only:
        compare_all_results()
        return
    
    # Determine which experiments to run
    if args.experiments:
        experiments_to_run = [e for e in EXPERIMENT_MATRIX if e['name'] in args.experiments]
        if not experiments_to_run:
            print(f"ERROR: No matching experiments found")
            return
    else:
        experiments_to_run = EXPERIMENT_MATRIX
    
    print("\n" + "=" * 80)
    print("THESIS BENCHMARKING - MASTER EXPERIMENT ORCHESTRATOR")
    print("=" * 80)
    print(f"Total experiments to run: {len(experiments_to_run)}")
    print(f"Events per experiment: {args.n_pspl + args.n_binary}")
    print(f"Training epochs: {args.epochs}")
    print("=" * 80)
    print()
    
    # Run experiments
    success_count = 0
    fail_count = 0
    
    for i, exp_config in enumerate(experiments_to_run, 1):
        print(f"\nPROGRESS: {i}/{len(experiments_to_run)}")
        
        if run_experiment(exp_config, 
                         n_pspl=args.n_pspl,
                         n_binary=args.n_binary,
                         epochs=args.epochs,
                         batch_size=args.batch_size):
            success_count += 1
        else:
            fail_count += 1
    
    # Final comparison
    print("\n")
    compare_all_results()
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 80)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
