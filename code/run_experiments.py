"""
Experiment automation for cadence studies
Generates datasets with different cadences and trains models
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.config import CADENCE_EXPERIMENTS, DATA_DIR, MODEL_DIR, RESULTS_DIR

def run_command(cmd, description):
    """Run shell command and print output"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True

def run_cadence_experiment(cadence_name, cadence_prob, n_pspl, n_binary, 
                          epochs, batch_size, skip_simulation=False):
    """Run complete experiment for a specific cadence"""
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT: {cadence_name.upper()} (cadence_prob={cadence_prob})")
    print(f"{'#'*80}\n")
    
    # File paths
    data_file = os.path.join(DATA_DIR, f'events_{cadence_name}.npz')
    model_name = f'model_{cadence_name}'
    
    # Step 1: Generate data (if not skipping)
    if not skip_simulation:
        print(f"\nStep 1: Generating {cadence_name} dataset...")
        sim_cmd = (
            f"python code/simulate_cadence.py "
            f"--n_pspl {n_pspl} "
            f"--n_binary {n_binary} "
            f"--cadence_prob {cadence_prob} "
            f"--output {data_file}"
        )
        if not run_command(sim_cmd, f"Simulating {cadence_name} dataset"):
            return False
    else:
        print(f"\nStep 1: Using existing dataset {data_file}")
    
    # Step 2: Train model
    print(f"\nStep 2: Training model for {cadence_name}...")
    train_cmd = (
        f"python code/train.py "
        f"--data {data_file} "
        f"--output {MODEL_DIR}/{model_name}.keras "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--experiment_name {cadence_name}"
    )
    if not run_command(train_cmd, f"Training {cadence_name} model"):
        return False
    
    # Step 3: Evaluate model
    print(f"\nStep 3: Evaluating {cadence_name} model...")
    
    # Find the most recent experiment directory for this cadence
    import glob
    experiment_dirs = glob.glob(os.path.join(RESULTS_DIR, f'{cadence_name}_*'))
    if experiment_dirs:
        latest_dir = max(experiment_dirs, key=os.path.getctime)
        model_path = os.path.join(latest_dir, 'best_model.keras')
        scaler_path = os.path.join(latest_dir, 'scaler.pkl')
        eval_output = os.path.join(latest_dir, 'evaluation')
    else:
        model_path = os.path.join(MODEL_DIR, f'{model_name}.keras')
        scaler_path = None
        eval_output = os.path.join(RESULTS_DIR, f'{cadence_name}_evaluation')
    
    eval_cmd = (
        f"python code/evaluate.py "
        f"--model {model_path} "
        f"--data {data_file} "
        f"--output_dir {eval_output}"
    )
    if scaler_path and os.path.exists(scaler_path):
        eval_cmd += f" --scaler {scaler_path}"
    
    if not run_command(eval_cmd, f"Evaluating {cadence_name} model"):
        return False
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {cadence_name.upper()} COMPLETE!")
    print(f"{'='*80}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run cadence experiments')
    parser.add_argument('--experiments', nargs='+', 
                       choices=list(CADENCE_EXPERIMENTS.keys()) + ['all'],
                       default=['all'],
                       help='Which experiments to run')
    parser.add_argument('--n_pspl', type=int, default=50000,
                       help='Number of PSPL events per experiment')
    parser.add_argument('--n_binary', type=int, default=50000,
                       help='Number of Binary events per experiment')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--skip_simulation', action='store_true',
                       help='Skip simulation if data already exists')
    args = parser.parse_args()
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        experiments = CADENCE_EXPERIMENTS.keys()
    else:
        experiments = args.experiments
    
    print(f"\n{'#'*80}")
    print(f"# CADENCE EXPERIMENT SUITE")
    print(f"# Running experiments: {', '.join(experiments)}")
    print(f"# Events per experiment: {args.n_pspl} PSPL + {args.n_binary} Binary")
    print(f"# Training: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"{'#'*80}\n")
    
    # Run experiments
    results = {}
    for cadence_name in experiments:
        cadence_prob = CADENCE_EXPERIMENTS[cadence_name]
        success = run_cadence_experiment(
            cadence_name=cadence_name,
            cadence_prob=cadence_prob,
            n_pspl=args.n_pspl,
            n_binary=args.n_binary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_simulation=args.skip_simulation
        )
        results[cadence_name] = success
    
    # Summary
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT SUITE COMPLETE")
    print(f"{'#'*80}\n")
    
    print("Results Summary:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name:15s}: {status}")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
