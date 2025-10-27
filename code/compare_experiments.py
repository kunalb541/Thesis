"""
Compare results across multiple experiments
Generates comparative plots for thesis
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.config import RESULTS_DIR

def load_experiment_results(experiment_dir):
    """Load results from an experiment directory"""
    eval_results_file = os.path.join(experiment_dir, 'evaluation', 'evaluation_results.json')
    config_file = os.path.join(experiment_dir, 'experiment_config.json')
    
    if not os.path.exists(eval_results_file):
        return None
    
    with open(eval_results_file, 'r') as f:
        eval_results = json.load(f)
    
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return {
        'evaluation': eval_results,
        'config': config,
        'directory': experiment_dir
    }

def find_all_experiments():
    """Find all completed experiments"""
    experiments = {}
    
    # Look for experiment directories
    pattern = os.path.join(RESULTS_DIR, '*_*')
    for exp_dir in glob.glob(pattern):
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir).split('_')[0]
            results = load_experiment_results(exp_dir)
            if results:
                if exp_name not in experiments:
                    experiments[exp_name] = []
                experiments[exp_name].append(results)
    
    # Get most recent experiment for each type
    latest_experiments = {}
    for name, exp_list in experiments.items():
        latest = max(exp_list, key=lambda x: os.path.getctime(x['directory']))
        latest_experiments[name] = latest
    
    return latest_experiments

def plot_cadence_comparison(experiments, output_dir):
    """Compare early detection performance across cadences"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Early detection curves
    ax1 = axes[0]
    for name, exp in experiments.items():
        early_det = exp['evaluation']['early_detection_accuracies']
        windows = sorted([int(k) for k in early_det.keys()])
        accuracies = [early_det[str(w)] for w in windows]
        ax1.plot(windows, accuracies, marker='o', linewidth=2, label=name)
    
    ax1.set_xlabel('Percentage of Observations (%)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('Early Detection: Accuracy vs Observation Time', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='80%')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='90%')
    
    # Plot 2: Detection time comparison
    ax2 = axes[1]
    names = []
    mean_times = []
    for name, exp in sorted(experiments.items()):
        names.append(name)
        mean_times.append(exp['evaluation']['detection_time_stats']['mean'])
    
    bars = ax2.bar(range(len(names)), mean_times, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Detection Time (observations)', fontsize=12)
    ax2.set_title('Average Time to Confident Detection', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_times)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cadence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cadence_comparison.png")

def plot_accuracy_roc_comparison(experiments, output_dir):
    """Compare final accuracy and ROC AUC"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Final accuracy comparison
    ax1 = axes[0]
    names = []
    accuracies = []
    for name, exp in sorted(experiments.items()):
        names.append(name)
        accuracies.append(exp['evaluation']['final_metrics']['accuracy'])
    
    bars = ax1.bar(range(len(names)), accuracies, color='forestgreen', alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Final Classification Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.legend()
    
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: ROC AUC comparison
    ax2 = axes[1]
    roc_aucs = []
    for name in names:
        roc_aucs.append(experiments[name]['evaluation']['final_metrics']['roc_auc'])
    
    bars = ax2.bar(range(len(names)), roc_aucs, color='darkorange', alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('ROC AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, roc_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_roc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/accuracy_roc_comparison.png")

def create_results_table(experiments, output_dir):
    """Create a summary table of all experiments"""
    
    # Prepare data
    rows = []
    for name, exp in sorted(experiments.items()):
        eval_res = exp['evaluation']
        row = {
            'Experiment': name,
            'Final Accuracy': f"{eval_res['final_metrics']['accuracy']:.4f}",
            'ROC AUC': f"{eval_res['final_metrics']['roc_auc']:.4f}",
            'Mean Detection Time': f"{eval_res['detection_time_stats']['mean']:.1f}",
            'Median Detection Time': f"{eval_res['detection_time_stats']['median']:.1f}",
            'Acc @ 50%': f"{eval_res['early_detection_accuracies'].get('50', 0):.4f}",
            'Acc @ 80%': f"{eval_res['early_detection_accuracies'].get('80', 0):.4f}",
        }
        rows.append(row)
    
    # Write to file
    table_file = os.path.join(output_dir, 'results_summary.txt')
    with open(table_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("="*120 + "\n\n")
        
        # Header
        headers = list(rows[0].keys())
        f.write(" | ".join(f"{h:20s}" for h in headers) + "\n")
        f.write("-"*120 + "\n")
        
        # Rows
        for row in rows:
            f.write(" | ".join(f"{str(row[h]):20s}" for h in headers) + "\n")
        
        f.write("\n" + "="*120 + "\n")
    
    print(f"Saved: {table_file}")
    
    # Also save as JSON for easy parsing
    json_file = os.path.join(output_dir, 'results_summary.json')
    with open(json_file, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {json_file}")

def main():
    print("="*80)
    print("COMPARING EXPERIMENT RESULTS")
    print("="*80 + "\n")
    
    # Find all experiments
    print("Searching for experiments...")
    experiments = find_all_experiments()
    
    if not experiments:
        print("No completed experiments found!")
        print(f"Looking in: {RESULTS_DIR}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for name in sorted(experiments.keys()):
        exp_dir = os.path.basename(experiments[name]['directory'])
        print(f"  - {name:15s} ({exp_dir})")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(RESULTS_DIR, f'comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Generate comparison plots
    print("Generating comparison plots...")
    plot_cadence_comparison(experiments, output_dir)
    plot_accuracy_roc_comparison(experiments, output_dir)
    
    # Create summary table
    print("\nCreating results summary table...")
    create_results_table(experiments, output_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - cadence_comparison.png")
    print("  - accuracy_roc_comparison.png")
    print("  - results_summary.txt")
    print("  - results_summary.json")
    print()

if __name__ == "__main__":
    main()
