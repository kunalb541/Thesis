
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_baseline_results(results_dir):
    """Generate comprehensive baseline analysis"""
    
    # Load training history
    with open(f'{results_dir}/history.json') as f:
        history = json.load(f)
    
    # Load metrics
    with open(f'{results_dir}/evaluation/metrics.json') as f:
        metrics = json.load(f)
    
    # Plot everything
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    axes[0,0].plot(history['train_loss'], label='Train')
    axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].legend()
    
    axes[0,1].plot(history['train_acc'], label='Train')
    axes[0,1].plot(history['val_acc'], label='Val')
    axes[0,1].set_title('Accuracy Curves')
    axes[0,1].legend()
    
    # Early detection
    if 'early_detection' in metrics:
        fracs = sorted(metrics['early_detection'].keys())
        accs = [metrics['early_detection'][f] for f in fracs]
        axes[0,2].plot([int(f.replace('pct','')) for f in fracs], accs, 'o-')
        axes[0,2].set_title('Early Detection Performance')
        axes[0,2].set_xlabel('% Observed')
        axes[0,2].set_ylabel('Accuracy')
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    if 'early_detection' in metrics:
        print("\nEarly Detection:")
        print(f"  33% observed: {metrics['early_detection']['33pct']:.4f}")
        print(f"  50% observed: {metrics['early_detection']['50pct']:.4f}")
        print(f"  67% observed: {metrics['early_detection']['67pct']:.4f}")
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/baseline_summary.png', dpi=300)
    print(f"\n✓ Summary saved to {results_dir}/baseline_summary.png")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/baseline_*"
    analyze_baseline_results(results_dir)