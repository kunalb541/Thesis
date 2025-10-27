"""
Utility functions for microlensing classification
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_training_history(history_dict, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history_dict['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history_dict['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history_dict['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_light_curve(timestamps, flux, title="Light Curve", save_path=None):
    """Plot a single light curve"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, flux, 'o-', markersize=3, linewidth=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Flux')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_sample_events(data_path, n_samples=6, save_path=None):
    """Plot sample PSPL and Binary events"""
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    # Get indices for each class
    pspl_idx = np.where(y == 'PSPL')[0]
    binary_idx = np.where(y == 'Binary')[0]
    
    # Random sample
    pspl_samples = np.random.choice(pspl_idx, n_samples // 2, replace=False)
    binary_samples = np.random.choice(binary_idx, n_samples // 2, replace=False)
    
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 6))
    
    # Plot PSPL samples
    for i, idx in enumerate(pspl_samples):
        timestamps = np.arange(X.shape[1])
        flux = X[idx, :, 0]
        axes[0, i].plot(timestamps, flux, 'b-', linewidth=1)
        axes[0, i].set_title(f'PSPL Event {idx}')
        axes[0, i].set_xlabel('Time Step')
        axes[0, i].set_ylabel('Flux')
        axes[0, i].grid(alpha=0.3)
    
    # Plot Binary samples
    for i, idx in enumerate(binary_samples):
        timestamps = np.arange(X.shape[1])
        flux = X[idx, :, 0]
        axes[1, i].plot(timestamps, flux, 'r-', linewidth=1)
        axes[1, i].set_title(f'Binary Event {idx}')
        axes[1, i].set_xlabel('Time Step')
        axes[1, i].set_ylabel('Flux')
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_experiment_results(results_dir):
    """Load all experiment results for comparison"""
    results = []
    
    for exp_name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        
        metrics_path = os.path.join(exp_path, 'metrics.json')
        config_path = os.path.join(exp_path, 'config.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            results.append({
                'experiment': exp_name,
                'metrics': metrics,
                'config': config
            })
    
    return results

def compare_cadence_experiments(results_dir, save_path=None):
    """Compare experiments with different cadences"""
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found to compare")
        return
    
    # Extract cadence experiments
    cadence_results = [r for r in results if 'cadence' in r['experiment']]
    
    if not cadence_results:
        print("No cadence experiments found")
        return
    
    # Sort by cadence probability
    cadence_results.sort(key=lambda x: x['config'].get('cadence_mask_prob', 0))
    
    # Extract data for plotting
    cadences = [r['config'].get('cadence_mask_prob', 0) * 100 for r in cadence_results]
    roc_aucs = [r['metrics'].get('roc_auc', 0) for r in cadence_results]
    pr_aucs = [r['metrics'].get('pr_auc', 0) for r in cadence_results]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cadences, roc_aucs, 'o-', label='ROC AUC', linewidth=2, markersize=8)
    ax.plot(cadences, pr_aucs, 's-', label='PR AUC', linewidth=2, markersize=8)
    
    ax.set_xlabel('Missing Observations (%)', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Classification Performance vs. Observing Cadence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cadence comparison plot saved to {save_path}")
    else:
        plt.show()

def check_gpu_availability():
    """Check if GPUs are available and configured correctly"""
    import tensorflow as tf
    
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  Details: {details}")
            except:
                print("  Could not retrieve device details")
    else:
        print("\nWARNING: No GPUs detected!")
        print("TensorFlow will run on CPU (very slow for training)")
    
    print("=" * 60)
    
    return len(gpus)

if __name__ == "__main__":
    # Quick test
    check_gpu_availability()
