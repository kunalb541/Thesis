"""
Utility functions for microlensing classification
Supports both AMD (ROCm) and NVIDIA (CUDA) GPUs
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

def detect_gpu_backend():
    """
    Detect which GPU backend is available
    Returns: 'amd', 'nvidia', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Check if it's AMD or NVIDIA
            device_name = torch.cuda.get_device_name(0).lower()
            if 'amd' in device_name or 'mi' in device_name or 'radeon' in device_name:
                return 'amd'
            else:
                return 'nvidia'
    except ImportError:
        pass
    
    return 'cpu'

def check_gpu_availability():
    """
    Check GPU availability and provide detailed information
    Works for both PyTorch (AMD/NVIDIA) and TensorFlow
    """
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    backend = detect_gpu_backend()
    print(f"\nDetected backend: {backend.upper()}")
    
    # Try PyTorch
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs detected: {num_gpus}")
            
            for i in range(num_gpus):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                
                # Get memory info
                if hasattr(torch.cuda, 'get_device_properties'):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / 1e9
                    print(f"  Total Memory: {total_memory:.2f} GB")
                
                # Current memory usage
                if hasattr(torch.cuda, 'memory_allocated'):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    cached = torch.cuda.memory_reserved(i) / 1e9
                    print(f"  Allocated: {allocated:.2f} GB")
                    print(f"  Cached: {cached:.2f} GB")
            
            # Test computation
            print("\nTesting GPU computation...")
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("✓ GPU computation test passed")
            except Exception as e:
                print(f"✗ GPU computation test failed: {e}")
            
            print("=" * 60)
            return num_gpus
        else:
            print("\n⚠ WARNING: PyTorch installed but no GPUs detected!")
            print("Training will run on CPU (very slow)")
            print("=" * 60)
            return 0
            
    except ImportError:
        print("\n⚠ PyTorch not installed")
        print("Install with:")
        if backend == 'amd':
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
        else:
            print("  pip install torch torchvision")
        print("=" * 60)
        return 0

def setup_gpu_environment():
    """
    Setup environment variables for optimal GPU performance
    Works for both AMD and NVIDIA
    """
    backend = detect_gpu_backend()
    
    print(f"\nConfiguring {backend.upper()} GPU environment...")
    
    if backend == 'amd':
        # AMD ROCm settings
        os.environ['ROCR_VISIBLE_DEVICES'] = '0,1,2,3'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
        print("  Set ROCR_VISIBLE_DEVICES=0,1,2,3")
        print("  Set HIP_VISIBLE_DEVICES=0,1,2,3")
    elif backend == 'nvidia':
        # NVIDIA CUDA settings
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        print("  Set CUDA_VISIBLE_DEVICES=0,1,2,3")
    
    # Common PyTorch optimizations
    os.environ['OMP_NUM_THREADS'] = '24'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    print("  Set OMP_NUM_THREADS=24")
    print("  Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    
    return backend

def get_device_name():
    """Get a descriptive name for the compute device"""
    backend = detect_gpu_backend()
    
    if backend in ['amd', 'nvidia']:
        try:
            import torch
            return f"{backend.upper()} - {torch.cuda.get_device_name(0)}"
        except:
            return backend.upper()
    else:
        return "CPU"

def plot_training_history(history_dict, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history_dict['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history_dict['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history_dict['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_light_curve(timestamps, flux, title="Light Curve", save_path=None):
    """Plot a single light curve"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, flux, 'o-', markersize=3, linewidth=1)
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Flux', fontsize=12)
    plt.title(title, fontsize=14)
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
        axes[0, i].set_title(f'PSPL Event {idx}', fontsize=10)
        axes[0, i].set_xlabel('Time Step', fontsize=9)
        axes[0, i].set_ylabel('Flux', fontsize=9)
        axes[0, i].grid(alpha=0.3)
    
    # Plot Binary samples
    for i, idx in enumerate(binary_samples):
        timestamps = np.arange(X.shape[1])
        flux = X[idx, :, 0]
        axes[1, i].plot(timestamps, flux, 'r-', linewidth=1)
        axes[1, i].set_title(f'Binary Event {idx}', fontsize=10)
        axes[1, i].set_xlabel('Time Step', fontsize=9)
        axes[1, i].set_ylabel('Flux', fontsize=9)
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
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results
    
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

def compare_experiments(results_dir, save_path=None):
    """Compare multiple experiments"""
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found to compare")
        return
    
    # Extract experiment names and accuracies
    names = [r['experiment'] for r in results]
    accuracies = [r['metrics'].get('accuracy', 0) for r in results]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    names = [names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(names)), accuracies, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Experiment Comparison', fontsize=14)
    ax.set_ylim([0.80, 1.0])
    ax.grid(alpha=0.3, axis='y')
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if accuracies[i] >= 0.95:
            bar.set_color('green')
        elif accuracies[i] >= 0.90:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Quick test
    print("\n" + "="*60)
    print("GPU DETECTION AND SETUP")
    print("="*60)
    
    num_gpus = check_gpu_availability()
    
    if num_gpus > 0:
        backend = setup_gpu_environment()
        device_name = get_device_name()
        print(f"\n✓ Ready to train on: {device_name}")
    else:
        print("\n⚠ No GPUs available. Training will be very slow on CPU.")
    
    print("="*60)