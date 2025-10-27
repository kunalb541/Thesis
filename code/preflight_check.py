#!/usr/bin/env python3
"""
Pre-flight check script for microlensing thesis project
Run this before submitting jobs to catch common issues
Works for both AMD (ROCm) and NVIDIA (CUDA) systems
"""

import sys
import os
from pathlib import Path
import subprocess

def print_section(title):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def check_status(condition, success_msg, fail_msg):
    """Print status with checkmark or X"""
    if condition:
        print(f"✓ {success_msg}")
        return True
    else:
        print(f"✗ {fail_msg}")
        return False

def main():
    print_section("MICROLENSING PROJECT PRE-FLIGHT CHECK")
    
    all_checks_passed = True
    
    # ========================================================================
    # 1. Directory Structure
    # ========================================================================
    print_section("1. Directory Structure")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'code', 'slurm', 'data/raw', 'data/processed', 
        'models', 'results', 'logs', 'docs'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        passed = check_status(
            full_path.exists(),
            f"{dir_path}/ exists",
            f"{dir_path}/ MISSING - create it with: mkdir -p {dir_path}"
        )
        all_checks_passed = all_checks_passed and passed
    
    # ========================================================================
    # 2. Python Environment
    # ========================================================================
    print_section("2. Python Environment")
    
    # Python version
    py_version = sys.version_info
    py_check = py_version.major == 3 and py_version.minor >= 8
    passed = check_status(
        py_check,
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
        f"Python version {py_version.major}.{py_version.minor} - need >= 3.8"
    )
    all_checks_passed = all_checks_passed and passed
    
    # Required packages
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'VBMicrolensing': 'VBMicrolensing'
    }
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT INSTALLED")
            if package == 'torch':
                print("  Install with:")
                print("  # For NVIDIA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                print("  # For AMD: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
            else:
                print(f"  Install with: pip install {package}")
            all_checks_passed = False
    
    # ========================================================================
    # 3. GPU Detection
    # ========================================================================
    print_section("3. GPU Detection")
    
    try:
        import torch
        
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            print(f"✓ PyTorch detected {num_gpus} GPU(s)")
            
            # Detect backend
            device_name = torch.cuda.get_device_name(0).lower()
            if 'amd' in device_name or 'mi' in device_name or 'radeon' in device_name:
                backend = 'AMD (ROCm)'
            else:
                backend = 'NVIDIA (CUDA)'
            
            print(f"  Backend: {backend}")
            
            # List GPUs
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
                
                # Try to get memory info
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / 1e9
                    print(f"    Memory: {total_mem:.1f} GB")
                except:
                    pass
            
            # Test GPU computation
            print("\n  Testing GPU computation...")
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("  ✓ GPU computation test passed")
            except Exception as e:
                print(f"  ✗ GPU computation test failed: {e}")
                all_checks_passed = False
        else:
            print("⚠ No GPUs detected - training will run on CPU (very slow)")
            print("  This is OK for testing but not recommended for full training")
            
    except ImportError:
        print("✗ PyTorch not installed - cannot check GPUs")
        all_checks_passed = False
    
    # ========================================================================
    # 4. Data Files
    # ========================================================================
    print_section("4. Data Files")
    
    data_dir = project_root / 'data' / 'raw'
    data_files = list(data_dir.glob('*.npz'))
    
    if data_files:
        print(f"✓ Found {len(data_files)} data file(s):")
        for f in data_files:
            size_mb = f.stat().st_size / 1e6
            print(f"    {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"⚠ No .npz data files found in {data_dir}")
        print(f"  Generate baseline data with:")
        print(f"    cd code")
        print(f"    python simulate.py --output ../data/raw/events_baseline_1M.npz")
    
    # Check for baseline data specifically
    baseline_files = [
        'events_baseline_1M.npz',
        'events_1M.npz',
        'test_2k.npz'
    ]
    
    baseline_found = False
    for filename in baseline_files:
        baseline_path = data_dir / filename
        if baseline_path.exists():
            print(f"✓ Found dataset: {filename}")
            baseline_found = True
            
            # Try to validate
            try:
                import numpy as np
                data = np.load(baseline_path)
                X = data['X']
                y = data['y']
                print(f"    Shape: {X.shape}, Labels: {len(set(y))}")
                
                if len(X) == len(y):
                    print(f"    ✓ Data validation passed")
                else:
                    print(f"    ✗ Data shape mismatch!")
                    all_checks_passed = False
            except Exception as e:
                print(f"    ✗ Could not load data: {e}")
                all_checks_passed = False
            break
    
    if not baseline_found:
        print(f"⚠ No baseline dataset found")
        print(f"  Generate with: python code/simulate.py")
    
    # ========================================================================
    # 5. Code Files
    # ========================================================================
    print_section("5. Code Files")
    
    code_dir = project_root / 'code'
    required_scripts = [
        'config.py',
        'simulate.py',
        'train.py',
        'evaluate.py',
        'utils.py'
    ]
    
    for script in required_scripts:
        script_path = code_dir / script
        passed = check_status(
            script_path.exists(),
            f"{script} exists",
            f"{script} MISSING"
        )
        all_checks_passed = all_checks_passed and passed
        
        # Check for syntax errors
        if script_path.exists():
            try:
                with open(script_path) as f:
                    compile(f.read(), script_path, 'exec')
                print(f"    ✓ No syntax errors")
            except SyntaxError as e:
                print(f"    ✗ Syntax error on line {e.lineno}: {e.msg}")
                all_checks_passed = False
    
    # ========================================================================
    # 6. SLURM Scripts
    # ========================================================================
    print_section("6. SLURM Scripts")
    
    slurm_dir = project_root / 'slurm'
    slurm_scripts = list(slurm_dir.glob('*.sh'))
    
    if slurm_scripts:
        print(f"✓ Found {len(slurm_scripts)} SLURM script(s)")
        for script in slurm_scripts:
            is_executable = os.access(script, os.X_OK)
            if is_executable:
                print(f"  ✓ {script.name} (executable)")
            else:
                print(f"  ⚠ {script.name} (not executable)")
                print(f"    Fix with: chmod +x {script}")
    else:
        print(f"⚠ No SLURM scripts found in {slurm_dir}")
        print(f"  This is OK if not using a cluster")
    
    # ========================================================================
    # 7. Configuration Check
    # ========================================================================
    print_section("7. Configuration")
    
    config_py = code_dir / 'config.py'
    if config_py.exists():
        try:
            sys.path.insert(0, str(code_dir))
            import config
            
            print(f"✓ config.py loaded successfully")
            print(f"    N_EVENTS_TOTAL: {getattr(config, 'N_EVENTS_TOTAL', 'NOT SET')}")
            print(f"    BATCH_SIZE: {getattr(config, 'BATCH_SIZE', 'NOT SET')}")
            print(f"    EPOCHS: {getattr(config, 'EPOCHS', 'NOT SET')}")
            
            # Check for experiments configuration
            if hasattr(config, 'EXPERIMENTS'):
                print(f"    EXPERIMENTS: {len(config.EXPERIMENTS)} configurations")
            
            # Check binary parameter sets
            if hasattr(config, 'BINARY_PARAM_SETS'):
                param_sets = config.BINARY_PARAM_SETS
                print(f"    BINARY_PARAM_SETS: {list(param_sets.keys())}")
            
        except Exception as e:
            print(f"✗ Error loading config.py: {e}")
            all_checks_passed = False
    
    # ========================================================================
    # 8. Disk Space
    # ========================================================================
    print_section("8. Disk Space")
    
    try:
        stat = os.statvfs(project_root)
        free_gb = (stat.f_bavail * stat.f_frsize) / 1e9
        total_gb = (stat.f_blocks * stat.f_frsize) / 1e9
        used_gb = total_gb - free_gb
        
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used:  {used_gb:.1f} GB")
        print(f"  Free:  {free_gb:.1f} GB")
        
        if free_gb < 50:
            print(f"⚠ Low disk space! Need at least 50 GB free")
            print(f"  Consider cleaning up or using different storage")
        else:
            print(f"✓ Sufficient disk space")
    except:
        print("⚠ Could not check disk space")
    
    # ========================================================================
    # 9. Memory Check
    # ========================================================================
    print_section("9. System Memory")
    
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
        
        for line in meminfo:
            if 'MemTotal' in line:
                total_kb = int(line.split()[1])
                total_gb = total_kb / 1e6
                print(f"  Total RAM: {total_gb:.1f} GB")
                
                if total_gb < 16:
                    print(f"⚠ Low RAM - recommend at least 16 GB")
                else:
                    print(f"✓ Sufficient RAM")
                break
    except:
        print("⚠ Could not check system memory")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_section("SUMMARY")
    
    if all_checks_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print("\n✓ System is ready for baseline training")
        print("\nNext steps:")
        print("  1. Generate data (if not done):")
        print("     cd code")
        print("     python simulate.py --output ../data/raw/events_baseline_1M.npz")
        print("\n  2. Start training:")
        print("     # Local:")
        print("     python train.py --data ../data/raw/events_baseline_1M.npz \\")
        print("                     --output ../models/baseline.pt \\")
        print("                     --experiment_name baseline")
        print("\n     # Cluster:")
        print("     sbatch slurm/train_baseline.sh")
        print("\n  3. Monitor progress:")
        print("     tail -f logs/baseline_*.out")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before starting training.")
        print("\nCommon fixes:")
        print("  - Install PyTorch:")
        print("    # NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("    # AMD: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        print("  - Install other packages:")
        print("    pip install -r requirements.txt")
        print("  - Create missing directories:")
        print("    mkdir -p data/raw data/processed models results logs")
        print("  - Make scripts executable:")
        print("    chmod +x slurm/*.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())