#!/usr/bin/env python3
"""
Pre-flight check script for microlensing thesis project
Run this before submitting SLURM jobs to catch common issues
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
            print(f"✗ {name} NOT INSTALLED - install with: pip install {package}")
            all_checks_passed = False
    
    # PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ PyTorch CUDA available ({torch.cuda.device_count()} GPUs)")
        else:
            print(f"⚠ PyTorch CUDA NOT available (OK for CPU testing)")
    except:
        pass
    
    # ========================================================================
    # 3. Data Files
    # ========================================================================
    print_section("3. Data Files")
    
    data_dir = project_root / 'data' / 'raw'
    data_files = list(data_dir.glob('*.npz'))
    
    if data_files:
        print(f"✓ Found {len(data_files)} data file(s):")
        for f in data_files:
            size_mb = f.stat().st_size / 1e6
            print(f"    - {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"✗ No .npz data files found in {data_dir}")
        print(f"  Generate data with: python code/simulate.py")
        all_checks_passed = False
    
    # Check for baseline data specifically
    baseline_data = data_dir / 'events_1M.npz'
    if baseline_data.exists():
        print(f"✓ Baseline dataset (events_1M.npz) exists")
        
        # Try to load and validate
        try:
            import numpy as np
            data = np.load(baseline_data)
            X = data['X']
            y = data['y']
            print(f"    Shape: {X.shape}, Labels: {len(y)}")
            
            # Basic validation
            if len(X) == len(y):
                print(f"✓ Data validation passed")
            else:
                print(f"✗ Data shape mismatch: X={len(X)}, y={len(y)}")
                all_checks_passed = False
                
        except Exception as e:
            print(f"✗ Could not load data: {e}")
            all_checks_passed = False
    else:
        print(f"⚠ Baseline dataset not found (events_1M.npz)")
        print(f"  You'll need this for baseline training")
    
    # ========================================================================
    # 4. Code Files
    # ========================================================================
    print_section("4. Code Files")
    
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
                print(f"    ✗ Syntax error: {e}")
                all_checks_passed = False
    
    # ========================================================================
    # 5. SLURM Scripts
    # ========================================================================
    print_section("5. SLURM Scripts")
    
    slurm_dir = project_root / 'slurm'
    slurm_scripts = list(slurm_dir.glob('*.sh'))
    
    if slurm_scripts:
        print(f"✓ Found {len(slurm_scripts)} SLURM script(s)")
        for script in slurm_scripts:
            # Check if executable
            is_executable = os.access(script, os.X_OK)
            if is_executable:
                print(f"  ✓ {script.name} (executable)")
            else:
                print(f"  ⚠ {script.name} (not executable - run: chmod +x {script})")
    else:
        print(f"✗ No SLURM scripts found in {slurm_dir}")
        all_checks_passed = False
    
    # ========================================================================
    # 6. File Extension Check
    # ========================================================================
    print_section("6. File Extension Consistency")
    
    # Check if train.py saves .pt files
    train_py = code_dir / 'train.py'
    if train_py.exists():
        with open(train_py) as f:
            content = f.read()
            if 'torch.save' in content and '.pt' in content:
                print(f"✓ train.py saves PyTorch .pt files")
                
                # Check SLURM scripts
                for script in slurm_scripts:
                    with open(script) as f:
                        script_content = f.read()
                        if '.keras' in script_content:
                            print(f"  ✗ {script.name} references .keras (should be .pt)")
                            all_checks_passed = False
                        elif '.pt' in script_content:
                            print(f"  ✓ {script.name} uses .pt extension")
            elif '.keras' in content:
                print(f"⚠ train.py might save .keras files (check if using TensorFlow)")
    
    # ========================================================================
    # 7. Path Consistency
    # ========================================================================
    print_section("7. Path Configuration")
    
    # Check for hardcoded paths in code
    print("Checking for hardcoded paths...")
    hardcoded_found = False
    
    for script in (code_dir / 'train.py', code_dir / 'evaluate.py'):
        if not script.exists():
            continue
            
        with open(script) as f:
            content = f.read()
            if '/u/hd_vm305' in content:
                print(f"  ⚠ {script.name} contains hardcoded path /u/hd_vm305")
                hardcoded_found = True
    
    if not hardcoded_found:
        print("  ✓ No hardcoded paths in Python scripts")
    
    # Check SLURM scripts
    for script in slurm_scripts:
        with open(script) as f:
            content = f.read()
            if '/u/hd_vm305' in content and 'PROJECT_DIR' not in content:
                print(f"  ⚠ {script.name} has hardcoded paths (consider using $PROJECT_DIR)")
    
    # ========================================================================
    # 8. Git Status
    # ========================================================================
    print_section("8. Git Repository")
    
    git_dir = project_root / '.git'
    if git_dir.exists():
        print(f"✓ Git repository initialized")
        
        # Check for uncommitted changes
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'], 
                cwd=project_root,
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                print(f"  ⚠ Uncommitted changes found:")
                for line in result.stdout.strip().split('\n')[:5]:
                    print(f"    {line}")
                if len(result.stdout.strip().split('\n')) > 5:
                    print(f"    ... and more")
            else:
                print(f"  ✓ Working directory clean")
        except:
            print(f"  ⚠ Could not check git status")
    else:
        print(f"⚠ Not a git repository (consider: git init)")
    
    # ========================================================================
    # 9. Configuration Check
    # ========================================================================
    print_section("9. Configuration")
    
    config_py = code_dir / 'config.py'
    if config_py.exists():
        try:
            import sys
            sys.path.insert(0, str(code_dir))
            import config
            
            # Check key parameters
            print(f"✓ config.py loaded successfully")
            print(f"    N_EVENTS_TOTAL: {getattr(config, 'N_EVENTS_TOTAL', 'NOT SET')}")
            print(f"    BATCH_SIZE: {getattr(config, 'BATCH_SIZE', 'NOT SET')}")
            print(f"    EPOCHS: {getattr(config, 'EPOCHS', 'NOT SET')}")
            
            # Check for experiments configuration
            if hasattr(config, 'EXPERIMENTS'):
                print(f"    EXPERIMENTS defined: {len(config.EXPERIMENTS)} configs")
            else:
                print(f"    ⚠ EXPERIMENTS dict not found")
                
        except Exception as e:
            print(f"✗ Error loading config.py: {e}")
            all_checks_passed = False
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_section("SUMMARY")
    
    if all_checks_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print("\nYou're ready to:")
        print("  1. Submit baseline training: sbatch slurm/train_baseline.sh")
        print("  2. Monitor with: squeue -u $USER")
        print("  3. Check logs: tail -f logs/train_baseline_*.out")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before submitting jobs.")
        print("Common fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Create missing directories: mkdir -p data/raw models results logs")
        print("  - Make scripts executable: chmod +x slurm/*.sh")
        print("  - Generate data: python code/simulate.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())