#!/usr/bin/env python
"""
Quick test script for debugging in interactive GPU session
Tests GPU availability, data loading, and model training on small sample
"""

import numpy as np
import tensorflow as tf
import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import check_gpu_availability
from train import setup_gpu, build_model, preprocess_data

def test_data_loading(data_path):
    """Test loading the 1M dataset"""
    print("=" * 60)
    print("TEST 1: DATA LOADING")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return False
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    X = data['X']
    y = data['y']
    
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {np.unique(y, return_counts=True)}")
    print(f"  Memory usage: {X.nbytes / 1e9:.2f} GB")
    
    return True

def test_gpu_setup():
    """Test GPU configuration"""
    print("\n" + "=" * 60)
    print("TEST 2: GPU SETUP")
    print("=" * 60)
    
    num_gpus = check_gpu_availability()
    
    if num_gpus == 0:
        print("✗ No GPUs detected")
        return False
    
    print(f"✓ {num_gpus} GPU(s) detected")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
    
    print("✓ GPU computation successful")
    
    return True

def test_model_build():
    """Test model building"""
    print("\n" + "=" * 60)
    print("TEST 3: MODEL BUILDING")
    print("=" * 60)
    
    try:
        model = build_model(sequence_length=1500, num_channels=1, num_classes=2)
        print("✓ Model built successfully")
        print(f"  Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False

def test_small_training(data_path, n_samples=1000):
    """Test training on small subset"""
    print("\n" + "=" * 60)
    print("TEST 4: SMALL TRAINING RUN")
    print("=" * 60)
    
    print(f"Training on {n_samples} samples for 2 epochs...")
    
    # Load small subset
    data = np.load(data_path)
    X = data['X'][:n_samples]
    y = data['y'][:n_samples]
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    n_train, n_time, n_feat = X_train.shape
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, n_time, n_feat)
    
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_feat)
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, n_time, n_feat)
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    
    # Repeat for TimeDistributed
    y_train_repeated = np.repeat(y_train_cat[:, np.newaxis, :], X_train_scaled.shape[1], axis=1)
    y_val_repeated = np.repeat(y_val_cat[:, np.newaxis, :], X_val_scaled.shape[1], axis=1)
    
    # Build and train model
    model = build_model(sequence_length=X_train_scaled.shape[1], num_channels=X_train_scaled.shape[2])
    
    history = model.fit(
        X_train_scaled, y_train_repeated,
        validation_data=(X_val_scaled, y_val_repeated),
        epochs=2,
        batch_size=32,
        verbose=1
    )
    
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"✓ Training completed")
    print(f"  Final train accuracy: {train_acc:.4f}")
    print(f"  Final val accuracy: {val_acc:.4f}")
    
    if val_acc > 0.5:  # Should be better than random
        print("✓ Model is learning (validation accuracy > 0.5)")
        return True
    else:
        print("✗ Model may not be learning properly")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MICROLENSING CLASSIFICATION - QUICK TEST SUITE")
    print("=" * 60)
    
    # Configuration
    data_path = "/u/hd_vm305/thesis-microlens/data/raw/events_1M.npz"
    
    # Run tests
    tests_passed = 0
    tests_total = 4
    
    if test_data_loading(data_path):
        tests_passed += 1
    
    if test_gpu_setup():
        tests_passed += 1
        # Setup GPUs for subsequent tests
        setup_gpu()
    
    if test_model_build():
        tests_passed += 1
    
    if test_small_training(data_path, n_samples=1000):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✓ ALL TESTS PASSED! Ready for full training.")
        print("\nNext steps:")
        print("1. Exit interactive session: exit")
        print("2. Submit batch job: sbatch slurm/slurm_train_baseline.sh")
        return 0
    else:
        print(f"\n✗ {tests_total - tests_passed} test(s) failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
