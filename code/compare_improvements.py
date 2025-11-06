#!/usr/bin/env python3
"""
Comprehensive Comparison of Baseline vs Enhanced Models

Tests each improvement incrementally to measure accuracy gains.

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

# Import modules
from model import TransformerClassifier  # Original
from model_enhanced import EnhancedTransformerClassifier  # Enhanced
from features import extract_batch_features
from augment import augment_dataset
from losses import FocalLoss


def create_test_data(n_samples=1000, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    T = 1500
    timestamps = np.linspace(0, 1000, T)
    
    X = []
    y = []
    
    # Generate PSPL events
    for i in range(n_samples // 2):
        t0 = np.random.uniform(400, 600)
        u0 = np.random.uniform(0.05, 0.3)
        tE = np.random.uniform(30, 80)
        
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        flux = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
        
        # Add noise
        flux += np.random.normal(0, 0.02, T)
        
        # Add padding
        flux[-100:] = -1.0
        
        X.append(flux)
        y.append(0)
    
    # Generate binary-like events (PSPL + spike)
    for i in range(n_samples // 2):
        t0 = np.random.uniform(400, 600)
        u0 = np.random.uniform(0.05, 0.3)
        tE = np.random.uniform(30, 80)
        
        u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
        flux = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
        
        # Add caustic spike
        spike_idx = int(len(flux) * 0.4)
        flux[spike_idx-10:spike_idx+10] *= np.random.uniform(1.5, 2.5)
        
        # Add noise
        flux += np.random.normal(0, 0.02, T)
        
        # Add padding
        flux[-100:] = -1.0
        
        X.append(flux)
        y.append(1)
    
    X = np.array(X)[:, None, :]  # [N, 1, T]
    y = np.array(y)
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    return X, y, timestamps


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    criterion,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device='cpu'
):
    """Train model and return test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        
        for i in range(0, len(X_train_t), batch_size):
            X_batch = X_train_t[i:i+batch_size].to(device)
            y_batch = y_train_t[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(X_batch, return_sequence=False)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            X_batch = X_test_t[i:i+batch_size].to(device)
            y_batch = y_test_t[i:i+batch_size].to(device)
            
            logits, _ = model(X_batch, return_sequence=False)
            pred = logits.argmax(dim=1)
            
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)
    
    accuracy = correct / total
    return accuracy


def run_comparison():
    """Run comprehensive comparison of all improvements."""
    print("="*80)
    print("COMPREHENSIVE COMPARISON: BASELINE VS ENHANCED")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Generate test data
    print("Generating test data...")
    X, y, timestamps = create_test_data(n_samples=2000, seed=42)
    
    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    results = {}
    
    # ========================================================================
    # Configuration 1: Baseline (Original)
    # ========================================================================
    print("="*80)
    print("1. BASELINE (Original 1-channel Transformer)")
    print("="*80)
    
    model = TransformerClassifier(
        in_channels=1,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    criterion = nn.CrossEntropyLoss()
    
    accuracy = train_and_evaluate(
        model, X_train, y_train, X_test, y_test,
        criterion, epochs=20, device=device
    )
    
    results['baseline'] = accuracy
    print(f"\n✓ Baseline Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # ========================================================================
    # Configuration 2: + Physics Features
    # ========================================================================
    print("="*80)
    print("2. + PHYSICS FEATURES (4-channel input)")
    print("="*80)
    
    print("Extracting physics features...")
    X_train_features = extract_batch_features(X_train, timestamps)
    X_test_features = extract_batch_features(X_test, timestamps)
    
    model = EnhancedTransformerClassifier(
        in_channels=4,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_multi_scale=False
    )
    
    criterion = nn.CrossEntropyLoss()
    
    accuracy = train_and_evaluate(
        model, X_train_features, y_train, X_test_features, y_test,
        criterion, epochs=20, device=device
    )
    
    results['physics_features'] = accuracy
    gain = accuracy - results['baseline']
    print(f"\n✓ + Physics Features: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Gain: {gain:+.4f} ({gain*100:+.2f}%)\n")
    
    # ========================================================================
    # Configuration 3: + Multi-scale Attention
    # ========================================================================
    print("="*80)
    print("3. + MULTI-SCALE ATTENTION")
    print("="*80)
    
    model = EnhancedTransformerClassifier(
        in_channels=4,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_multi_scale=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    accuracy = train_and_evaluate(
        model, X_train_features, y_train, X_test_features, y_test,
        criterion, epochs=20, device=device
    )
    
    results['multi_scale'] = accuracy
    gain = accuracy - results['physics_features']
    print(f"\n✓ + Multi-scale: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Gain: {gain:+.4f} ({gain*100:+.2f}%)\n")
    
    # ========================================================================
    # Configuration 4: + Focal Loss
    # ========================================================================
    print("="*80)
    print("4. + FOCAL LOSS (α=0.25, γ=2.0)")
    print("="*80)
    
    model = EnhancedTransformerClassifier(
        in_channels=4,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_multi_scale=True
    )
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    accuracy = train_and_evaluate(
        model, X_train_features, y_train, X_test_features, y_test,
        criterion, epochs=20, device=device
    )
    
    results['focal_loss'] = accuracy
    gain = accuracy - results['multi_scale']
    print(f"\n✓ + Focal Loss: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Gain: {gain:+.4f} ({gain*100:+.2f}%)\n")
    
    # ========================================================================
    # Configuration 5: + Data Augmentation
    # ========================================================================
    print("="*80)
    print("5. + DATA AUGMENTATION (3x per sample)")
    print("="*80)
    
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_dataset(
        X_train_features, y_train, timestamps,
        n_augmentations=3
    )
    
    model = EnhancedTransformerClassifier(
        in_channels=4,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_multi_scale=True
    )
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    accuracy = train_and_evaluate(
        model, X_train_aug, y_train_aug, X_test_features, y_test,
        criterion, epochs=20, device=device
    )
    
    results['augmentation'] = accuracy
    gain = accuracy - results['focal_loss']
    print(f"\n✓ + Augmentation: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Gain: {gain:+.4f} ({gain*100:+.2f}%)\n")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*80)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*80)
    
    print(f"\n{'Configuration':<30} {'Accuracy':<12} {'Gain':<10}")
    print("-"*52)
    
    baseline_acc = results['baseline']
    
    for name, acc in results.items():
        gain = acc - baseline_acc
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<30} {acc:.4f} ({acc*100:>5.2f}%)  "
              f"{gain:+.4f} ({gain*100:+4.1f}%)")
    
    total_gain = results['augmentation'] - results['baseline']
    print("-"*52)
    print(f"{'TOTAL IMPROVEMENT':<30} {results['augmentation']:.4f} "
          f"({results['augmentation']*100:>5.2f}%)  "
          f"{total_gain:+.4f} ({total_gain*100:+4.1f}%)")
    
    # Save results
    output_dir = Path("../results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            'results': {k: float(v) for k, v in results.items()},
            'total_gain': float(total_gain),
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    run_comparison()
