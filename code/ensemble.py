#!/usr/bin/env python3
"""
Ensemble Training and Inference

Implements ensemble methods for improved accuracy and uncertainty estimation.

Methods:
1. Multi-seed ensemble: Train same architecture with different seeds
2. Architecture ensemble: Combine different model sizes
3. Voting strategies: Hard voting, soft voting, weighted voting

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0 - Ensemble methods
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class EnsembleClassifier:
    """
    Ensemble of multiple trained models.
    
    Supports:
    - Soft voting (average probabilities)
    - Hard voting (majority vote)
    - Weighted voting (based on validation accuracy)
    - Uncertainty estimation (prediction variance)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal)
            device: Device for inference
        """
        self.models = [model.to(device).eval() for model in models]
        self.device = device
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        
        self.weights = self.weights.to(device)
        
        print(f"✓ Ensemble initialized with {len(models)} models")
        print(f"  Weights: {self.weights.cpu().numpy()}")
    
    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        return_uncertainty: bool = False,
        voting: str = 'soft'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input tensor [B, C, T]
            return_uncertainty: If True, return prediction uncertainty
            voting: 'soft' (average probs) or 'hard' (majority vote)
            
        Returns:
            predictions: Predicted classes [B]
            uncertainty: Prediction uncertainty [B] (optional)
        """
        X = X.to(self.device)
        
        # Collect predictions from all models
        all_logits = []
        all_probs = []
        
        for model in self.models:
            logits, _ = model(X, return_sequence=False)
            probs = torch.softmax(logits, dim=1)
            
            all_logits.append(logits)
            all_probs.append(probs)
        
        # Stack predictions [n_models, B, C]
        all_probs = torch.stack(all_probs, dim=0)
        
        if voting == 'soft':
            # Weighted average of probabilities
            weighted_probs = (all_probs * self.weights.view(-1, 1, 1)).sum(dim=0)
            predictions = weighted_probs.argmax(dim=1).cpu().numpy()
            
            if return_uncertainty:
                # Uncertainty = variance across models
                uncertainty = all_probs.var(dim=0).max(dim=1)[0].cpu().numpy()
            else:
                uncertainty = None
        
        elif voting == 'hard':
            # Majority vote
            all_preds = all_probs.argmax(dim=2)  # [n_models, B]
            
            # Weighted vote
            votes = torch.zeros(X.size(0), 2, device=self.device)
            for i, preds in enumerate(all_preds):
                votes.scatter_add_(1, preds.unsqueeze(1), 
                                  self.weights[i].expand(X.size(0), 1))
            
            predictions = votes.argmax(dim=1).cpu().numpy()
            
            if return_uncertainty:
                # Uncertainty = vote disagreement
                max_votes = votes.max(dim=1)[0]
                uncertainty = (1.0 - max_votes / self.weights.sum()).cpu().numpy()
            else:
                uncertainty = None
        
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")
        
        return predictions, uncertainty
    
    @torch.no_grad()
    def predict_proba(
        self,
        X: torch.Tensor,
        return_all: bool = False
    ) -> np.ndarray:
        """
        Get class probabilities from ensemble.
        
        Args:
            X: Input tensor [B, C, T]
            return_all: If True, return probs from all models
            
        Returns:
            probs: Class probabilities [B, C] or [n_models, B, C]
        """
        X = X.to(self.device)
        
        all_probs = []
        for model in self.models:
            logits, _ = model(X, return_sequence=False)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=0)
        
        if return_all:
            return all_probs.cpu().numpy()
        else:
            # Weighted average
            weighted_probs = (all_probs * self.weights.view(-1, 1, 1)).sum(dim=0)
            return weighted_probs.cpu().numpy()


def train_ensemble(
    model_class: type,
    model_kwargs: Dict,
    train_loader,
    val_loader,
    n_models: int = 5,
    seeds: Optional[List[int]] = None,
    epochs: int = 50,
    device: torch.device = torch.device('cuda'),
    save_dir: Optional[Path] = None
) -> Tuple[List[nn.Module], List[float]]:
    """
    Train ensemble of models with different seeds.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model initialization
        train_loader: Training data loader
        val_loader: Validation data loader
        n_models: Number of models in ensemble
        seeds: Random seeds for each model (auto-generated if None)
        epochs: Training epochs per model
        device: Device for training
        save_dir: Directory to save models (optional)
        
    Returns:
        models: List of trained models
        val_accs: Validation accuracies for weighting
    """
    if seeds is None:
        seeds = [42 + i * 100 for i in range(n_models)]
    
    assert len(seeds) == n_models, "Number of seeds must match n_models"
    
    models = []
    val_accs = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Training ensemble member {i+1}/{n_models} (seed={seed})")
        print(f"{'='*80}")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model
        model = model_class(**model_kwargs).to(device)
        
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(X, return_sequence=False)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    logits, _ = model(X, return_sequence=False)
                    pred = logits.argmax(1)
                    val_correct += (pred == y).sum().item()
                    val_total += len(y)
            
            val_acc = val_correct / val_total
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {train_loss/len(train_loader):.4f}, "
                      f"Val Acc = {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                if save_dir is not None:
                    save_path = save_dir / f"ensemble_model_{i}_seed{seed}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'val_acc': val_acc,
                        'seed': seed,
                        'epoch': epoch
                    }, save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"  ✓ Best validation accuracy: {best_val_acc:.4f}")
        
        models.append(model)
        val_accs.append(best_val_acc)
    
    print(f"\n{'='*80}")
    print(f"✓ Ensemble training complete!")
    print(f"  Mean validation accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"{'='*80}\n")
    
    return models, val_accs


def load_ensemble(
    model_class: type,
    model_kwargs: Dict,
    model_paths: List[Path],
    device: torch.device = torch.device('cpu')
) -> Tuple[List[nn.Module], List[float]]:
    """
    Load ensemble from saved checkpoints.
    
    Args:
        model_class: Model class
        model_kwargs: Model initialization arguments
        model_paths: Paths to saved models
        device: Device to load models on
        
    Returns:
        models: List of loaded models
        val_accs: Validation accuracies
    """
    models = []
    val_accs = []
    
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device)
        
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        
        models.append(model)
        val_accs.append(checkpoint.get('val_acc', 1.0))
    
    print(f"✓ Loaded {len(models)} models for ensemble")
    
    return models, val_accs


if __name__ == "__main__":
    print("=" * 80)
    print("ENSEMBLE MODULE SELF-TEST")
    print("=" * 80)
    
    from model_enhanced import EnhancedTransformerClassifier
    
    torch.manual_seed(42)
    
    # Create synthetic models
    print("\nCreating ensemble of 3 models...")
    models = []
    for i in range(3):
        model = EnhancedTransformerClassifier(
            in_channels=1,
            n_classes=2,
            d_model=32,
            nhead=2,
            num_layers=1,
            use_multi_scale=False
        )
        models.append(model)
    
    # Create ensemble
    weights = [0.4, 0.3, 0.3]  # Based on validation performance
    ensemble = EnsembleClassifier(models, weights=weights)
    
    # Test inference
    print("\nTesting ensemble inference...")
    X = torch.randn(16, 1, 1500)
    X[:, :, -100:] = -1.0  # Padding
    
    # Soft voting
    preds_soft, uncertainty_soft = ensemble.predict(
        X, return_uncertainty=True, voting='soft'
    )
    print(f"  Soft voting predictions: {preds_soft[:5]}")
    print(f"  Uncertainty: mean={uncertainty_soft.mean():.4f}, std={uncertainty_soft.std():.4f}")
    
    # Hard voting
    preds_hard, uncertainty_hard = ensemble.predict(
        X, return_uncertainty=True, voting='hard'
    )
    print(f"  Hard voting predictions: {preds_hard[:5]}")
    print(f"  Disagreement: mean={uncertainty_hard.mean():.4f}, std={uncertainty_hard.std():.4f}")
    
    # Probabilities
    probs = ensemble.predict_proba(X)
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  First 3 samples: {probs[:3]}")
    
    print("\n✓ All tests passed!")
    print("=" * 80)
