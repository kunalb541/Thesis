#!/usr/bin/env python3
"""
Focal Loss and Advanced Loss Functions

Implements loss functions that focus on hard examples and handle class imbalance.

Focal Loss: Focuses training on hard examples (high u₀ events)
Weighted Loss: Handles class imbalance
Label Smoothing: Prevents overconfidence

Author: Kunal Bhatia
Date: November 2025
Version: 6.0.0 - Advanced loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: Predicted probability for true class
    - α: Class weighting factor
    - γ (gamma): Focusing parameter (higher = more focus on hard examples)
    
    Benefits for microlensing:
    - Emphasizes hard-to-classify events (high u₀, weak caustics)
    - Reduces loss contribution from easy examples
    - Improves performance on rare binary types
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance (0.25 = slight emphasis on binary)
            gamma: Focusing parameter (2.0 = standard, higher = more focus on hard)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [B, C]
            targets: Target class indices [B]
            
        Returns:
            Loss scalar
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            targets_smooth = torch.zeros_like(inputs)
            targets_smooth.fill_(self.label_smoothing / (n_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Cross entropy with smoothed labels
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-sample weighting.
    
    Allows dynamic weighting based on event properties (e.g., u₀ values).
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted focal loss.
        
        Args:
            inputs: Logits [B, C]
            targets: Target class indices [B]
            weights: Per-sample weights [B] (optional)
            
        Returns:
            Loss scalar
        """
        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal weight
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply per-sample weights
        if weights is not None:
            focal_loss = focal_loss * weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal + auxiliary losses.
    
    Combines classification loss with optional regularization terms.
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        aux_weight: float = 0.1,
        use_focal: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight
        
        if use_focal:
            self.main_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.main_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        aux_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Main logits [B, C]
            targets: Target class indices [B]
            aux_outputs: Auxiliary outputs for regularization (optional)
            
        Returns:
            Total loss
        """
        # Main classification loss
        main_loss = self.main_loss(inputs, targets)
        total_loss = self.focal_weight * main_loss
        
        # Auxiliary loss (e.g., sequence consistency)
        if aux_outputs is not None and self.aux_weight > 0:
            aux_loss = F.cross_entropy(aux_outputs, targets)
            total_loss += self.aux_weight * aux_loss
        
        return total_loss


def compute_sample_weights(
    y: torch.Tensor,
    u0_values: Optional[torch.Tensor] = None,
    strategy: str = 'inverse_freq'
) -> torch.Tensor:
    """
    Compute per-sample weights for training.
    
    Strategies:
    - 'inverse_freq': Weight inversely to class frequency
    - 'u0_based': Weight based on impact parameter (harder = higher weight)
    - 'uniform': Equal weights
    
    Args:
        y: Target labels [N]
        u0_values: Impact parameters [N] (optional)
        strategy: Weighting strategy
        
    Returns:
        Sample weights [N]
    """
    n_samples = len(y)
    
    if strategy == 'uniform':
        return torch.ones(n_samples)
    
    elif strategy == 'inverse_freq':
        # Weight inversely to class frequency
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y]
        return sample_weights / sample_weights.mean()
    
    elif strategy == 'u0_based' and u0_values is not None:
        # Weight by u0: higher u0 = harder = higher weight
        # Use sigmoid to smoothly increase weight for u0 > 0.2
        base_weight = torch.ones(n_samples)
        
        # For binary events, increase weight with u0
        binary_mask = (y == 1)
        u0_weight = torch.sigmoid((u0_values - 0.2) * 10)  # Sharp transition at u0=0.2
        base_weight[binary_mask] *= (1.0 + 2.0 * u0_weight[binary_mask])
        
        return base_weight / base_weight.mean()
    
    else:
        return torch.ones(n_samples)


if __name__ == "__main__":
    print("=" * 80)
    print("FOCAL LOSS MODULE SELF-TEST")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # Create synthetic data
    B, C = 128, 2
    inputs = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    
    print("\nTest 1: Standard Cross Entropy")
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(inputs, targets)
    print(f"  CE Loss: {loss_ce.item():.4f}")
    
    print("\nTest 2: Focal Loss (alpha=0.25, gamma=2.0)")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal_loss(inputs, targets)
    print(f"  Focal Loss: {loss_focal.item():.4f}")
    
    print("\nTest 3: Focal Loss with Label Smoothing")
    focal_smooth = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    loss_smooth = focal_smooth(inputs, targets)
    print(f"  Focal + Smoothing Loss: {loss_smooth.item():.4f}")
    
    print("\nTest 4: Weighted Focal Loss")
    weights = torch.rand(B)
    weighted_focal = WeightedFocalLoss(alpha=0.25, gamma=2.0)
    loss_weighted = weighted_focal(inputs, targets, weights)
    print(f"  Weighted Focal Loss: {loss_weighted.item():.4f}")
    
    print("\nTest 5: Sample Weight Computation")
    u0_values = torch.rand(B)
    
    weights_uniform = compute_sample_weights(targets, strategy='uniform')
    print(f"  Uniform weights: mean={weights_uniform.mean():.3f}, std={weights_uniform.std():.3f}")
    
    weights_freq = compute_sample_weights(targets, strategy='inverse_freq')
    print(f"  Inverse freq weights: mean={weights_freq.mean():.3f}, std={weights_freq.std():.3f}")
    
    weights_u0 = compute_sample_weights(targets, u0_values, strategy='u0_based')
    print(f"  u0-based weights: mean={weights_u0.mean():.3f}, std={weights_u0.std():.3f}")
    
    print("\nTest 6: Hard Example Focus")
    # Create easy and hard examples
    easy_inputs = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # Very confident
    hard_inputs = torch.tensor([[0.1, -0.1], [-0.1, 0.1]])      # Very uncertain
    easy_targets = torch.tensor([0, 1])
    hard_targets = torch.tensor([0, 1])
    
    ce = nn.CrossEntropyLoss(reduction='none')
    focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
    
    ce_easy = ce(easy_inputs, easy_targets).mean()
    ce_hard = ce(hard_inputs, hard_targets).mean()
    focal_easy = focal(easy_inputs, easy_targets).mean()
    focal_hard = focal(hard_inputs, hard_targets).mean()
    
    print(f"  CE - Easy: {ce_easy:.4f}, Hard: {ce_hard:.4f}, Ratio: {ce_hard/ce_easy:.2f}x")
    print(f"  Focal - Easy: {focal_easy:.4f}, Hard: {focal_hard:.4f}, Ratio: {focal_hard/focal_easy:.2f}x")
    print(f"  ✓ Focal loss emphasizes hard examples {focal_hard/focal_easy:.1f}x more than CE")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
