#!/usr/bin/env python3
"""
Streaming Loss Functions for Binary Microlensing Detection

Includes:
- Early detection loss (rewards early classification)
- Caustic focal loss (emphasizes caustic regions)
- Temporal consistency loss (smooth predictions)

Author: Kunal Bhatia
Version: 6.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import config as CFG


class EarlyDetectionLoss(nn.Module):
    """
    Rewards early detection of binary events.
    Loss decreases linearly with detection time.
    """
    
    def __init__(self, zero_loss_fraction: float = 0.3):
        """
        Args:
            zero_loss_fraction: Fraction of sequence with zero loss (early detection reward)
        """
        super().__init__()
        self.zero_loss_fraction = zero_loss_fraction
    
    def forward(
        self,
        logits_seq: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits_seq: [batch, seq, 2] sequence of predictions
            targets: [batch] true labels
            padding_mask: [batch, seq] mask for padded positions
        """
        B, T, C = logits_seq.shape
        device = logits_seq.device
        
        # Compute probabilities
        probs = F.softmax(logits_seq, dim=-1)
        
        # Get probability of correct class
        targets_expanded = targets.unsqueeze(1).expand(B, T)
        correct_probs = probs.gather(2, targets_expanded.unsqueeze(-1)).squeeze(-1)
        
        # Create time penalty
        time_weight = torch.linspace(0, 1, T, device=device).unsqueeze(0)
        time_weight[:, :int(T * self.zero_loss_fraction)] = 0  # Zero loss for early detection
        
        # Compute loss for each timestep
        losses = -torch.log(correct_probs + 1e-8) * time_weight
        
        # Apply padding mask if provided
        if padding_mask is not None:
            losses = losses.masked_fill(padding_mask, 0)
            valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            loss = (losses.sum(dim=1) / valid_counts.squeeze()).mean()
        else:
            loss = losses.mean()
        
        return loss


class CausticFocalLoss(nn.Module):
    """
    Focal loss that emphasizes caustic crossing regions.
    Higher weight on hard examples (low confidence predictions).
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        caustic_weight: float = 2.0
    ):
        """
        Args:
            alpha: Class balance weight
            gamma: Focusing parameter
            caustic_weight: Extra weight for caustic regions
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.caustic_weight = caustic_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        caustic_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq, 2] or [batch, 2]
            targets: [batch]
            caustic_probs: [batch, seq] probability of caustic at each timestep
        """
        if logits.dim() == 3:
            # Sequence predictions - use last timestep
            logits = logits[:, -1, :]
        
        # Standard focal loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply caustic weighting if provided
        if caustic_probs is not None:
            # Use maximum caustic probability as weight
            if caustic_probs.dim() == 2:
                caustic_weight = caustic_probs.max(dim=1)[0]
            else:
                caustic_weight = caustic_probs
            
            # Extra weight for binary events with high caustic probability
            is_binary = targets == 1
            weight = torch.ones_like(focal_loss)
            weight[is_binary] = 1 + (self.caustic_weight - 1) * caustic_weight[is_binary]
            focal_loss = focal_loss * weight
        
        return focal_loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Ensures smooth probability transitions over time.
    Prevents flickering predictions.
    """
    
    def __init__(self, smoothing_weight: float = 0.1):
        """
        Args:
            smoothing_weight: Weight for smoothness penalty
        """
        super().__init__()
        self.smoothing_weight = smoothing_weight
    
    def forward(
        self,
        logits_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits_seq: [batch, seq, 2] sequence of predictions
            padding_mask: [batch, seq] mask for padded positions
        """
        B, T, C = logits_seq.shape
        
        if T < 2:
            return torch.tensor(0.0, device=logits_seq.device)
        
        # Compute probabilities
        probs = F.softmax(logits_seq, dim=-1)
        
        # Compute differences between consecutive timesteps
        prob_diff = torch.diff(probs, dim=1)  # [B, T-1, C]
        
        # L2 penalty on differences
        consistency_loss = (prob_diff ** 2).sum(dim=-1)  # [B, T-1]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Mask for valid transitions (both timesteps must be valid)
            valid_transitions = ~(padding_mask[:, :-1] | padding_mask[:, 1:])
            consistency_loss = consistency_loss * valid_transitions
            
            # Average over valid transitions
            valid_counts = valid_transitions.sum(dim=1).clamp(min=1)
            loss = (consistency_loss.sum(dim=1) / valid_counts).mean()
        else:
            loss = consistency_loss.mean()
        
        return loss * self.smoothing_weight


class CombinedLoss(nn.Module):
    """
    Combined loss for streaming binary microlensing detection.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            weights: Loss component weights
        """
        super().__init__()
        
        if weights is None:
            weights = CFG.LOSS_WEIGHTS
        
        self.weights = weights
        
        # Initialize loss components
        self.classification_loss = nn.CrossEntropyLoss()
        self.early_detection = EarlyDetectionLoss()
        self.caustic_focal = CausticFocalLoss()
        self.temporal_consistency = TemporalConsistencyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        inputs: torch.Tensor,
        pad_value: float = -1.0
    ) -> torch.Tensor:
        """
        Args:
            outputs: Model outputs dict with 'binary', 'anomaly', 'caustic'
            targets: [batch] true labels
            inputs: [batch, seq] original inputs (for padding mask)
            pad_value: Value indicating padding
        """
        # Get padding mask
        padding_mask = (inputs == pad_value)
        
        # Get binary logits
        binary_logits = outputs['binary']
        
        total_loss = 0
        
        # 1. Classification loss
        if binary_logits.dim() == 3:
            # Use last valid timestep for each sample
            B, T, C = binary_logits.shape
            last_valid = (~padding_mask).sum(dim=1) - 1
            last_valid = last_valid.clamp(min=0, max=T-1)
            
            batch_idx = torch.arange(B, device=binary_logits.device)
            final_logits = binary_logits[batch_idx, last_valid]
            cls_loss = self.classification_loss(final_logits, targets)
        else:
            cls_loss = self.classification_loss(binary_logits, targets)
        
        total_loss += self.weights['classification'] * cls_loss
        
        # 2. Early detection loss (if sequence predictions available)
        if binary_logits.dim() == 3 and self.weights.get('early_detection', 0) > 0:
            early_loss = self.early_detection(binary_logits, targets, padding_mask)
            total_loss += self.weights['early_detection'] * early_loss
        
        # 3. Caustic focal loss (if caustic predictions available)
        if 'caustic' in outputs and self.weights.get('caustic_focal', 0) > 0:
            if binary_logits.dim() == 3:
                final_logits = binary_logits[:, -1, :]
            else:
                final_logits = binary_logits
            
            caustic_loss = self.caustic_focal(
                final_logits, targets, outputs.get('caustic')
            )
            total_loss += self.weights['caustic_focal'] * caustic_loss
        
        # 4. Temporal consistency loss
        if binary_logits.dim() == 3 and self.weights.get('temporal_consistency', 0) > 0:
            consistency_loss = self.temporal_consistency(binary_logits, padding_mask)
            total_loss += self.weights['temporal_consistency'] * consistency_loss
        
        return total_loss


if __name__ == "__main__":
    print("Testing streaming losses...")
    
    # Create dummy data
    B, T, C = 4, 100, 2
    logits = torch.randn(B, T, C)
    targets = torch.randint(0, 2, (B,))
    inputs = torch.randn(B, T)
    inputs[:, -20:] = -1.0  # Padding
    
    # Test individual losses
    print("\n1. Early Detection Loss")
    early_loss = EarlyDetectionLoss()
    loss = early_loss(logits, targets, inputs == -1.0)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. Caustic Focal Loss")
    focal_loss = CausticFocalLoss()
    caustic_probs = torch.rand(B, T)
    loss = focal_loss(logits, targets, caustic_probs)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Temporal Consistency Loss")
    consistency_loss = TemporalConsistencyLoss()
    loss = consistency_loss(logits, inputs == -1.0)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n4. Combined Loss")
    outputs = {
        'binary': logits,
        'caustic': caustic_probs
    }
    combined = CombinedLoss()
    loss = combined(outputs, targets, inputs)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ All loss functions working!")
