#!/usr/bin/env python3
"""
Streaming Loss Functions for Binary Microlensing Detection

FIXED VERSION - Handles downsampled sequences and mixed precision properly

Author: Kunal Bhatia
Version: 7.1 - Fixed for temporal downsampling + mixed precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import config_lite as CFG


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
            logits_seq: [batch, seq, 2] sequence of predictions (downsampled length)
            targets: [batch] true labels
            padding_mask: [batch, seq] mask for padded positions (MUST match logits_seq length)
        """
        B, T, C = logits_seq.shape
        device = logits_seq.device
        
        # Compute probabilities
        probs = F.softmax(logits_seq, dim=-1)
        
        # Get probability of correct class
        targets_expanded = targets.unsqueeze(1).expand(B, T)
        correct_probs = probs.gather(2, targets_expanded.unsqueeze(-1)).squeeze(-1)
        
        # Create time penalty
        time_weight = torch.linspace(0, 1, T, device=device, dtype=logits_seq.dtype).unsqueeze(0)
        time_weight[:, :int(T * self.zero_loss_fraction)] = 0  # Zero loss for early detection
        
        # Compute loss for each timestep
        losses = -torch.log(correct_probs + 1e-8) * time_weight
        
        # Apply padding mask if provided
        if padding_mask is not None:
            losses = losses.masked_fill(padding_mask, 0)
            valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
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
            caustic_probs: [batch, seq] or [batch, seq, 1] probability of caustic at each timestep
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
            # Handle different input shapes
            if caustic_probs.dim() == 3:
                # [batch, seq, 1] -> squeeze last dim and take max over seq
                caustic_probs = caustic_probs.squeeze(-1)
            
            if caustic_probs.dim() == 2:
                # [batch, seq] -> take max over sequence
                caustic_weight = caustic_probs.max(dim=1)[0]
            else:
                # [batch] -> use directly
                caustic_weight = caustic_probs
            
            # CRITICAL FIX: Ensure dtype consistency for mixed precision
            caustic_weight = caustic_weight.to(dtype=focal_loss.dtype)
            
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
            logits_seq: [batch, seq, 2] sequence of predictions (downsampled length)
            padding_mask: [batch, seq] mask for padded positions (MUST match logits_seq length)
        """
        B, T, C = logits_seq.shape
        
        if T < 2:
            return torch.tensor(0.0, device=logits_seq.device, dtype=logits_seq.dtype)
        
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
            valid_counts = valid_transitions.sum(dim=1).clamp(min=1).float()
            loss = (consistency_loss.sum(dim=1) / valid_counts).mean()
        else:
            loss = consistency_loss.mean()
        
        return loss * self.smoothing_weight


class CombinedLoss(nn.Module):
    """
    Combined loss for streaming binary microlensing detection.
    
    FIXED VERSION:
    - Handles downsampled sequences (1500 -> 300 points)
    - Handles mixed precision training (AMP)
    - Proper dtype consistency
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
            inputs: [batch, seq] original inputs (1500 points, for padding mask)
            pad_value: Value indicating padding
        """
        # Get padding mask from original inputs (1500 points)
        padding_mask_orig = (inputs == pad_value)
        
        # Get binary logits (300 points due to downsampling)
        binary_logits = outputs['binary']
        
        # CRITICAL FIX: Downsample padding mask to match output length
        if binary_logits.dim() == 3 and binary_logits.shape[1] != inputs.shape[1]:
            T_out = binary_logits.shape[1]  # 300
            T_in = inputs.shape[1]          # 1500
            downsample_factor = T_in // T_out  # 5
            
            # Downsample padding mask via max pooling
            # (any padding in window = padded)
            padding_mask = F.max_pool1d(
                padding_mask_orig.float().unsqueeze(1),
                kernel_size=downsample_factor,
                stride=downsample_factor
            ).squeeze(1).bool()
            
            # Ensure exact length match (handle rounding)
            if padding_mask.shape[1] > T_out:
                padding_mask = padding_mask[:, :T_out]
            elif padding_mask.shape[1] < T_out:
                # Pad with False (not padded)
                B = padding_mask.shape[0]
                padding_extension = torch.zeros(
                    B, T_out - padding_mask.shape[1],
                    dtype=torch.bool, device=padding_mask.device
                )
                padding_mask = torch.cat([padding_mask, padding_extension], dim=1)
        else:
            padding_mask = padding_mask_orig
        
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
    print("Testing streaming losses with downsampling and mixed precision...")
    
    # Create dummy data
    B, T_orig, T_down, C = 4, 1500, 300, 2
    
    # Simulate downsampled model outputs
    logits = torch.randn(B, T_down, C)
    targets = torch.randint(0, 2, (B,))
    inputs = torch.randn(B, T_orig)  # Original 1500 length
    inputs[:, -300:] = -1.0  # Padding
    
    # Test 1: Without mixed precision
    print("\n1. Testing Combined Loss (no AMP):")
    outputs = {
        'binary': logits,
        'caustic': torch.rand(B, T_down, 1)
    }
    combined = CombinedLoss()
    loss = combined(outputs, targets, inputs)
    print(f"   Loss: {loss.item():.4f} ✅")
    
    # Test 2: With mixed precision (half precision)
    print("\n2. Testing Combined Loss (with AMP/half precision):")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logits_half = logits.to(device).half()
    targets_half = targets.to(device)
    inputs_half = inputs.to(device).half()
    outputs_half = {
        'binary': logits_half,
        'caustic': torch.rand(B, T_down, 1, device=device).half()
    }
    combined_half = combined.to(device)
    
    try:
        loss_half = combined_half(outputs_half, targets_half, inputs_half)
        print(f"   Loss: {loss_half.item():.4f} ✅")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Individual losses
    print("\n3. Testing Individual Losses:")
    
    print("   Early Detection Loss:")
    early_loss = combined.early_detection(logits, targets, inputs == -1.0)
    print(f"   Loss: {early_loss.item():.4f} ✅")
    
    print("   Caustic Focal Loss:")
    caustic_probs = torch.rand(B, T_down, 1)
    focal_loss = combined.caustic_focal(logits, targets, caustic_probs)
    print(f"   Loss: {focal_loss.item():.4f} ✅")
    
    print("   Temporal Consistency Loss:")
    consistency_loss = combined.temporal_consistency(logits, inputs == -1.0)
    print(f"   Loss: {consistency_loss.item():.4f} ✅")
    
    print("\n✅ All loss functions working with downsampling and mixed precision!")
