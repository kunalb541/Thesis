#!/usr/bin/env python3
"""
model.py - FIXED TimeDistributed CNN (v6.2 - Matches TensorFlow)

CRITICAL FIX: Changed from causal padding to symmetric padding
to match the WORKING TensorFlow implementation.

Author: Kunal Bhatia (fixed by Claude)
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    """
    TimeDistributed CNN for microlensing classification.
    
    **FIXED v6.2**: Now uses symmetric padding (like TensorFlow padding="same")
    instead of causal padding. This matches the working TensorFlow model.
    
    Architecture (matches TensorFlow exactly):
    - Conv1D(128, kernel=5, padding=same) + ReLU + Dropout(0.3)
    - Conv1D(64, kernel=3, padding=same) + ReLU + Dropout(0.3)
    - Conv1D(32, kernel=3, padding=same) + ReLU + Dropout(0.3)
    - Linear(32, 2) per timestep
    
    Input:  [B, 1, T] - full time series
    Output (seq=True): [B, T, 2] - class probabilities at each timestep
    Output (seq=False): [B, 2] - class probabilities at final timestep
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        # Match TensorFlow architecture exactly
        conv_channels: list = [128, 64, 32],
        kernel_sizes: list = [5, 3, 3],
        dropout: float = 0.3,
        # Unused args (kept for config.py compatibility)
        window_size: int = 50, 
        use_lstm: bool = False 
    ):
        super().__init__()
        
        # --- 1. 1D CNN Feature Extractor (FIXED: symmetric padding) ---
        layers = []
        C_in = in_channels
        for i in range(len(conv_channels)):
            C_out = conv_channels[i]
            k = kernel_sizes[i]
            
            # CRITICAL FIX: Use symmetric padding like TensorFlow
            padding = (k - 1) // 2  # This gives "same" padding
            
            layers.append(
                nn.Conv1d(C_in, C_out, kernel_size=k, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            C_in = C_out
            
        self.feature_extractor = nn.Sequential(*layers)
        
        feature_dim = conv_channels[-1]  # 32
        
        # --- 2. Classifier (applied at each timestep) ---
        self.classifier = nn.Linear(feature_dim, n_classes)

    
    def forward(self, x, return_sequence=True):
        """
        Forward pass.
        
        Args:
            x: [B, C, T] input sequences
            return_sequence: if True, return predictions at all timesteps
                           if False, return only final prediction
        
        Returns:
            if return_sequence:
                logits: [B, T, n_classes] - predictions at each timestep
                final_hidden: None (to match tuple expected by train.py)
            else:
                logits: [B, n_classes] - final aggregated prediction
                final_hidden: None (to match tuple expected by train.py)
        """
        B, C, T = x.shape
        
        # 1. Extract features at each timestep
        # Input: [B, C, T]
        features = self.feature_extractor(x)  # Output: [B, feature_dim, T]
        
        # 2. Reshape for classifier
        # permute(0, 2, 1) swaps T and feature_dim
        features = features.permute(0, 2, 1)  # Output: [B, T, feature_dim]
        
        # 3. Apply classifier to all timesteps
        logits = self.classifier(features)  # Output: [B, T, n_classes]
        
        if return_sequence:
            return logits, None
        else:
            # Return last timestep's prediction
            return logits[:, -1, :], None


def test_timedistributed():
    """Test TimeDistributedCNN"""
    print("="*80)
    print("Testing TimeDistributedCNN (v6.2 - Fixed Padding)")
    print("="*80)
    
    B, C, T = 4, 1, 1500
    x = torch.randn(B, C, T)
    
    print("\n1. TimeDistributedCNN (Full Sequence):")
    model1 = TimeDistributedCNN(
        in_channels=1,
        n_classes=2
    )
    
    out_seq, _ = model1(x, return_sequence=True)
    out_final, _ = model1(x, return_sequence=False)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output (sequence): {out_seq.shape}  # Expected: [B, T, n_classes]")
    print(f"   Output (final):    {out_final.shape}  # Expected: [B, n_classes]")
    
    assert out_seq.shape == (B, T, 2)
    assert out_final.shape == (B, 2)
    
    # Test partial sequence (for early detection)
    print("\n2. TimeDistributedCNN (Partial Sequence):")
    T_partial = 100
    x_partial = torch.randn(B, C, T_partial)
    out_final_partial, _ = model1(x_partial, return_sequence=False)
    print(f"   Input shape:  {x_partial.shape}")
    print(f"   Output (final):    {out_final_partial.shape}  # Expected: [B, n_classes]")
    assert out_final_partial.shape == (B, 2)

    # Check that output varies with input
    print("\n3. Checking model responsiveness:")
    x1 = torch.randn(2, 1, 100)
    x2 = torch.randn(2, 1, 100)
    
    with torch.no_grad():
        model1.eval()
        out1, _ = model1(x1, return_sequence=False)
        out2, _ = model1(x2, return_sequence=False)
    
    diff = torch.abs(out1 - out2).max().item()
    print(f"   Max output difference for different inputs: {diff:.4f}")
    
    if diff < 1e-6:
        print("   ⚠️ WARNING: Model produces same output for different inputs!")
    else:
        print("   ✓ Model outputs vary with input")
    
    print("\n" + "="*80)
    print("✅ TimeDistributed model working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_timedistributed()
