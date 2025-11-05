#!/usr/bin/env python3
"""
model.py - FIXED TimeDistributed CNN with Global Pooling

CRITICAL FIX: Added proper global pooling to capture mean flux signal.
The mean flux difference between PSPL and Binary events is the primary
discriminative feature.

Author: Kunal Bhatia (fixed by Claude)
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    """
    TimeDistributed CNN for microlensing classification.
    
    FIXED: Now properly supports global pooling over timesteps to capture
    the mean flux signal (16% difference between PSPL and Binary).
    
    Architecture:
    - Conv1D(128, kernel=5, padding=same) + ReLU + Dropout(0.3)
    - Conv1D(64, kernel=3, padding=same) + ReLU + Dropout(0.3)
    - Conv1D(32, kernel=3, padding=same) + ReLU + Dropout(0.3)
    - Global pooling over timesteps (captures mean flux)
    - Linear(32, 2) for classification
    
    Input:  [B, C, T] - batch of time series (C channels, T timesteps)
    Output: [B, T, 2] (if return_sequence=True) - predictions at each timestep
            [B, 2] (if return_sequence=False) - global prediction using mean pooling
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        conv_channels: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.3,
        # Unused args (kept for compatibility)
        window_size: int = 50, 
        use_lstm: bool = False 
    ):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [128, 64, 32]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]
        
        # --- 1D CNN Feature Extractor ---
        layers = []
        C_in = in_channels
        for i in range(len(conv_channels)):
            C_out = conv_channels[i]
            k = kernel_sizes[i]
            
            # Symmetric padding (like TensorFlow "same")
            padding = (k - 1) // 2
            
            layers.append(
                nn.Conv1d(C_in, C_out, kernel_size=k, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            C_in = C_out
            
        self.feature_extractor = nn.Sequential(*layers)
        
        feature_dim = conv_channels[-1]  # 32
        
        # --- Classifier ---
        self.classifier = nn.Linear(feature_dim, n_classes)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better scaling"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_sequence=True):
        B, C, T = x.shape
        features = self.feature_extractor(x)  # [B, feature_dim, T]
        
        if return_sequence:
            features = features.permute(0, 2, 1)  # [B, T, feature_dim]
            logits = self.classifier(features)  # [B, T, n_classes]
            return logits, None
        else:
            # Global average pooling over time
            features_pooled = features.mean(dim=2)  # [B, feature_dim]
            logits = self.classifier(features_pooled)  # [B, n_classes]
            return logits, None


def test_timedistributed():
    """Test TimeDistributedCNN"""
    print("="*80)
    print("Testing TimeDistributedCNN (Fixed with Global Pooling)")
    print("="*80)
    
    B, C, T = 4, 10, 100
    x = torch.randn(B, C, T)
    
    print("\n1. TimeDistributedCNN Test:")
    model = TimeDistributedCNN(
        in_channels=10,
        n_classes=2
    )
    
    # Test sequence output
    out_seq, _ = model(x, return_sequence=True)
    print(f"   Input shape:       {x.shape}")
    print(f"   Output (sequence): {out_seq.shape}  # [B, T, n_classes]")
    assert out_seq.shape == (B, T, 2)
    
    # Test global pooling output
    out_global, _ = model(x, return_sequence=False)
    print(f"   Output (global):   {out_global.shape}  # [B, n_classes]")
    assert out_global.shape == (B, 2)
    
    # Verify global output equals mean of sequence
    out_seq_mean = out_seq.mean(dim=1)
    diff = torch.abs(out_global - out_seq_mean).max().item()
    print(f"\n2. Global pooling verification:")
    print(f"   Max difference between global and mean(sequence): {diff:.6f}")
    assert diff < 1e-5, "Global pooling should equal mean of sequence!"
    print(f"   ✓ Global pooling working correctly")
    
    # Test with different inputs
    print("\n3. Testing discriminative power:")
    x1 = torch.ones(2, 10, 100) * 0.96  # Simulate PSPL (lower flux)
    x2 = torch.ones(2, 10, 100) * 1.13  # Simulate Binary (higher flux)
    
    model.eval()
    with torch.no_grad():
        out1, _ = model(x1, return_sequence=False)
        out2, _ = model(x2, return_sequence=False)
    
    pred1 = out1.argmax(dim=1)
    pred2 = out2.argmax(dim=1)
    
    print(f"   Low flux input (0.96):  predictions = {pred1.tolist()}")
    print(f"   High flux input (1.13): predictions = {pred2.tolist()}")
    
    if (pred1 == pred2).all():
        print(f"   ⚠️  Model predicts same for different flux levels (needs training)")
    else:
        print(f"   ✓ Model responds to flux differences")
    
    print("\n" + "="*80)
    print("✅ Model architecture working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_timedistributed()
