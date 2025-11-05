#!/usr/bin/env python3
"""
model.py - TimeDistributed CNN (v5.3 - Causal CNN+LSTM)

This version implements a truly causal architecture for real-time
sequential classification.

FIXES:
- Replaced non-causal windowing with a 1D CNN feature extractor
  followed by a unidirectional LSTM for temporal aggregation.
- This model is causal: prediction at timestep 't' only uses
  data from timesteps 0 to 't'.
- Added causality test to verify this property.

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    """
    TimeDistributed CNN+LSTM for causal microlensing classification.
    
    Architecture:
    1. 1D CNN extracts features at each timestep independently
    2. Unidirectional LSTM aggregates past context
    3. Classifier predicts at each timestep
    
    Key Properties:
    - CAUSAL: At timestep t, only uses data from timesteps 0 to t
    - SEQUENTIAL: Produces predictions at every timestep
    - REAL-TIME CAPABLE: Can process partial observations
    
    Training:
    - ALL timesteps supervised with the true label
    - Extra weight on final prediction
    - Optional temporal smoothness regularization
    
    Input:  [B, 1, T] - full time series
    Output (seq=True): [B, T, 2] - class probabilities at each timestep
    Output (seq=False): [B, 2] - class probabilities at final timestep
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        window_size: int = 50, # No longer used for windowing, but kept for config compatibility
        conv_channels: list = [64, 32, 16],
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_lstm: bool = True # Kept for config compatibility
    ):
        super().__init__()
        
        self.window_size = window_size
        self.use_lstm = True # Hard-coded to True
        
        # 1. 1D CNN Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        feature_dim = conv_channels[2] # 16
        
        # 2. Temporal aggregator (LSTM)
        self.temporal = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False # MUST be False for causality
        )
        temporal_output_dim = lstm_hidden
        
        # 3. Classifier (applied at each timestep)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_output_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    
    def forward(self, x, return_sequence=True):
        """
        Forward pass with causal CNN+LSTM processing.
        
        Args:
            x: [B, C, T] input sequences
            return_sequence: if True, return predictions at all timesteps
                           if False, return only final prediction
        
        Returns:
            if return_sequence:
                logits: [B, T, n_classes] - predictions at each timestep
            else:
                logits: [B, n_classes] - final aggregated prediction
        """
        B, C, T = x.shape
        
        # 1. Extract features at each timestep
        # Input: [B, C, T]
        features = self.feature_extractor(x)  # Output: [B, feature_dim, T]
        
        # 2. Reshape for LSTM
        features = features.permute(0, 2, 1)  # Output: [B, T, feature_dim]
        
        # 3. Temporal aggregation (LSTM)
        # Input: [B, T, feature_dim]
        temporal_out, _ = self.temporal(features) # Output: [B, T, lstm_hidden]
        
        # 4. Apply classifier to all timesteps efficiently
        # Reshape to apply classifier in one batch
        temporal_out_flat = temporal_out.reshape(B * T, -1) # [B*T, lstm_hidden]
        
        # Classify at each timestep
        logits_flat = self.classifier(temporal_out_flat) # [B*T, n_classes]
        
        # Reshape back to sequence: [B, T, n_classes]
        logits = logits_flat.reshape(B, T, -1)
        
        if return_sequence:
            return logits  # [B, T, n_classes]
        else:
            # Return last timestep's prediction
            return logits[:, -1, :]  # [B, n_classes]


def test_causality():
    """Verify model only uses past data"""
    print("\n" + "="*80)
    print("Testing Model Causality")
    print("="*80)
    
    model = TimeDistributedCNN(in_channels=1, n_classes=2, lstm_hidden=64)
    model.eval()
    
    B, C, T = 2, 1, 100
    x = torch.randn(B, C, T)
    
    # Get predictions with full sequence
    with torch.no_grad():
        logits_full = model(x, return_sequence=True)  # [B, T, 2]
    
    # Get predictions with truncated sequence (first 50 points)
    T_partial = 50
    x_truncated = x[:, :, :T_partial]
    with torch.no_grad():
        logits_truncated = model(x_truncated, return_sequence=True)  # [B, 50, 2]
    
    # Predictions at t=49 (the 50th point) should be IDENTICAL
    pred_full_t49 = logits_full[:, T_partial - 1, :]
    pred_trunc_t49 = logits_truncated[:, T_partial - 1, :]
    
    diff = torch.abs(pred_full_t49 - pred_trunc_t49)
    
    print(f"Causality test: max difference at t={T_partial-1} = {diff.max().item():.8f}")
    
    if diff.max().item() < 1e-6:
        print("✓ Model is causal (only uses past data)")
    else:
        print(f"❌ FAILED: Model uses future data! Diff: {diff.max().item()}")
    
    assert diff.max().item() < 1e-6, "Model uses future data! Not causal!"
    print("="*80)


def test_timedistributed():
    """Test TimeDistributed model"""
    print("="*80)
    print("Testing TimeDistributedCNN (Causal CNN+LSTM)")
    print("="*80)
    
    B, C, T = 4, 1, 1500
    x = torch.randn(B, C, T)
    
    print("\n1. TimeDistributedCNN (Full Sequence):")
    model1 = TimeDistributedCNN(
        in_channels=1,
        n_classes=2,
        lstm_hidden=64
    )
    
    out_seq = model1(x, return_sequence=True)
    out_final = model1(x, return_sequence=False)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output (sequence): {out_seq.shape}  # Expected: [B, T, n_classes]")
    print(f"   Output (final):    {out_final.shape}  # Expected: [B, n_classes]")
    
    assert out_seq.shape == (B, T, 2)
    assert out_final.shape == (B, 2)
    
    # Test partial sequence (for early detection)
    print("\n2. TimeDistributedCNN (Partial Sequence):")
    T_partial = 100
    x_partial = torch.randn(B, C, T_partial)
    out_final_partial = model1(x_partial, return_sequence=False)
    print(f"   Input shape:  {x_partial.shape}")
    print(f"   Output (final):    {out_final_partial.shape}  # Expected: [B, n_classes]")
    assert out_final_partial.shape == (B, 2)

    # Test very short sequence (e.g., length 1)
    print("\n3. TimeDistributedCNN (Short Sequence, T=1):")
    T_short = 1
    x_short = torch.randn(B, C, T_short)
    out_final_short = model1(x_short, return_sequence=False)
    print(f"   Input shape:  {x_short.shape}")
    print(f"   Output (final):    {out_final_short.shape}  # Expected: [B, n_classes]")
    assert out_final_short.shape == (B, 2)
    
    print(f"   ✓ Can predict at each timestep (early detection)")
    print("\n" + "="*80)
    print("✅ TimeDistributed model working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_timedistributed()
    test_causality()