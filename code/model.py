#!/usr/bin/env python3
"""
model.py - TimeDistributed CNN (v5.2 - LSTM Only, Corrected)

This version removes the 'Simple' model to enforce the use of the
correct, efficient LSTM-based sequential architecture.

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    """
    TimeDistributed CNN for sequential microlensing classification.
    
    This is the primary model for the thesis. It processes sliding
    windows and uses an LSTM to aggregate temporal features,
    enabling predictions at every timestep.
    
    Input:  [B, 1, T] - full time series
    Output (seq=True): [B, T, 2] - class probabilities at each timestep
    Output (seq=False): [B, 2] - class probabilities at final timestep
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        window_size: int = 50,
        conv_channels: list = [64, 32, 16],
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_lstm: bool = True # Kept for config compatibility
    ):
        super().__init__()
        
        self.window_size = window_size
        self.use_lstm = True # Hard-coded to True
        
        # Feature extractor (applied to each window)
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
            
            nn.AdaptiveAvgPool1d(1)  # Pool to single value per channel
        )
        
        feature_dim = conv_channels[2]
        
        # Temporal aggregator
        self.temporal = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        temporal_output_dim = lstm_hidden
        
        # Classifier (applied at each timestep)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_output_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def extract_windows(self, x):
        """
        Extract sliding windows from sequence.
        
        Args:
            x: [B, C, T] input sequences
            
        Returns:
            windows: [B, T, C, W] where W is window_size
        """
        B, C, T = x.shape
        W = self.window_size
        
        # Pad beginning so first window can be computed
        # (W-1) padding on the left
        x_padded = F.pad(x, (W-1, 0), mode='constant', value=0)
        # Now x_padded: [B, C, W-1+T]
        
        # Extract windows using unfold
        # unfold(dimension, size, step)
        windows = x_padded.unfold(2, W, 1)  # [B, C, T, W]
        windows = windows.permute(0, 2, 1, 3)  # [B, T, C, W]
        
        return windows
    
    def forward(self, x, return_sequence=True):
        """
        Forward pass with TimeDistributed processing.
        
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
        
        # Handle partial sequences: if T < window_size, pad to window_size
        if T < self.window_size:
            padding_needed = self.window_size - T
            # Pad on the right to simulate "future" empty data
            x = F.pad(x, (0, padding_needed), mode='constant', value=0)
            T = self.window_size # Update T
            
        # Extract windows: [B, T, C, W]
        windows = self.extract_windows(x)
        
        # Reshape for CNN processing: [B*T, C, W]
        windows_flat = windows.reshape(B * T, C, self.window_size)
        
        # Extract features from each window: [B*T, feature_dim, 1]
        features = self.feature_extractor(windows_flat)  # [B*T, feature_dim, 1]
        features = features.squeeze(-1)  # [B*T, feature_dim]
        
        # Reshape back to sequence: [B, T, feature_dim]
        features_seq = features.reshape(B, T, -1)
        
        # Temporal aggregation (LSTM): [B, T, lstm_hidden]
        temporal_out, _ = self.temporal(features_seq)
        
        # --- Apply classifier to all timesteps efficiently ---
        # Reshape to apply classifier in one batch
        temporal_out_flat = temporal_out.reshape(B * T, -1) # [B*T, lstm_hidden]
        
        # Classify at each timestep
        logits_flat = self.classifier(temporal_out_flat) # [B*T, n_classes]
        
        # Reshape back to sequence: [B, T, n_classes]
        logits = logits_flat.reshape(B, T, -1)
        
        if return_sequence:
            return logits  # [B, T, n_classes]
        else:
            # Aggregate over time (use last timestep)
            return logits[:, -1, :]  # [B, n_classes]


def test_timedistributed():
    """Test TimeDistributed model"""
    print("="*80)
    print("Testing TimeDistributedCNN (LSTM-Only)")
    print("="*80)
    
    B, C, T = 4, 1, 1500
    x = torch.randn(B, C, T)
    
    # Test full TimeDistributed with LSTM
    print("\n1. TimeDistributedCNN (Full Sequence):")
    model1 = TimeDistributedCNN(
        in_channels=1,
        n_classes=2,
        window_size=50
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

    # Test very short sequence (less than window size)
    print("\n3. TimeDistributedCNN (Short Sequence < Window):")
    T_short = 30
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