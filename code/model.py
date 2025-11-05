#!/usr/bin/env python3
"""
model_timedistributed.py - True TimeDistributed CNN

This processes sequences in temporal windows and makes predictions
at each timestep, enabling early detection.

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    """
    TimeDistributed CNN for sequential microlensing classification.
    
    Key Features:
    - Processes sliding windows over time sequence
    - Makes predictions at each timestep
    - Enables early detection (classify with partial observations)
    
    Architecture:
    1. Conv1D feature extractor (learns light curve patterns)
    2. Applied to windows of size W at each timestep
    3. Temporal aggregation (LSTM or attention)
    4. Classification head at each timestep
    
    Input:  [B, 1, T] - full time series
    Output: [B, T, 2] - class probabilities at each timestep
    
    For final prediction: aggregate over all timesteps (mean/max/last)
    For early detection: use prediction at timestep t
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        window_size: int = 50,
        conv_channels: list = [64, 32, 16],
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.window_size = window_size
        self.use_lstm = use_lstm
        
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
        if use_lstm:
            self.temporal = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            temporal_output_dim = lstm_hidden
        else:
            # Simple feed-forward if no LSTM
            self.temporal = None
            temporal_output_dim = feature_dim
        
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
                    For each timestep t, window is x[:, :, max(0, t-W+1):t+1]
        """
        B, C, T = x.shape
        W = self.window_size
        
        # Pad beginning so first window can be computed
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
        
        # Extract windows: [B, T, C, W]
        windows = self.extract_windows(x)
        
        # Reshape for CNN processing: [B*T, C, W]
        windows_flat = windows.reshape(B * T, C, self.window_size)
        
        # Extract features from each window: [B*T, feature_dim, 1]
        features = self.feature_extractor(windows_flat)  # [B*T, feature_dim, 1]
        features = features.squeeze(-1)  # [B*T, feature_dim]
        
        # Reshape back to sequence: [B, T, feature_dim]
        features_seq = features.reshape(B, T, -1)
        
        # Temporal aggregation
        if self.use_lstm:
            # LSTM processes sequence: [B, T, lstm_hidden]
            temporal_out, _ = self.temporal(features_seq)
        else:
            temporal_out = features_seq
        
        # Classify at each timestep: [B, T, n_classes]
        logits = self.classifier(temporal_out)
        
        if return_sequence:
            return logits  # [B, T, n_classes]
        else:
            # Aggregate over time (use last timestep)
            return logits[:, -1, :]  # [B, n_classes]


class TimeDistributedCNNSimple(nn.Module):
    """
    Simpler version without LSTM - just applies CNN to growing windows.
    
    Good for faster inference, still supports early detection.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward_at_timestep(self, x, timestep):
        """
        Make prediction using data up to timestep.
        
        Args:
            x: [B, C, T] full sequence
            timestep: int, predict using data[:, :, :timestep]
        
        Returns:
            logits: [B, n_classes]
        """
        # Use only data up to this timestep
        x_partial = x[:, :, :timestep]
        
        # Extract features
        h = self.features(x_partial)  # [B, 32, timestep]
        
        # Aggregate
        mean_pool = torch.mean(h, dim=-1)  # [B, 32]
        max_pool, _ = torch.max(h, dim=-1)  # [B, 32]
        z = torch.cat([mean_pool, max_pool], dim=1)  # [B, 64]
        
        # Classify
        return self.classifier(z)  # [B, n_classes]
    
    def forward(self, x, return_sequence=True):
        """
        Forward pass.
        
        Args:
            x: [B, C, T] input sequences
            return_sequence: if True, return predictions at multiple timesteps
                           if False, return only final prediction
        
        Returns:
            if return_sequence:
                logits: [B, num_checkpoints, n_classes]
            else:
                logits: [B, n_classes]
        """
        B, C, T = x.shape
        
        if not return_sequence:
            # Just return final prediction
            return self.forward_at_timestep(x, T)
        else:
            # Return predictions at checkpoints: 10%, 25%, 50%, 75%, 100%
            checkpoints = [
                max(int(T * 0.10), 10),
                max(int(T * 0.25), 25),
                max(int(T * 0.50), 50),
                max(int(T * 0.75), 75),
                T
            ]
            
            predictions = []
            for t in checkpoints:
                if t <= T:
                    pred_t = self.forward_at_timestep(x, t)
                    predictions.append(pred_t)
            
            # Stack: [B, num_checkpoints, n_classes]
            return torch.stack(predictions, dim=1)


# Alias for backwards compatibility
TDConvClassifier = TimeDistributedCNNSimple


def test_timedistributed():
    """Test TimeDistributed models"""
    print("="*80)
    print("Testing TimeDistributed Models")
    print("="*80)
    
    B, C, T = 4, 1, 1500
    x = torch.randn(B, C, T)
    
    # Test full TimeDistributed with LSTM
    print("\n1. TimeDistributedCNN (with LSTM):")
    model1 = TimeDistributedCNN(
        in_channels=1,
        n_classes=2,
        window_size=50,
        use_lstm=True
    )
    
    out_seq = model1(x, return_sequence=True)
    out_final = model1(x, return_sequence=False)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output (sequence): {out_seq.shape}  # [B, T, n_classes]")
    print(f"   Output (final):    {out_final.shape}  # [B, n_classes]")
    print(f"   ✓ Can predict at each timestep (early detection)")
    
    # Test simple version
    print("\n2. TimeDistributedCNNSimple (no LSTM):")
    model2 = TimeDistributedCNNSimple(in_channels=1, n_classes=2)
    
    out_checkpoints = model2(x, return_sequence=True)
    out_final2 = model2(x, return_sequence=False)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output (checkpoints): {out_checkpoints.shape}  # [B, 5, n_classes]")
    print(f"   Output (final):       {out_final2.shape}  # [B, n_classes]")
    print(f"   ✓ Predicts at: 10%, 25%, 50%, 75%, 100% observation")
    
    # Test early detection
    print("\n3. Early Detection Test:")
    model2.eval()
    with torch.no_grad():
        pred_10pct = model2.forward_at_timestep(x, int(T * 0.10))
        pred_50pct = model2.forward_at_timestep(x, int(T * 0.50))
        pred_100pct = model2.forward_at_timestep(x, T)
    
    print(f"   Prediction at 10%:  {pred_10pct.shape}")
    print(f"   Prediction at 50%:  {pred_50pct.shape}")
    print(f"   Prediction at 100%: {pred_100pct.shape}")
    print(f"   ✓ Can classify at any point during observation")
    
    print("\n" + "="*80)
    print("✅ TimeDistributed models working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_timedistributed()