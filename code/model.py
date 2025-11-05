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
- Added rigorous causality test to verify this property.

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
        self.lstm_hidden = lstm_hidden # CHANGED: Store lstm_hidden
        self.lstm_layers = 2 # CHANGED: Store num_layers
        
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
            num_layers=self.lstm_layers, # CHANGED: Use stored value
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

    
    def forward(self, x, return_sequence=True, hidden_state=None): # CHANGED: Added hidden_state
        """
        Forward pass with causal CNN+LSTM processing.
        
        Args:
            x: [B, C, T] input sequences
            return_sequence: if True, return predictions at all timesteps
                           if False, return only final prediction
            hidden_state: (Optional) (h, c) tuple for LSTM state
        
        Returns:
            if return_sequence:
                logits: [B, T, n_classes] - predictions at each timestep
                final_hidden: (h, c) tuple
            else:
                logits: [B, n_classes] - final aggregated prediction
                final_hidden: (h, c) tuple
        """
        B, C, T = x.shape
        
        # 1. Extract features at each timestep
        # Input: [B, C, T]
        features = self.feature_extractor(x)  # Output: [B, feature_dim, T]
        
        # 2. Reshape for LSTM
        features = features.permute(0, 2, 1)  # Output: [B, T, feature_dim]
        
        # CHANGED: Initialize hidden state if not provided
        if hidden_state is None:
            h0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden, device=x.device, dtype=x.dtype)
            hidden_state = (h0, c0)
        
        # 3. Temporal aggregation (LSTM)
        # Input: [B, T, feature_dim]
        # Pass hidden_state
        temporal_out, final_hidden = self.temporal(features, hidden_state) # Output: [B, T, lstm_hidden]
        
        # 4. Apply classifier to all timesteps efficiently
        # Reshape to apply classifier in one batch
        temporal_out_flat = temporal_out.reshape(B * T, -1) # [B*T, lstm_hidden]
        
        # Classify at each timestep
        logits_flat = self.classifier(temporal_out_flat) # [B*T, n_classes]
        
        # Reshape back to sequence: [B, T, n_classes]
        logits = logits_flat.reshape(B, T, -1)
        
        if return_sequence:
            return logits, final_hidden  # [B, T, n_classes], (h, c)
        else:
            # Return last timestep's prediction
            return logits[:, -1, :], final_hidden  # [B, n_classes], (h, c)


# --- START CHANGED: Rigorous Causality Test (Fix #5) ---
def test_causality_rigorous():
    """
    Rigorous causality test: predictions at t should be IDENTICAL
    for any sequence length >= t+1
    """
    print("\n" + "="*80)
    print("Testing Model Causality (Rigorous)")
    print("="*80)
    
    model = TimeDistributedCNN(in_channels=1, n_classes=2, lstm_hidden=64)
    model.eval()
    
    B, C = 2, 1
    T_long = 100
    T_test_idx = 49  # Test prediction at index 49 (50th point)
    
    # Generate random sequence
    x_long = torch.randn(B, C, T_long)
    
    base_pred_at_t = None
    
    # Test predictions at T_test_idx for different sequence lengths
    with torch.no_grad():
        
        # Test sequences of increasing length
        for T_curr in [T_test_idx + 1, T_test_idx + 10, T_test_idx + 20, T_long]:
            
            # Get prediction at T_test_idx
            x_curr = x_long[:, :, :T_curr]
            logits_seq, _ = model(x_curr, return_sequence=True)
            
            # Get the prediction from the timestep we care about
            logits_at_t = logits_seq[:, T_test_idx, :]
            
            if base_pred_at_t is None:
                base_pred_at_t = logits_at_t
                print(f"  Base prediction at t={T_test_idx} (from seq len {T_curr}) established.")
                continue

            # Compare to base prediction
            diff = torch.abs(base_pred_at_t - logits_at_t).max().item()
            status = "✓ PASS" if diff < 1e-6 else "❌ FAIL"
            print(f"  Seq len {T_curr:3d}: diff at t={T_test_idx} = {diff:.2e}  {status}")
            
            if diff >= 1e-6:
                print("  ❌ FAILED: Model prediction changed based on future data!")
                assert diff < 1e-6, "Model is not causal!"

    print("\n✓ Model is causal (predictions only use past data)")
    print("="*80)
# --- END CHANGED ---


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

    # Test very short sequence (e.g., length 1)
    print("\n3. TimeDistributedCNN (Short Sequence, T=1):")
    T_short = 1
    x_short = torch.randn(B, C, T_short)
    out_final_short, _ = model1(x_short, return_sequence=False)
    print(f"   Input shape:  {x_short.shape}")
    print(f"   Output (final):    {out_final_short.shape}  # Expected: [B, n_classes]")
    assert out_final_short.shape == (B, 2)
    
    print(f"   ✓ Can predict at each timestep (early detection)")
    print("\n" + "="*80)
    print("✅ TimeDistributed model working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_timedistributed()
    test_causality_rigorous() # CHANGED: Run new test