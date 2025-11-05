#!/usr/bin/env python3
"""
model.py - SIMPLIFIED TimeDistributed CNN (v6.0 - No LSTM)

This version implements the simpler, proven architecture based on
the user's request, removing the LSTM entirely.

FIXES:
- Replaced CNN+LSTM with a purely CNN-based feature extractor.
- Implemented CausalConv1d to ensure causality is maintained
  at the convolution-level, as the LSTM is no longer present.
- Removed all hidden_state management and LSTM parameters.
- Model now matches the user's simpler TensorFlow example
  (Conv1D -> ReLU -> Dropout, repeated).
- Matches the TF example's filters=[128, 64, 32] and kernels=[5, 3, 3].

Author: Kunal Bhatia (modified by Gemini)
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    A 1D causal convolution layer.
    Pads *only* on the left, ensuring output at timestep 't'
    only depends on inputs from 't' and earlier.
    
    This implementation is crucial now that the LSTM is removed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Calculate left padding
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=0, # We handle padding manually
                             dilation=dilation)

    def forward(self, x):
        # x shape: [B, C, T]
        # Apply left padding
        x_padded = F.pad(x, (self.padding, 0)) # Pad (left, right) on the last dim
        return self.conv(x_padded)


class TimeDistributedCNN(nn.Module):
    """
    SIMPLIFIED TimeDistributed CNN for causal microlensing classification.
    
    Architecture (based on user's working TF model):
    1. 3-layer Causal 1D CNN extracts features (128->64->32 filters)
    2. Linear classifier predicts at EACH timestep
    
    Key Properties:
    - CAUSAL: At timestep t, only uses data from timesteps 0 to t
    - NO LSTM: Simpler, faster architecture
    
    Input:  [B, 1, T] - full time series
    Output (seq=True): [B, T, 2] - class probabilities at each timestep
    Output (seq=False): [B, 2] - class probabilities at final timestep
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        # Use TF-based architecture
        conv_channels: list = [128, 64, 32],
        kernel_sizes: list = [5, 3, 3],
        dropout: float = 0.3,
        # Unused args (kept for config.py compatibility)
        window_size: int = 50, 
        use_lstm: bool = False 
    ):
        super().__init__()
        
        # --- 1. 1D Causal CNN Feature Extractor ---
        layers = []
        C_in = in_channels
        for i in range(len(conv_channels)):
            C_out = conv_channels[i]
            k = kernel_sizes[i]
            
            layers.append(
                CausalConv1d(C_in, C_out, kernel_size=k)
            )
            # No BatchNorm, as per user's TF example
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            C_in = C_out
            
        self.feature_extractor = nn.Sequential(*layers)
        
        feature_dim = conv_channels[-1] # 32
        
        # --- 2. Classifier (applied at each timestep) ---
        self.classifier = nn.Linear(feature_dim, n_classes)

    
    def forward(self, x, return_sequence=True):
        """
        Forward pass with causal CNN processing.
        Removed 'hidden_state' argument as LSTM is gone.
        
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
        logits = self.classifier(features) # Output: [B, T, n_classes]
        
        if return_sequence:
            # Return (logits, None) to match the (data, hidden_state)
            # tuple signature expected by train.py
            return logits, None
        else:
            # Return last timestep's prediction
            return logits[:, -1, :], None


# --- START CHANGED: Rigorous Causality Test (Fix #5) ---
def test_causality_rigorous():
    """
    Rigorous causality test: predictions at t should be IDENTICAL
    for any sequence length >= t+1
    """
    print("\n" + "="*80)
    print("Testing Model Causality (Rigorous - No LSTM)")
    print("="*80)
    
    # --- CHANGED: Instantiate the new, simpler model ---
    model = TimeDistributedCNN(in_channels=1, n_classes=2)
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
                print("  This means CausalConv1d is not working correctly.")
                assert diff < 1e-6, "Model is not causal!"

    print("\n✓ Model is causal (predictions only use past data)")
    print("="*80)
# --- END CHANGED ---


def test_timedistributed():
    """Test TimeDistributed model"""
    print("="*80)
    print("Testing TimeDistributedCNN (SIMPLIFIED - No LSTM)")
    print("="*80)
    
    B, C, T = 4, 1, 1500
    x = torch.randn(B, C, T)
    
    print("\n1. TimeDistributedCNN (Full Sequence):")
    # --- CHANGED: Instantiate new model ---
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