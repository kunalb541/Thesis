#!/usr/bin/env python3
"""
Transformer Model for Binary Microlensing Classification

Author: Kunal Bhatia
University of Heidelberg
Date: November 2025
Version: 5.6.1 - Enhanced numerical stability in positional encoding
"""

import torch
import torch.nn as nn
import math


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding with enhanced numerical stability.
    
    Uses log-space computation to prevent overflow/underflow in the exponential term.
    Implements the formula from "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings once
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        # Improved numerical stability: use log-space computation
        # div_term = exp(log(10000) * -2i/d_model) for even indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / d_model)
        )
        
        # Clamp to prevent numerical issues (though exp with negative values is stable)
        div_term = torch.clamp(div_term, min=1e-10, max=1e10)

        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Tensor of same shape with positional information added
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.pe.size(1)}. "
                f"Increase max_len in PositionalEncoding initialization."
            )
        
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================
# TRANSFORMER CLASSIFIER
# ============================================================

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for PSPL vs Binary microlensing.
    
    Architecture:
    1. Conv1D Downsampler: Reduces sequence length (1500 → 500)
       - This is PREPROCESSING, not the main model
       - Makes computation tractable for transformers
       
    2. Positional Encoding: Adds position information
    
    3. Transformer Encoder: The actual classification model
       - Multi-head self-attention
       - Feed-forward networks
       - Layer normalization
       
    4. Classification Heads:
       - Per-timestep head: For early detection analysis
       - Global head: For final classification
    
    Features:
    - Handles padding via mask propagation
    - Supports per-timestep outputs (for decision-time analysis)
    - Parallel processing of entire sequence (no recurrence)
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        downsample_factor=3,
        dropout=0.3,
        max_len=2000,
    ):
        super().__init__()

        self.d_model = d_model
        self.downsample_factor = downsample_factor

        # --- 1. Conv1D Downsampler (PREPROCESSING LAYER) ---
        # Reduces sequence length and projects to embedding dimension
        # This is NOT the main model, just makes transformers computationally feasible
        self.downsample = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=downsample_factor * 2 + 1,
            stride=downsample_factor,
            padding=downsample_factor,
            bias=True,
        )

        # --- 2. Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # --- 3. Transformer Encoder (THE ACTUAL MODEL) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 4. Classification Heads ---
        # Per-timestep head for sequential predictions
        self.per_step_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Global head for final classification
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _init_weights(self):
        """Xavier uniform initialization for better convergence"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    # ============================================================
    # FORWARD PASS
    # ============================================================

    def forward(self, x, return_sequence=False, pad_value=-1.0):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch, channels, time]
            return_sequence: If True, return per-timestep predictions
            pad_value: Value used for padding (default -1.0)
            
        Returns:
            logits: 
                - [batch, n_classes] if return_sequence=False
                - [batch, time_downsampled, n_classes] if return_sequence=True
            None: Placeholder for compatibility
        """
        B, C, T = x.shape

        # --- 1. Create Padding Mask ---
        pad_mask = (x[:, 0, :] == pad_value)  # [B, T]

        # --- 2. Downsample Sequence ---
        x = self.downsample(x)  # [B, d_model, T_down]
        B, D, T_down = x.shape

        # --- 3. Downsample Padding Mask ---
        # Use max pooling to preserve padding information
        pad_mask_f = pad_mask.float().unsqueeze(1)
        pad_mask_down = nn.functional.max_pool1d(
            pad_mask_f, 
            kernel_size=self.downsample_factor, 
            stride=self.downsample_factor
        ).squeeze(1).bool()  # [B, T_down]

        # --- 4. Transformer Encoding ---
        x = x.transpose(1, 2)  # [B, T_down, d_model]
        
        # Scale embeddings (standard practice in transformers)
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=pad_mask_down)

        # --- 5. Classification ---
        if return_sequence:
            # Return per-timestep predictions for early detection analysis
            return self.per_step_head(x), None

        # Global classification: masked average pooling
        mask = (~pad_mask_down).unsqueeze(-1).float()  # [B, T_down, 1]
        x_masked = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-8)
        x_pooled = x_masked.sum(dim=1) / denom  # [B, d_model]
        
        logits = self.global_head(x_pooled)
        return logits, None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# TEST / DEBUG ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Transformer Model Architecture Self-Test")
    print("=" * 80)

    B, C, T = 8, 1, 1500
    x = torch.randn(B, C, T)
    x[:, :, -100:] = -1.0  # Simulate padding

    configs = [
        ("Small", {"d_model": 64, "nhead": 4, "num_layers": 2, "downsample_factor": 3}),
        ("Medium", {"d_model": 128, "nhead": 8, "num_layers": 3, "downsample_factor": 3}),
        ("Large", {"d_model": 128, "nhead": 8, "num_layers": 4, "downsample_factor": 5}),
    ]

    for name, cfg in configs:
        print(f"\n{name} configuration:")
        model = TransformerClassifier(
            in_channels=1,
            n_classes=2,
            max_len=(T // cfg["downsample_factor"]) + 1,
            **cfg,
        )
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        model.eval()
        with torch.no_grad():
            out_global, _ = model(x, return_sequence=False)
            out_seq, _ = model(x, return_sequence=True)

        print(f"  Global output shape: {tuple(out_global.shape)}")
        print(f"  Sequence output shape: {tuple(out_seq.shape)}")

        expected_T_down = math.ceil(T / cfg["downsample_factor"])
        assert out_seq.shape[1] == expected_T_down, (
            f"Expected downsampled length ~{expected_T_down}, got {out_seq.shape[1]}"
        )

        print(f"  Downsampled T: {expected_T_down}")
        print("  ✅ Forward pass OK.")

    print("=" * 80)
    print("✅ All configuration tests passed successfully!")
    print("=" * 80)