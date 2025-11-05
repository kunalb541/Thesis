#!/usr/bin/env python3
"""
Transformer Model for Binary Microlensing Classification

Author: Kunal Bhatia
University of Heidelberg
Date: November 2025
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Efficient Transformer for microlensing classification
    
    Handles padded sequences (-1.0 padding value) and provides:
    - Per-timestep predictions (for decision-time analysis)
    - Final classification output
    
    Args:
        in_channels: Input channels (1 for single light curve)
        n_classes: Number of classes (2: PSPL vs Binary)
        d_model: Transformer embedding dimension (64-128)
        nhead: Number of attention heads (4-8)
        num_layers: Number of Transformer layers (2-4)
        dim_feedforward: FFN dimension (256-512)
        downsample_factor: Reduce sequence length (2-5)
        dropout: Dropout rate
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
        max_len=2000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.downsample_factor = downsample_factor
        
        # Downsample input to reduce memory (1500 -> 500 timesteps)
        self.downsample = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=downsample_factor * 2 + 1,
            stride=downsample_factor,
            padding=downsample_factor,
            bias=True
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads
        self.per_step_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
    
    def forward(self, x, return_sequence=False):
        """
        Args:
            x: [batch, channels, time] with -1.0 as padding
            return_sequence: If True, return per-timestep predictions
        
        Returns:
            logits: [batch, n_classes] or [batch, downsampled_time, n_classes]
            attention_weights: None (for compatibility)
        """
        B, C, T = x.shape
        
        # Create padding mask BEFORE downsampling
        pad_mask = (x[:, 0, :] == -1.0)  # [B, T]
        
        # Downsample: [B, C, T] -> [B, d_model, T_down]
        x = self.downsample(x)
        T_down = x.size(2)
        
        # Downsample mask
        pad_mask_down = nn.functional.max_pool1d(
            pad_mask.float().unsqueeze(1),
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        ).squeeze(1).bool()  # [B, T_down]
        
        # Transpose to [B, T_down, d_model]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=pad_mask_down)
        
        if return_sequence:
            # Per-timestep classification
            logits = self.per_step_head(x)  # [B, T_down, n_classes]
            return logits, None
        else:
            # Global pooling (masked average)
            mask_expanded = (~pad_mask_down).unsqueeze(-1).float()  # [B, T_down, 1]
            x_masked = x * mask_expanded
            
            x_pooled = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            
            # Global classification
            logits = self.global_head(x_pooled)  # [B, n_classes]
            return logits, None


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*80)
    print("Transformer Model Architecture Test")
    print("="*80)
    
    # Test configurations
    configs = [
        ("Small", {"d_model": 64, "nhead": 4, "num_layers": 2, "downsample_factor": 3}),
        ("Medium", {"d_model": 128, "nhead": 8, "num_layers": 3, "downsample_factor": 3}),
        ("Large", {"d_model": 128, "nhead": 8, "num_layers": 4, "downsample_factor": 5}),
    ]
    
    B, C, T = 32, 1, 1500
    x = torch.randn(B, C, T)
    x[:, :, -100:] = -1.0  # Add padding
    
    print(f"\nTest input: {x.shape}")
    print(f"Padded positions: {(x == -1.0).sum().item()}\n")
    
    for name, config in configs:
        print(f"{name} Configuration:")
        print(f"  {config}")
        
        model = TransformerClassifier(in_channels=1, n_classes=2, **config)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            # Test global classification
            logits, _ = model(x, return_sequence=False)
            print(f"  Global output: {logits.shape}")
            
            # Test per-timestep classification
            seq_logits, _ = model(x, return_sequence=True)
            print(f"  Sequence output: {seq_logits.shape}")
        
        # Estimate memory
        T_down = T // config['downsample_factor']
        attention_mem = B * config['nhead'] * T_down * T_down * 4 / 1024**3
        print(f"  Downsampled length: {T_down}")
        print(f"  Est. attention memory: {attention_mem:.3f} GB\n")
    
    print("="*80)
    print("✅ All tests passed!")
    print("="*80)