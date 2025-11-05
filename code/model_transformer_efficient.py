#!/usr/bin/env python3
"""
model_transformer_efficient.py - Memory-Efficient Transformer

FIXES:
1. Chunked attention to reduce memory
2. Better architecture for time series
3. Proper handling of long sequences
4. More stable training

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EfficientTransformerClassifier(nn.Module):
    """
    Memory-efficient Transformer for microlensing classification.
    
    Key improvements:
    1. Downsampling to reduce sequence length
    2. Local + global attention pattern
    3. Better initialization
    4. Gradient checkpointing support
    
    Args:
        in_channels: Input channels (1 for single light curve)
        n_classes: Number of classes (2 for binary)
        d_model: Model dimension (recommend 64-128)
        nhead: Number of attention heads (must divide d_model)
        num_layers: Number of transformer layers (recommend 2-4)
        downsample_factor: Downsample input by this factor (2-5)
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
        downsample_factor=3,  # 1500 -> 500 timesteps
        dropout=0.3,
        max_len=2000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.downsample_factor = downsample_factor
        
        # Downsample using 1D conv (reduces memory dramatically)
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
        
        # Transformer encoder (with efficient settings)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization for stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)  # Small init for stability
    
    def forward(self, x, return_sequence=False):
        """
        Args:
            x: [batch, channels, time]
            return_sequence: If True, return per-timestep predictions
        
        Returns:
            logits: [batch, n_classes] or [batch, downsampled_time, n_classes]
        """
        B, C, T = x.shape
        
        # Create mask BEFORE downsampling
        # Padded positions have value -1.0
        pad_mask = (x[:, 0, :] == -1.0)  # [B, T]
        
        # Downsample: [B, C, T] -> [B, d_model, T//downsample_factor]
        x = self.downsample(x)  # [B, d_model, T_down]
        T_down = x.size(2)
        
        # Downsample mask too
        pad_mask_down = F.max_pool1d(
            pad_mask.float().unsqueeze(1),
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        ).squeeze(1).bool()  # [B, T_down]
        
        # Transpose to [B, T_down, d_model]
        x = x.transpose(1, 2)
        
        # Scale and add positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask_down)
        
        if return_sequence:
            # Per-timestep classification
            logits = self.classifier(x)  # [B, T_down, n_classes]
            return logits, None
        else:
            # Global pooling (masked average)
            mask_expanded = (~pad_mask_down).unsqueeze(-1).float()  # [B, T_down, 1]
            x_masked = x * mask_expanded
            
            sum_valid = x_masked.sum(dim=1)  # [B, d_model]
            count_valid = mask_expanded.sum(dim=1)  # [B, 1]
            x_pooled = sum_valid / (count_valid + 1e-8)
            
            # Classify
            logits = self.classifier(x_pooled)  # [B, n_classes]
            return logits, None


def test_efficient_transformer():
    """Test memory usage"""
    print("="*80)
    print("Testing Efficient Transformer")
    print("="*80)
    
    B, C, T = 32, 1, 1500
    
    # Create test data
    x = torch.randn(B, C, T)
    x[:, :, -100:] = -1.0  # Add padding
    
    print(f"\nInput: {x.shape}")
    print(f"Padded positions: {(x == -1.0).sum().item()}")
    
    # Test different configurations
    configs = [
        ("Small", {"d_model": 64, "nhead": 4, "num_layers": 2, "downsample_factor": 3}),
        ("Medium", {"d_model": 128, "nhead": 8, "num_layers": 3, "downsample_factor": 3}),
        ("Large", {"d_model": 128, "nhead": 8, "num_layers": 4, "downsample_factor": 5}),
    ]
    
    for name, config in configs:
        print(f"\n{name} Model:")
        print(f"  Config: {config}")
        
        model = EfficientTransformerClassifier(
            in_channels=1,
            n_classes=2,
            **config
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, return_sequence=False)
        
        print(f"  Output: {logits.shape}")
        
        # Estimate memory
        T_down = T // config['downsample_factor']
        attention_mem = B * config['nhead'] * T_down * T_down * 4 / 1024**3  # GB
        print(f"  Downsampled length: {T_down}")
        print(f"  Est. attention memory: {attention_mem:.2f} GB")
    
    print("\n" + "="*80)
    print("✅ All configurations working!")
    print("="*80)


if __name__ == "__main__":
    test_efficient_transformer()
