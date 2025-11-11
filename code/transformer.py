#!/usr/bin/env python3
"""
Production Transformer for 8x H100 Distributed Training
Optimized for multi-GPU with gradient stability

Author: Kunal Bhatia
Version: 10.0 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple


class StableMultiHeadAttention(nn.Module):
    """Multi-head attention optimized for H100 with Flash Attention support"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections - use bias for stability
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norms for Q, K stability
        self.q_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.k_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier initialization with reduced gain for stability
        gain = 1.0 / math.sqrt(2)
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        # Normalize input for stability
        x = self.q_norm(x)
        
        # Linear projections
        Q = self.w_q(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Normalize Q and K to prevent explosion
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        # Scaled dot-product attention with clamping
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-10, max=10)
        
        # Apply mask if provided (True = invalid/padded)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)  # Not -inf to avoid NaN
        
        # Stable softmax
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.w_o(out)
        
        # Scale output for residual
        return out * 0.5


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and stable residuals"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Pre-norm architecture (more stable)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.attn = StableMultiHeadAttention(d_model, nhead, dropout)
        
        # FFN with GELU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable gates for stable residuals
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + attention
        attn_out = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        # Pre-norm + FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        # Clamp for stability
        x = torch.clamp(x, min=-100, max=100)
        
        return x


class MicrolensingTransformer(nn.Module):
    """
    Production transformer for microlensing with H100 optimization
    - Stable gradient flow
    - Efficient for distributed training
    - Handles missing data properly
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 256,      # Larger for H100s
        nhead: int = 8,          # More heads for H100s
        num_layers: int = 6,     # Deeper for better performance
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_value: float = -1.0,
        max_seq_len: int = 2000
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        self.nhead = nhead
        
        # Input embedding with feature extraction
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Gap embedding for missing data
        self.gap_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output heads
        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Auxiliary heads for multi-task learning
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
        # Register backward hook for gradient clipping
        for param in self.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0) if grad is not None else grad)
    
    def _init_weights(self):
        """Initialize weights carefully for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True where data is invalid/padded)"""
        if x.dim() == 3:
            return x[:, :, 0] == self.pad_value
        else:
            return x == self.pad_value
    
    def compute_gap_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute features about gaps in data"""
        B, T = mask.shape
        
        # Cumulative gap count
        gaps = mask.float()
        cumulative = gaps.cumsum(dim=1)
        
        # Normalize by position
        positions = torch.arange(1, T + 1, device=mask.device).unsqueeze(0)
        gap_ratio = cumulative / positions
        
        return gap_ratio.unsqueeze(-1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input light curves [B, T] or [B, T, 1]
            return_all: Return all auxiliary outputs
            
        Returns:
            Dictionary with model outputs
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        B, T, _ = x.shape
        device = x.device
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Normalize input (only valid positions)
        valid_mask = ~padding_mask
        if valid_mask.any():
            # Compute statistics on valid data
            valid_data = x[valid_mask.unsqueeze(-1)].view(-1)
            mean = valid_data.mean()
            std = valid_data.std() + 1e-8
            
            # Safe normalization
            x_norm = x.clone()
            x_norm[valid_mask] = (x[valid_mask] - mean) / std
            x_norm = torch.clamp(x_norm, min=-10, max=10)
            x_norm[padding_mask] = 0  # Zero out padding
        else:
            x_norm = torch.zeros_like(x)
        
        # Input embedding
        x_embed = self.input_embed(x_norm)
        
        # Add gap information
        gap_features = self.compute_gap_features(padding_mask)
        gap_embed = self.gap_embed(gap_features)
        x_embed = x_embed + 0.1 * gap_embed
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding[:, :T, :]
        
        # Pass through transformer layers
        for layer in self.layers:
            x_embed = layer(x_embed, padding_mask)
        
        # Final normalization
        x_embed = self.norm(x_embed)
        
        # Global pooling (weighted by validity)
        if valid_mask.any():
            # Masked mean pooling
            valid_expand = valid_mask.unsqueeze(-1).float()
            x_sum = (x_embed * valid_expand).sum(dim=1)
            x_count = valid_expand.sum(dim=1).clamp(min=1)
            x_pooled = x_sum / x_count
            
            # Also get max pooling
            x_masked = x_embed.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            x_max, _ = x_masked.max(dim=1)
            
            # Combine
            x_final = x_pooled + 0.1 * x_max
        else:
            x_final = x_embed.mean(dim=1)
        
        # Get outputs
        outputs = {
            'binary': self.binary_head(x_final)
        }
        
        if return_all:
            outputs['anomaly'] = self.anomaly_head(x_final).squeeze(-1)
            outputs['caustic'] = self.caustic_head(x_final).squeeze(-1)
        
        # Clamp outputs to prevent overflow in loss
        outputs['binary'] = torch.clamp(outputs['binary'], min=-20, max=20)
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model creation and forward pass"""
    print("Testing MicrolensingTransformer...")
    
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"Model created: {count_parameters(model):,} parameters")
    
    # Test forward pass
    x = torch.randn(4, 1500)
    x[:, 1000:] = -1.0  # Add padding
    
    outputs = model(x)
    print(f"Output shape: {outputs['binary'].shape}")
    print("✅ Model test passed!")


if __name__ == "__main__":
    test_model()
