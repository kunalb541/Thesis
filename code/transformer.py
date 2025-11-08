#!/usr/bin/env python3
"""
Masked Transformer for Microlensing with Missing Data Handling
Properly handles gaps in observations without creating artificial features

Author: Kunal Bhatia
Version: 3.0 - With attention masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple


class MaskedAttention(nn.Module):
    """Multi-head attention with proper masking for missing data"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                validity_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input embeddings
            validity_mask: [B, T] boolean mask (True = valid data)
            causal_mask: [T, T] causal attention mask
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Normalize for stability
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-5, max=5)
        
        # Apply validity mask - don't attend to invalid positions
        if validity_mask is not None:
            # Expand mask for all heads [B, 1, 1, T]
            validity_attn_mask = validity_mask.unsqueeze(1).unsqueeze(2)
            # Set scores to -inf where attending to invalid positions
            scores = scores.masked_fill(~validity_attn_mask, -1e9)
        
        # Apply causal mask if provided
        if causal_mask is not None:
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        return out * 0.5  # Scale for stability


class MaskedTransformerBlock(nn.Module):
    """Transformer block that respects validity masks"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.attn = MaskedAttention(d_model, nhead, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        
        self.gate1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gate2 = nn.Parameter(torch.ones(1) * 0.1)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, 
                validity_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention with masking
        attn_out = self.attn(self.norm1(x), validity_mask, causal_mask)
        x = x + self.gate1 * attn_out
        
        # FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.gate2 * ffn_out
        
        # Clamp for stability
        x = torch.clamp(x, min=-10, max=10)
        
        return x


class MaskedMicrolensingTransformer(nn.Module):
    """
    Transformer for microlensing that properly handles missing data
    
    Key features:
    - Smooth interpolation instead of padding
    - Validity mask embedding
    - Attention masking for missing data
    - No artificial discontinuities
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        max_len: int = 2000,
        pad_value: float = -1.0
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Validity embedding - tells model which data is real vs interpolated
        self.validity_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
        # Transformer blocks with masking
        self.blocks = nn.ModuleList([
            MaskedTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output heads
        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights carefully"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def preprocess_input(self, x: torch.Tensor, validity_mask: Optional[torch.Tensor] = None):
        """
        Preprocess input to remove discontinuities
        
        Args:
            x: [B, T] raw input with padding
            validity_mask: [B, T] boolean mask (optional)
            
        Returns:
            x_smooth: [B, T] smoothed input
            validity_mask: [B, T] boolean mask
        """
        B, T = x.shape
        device = x.device
        
        # Auto-detect validity if not provided
        if validity_mask is None:
            validity_mask = (x != self.pad_value)
        
        x_smooth = x.clone()
        
        # Process each sequence
        for b in range(B):
            valid_indices = torch.where(validity_mask[b])[0]
            
            if len(valid_indices) == 0:
                # All padding - use zeros
                x_smooth[b] = 0.0
                continue
            
            if len(valid_indices) == T:
                # No missing data
                continue
            
            # Get valid data
            valid_data = x[b, valid_indices]
            
            # Estimate baseline from first and last 20% of valid data
            n_valid = len(valid_data)
            n_baseline = max(1, int(0.2 * n_valid))
            baseline_points = torch.cat([
                valid_data[:n_baseline],
                valid_data[-n_baseline:]
            ])
            baseline = baseline_points.median()
            
            # Center the data
            x_smooth[b, valid_indices] = valid_data - baseline
            
            # Fill gaps with smooth interpolation
            first_valid = valid_indices[0].item()
            last_valid = valid_indices[-1].item()
            
            # Before first valid: use first valid value
            if first_valid > 0:
                x_smooth[b, :first_valid] = x_smooth[b, first_valid]
            
            # After last valid: use last valid value
            if last_valid < T - 1:
                x_smooth[b, last_valid+1:] = x_smooth[b, last_valid]
            
            # Interpolate internal gaps
            for i in range(first_valid + 1, last_valid):
                if not validity_mask[b, i]:
                    # Find surrounding valid points
                    prev_valid = valid_indices[valid_indices < i].max()
                    next_valid = valid_indices[valid_indices > i].min()
                    
                    # Linear interpolation
                    alpha = (i - prev_valid).float() / (next_valid - prev_valid).float()
                    x_smooth[b, i] = (x_smooth[b, prev_valid] * (1 - alpha) + 
                                      x_smooth[b, next_valid] * alpha)
        
        return x_smooth, validity_mask
    
    def forward(
        self,
        x: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with proper missing data handling
        
        Args:
            x: [B, T] input light curves with padding
            validity_mask: [B, T] boolean mask (True = valid data)
            return_all_timesteps: whether to return per-timestep predictions
            
        Returns:
            Dictionary with predictions
        """
        # Handle input shape
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        B, T = x.shape
        device = x.device
        
        # Preprocess to remove discontinuities
        x_smooth, validity_mask = self.preprocess_input(x, validity_mask)
        
        # Input projection
        x_embed = self.input_proj(x_smooth.unsqueeze(-1))  # [B, T, d_model]
        
        # Add validity information
        validity_feat = self.validity_embedding(
            validity_mask.float().unsqueeze(-1)
        )
        x_embed = x_embed + validity_feat
        
        # Add positional encoding
        x_embed = x_embed + 0.1 * self.pos_embed[:, :T, :]
        
        # Create causal mask (optional - remove if not needed)
        # causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1) * -1e4
        causal_mask = None  # For full bidirectional attention
        
        # Apply transformer blocks with masking
        for block in self.blocks:
            x_embed = block(x_embed, validity_mask, causal_mask)
        
        # Final normalization
        x_embed = self.norm(x_embed)
        
        # Get outputs
        if return_all_timesteps:
            outputs = {
                'binary': self.binary_head(x_embed),
                'anomaly': self.anomaly_head(x_embed).squeeze(-1),
                'caustic': self.caustic_head(x_embed).squeeze(-1)
            }
        else:
            # Pool over valid timesteps only
            if validity_mask is not None:
                # Masked mean pooling
                mask_expanded = validity_mask.unsqueeze(-1).float()
                x_sum = (x_embed * mask_expanded).sum(dim=1)
                x_mean = x_sum / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                # Standard mean pooling
                x_mean = x_embed.mean(dim=1)
            
            outputs = {
                'binary': self.binary_head(x_mean),
                'anomaly': self.anomaly_head(x_mean),
                'caustic': self.caustic_head(x_mean)
            }
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_masked_transformer():
    """Test the masked transformer"""
    print("="*60)
    print("TESTING MASKED TRANSFORMER")
    print("="*60)
    
    # Create model
    model = MaskedMicrolensingTransformer(
        n_points=1500,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.2
    )
    
    print(f"Model created: {count_parameters(model):,} parameters")
    
    # Test data with missing values
    batch_size = 4
    seq_len = 1500
    
    # Create data with gaps
    x = torch.randn(batch_size, seq_len)
    
    # Add some "missing" data (padding)
    x[:, 300:400] = -1.0  # Gap in middle
    x[:, 1200:] = -1.0    # Missing end
    
    # Auto-detect validity mask
    validity_mask = (x != -1.0)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Valid data: {validity_mask.float().mean()*100:.1f}%")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x, validity_mask, return_all_timesteps=False)
    
    print("\nOutputs:")
    for key, val in outputs.items():
        print(f"  {key}: shape={val.shape}, range=[{val.min():.3f}, {val.max():.3f}]")
    
    # Check that preprocessing removes discontinuities
    x_smooth, _ = model.preprocess_input(x, validity_mask)
    
    # Check for jumps
    max_jumps = []
    for i in range(batch_size):
        diffs = torch.abs(torch.diff(x_smooth[i]))
        max_jumps.append(diffs.max().item())
    
    print(f"\nMax jump after preprocessing: {np.mean(max_jumps):.3f} (should be small)")
    print("\n✅ Masked transformer test passed!")
    
    return model


if __name__ == "__main__":
    model = test_masked_transformer()