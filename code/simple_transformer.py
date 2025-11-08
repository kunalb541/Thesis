#!/usr/bin/env python3
"""
Simple Stable Transformer for Binary Microlensing
Carefully initialized to avoid NaN issues

Author: Kunal Bhatia
Version: 1.0 - Stable version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class StableAttention(nn.Module):
    """Numerically stable attention mechanism"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Initialize with smaller scale for stability
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights carefully
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier uniform with gain < 1 for stability
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Clamp scores to prevent overflow in softmax
        scores = torch.clamp(scores, min=-10, max=10)
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1) * -1e9
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores + mask
        
        # Stable softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        out = self.out_proj(out)
        
        return out


class StableTransformerBlock(nn.Module):
    """Transformer block with numerical stability"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Pre-norm architecture (more stable than post-norm)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.attn = StableAttention(d_model, nhead, dropout)
        
        # FFN with careful initialization
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Initialize FFN carefully
        self._reset_parameters()
        
        # Residual scaling for stability
        self.residual_scale = 0.9
    
    def _reset_parameters(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + residual with scaling
        x = x + self.residual_scale * self.attn(self.norm1(x), mask)
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x


class SimpleStableTransformer(nn.Module):
    """
    Simple, stable transformer for binary microlensing classification
    
    Features:
    - Careful initialization to prevent NaN
    - Numerical stability throughout
    - Multi-head outputs (binary, anomaly, caustic)
    - No fancy tricks that could cause issues
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 64,      # Small for stability
        nhead: int = 4,         # Fewer heads
        num_layers: int = 3,    # Shallow
        dim_ff: int = 256,      # Moderate FFN
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        
        # Input projection with careful init
        self.input_proj = nn.Linear(1, d_model, bias=True)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)  # Very small init
        nn.init.zeros_(self.input_proj.bias)
        
        # Learned positional encoding (more stable than sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0, std=0.02)  # Small init
        
        # Token type embedding for padding
        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pad_token, mean=0, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            StableTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Output heads with careful initialization
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Caustic detection head
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Probability
        )
        
        # Initialize output heads carefully
        self._init_heads()
    
    def _init_heads(self):
        """Initialize output heads with small weights"""
        for head in [self.binary_head, self.anomaly_head, self.caustic_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_timesteps: bool = False,
        pad_value: float = -1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with numerical stability
        
        Args:
            x: Input [batch, seq_len] or [batch, 1, seq_len]
            return_all_timesteps: Return sequence outputs
            pad_value: Padding sentinel value
        """
        # Handle input shape
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        B, T = x.shape
        device = x.device
        
        # Create padding mask
        pad_mask = (x == pad_value)
        
        # Expand to [B, T, 1] for projection
        x_expanded = x.unsqueeze(-1)
        
        # Input projection
        x_embed = self.input_proj(x_expanded)  # [B, T, d_model]
        
        # Add positional encoding
        x_embed = x_embed + self.pos_embed[:, :T, :]
        
        # Replace padded positions with pad token
        for b in range(B):
            if pad_mask[b].any():
                x_embed[b, pad_mask[b]] = self.pad_token[0, 0]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1) * -1e9
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.blocks:
            x_embed = block(x_embed, causal_mask)
            
            # Clamp to prevent explosion
            x_embed = torch.clamp(x_embed, min=-10, max=10)
        
        # Final normalization
        x_embed = self.norm(x_embed)
        
        # Apply output heads
        binary_logits = self.binary_head(x_embed)  # [B, T, 2]
        anomaly_scores = self.anomaly_head(x_embed)  # [B, T, 1]
        caustic_probs = self.caustic_head(x_embed)  # [B, T, 1]
        
        # Prepare outputs
        if return_all_timesteps:
            outputs = {
                'binary': binary_logits,
                'anomaly': anomaly_scores,
                'caustic': caustic_probs
            }
        else:
            # Use last non-padded timestep
            last_valid = (~pad_mask).sum(dim=1) - 1
            last_valid = last_valid.clamp(min=0, max=T-1)
            
            batch_idx = torch.arange(B, device=device)
            outputs = {
                'binary': binary_logits[batch_idx, last_valid],  # [B, 2]
                'anomaly': anomaly_scores[batch_idx, last_valid],  # [B, 1]
                'caustic': caustic_probs[batch_idx, last_valid]    # [B, 1]
            }
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model for NaN issues"""
    print("="*60)
    print("TESTING SIMPLE STABLE TRANSFORMER")
    print("="*60)
    
    # Create model
    model = SimpleStableTransformer(
        n_points=1500,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.1
    )
    
    print(f"Model created: {count_parameters(model):,} parameters")
    
    # Test data
    batch_size = 4
    seq_len = 1500
    x = torch.randn(batch_size, seq_len) * 2  # Reasonable input range
    x[:, -500:] = -1.0  # Padding
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_all_timesteps=False)
    
    print("\nOutputs:")
    for key, val in outputs.items():
        has_nan = torch.isnan(val).any()
        has_inf = torch.isinf(val).any()
        print(f"  {key}: shape={val.shape}, NaN={has_nan}, Inf={has_inf}")
        if not has_nan and not has_inf:
            print(f"    Range: [{val.min():.3f}, {val.max():.3f}]")
    
    # Test loss
    criterion = nn.CrossEntropyLoss()
    y = torch.randint(0, 2, (batch_size,))
    loss = criterion(outputs['binary'], y)
    print(f"\nLoss: {loss.item():.4f}, NaN={torch.isnan(loss)}")
    
    if not torch.isnan(loss):
        print("\n✅ Model is stable!")
    else:
        print("\n❌ Model still has issues")
    
    return model


if __name__ == "__main__":
    model = test_model()
