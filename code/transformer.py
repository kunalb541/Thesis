#!/usr/bin/env python3
"""
Stable Transformer for Binary Microlensing with Gradient Fix
Fixed numerical stability and gradient explosion issues

Author: Kunal Bhatia
Version: 2.0 - Fixed gradient and stability issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class StableAttention(nn.Module):
    """Numerically stable attention with gradient-safe operations"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Smaller scale factor for more stable gradients
        self.scale = 1.0 / math.sqrt(self.d_k * 2)  # Extra scaling for stability
        
        # Separate Q, K, V projections (more stable than combined)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with very small values
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Much smaller initialization to prevent gradient explosion
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project to Q, K, V with gradient-safe operations
        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Normalize Q and K to prevent explosion
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        
        # Scaled dot-product with extra stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # More aggressive clamping for stability
        scores = torch.clamp(scores, min=-5, max=5)
        
        # Apply mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e4)
        else:
            scores = scores + mask
        
        # Stable softmax with temperature
        attn = F.softmax(scores / 1.5, dim=-1)  # Temperature scaling
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        # Residual scaling built-in
        return out * 0.5


class StableTransformerBlock(nn.Module):
    """Transformer block with improved gradient flow"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Layer norms with higher epsilon for stability
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.attn = StableAttention(d_model, nhead, dropout)
        
        # Smaller FFN with better initialization
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        
        # Learnable residual gates for stability
        self.gate1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gate2 = nn.Parameter(torch.ones(1) * 0.1)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm with gated residuals for stable gradients
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.gate1 * attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.gate2 * ffn_out
        
        # Prevent gradient explosion
        x = torch.clamp(x, min=-10, max=10)
        
        return x


class SimpleStableTransformer(nn.Module):
    """
    Improved transformer with gradient stability fixes
    
    Key improvements:
    - Better initialization (much smaller)
    - Gradient-safe normalization
    - Learnable gating for residuals
    - Temperature scaling in softmax
    - Separate Q/K/V projections
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,  # Increased dropout
        max_len: int = 2000
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        
        # Input projection with LayerNorm for stability
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding (smaller init)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
        # Token for padding
        self.pad_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            StableTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling for final representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Simplified output heads with better init
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
        """Initialize all weights carefully"""
        # Input projection
        for m in self.input_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output heads - very small init
        for head in [self.binary_head, self.anomaly_head, self.caustic_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_timesteps: bool = False,
        pad_value: float = -1.0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with gradient safety checks"""
        
        # Handle input shape
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        B, T = x.shape
        device = x.device
        
        # Normalize input to prevent large values
        x = torch.clamp(x, min=-10, max=10)
        
        # Create padding mask
        pad_mask = (x == pad_value)
        
        # Input projection
        x_expanded = x.unsqueeze(-1)
        x_embed = self.input_proj(x_expanded)  # [B, T, d_model]
        
        # Add positional encoding (scaled)
        x_embed = x_embed + 0.1 * self.pos_embed[:, :T, :]
        
        # Replace padding with learned token
        for b in range(B):
            if pad_mask[b].any():
                x_embed[b, pad_mask[b]] = self.pad_token[0, 0]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1) * -1e4
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply transformer blocks with gradient checkpointing
        for i, block in enumerate(self.blocks):
            x_embed = block(x_embed, causal_mask)
            
            # Check for NaN/Inf
            if torch.isnan(x_embed).any() or torch.isinf(x_embed).any():
                print(f"NaN/Inf detected in block {i}")
                x_embed = torch.nan_to_num(x_embed, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Final norm
        x_embed = self.norm(x_embed)
        
        # Get outputs
        if return_all_timesteps:
            outputs = {
                'binary': self.binary_head(x_embed),
                'anomaly': self.anomaly_head(x_embed).squeeze(-1),
                'caustic': self.caustic_head(x_embed).squeeze(-1)
            }
        else:
            # Global pooling over non-padded timesteps
            mask_expanded = (~pad_mask).float().unsqueeze(-1)
            x_masked = x_embed * mask_expanded
            x_sum = x_masked.sum(dim=1)
            x_mean = x_sum / mask_expanded.sum(dim=1).clamp(min=1)
            
            outputs = {
                'binary': self.binary_head(x_mean),
                'anomaly': self.anomaly_head(x_mean),
                'caustic': self.caustic_head(x_mean)
            }
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model for stability"""
    print("="*60)
    print("TESTING IMPROVED STABLE TRANSFORMER")
    print("="*60)
    
    # Create model
    model = SimpleStableTransformer(
        n_points=1500,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.2
    )
    
    print(f"Model created: {count_parameters(model):,} parameters")
    
    # Test data
    batch_size = 8
    seq_len = 1500
    x = torch.randn(batch_size, seq_len) * 0.5  # Smaller input range
    x[:, -500:] = -1.0  # Padding
    
    # Test forward pass
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
    
    # Test backward pass
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    y = torch.randint(0, 2, (batch_size,))
    
    # Forward
    outputs = model(x, return_all_timesteps=False)
    loss = criterion(outputs['binary'], y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            grad_norms.append(grad_norm.item())
            if grad_norm > 10:
                print(f"  Large gradient in {name}: {grad_norm:.2f}")
    
    max_grad = max(grad_norms) if grad_norms else 0
    mean_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    print(f"\nGradient Statistics:")
    print(f"  Max gradient norm: {max_grad:.2f}")
    print(f"  Mean gradient norm: {mean_grad:.2f}")
    print(f"  Loss: {loss.item():.4f}")
    
    if max_grad < 100 and not torch.isnan(loss):
        print("\n✅ Model is stable with improved gradients!")
    else:
        print("\n⚠️ Model may still have issues")
    
    return model


if __name__ == "__main__":
    model = test_model()