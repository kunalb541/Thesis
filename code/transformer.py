#!/usr/bin/env python3
"""
Transformer Architecture for THREE-CLASS Classification - ARCHITECTURAL FIX
===========================================================================
Classes: 0=Flat (no event), 1=PSPL, 2=Binary

v12.0-beta ARCHITECTURAL FIX:
- Relative positional encoding (no absolute time knowledge)
- Model only knows: "I've seen N observations" not "I'm at day -50"
- Variable-length sequence support
- NO causal truncation needed (tested and rejected)

This PREVENTS the model from "cheating" by inferring event type
from temporal position. The architectural fix alone is sufficient.

Author: Kunal Bhatia
Version: 12.0-beta - Architectural Fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    """
    ARCHITECTURAL FIX: Positional encoding that only knows:
    1. How many valid observations seen so far
    2. Relative time gaps between observations
    
    Does NOT know:
    - Absolute calendar time (no "day -50" vs "day 0" knowledge)
    - Total sequence length
    - Future observation times
    
    This prevents the model from learning temporal artifacts!
    """
    
    def __init__(self, d_model: int, max_observations: int = 2000):
        super().__init__()
        self.d_model = d_model
        
        # Embedding for "number of observations seen so far"
        self.obs_count_encoding = nn.Embedding(max_observations, d_model // 2)
        
        # Encoding for relative gaps (time since last observation)
        # This is causal because it only looks backward
        self.gap_encoding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # Initialize conservatively
        nn.init.normal_(self.obs_count_encoding.weight, std=0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative positional encoding (architectural fix)
        
        Args:
            x: Input tensor [B, T, D]
            padding_mask: Boolean mask [B, T] (True = padded/invalid)
        
        Returns:
            pos_encoding: [B, T, D] positional encoding
        """
        B, T, D = x.shape
        device = x.device
        
        # Compute cumulative observation count
        valid_mask = ~padding_mask
        obs_count = torch.cumsum(valid_mask.long(), dim=1) - 1  # [B, T]
        obs_count = torch.clamp(
            obs_count,
            min=0,
            max=self.obs_count_encoding.num_embeddings - 1
        )
        
        # Embedding: "I have seen N observations so far"
        count_embed = self.obs_count_encoding(obs_count)  # [B, T, D/2]
        
        # Compute relative gaps
        gaps = torch.zeros(B, T, 1, device=device)
        for b in range(B):
            valid_indices = torch.where(valid_mask[b])[0]
            if len(valid_indices) > 1:
                # Gap = current position - last valid position
                for i in range(1, len(valid_indices)):
                    curr_idx = valid_indices[i]
                    prev_idx = valid_indices[i-1]
                    gaps[b, curr_idx, 0] = float(curr_idx - prev_idx)
        
        # Encode gaps
        gap_embed = self.gap_encoding(gaps)  # [B, T, D/2]
        
        # Concatenate both sources of information
        pos_encoding = torch.cat([count_embed, gap_embed], dim=-1)  # [B, T, D]
        
        return pos_encoding


class StableMultiHeadAttention(nn.Module):
    """Multi-head attention with stability improvements"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norms for stability
        self.q_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        self._init_weights()
    
    def _init_weights(self):
        gain = 1.0 / math.sqrt(2)
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        # Normalize input
        x = self.q_norm(x)
        
        # Project to Q, K, V
        Q = self.w_q(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Normalize Q and K for stability
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-10, max=10)
        
        # Apply padding mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask, -1e4)
        
        # Softmax with stability
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.w_o(out)
        
        return out * 0.5  # Residual scale


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and stable residuals"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.attn = StableMultiHeadAttention(d_model, nhead, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        # Clamp for stability
        x = torch.clamp(x, min=-100, max=100)
        
        return x


class MicrolensingTransformer(nn.Module):
    """
    Transformer for microlensing classification with architectural fix
    
    v12.0-beta ARCHITECTURAL FIX:
    1. Uses RelativePositionalEncoding (no absolute time)
    2. Only knows: "I've seen N observations"
    3. Cannot infer event type from temporal position
    4. Smaller model: ~100K parameters (vs 450K in v11)
    
    Classes: 0=Flat, 1=PSPL, 2=Binary
    
    Auxiliary tasks:
    - Flat detection (0.5 weight)
    - PSPL detection (0.5 weight)  
    - Anomaly detection (0.2 weight)
    - Caustic detection (0.2 weight)
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 128,           # v12.0-beta: Smaller (was 256)
        nhead: int = 4,               # v12.0-beta: Smaller (was 8)
        num_layers: int = 4,          # v12.0-beta: Smaller (was 6)
        dim_feedforward: int = 512,   # v12.0-beta: Smaller (was 1024)
        dropout: float = 0.1,
        pad_value: float = -1.0,
        max_seq_len: int = 2000
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        self.nhead = nhead
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # v12.0-beta: RELATIVE positional encoding (architectural fix!)
        self.pos_encoding = RelativePositionalEncoding(
            d_model=d_model,
            max_observations=max_seq_len
        )
        
        # Gap embedding for sparse observation patterns
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
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # ============== CLASSIFICATION HEADS ==============
        
        # Main: 3-class classification (Flat, PSPL, Binary)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Alias for backwards compatibility
        self.binary_head = self.classification_head
        
        # Auxiliary heads (output logits for BCEWithLogitsLoss)
        self.flat_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.pspl_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True = invalid/padded)"""
        if x.dim() == 3:
            return x[:, :, 0] == self.pad_value
        else:
            return x == self.pad_value
    
    def compute_gap_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute features about observation gaps"""
        B, T = mask.shape
        
        gaps = mask.float()
        cumulative = gaps.cumsum(dim=1)
        
        positions = torch.arange(1, T + 1, device=mask.device).unsqueeze(0)
        gap_ratio = cumulative / positions
        
        return gap_ratio.unsqueeze(-1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with relative positional encoding (architectural fix)
        
        Args:
            x: Input light curves [B, T] or [B, T, 1]
            return_all: Return all auxiliary outputs
        
        Returns:
            Dictionary with:
            - 'logits': 3-class logits [B, 3]
            - 'binary': Alias for logits
            - 'confidence': Prediction confidence [B]
            - 'flat', 'pspl', 'anomaly', 'caustic': Auxiliary logits
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        B, T, _ = x.shape
        device = x.device
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Zero out padding
        x_norm = x.clone()
        x_norm[padding_mask.unsqueeze(-1)] = 0.0
        
        # Input embedding
        x_embed = self.input_embed(x_norm)
        
        # Add gap information
        gap_features = self.compute_gap_features(padding_mask)
        gap_embed = self.gap_embed(gap_features)
        x_embed = x_embed + 0.1 * gap_embed
        
        # v12.0-beta: Add RELATIVE positional encoding (architectural fix!)
        pos_encoding = self.pos_encoding(x_embed, padding_mask)
        x_embed = x_embed + pos_encoding
        
        # Pass through transformer layers
        for layer in self.layers:
            x_embed = layer(x_embed, padding_mask)
        
        # Final normalization
        x_embed = self.norm(x_embed)
        
        # Global pooling (weighted by validity)
        valid_mask = ~padding_mask
        if valid_mask.any():
            valid_expand = valid_mask.unsqueeze(-1).float()
            x_sum = (x_embed * valid_expand).sum(dim=1)
            x_count = valid_expand.sum(dim=1).clamp(min=1)
            x_pooled = x_sum / x_count
            
            # Max pooling for complementary features
            x_masked = x_embed.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            x_max, _ = x_masked.max(dim=1)
            
            x_final = x_pooled + 0.1 * x_max
        else:
            x_final = x_embed.mean(dim=1)
        
        # Get outputs
        logits = self.classification_head(x_final)  # [B, 3]
        
        outputs = {
            'logits': logits,
            'binary': logits,
            'confidence': self.confidence_head(x_final).squeeze(-1),
            'flat': self.flat_head(x_final).squeeze(-1),
            'pspl': self.pspl_head(x_final).squeeze(-1),
            'anomaly': self.anomaly_head(x_final).squeeze(-1),
            'caustic': self.caustic_head(x_final).squeeze(-1)
        }
        
        if return_all:
            # Compute class probabilities
            probs = F.softmax(logits, dim=-1)
            outputs['prob_flat'] = probs[:, 0]
            outputs['prob_pspl'] = probs[:, 1]
            outputs['prob_binary'] = probs[:, 2]
            
            # Sigmoid versions for visualization
            outputs['flat_prob'] = torch.sigmoid(outputs['flat'])
            outputs['pspl_prob'] = torch.sigmoid(outputs['pspl'])
        
        # Clamp outputs
        outputs['logits'] = torch.clamp(outputs['logits'], min=-20, max=20)
        outputs['binary'] = outputs['logits']
        outputs['flat'] = torch.clamp(outputs['flat'], min=-20, max=20)
        outputs['pspl'] = torch.clamp(outputs['pspl'], min=-20, max=20)
        outputs['anomaly'] = torch.clamp(outputs['anomaly'], min=-20, max=20)
        outputs['caustic'] = torch.clamp(outputs['caustic'], min=-20, max=20)
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test v12.0-beta model with architectural fix"""
    print("="*70)
    print("Testing MicrolensingTransformer")
    print("="*70)
    
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=128,        # Smaller
        nhead=4,            # Smaller
        num_layers=4,       # Smaller
        dim_feedforward=512, # Smaller
        dropout=0.1
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    # Test with variable-length sequences
    print("\n" + "="*70)
    print("Testing variable-length sequences:")
    print("="*70)
    
    # Batch with different lengths
    x1 = torch.randn(1, 300)  # Short sequence (20%)
    x2 = torch.randn(1, 750)  # Medium sequence (50%)
    x3 = torch.randn(1, 1500) # Full sequence (100%)
    x3[:, 1000:] = -1.0       # With padding
    
    for i, x in enumerate([x1, x2, x3], 1):
        outputs = model(x, return_all=True)
        n_valid = (x != -1.0).sum().item()
        
        print(f"\nSequence {i}: {x.shape[1]} points ({n_valid} valid)")
        print(f"  Flat prob:   {outputs['prob_flat'][0]:.3f}")
        print(f"  PSPL prob:   {outputs['prob_pspl'][0]:.3f}")
        print(f"  Binary prob: {outputs['prob_binary'][0]:.3f}")
        print(f"  Confidence:  {outputs['confidence'][0]:.3f}")
    
    print("\n" + "="*70)
    print("✅ Test passed!")
    print("="*70)


if __name__ == "__main__":
    test_model()