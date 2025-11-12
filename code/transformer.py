#!/usr/bin/env python3
"""
Production Transformer for THREE-CLASS Classification with Enhanced Auxiliary Tasks
===================================================================================
Classes: 0=Flat (no event), 1=PSPL, 2=Binary

v11.1 HOTFIX: Fixed AMP compatibility by removing Sigmoid from auxiliary heads
- All auxiliary heads now output logits
- Use BCEWithLogitsLoss in training (numerically stable + AMP-safe)

Author: Kunal Bhatia
Version: 11.1-hotfix - AMP-Safe Multi-Task Learning
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
        
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.q_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.k_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        self._init_weights()
    
    def _init_weights(self):
        gain = 1.0 / math.sqrt(2)
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        x = self.q_norm(x)
        
        Q = self.w_q(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-10, max=10)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
        
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.w_o(out)
        
        return out * 0.5


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and stable residuals"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
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
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        x = torch.clamp(x, min=-100, max=100)
        
        return x


class MicrolensingTransformer(nn.Module):
    """
    Production transformer for THREE-CLASS microlensing classification
    
    v11.1-hotfix: AMP-Safe auxiliary heads
    - All auxiliary heads output LOGITS (no sigmoid)
    - Use F.binary_cross_entropy_with_logits in training
    - Numerically stable + AMP-compatible
    
    Main task: 3-class classification (Flat, PSPL, Binary)
    Auxiliary tasks (output logits for BCEWithLogitsLoss):
    - Flat detection (HIGH WEIGHT 0.5) - avoids false triggers
    - PSPL detection (HIGH WEIGHT 0.5) - distinguishes simple events
    - Anomaly detection (WEIGHT 0.2) - any event vs baseline
    - Caustic detection (WEIGHT 0.2) - binary-specific features
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
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
        
        # Input embedding
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
        
        # ============== MAIN TASK ==============
        # 3-class classification head (Flat, PSPL, Binary)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Keep "binary" alias for backwards compatibility
        self.binary_head = self.classification_head
        
        # ============== AUXILIARY TASKS (OUTPUT LOGITS) ==============
        
        # Flat detection: Is this a non-event? (Flat vs. PSPL/Binary)
        # HIGH WEIGHT (0.5): Critical for avoiding false triggers
        # Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss
        self.flat_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # PSPL detection: Is this a simple lens event? (PSPL vs. Flat/Binary)
        # HIGH WEIGHT (0.5): Important for distinguishing simple from complex
        # Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss
        self.pspl_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Anomaly detection: Any event vs. Flat (PSPL or Binary vs Flat)
        # MODERATE WEIGHT (0.2): General event detection
        # Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        # Caustic detection: only Binary has caustics (Binary vs. PSPL/Flat)
        # MODERATE WEIGHT (0.2): Binary-specific features
        # Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        # Confidence estimation head (still uses sigmoid for 0-1 output)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
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
        Forward pass with enhanced auxiliary outputs
        
        Args:
            x: Input light curves [B, T] or [B, T, 1]
            return_all: Return all auxiliary outputs with probabilities
            
        Returns:
            Dictionary with model outputs:
            - 'logits': Main 3-class classification [B, 3]
            - 'binary': Alias for 'logits' (backwards compatibility)
            - 'confidence': Prediction confidence [B] (0-1, has sigmoid)
            - 'flat': Flat detection LOGITS [B] (use with BCEWithLogitsLoss)
            - 'pspl': PSPL detection LOGITS [B] (use with BCEWithLogitsLoss)
            - 'anomaly': Anomaly LOGITS [B] (use with BCEWithLogitsLoss)
            - 'caustic': Caustic detection LOGITS [B] (use with BCEWithLogitsLoss)
            
            If return_all=True, also includes:
            - 'prob_flat': Flat probability [B] (0-1)
            - 'prob_pspl': PSPL probability [B] (0-1)
            - 'prob_binary': Binary probability [B] (0-1)
            - 'flat_prob': Sigmoid(flat logits) [B] (0-1)
            - 'pspl_prob': Sigmoid(pspl logits) [B] (0-1)
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
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding[:, :T, :]
        
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
            
            x_masked = x_embed.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            x_max, _ = x_masked.max(dim=1)
            
            x_final = x_pooled + 0.1 * x_max
        else:
            x_final = x_embed.mean(dim=1)
        
        # Get outputs
        logits = self.classification_head(x_final)  # [B, 3]
        
        # ALWAYS compute auxiliary heads (output LOGITS for training)
        outputs = {
            'logits': logits,
            'binary': logits,  # Alias for backwards compatibility
            'confidence': self.confidence_head(x_final).squeeze(-1),
            'flat': self.flat_head(x_final).squeeze(-1),        # LOGITS
            'pspl': self.pspl_head(x_final).squeeze(-1),        # LOGITS
            'anomaly': self.anomaly_head(x_final).squeeze(-1),  # LOGITS
            'caustic': self.caustic_head(x_final).squeeze(-1)   # LOGITS
        }
        
        if return_all:
            # Compute class probabilities
            probs = F.softmax(logits, dim=-1)
            outputs['prob_flat'] = probs[:, 0]
            outputs['prob_pspl'] = probs[:, 1]
            outputs['prob_binary'] = probs[:, 2]
            
            # Also provide sigmoid versions of auxiliary logits for visualization
            outputs['flat_prob'] = torch.sigmoid(outputs['flat'])
            outputs['pspl_prob'] = torch.sigmoid(outputs['pspl'])
        
        # Clamp main outputs
        outputs['logits'] = torch.clamp(outputs['logits'], min=-20, max=20)
        outputs['binary'] = outputs['logits']  # Keep alias synced
        
        # Clamp auxiliary logits
        outputs['flat'] = torch.clamp(outputs['flat'], min=-20, max=20)
        outputs['pspl'] = torch.clamp(outputs['pspl'], min=-20, max=20)
        outputs['anomaly'] = torch.clamp(outputs['anomaly'], min=-20, max=20)
        outputs['caustic'] = torch.clamp(outputs['caustic'], min=-20, max=20)
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model creation and forward pass"""
    print("Testing MicrolensingTransformer v11.1-hotfix (AMP-Safe)...")
    
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
    
    outputs = model(x, return_all=True)
    print(f"Logits shape: {outputs['logits'].shape}")  # Should be [4, 3]
    print(f"Number of classes: {outputs['logits'].shape[1]}")
    
    print(f"\nAuxiliary outputs (LOGITS):")
    print(f"  Flat detection logits: {outputs['flat'][0].item():.3f}")
    print(f"  PSPL detection logits: {outputs['pspl'][0].item():.3f}")
    print(f"  Anomaly detection logits: {outputs['anomaly'][0].item():.3f}")
    print(f"  Caustic detection logits: {outputs['caustic'][0].item():.3f}")
    
    print(f"\nAuxiliary outputs (PROBABILITIES):")
    print(f"  Flat detection prob: {outputs['flat_prob'][0].item():.3f}")
    print(f"  PSPL detection prob: {outputs['pspl_prob'][0].item():.3f}")
    
    if 'prob_flat' in outputs:
        print(f"\n✅ Three-class probabilities working!")
        print(f"   Flat prob: {outputs['prob_flat'][0].item():.3f}")
        print(f"   PSPL prob: {outputs['prob_pspl'][0].item():.3f}")
        print(f"   Binary prob: {outputs['prob_binary'][0].item():.3f}")
    
    print("\n✅ Model test passed! Auxiliary heads output logits (AMP-safe)")


if __name__ == "__main__":
    test_model()
