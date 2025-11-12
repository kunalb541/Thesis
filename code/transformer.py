"""
Transformer Architecture for Microlensing Classification

Three-Class Classification: 0=Flat (no event), 1=PSPL, 2=Binary

Key Innovation: Relative positional encoding prevents temporal information
leakage. Model learns from magnification patterns, not absolute time positions.

Architecture:
- Embedding: Input light curve → d_model dimensional space
- Positional Encoding: Relative (observation count + gaps), not absolute time
- Transformer Blocks: Multi-head attention + feed-forward networks
- Pooling: Global (average + max) over valid observations
- Heads: Classification (3-class) + Auxiliary tasks (5 outputs)

Auxiliary Tasks (Multi-Task Learning):
- Flat Detection: Binary classifier for non-events (weight=0.5)
- PSPL Detection: Binary classifier for single lens (weight=0.5)
- Anomaly Detection: General event detection (weight=0.2)
- Caustic Detection: Binary-specific features (weight=0.2)
- Confidence Estimation: Self-assessment of prediction quality

Author: Kunal Bhatia
Version: 13.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding - prevents temporal information leakage
    
    Encodes two types of information:
    1. Cumulative observation count (how many valid observations seen so far)
    2. Relative gaps (time since last valid observation)
    
    Unlike absolute positional encoding, this prevents the model from
    "cheating" by inferring event type from peak timing.
    """
    
    def __init__(self, d_model: int, max_observations: int = 2000):
        super().__init__()
        self.d_model = d_model
        
        # Embedding for observation count
        self.obs_count_encoding = nn.Embedding(max_observations, d_model // 2)
        
        # MLP for gap encoding
        self.gap_encoding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # Initialize with small values
        nn.init.normal_(self.obs_count_encoding.weight, std=0.01)
        for m in self.gap_encoding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
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
        obs_count = torch.cumsum(valid_mask.long(), dim=1) - 1
        obs_count = torch.clamp(obs_count, min=0, max=self.obs_count_encoding.num_embeddings - 1)
        count_embed = self.obs_count_encoding(obs_count)
        
        # Compute relative gaps
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        valid_positions = torch.where(
            valid_mask,
            positions.float(),
            torch.tensor(float('-inf'), device=device)
        )
        
        last_valid_pos, _ = torch.cummax(valid_positions, dim=1)
        gaps = positions.float() - last_valid_pos
        gaps = torch.where(
            torch.isfinite(gaps) & (gaps >= 0),
            gaps,
            torch.zeros_like(gaps)
        )
        gaps = torch.clamp(gaps, min=0, max=100)
        gaps_norm = gaps / (gaps.max() + 1e-8)
        gap_embed = self.gap_encoding(gaps_norm.unsqueeze(-1))
        
        # Concatenate
        pos_encoding = torch.cat([count_embed, gap_embed], dim=-1)
        
        return pos_encoding


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Flash Attention support (PyTorch 2.0+)
    
    Uses scaled_dot_product_attention when available for 2-4× speedup.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Check Flash Attention availability
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
        self._init_weights()
    
    def _init_weights(self):
        gain = 1.0 / math.sqrt(2)
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, D]
            mask: Padding mask [B, T] (True = invalid)
        
        Returns:
            Output [B, T, D]
        """
        B, T, D = x.shape
        
        # Normalize and project
        x_norm = self.q_norm(x)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, T, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Normalize Q and K for stability
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        if self.use_flash and mask is None:
            # Flash Attention path
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
        else:
            # Manual attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            scores = torch.clamp(scores, min=-10, max=10)
            
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask_expanded, -1e4)
            
            scores_max = scores.max(dim=-1, keepdim=True)[0]
            scores = scores - scores_max
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = torch.matmul(attn, V)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(B, T, D)
        out = self.out_proj(attn_out)
        
        return out * 0.5  # Scale for residual


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm architecture
    """
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable residual gates
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, D]
            mask: Padding mask [B, T]
        
        Returns:
            Output [B, T, D]
        """
        # Attention + residual
        attn_out = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        # FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        # Clamp for stability
        x = torch.clamp(x, min=-100, max=100)
        
        return x


class MicrolensingTransformer(nn.Module):
    """
    Transformer for three-class microlensing classification
    
    Classes: 0=Flat (no event), 1=PSPL, 2=Binary
    
    Key features:
    - Relative positional encoding (no temporal leakage)
    - Multi-task learning with auxiliary heads
    - Variable-length sequence support
    - Flash Attention when available
    """
    
    def __init__(
        self,
        n_points: int = 1500,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_value: float = -1.0,
        max_seq_len: int = 2000,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        self.nhead = nhead
        self.use_checkpoint = use_checkpoint
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2, eps=1e-6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        
        # Gap embedding
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
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Alias for compatibility
        self.binary_head = self.classification_head
        
        # Auxiliary heads
        self.flat_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.pspl_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 1)
        )
        
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True = invalid/padded)"""
        if x.dim() == 3:
            return x[:, :, 0] == self.pad_value
        else:
            return x == self.pad_value
    
    def compute_gap_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute gap features for sparse observations"""
        gaps = mask.float()
        cumulative = gaps.cumsum(dim=1)
        positions = torch.arange(1, mask.shape[1] + 1, device=mask.device).unsqueeze(0)
        gap_ratio = cumulative / positions
        return gap_ratio.unsqueeze(-1)
    
    def global_pooling(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Global pooling combining average and max
        
        Args:
            x: Features [B, T, D]
            padding_mask: Mask [B, T]
        
        Returns:
            Pooled features [B, D]
        """
        valid_mask = ~padding_mask
        
        if valid_mask.any():
            # Average pooling
            valid_expand = valid_mask.unsqueeze(-1).float()
            x_sum = (x * valid_expand).sum(dim=1)
            x_count = valid_expand.sum(dim=1).clamp(min=1)
            x_avg = x_sum / x_count
            
            # Max pooling
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            x_max, _ = x_masked.max(dim=1)
            
            # Combine
            x_pooled = x_avg + 0.1 * x_max
        else:
            x_pooled = x.mean(dim=1)
        
        return x_pooled
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input light curves [B, T] or [B, T, 1]
            return_all: Return all auxiliary outputs
        
        Returns:
            Dictionary with classification and auxiliary outputs
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        B, T, _ = x.shape
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Zero out padding
        x_clean = x.clone()
        x_clean[padding_mask.unsqueeze(-1)] = 0.0
        
        # Input embedding
        x_embed = self.input_embed(x_clean)
        
        # Add gap information
        gap_features = self.compute_gap_features(padding_mask)
        gap_embed = self.gap_embed(gap_features)
        x_embed = x_embed + 0.1 * gap_embed
        
        # Add relative positional encoding
        pos_encoding = self.pos_encoding(x_embed, padding_mask)
        x_embed = x_embed + pos_encoding
        
        # Transformer layers
        if self.use_checkpoint and self.training:
            for layer in self.layers:
                x_embed = torch.utils.checkpoint.checkpoint(
                    layer,
                    x_embed,
                    padding_mask,
                    use_reentrant=False
                )
        else:
            for layer in self.layers:
                x_embed = layer(x_embed, padding_mask)
        
        # Final norm
        x_embed = self.norm(x_embed)
        
        # Global pooling
        x_pooled = self.global_pooling(x_embed, padding_mask)
        
        # Get outputs
        logits = self.classification_head(x_pooled)
        
        outputs = {
            'logits': logits,
            'binary': logits,  # Alias
            'confidence': self.confidence_head(x_pooled).squeeze(-1),
            'flat': self.flat_head(x_pooled).squeeze(-1),
            'pspl': self.pspl_head(x_pooled).squeeze(-1),
            'anomaly': self.anomaly_head(x_pooled).squeeze(-1),
            'caustic': self.caustic_head(x_pooled).squeeze(-1)
        }
        
        if return_all:
            probs = F.softmax(logits, dim=-1)
            outputs['prob_flat'] = probs[:, 0]
            outputs['prob_pspl'] = probs[:, 1]
            outputs['prob_binary'] = probs[:, 2]
            outputs['flat_prob'] = torch.sigmoid(outputs['flat'])
            outputs['pspl_prob'] = torch.sigmoid(outputs['pspl'])
        
        # Clamp for stability
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


if __name__ == "__main__":
    print("="*70)
    print("MicrolensingTransformer")
    print("="*70)
    
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    print(f"\nParameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    x[:, 500:] = -1.0  # Add padding
    
    outputs = model(x, return_all=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Confidence: {outputs['confidence'].shape}")
    print(f"  Flat prob: {outputs['prob_flat'].shape}")
    print(f"  PSPL prob: {outputs['prob_pspl'].shape}")
    print(f"  Binary prob: {outputs['prob_binary'].shape}")
    
    print("\n" + "="*70)
    print("✅ Model initialized successfully")
    print("="*70)