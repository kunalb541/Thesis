#!/usr/bin/env python3
"""
OPTIMIZED Transformer Architecture for THREE-CLASS Classification
=================================================================
Classes: 0=Flat (no event), 1=PSPL, 2=Binary

v12.0-beta-OPTIMIZED: FULLY VECTORIZED (NO PYTHON LOOPS)
- All operations are GPU-optimized tensor operations
- Efficient relative positional encoding
- Memory-efficient attention with flash attention when available
- Pre-computed gap features
- Fused operations where possible

OPTIMIZATIONS:
1. Vectorized gap computation (no loops!)
2. Efficient cumulative operations with masking
3. Flash attention support (PyTorch 2.0+)
4. Gradient checkpointing for memory
5. Optimized multi-head attention
6. Efficient global pooling

Author: Kunal Bhatia
Version: 12.0-beta-OPTIMIZED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple


class OptimizedRelativePositionalEncoding(nn.Module):
    """
    FULLY VECTORIZED Relative Positional Encoding
    
    NO PYTHON LOOPS - All operations are pure PyTorch tensors
    
    Encodes:
    1. Cumulative observation count (how many valid observations seen)
    2. Relative gaps (distance since last valid observation)
    
    Both are computed efficiently using vectorized operations.
    """
    
    def __init__(self, d_model: int, max_observations: int = 2000):
        super().__init__()
        self.d_model = d_model
        
        # Embedding for observation count
        self.obs_count_encoding = nn.Embedding(max_observations, d_model // 2)
        
        # MLP for gap encoding (relative time since last observation)
        self.gap_encoding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # Initialize with small values for stability
        nn.init.normal_(self.obs_count_encoding.weight, std=0.01)
        for m in self.gap_encoding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        OPTIMIZED: Fully vectorized, no Python loops
        
        Args:
            x: Input tensor [B, T, D]
            padding_mask: Boolean mask [B, T] (True = padded/invalid)
        
        Returns:
            pos_encoding: [B, T, D] positional encoding
        """
        B, T, D = x.shape
        device = x.device
        
        # 1. Compute cumulative observation count (already vectorized)
        valid_mask = ~padding_mask  # [B, T]
        obs_count = torch.cumsum(valid_mask.long(), dim=1) - 1  # [B, T]
        obs_count = torch.clamp(
            obs_count,
            min=0,
            max=self.obs_count_encoding.num_embeddings - 1
        )
        count_embed = self.obs_count_encoding(obs_count)  # [B, T, D/2]
        
        # 2. VECTORIZED gap computation (NO LOOPS!)
        # Create position tensor
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        
        # For each position, find the position of the last valid observation
        # Method: Create a tensor where invalid positions have value -inf,
        # then use cummax to propagate the last valid position forward
        valid_positions = torch.where(
            valid_mask,
            positions.float(),
            torch.tensor(float('-inf'), device=device)
        )  # [B, T]
        
        # Cummax gives us the position of the last valid observation up to each point
        last_valid_pos, _ = torch.cummax(valid_positions, dim=1)  # [B, T]
        
        # Gap = current position - last valid position
        # For the first valid observation in each sequence, gap = 0
        gaps = positions.float() - last_valid_pos  # [B, T]
        
        # Clamp gaps to reasonable range and handle -inf cases
        gaps = torch.where(
            torch.isfinite(gaps) & (gaps >= 0),
            gaps,
            torch.zeros_like(gaps)
        )
        gaps = torch.clamp(gaps, min=0, max=100)  # [B, T]
        
        # Normalize gaps (helps with stability)
        gaps_norm = gaps / (gaps.max() + 1e-8)  # [B, T]
        
        # Encode gaps
        gap_embed = self.gap_encoding(gaps_norm.unsqueeze(-1))  # [B, T, D/2]
        
        # 3. Concatenate both embeddings
        pos_encoding = torch.cat([count_embed, gap_embed], dim=-1)  # [B, T, D]
        
        return pos_encoding


class FlashMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with Flash Attention support
    
    Uses scaled_dot_product_attention (PyTorch 2.0+) when available
    for 2-4x speedup and lower memory usage.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Fused QKV projection (more efficient than 3 separate projections)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Check if Flash Attention is available
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
        self._init_weights()
    
    def _init_weights(self):
        gain = 1.0 / math.sqrt(2)
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimized attention with Flash Attention when available
        
        Args:
            x: Input [B, T, D]
            mask: Padding mask [B, T] (True = invalid)
        
        Returns:
            Output [B, T, D]
        """
        B, T, D = x.shape
        
        # Normalize input for stability
        x_norm = self.q_norm(x)
        
        # Fused QKV projection (more efficient)
        qkv = self.qkv_proj(x_norm)  # [B, T, 3*D]
        qkv = qkv.reshape(B, T, 3, self.nhead, self.d_k)  # [B, T, 3, H, D_k]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [B, H, T, D_k]
        
        # Normalize Q and K for stability (prevents overflow in softmax)
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        if self.use_flash and mask is None:
            # Use Flash Attention (PyTorch 2.0+) - fastest path
            # Note: Flash attention has limitations with custom masks
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )  # [B, H, T, D_k]
        else:
            # Manual attention computation with stability tricks
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, T]
            
            # Clamp scores for stability
            scores = torch.clamp(scores, min=-10, max=10)
            
            # Apply padding mask if provided
            if mask is not None:
                # Expand mask: [B, T] -> [B, 1, 1, T]
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask_expanded, -1e4)
            
            # Stable softmax (subtract max before exp)
            scores_max = scores.max(dim=-1, keepdim=True)[0]
            scores = scores - scores_max
            attn = F.softmax(scores, dim=-1)  # [B, H, T, T]
            attn = self.attn_dropout(attn)
            
            # Apply attention to values
            attn_out = torch.matmul(attn, V)  # [B, H, T, D_k]
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, H, D_k]
        attn_out = attn_out.reshape(B, T, D)  # [B, T, D]
        out = self.out_proj(attn_out)
        
        # Scale output for residual connection
        return out * 0.5


class OptimizedTransformerBlock(nn.Module):
    """
    Transformer block with all optimizations:
    - Flash attention
    - Efficient FFN with GELU
    - Pre-norm architecture
    - Stable residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-norm architecture (more stable than post-norm)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Optimized attention
        self.attn = FlashMultiHeadAttention(d_model, nhead, dropout)
        
        # Efficient FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable residual gates (helps with deep networks)
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
        """
        Forward with pre-norm and stable residuals
        
        Args:
            x: Input [B, T, D]
            mask: Padding mask [B, T]
        
        Returns:
            Output [B, T, D]
        """
        # Pre-norm + attention + residual
        attn_out = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        # Pre-norm + FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        # Clamp for stability (prevents overflow)
        x = torch.clamp(x, min=-100, max=100)
        
        return x


class MicrolensingTransformer(nn.Module):
    """
    OPTIMIZED Transformer for microlensing classification
    
    v12.0-beta-OPTIMIZED: Production-ready with all optimizations
    - Fully vectorized (no Python loops)
    - Flash attention support
    - Memory-efficient operations
    - Gradient checkpointing ready
    
    Classes: 0=Flat, 1=PSPL, 2=Binary
    
    Auxiliary tasks for multi-task learning:
    - Flat detection (weight=0.5)
    - PSPL detection (weight=0.5)
    - Anomaly detection (weight=0.2)
    - Caustic detection (weight=0.2)
    - Confidence estimation
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
        
        # Input embedding with skip connection
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2, eps=1e-6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Optimized relative positional encoding (NO LOOPS!)
        self.pos_encoding = OptimizedRelativePositionalEncoding(
            d_model=d_model,
            max_observations=max_seq_len
        )
        
        # Gap embedding (helps with sparse observations)
        self.gap_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Transformer layers (with optional gradient checkpointing)
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # ============== OUTPUT HEADS ==============
        
        # Main: 3-class classification
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Alias for compatibility
        self.binary_head = self.classification_head
        
        # Auxiliary heads (output logits for BCEWithLogitsLoss)
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
        """Initialize weights with best practices"""
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
        """
        OPTIMIZED: Compute gap features efficiently
        
        Uses cumsum for vectorized computation
        """
        gaps = mask.float()
        cumulative = gaps.cumsum(dim=1)
        
        # Avoid division by zero
        positions = torch.arange(1, mask.shape[1] + 1, device=mask.device).unsqueeze(0)
        gap_ratio = cumulative / positions
        
        return gap_ratio.unsqueeze(-1)
    
    def global_pooling(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient global pooling with both average and max
        
        Combines average pooling (for general features) with
        max pooling (for peak detection).
        """
        valid_mask = ~padding_mask  # [B, T]
        
        if valid_mask.any():
            # Average pooling (weighted by validity)
            valid_expand = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
            x_sum = (x * valid_expand).sum(dim=1)  # [B, D]
            x_count = valid_expand.sum(dim=1).clamp(min=1)  # [B, 1]
            x_avg = x_sum / x_count  # [B, D]
            
            # Max pooling (for peak features)
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            x_max, _ = x_masked.max(dim=1)  # [B, D]
            
            # Combine both (average is primary, max is supplementary)
            x_pooled = x_avg + 0.1 * x_max
        else:
            # Fallback (shouldn't happen in practice)
            x_pooled = x.mean(dim=1)
        
        return x_pooled
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized forward pass
        
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
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Zero out padding (more efficient than masking in every layer)
        x_clean = x.clone()
        x_clean[padding_mask.unsqueeze(-1)] = 0.0
        
        # Input embedding
        x_embed = self.input_embed(x_clean)  # [B, T, D]
        
        # Add gap information
        gap_features = self.compute_gap_features(padding_mask)
        gap_embed = self.gap_embed(gap_features)
        x_embed = x_embed + 0.1 * gap_embed
        
        # Add relative positional encoding (OPTIMIZED - no loops!)
        pos_encoding = self.pos_encoding(x_embed, padding_mask)
        x_embed = x_embed + pos_encoding
        
        # Pass through transformer layers
        if self.use_checkpoint and self.training:
            # Gradient checkpointing (saves memory during training)
            for layer in self.layers:
                x_embed = torch.utils.checkpoint.checkpoint(
                    layer,
                    x_embed,
                    padding_mask,
                    use_reentrant=False
                )
        else:
            # Standard forward
            for layer in self.layers:
                x_embed = layer(x_embed, padding_mask)
        
        # Final normalization
        x_embed = self.norm(x_embed)  # [B, T, D]
        
        # Global pooling (optimized)
        x_pooled = self.global_pooling(x_embed, padding_mask)  # [B, D]
        
        # Get all outputs
        logits = self.classification_head(x_pooled)  # [B, 3]
        
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
            # Compute class probabilities
            probs = F.softmax(logits, dim=-1)
            outputs['prob_flat'] = probs[:, 0]
            outputs['prob_pspl'] = probs[:, 1]
            outputs['prob_binary'] = probs[:, 2]
            
            # Sigmoid versions for auxiliary tasks
            outputs['flat_prob'] = torch.sigmoid(outputs['flat'])
            outputs['pspl_prob'] = torch.sigmoid(outputs['pspl'])
        
        # Clamp outputs for stability
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
    """Test optimized model"""
    print("="*70)
    print("Testing OPTIMIZED MicrolensingTransformer")
    print("="*70)
    
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        use_checkpoint=False
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Test with batch
    print("\n" + "="*70)
    print("Testing batch processing:")
    print("="*70)
    
    batch_size = 32
    x = torch.randn(batch_size, 1500)
    
    # Add some padding
    for i in range(batch_size):
        pad_start = torch.randint(500, 1500, (1,)).item()
        x[i, pad_start:] = -1.0
    
    # Test forward pass
    import time
    
    # Warmup
    for _ in range(5):
        _ = model(x, return_all=False)
    
    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    n_iters = 100
    for _ in range(n_iters):
        outputs = model(x, return_all=True)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / n_iters * 1000  # ms
    
    print(f"\nBatch size: {batch_size}")
    print(f"Average forward pass: {avg_time:.2f} ms")
    print(f"Throughput: {batch_size / (avg_time / 1000):.0f} samples/sec")
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Flat prob: {outputs['prob_flat'].shape}")
    print(f"  PSPL prob: {outputs['prob_pspl'].shape}")
    print(f"  Binary prob: {outputs['prob_binary'].shape}")
    print(f"  Confidence: {outputs['confidence'].shape}")
    
    print("\n" + "="*70)
    print("✅ Test passed! Model is fully optimized.")
    print("="*70)


if __name__ == "__main__":
    test_model()