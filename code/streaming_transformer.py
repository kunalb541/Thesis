#!/usr/bin/env python3
"""
Streaming Transformer for Real-Time Binary Microlensing Detection

Features:
- Causal self-attention (no future leakage)
- Multi-head output (binary, anomaly, caustic)
- Sliding window attention
- Per-timestep predictions

Author: Kunal Bhatia
Version: 6.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import config as CFG


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with sliding window support.
    Ensures strictly causal predictions for real-time processing.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.window_size = window_size
        
        # Multi-head projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Register causal mask buffer
        self.register_buffer("causal_mask", None)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create strictly causal mask (upper triangular)"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Apply sliding window if specified
        if self.window_size is not None and self.window_size < seq_len:
            for i in range(seq_len):
                start = max(0, i - self.window_size + 1)
                if start > 0:
                    mask[i, :start] = float('-inf')
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            key_padding_mask: [batch, seq] binary mask (1 = padded)
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.W_k(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]
        
        # Apply causal mask
        if self.causal_mask is None or self.causal_mask.shape[0] != T:
            self.causal_mask = self.create_causal_mask(T, x.device)
        
        scores = scores + self.causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, T, d_k]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # Output projection
        out = self.W_o(out)
        
        return out


class StreamingTransformerBlock(nn.Module):
    """
    Transformer block with causal attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = CausalSelfAttention(
            d_model, nhead, window_size, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), key_padding_mask)
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class StreamingTransformer(nn.Module):
    """
    Streaming Transformer for real-time binary microlensing detection.
    
    Features:
    - NO downsampling (processes at full temporal resolution)
    - Causal attention (strictly no future information)
    - Multi-head outputs (binary, anomaly, caustic)
    - Per-timestep predictions for streaming
    """
    
    def __init__(
        self,
        n_points: int = CFG.N_POINTS,
        d_model: int = CFG.D_MODEL,
        nhead: int = CFG.NHEAD,
        num_layers: int = CFG.NUM_LAYERS,
        dim_feedforward: int = CFG.DIM_FEEDFORWARD,
        window_size: int = CFG.WINDOW_SIZE,
        dropout: float = CFG.DROPOUT,
        use_multi_head: bool = CFG.USE_MULTI_HEAD
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.use_multi_head = use_multi_head
        
        # Input projection (directly from time series)
        self.input_projection = nn.Linear(1, d_model)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, n_points, d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            StreamingTransformerBlock(
                d_model, nhead, dim_feedforward, window_size, dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.binary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Binary classification
        )
        
        if use_multi_head:
            self.anomaly_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)  # Anomaly score
            )
            
            self.caustic_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)  # Caustic probability
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_timesteps: bool = True,
        pad_value: float = -1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through streaming transformer.
        
        Args:
            x: Input tensor [batch, seq_len] or [batch, 1, seq_len]
            return_all_timesteps: Return predictions for all timesteps
            pad_value: Value used for padding
            
        Returns:
            Dictionary with:
                - 'binary': [batch, seq, 2] or [batch, 2]
                - 'anomaly': [batch, seq, 1] or [batch, 1] (if multi-head)
                - 'caustic': [batch, seq, 1] or [batch, 1] (if multi-head)
        """
        # Handle input shape
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [batch, 1, seq] -> [batch, seq]
        
        B, T = x.shape
        device = x.device
        
        # Create padding mask
        padding_mask = (x == pad_value)
        
        # Project input to d_model
        x_embed = self.input_projection(x.unsqueeze(-1))  # [B, T, d_model]
        
        # Add positional embeddings
        x_embed = x_embed + self.pos_embedding[:, :T, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x_embed = block(x_embed, padding_mask)
        
        # Apply output heads
        outputs = {}
        
        # Binary classification (always present)
        binary_logits = self.binary_head(x_embed)  # [B, T, 2]
        
        if self.use_multi_head:
            anomaly_scores = self.anomaly_head(x_embed)  # [B, T, 1]
            caustic_probs = torch.sigmoid(self.caustic_head(x_embed))  # [B, T, 1]
            
            outputs['anomaly'] = anomaly_scores
            outputs['caustic'] = caustic_probs
        
        if return_all_timesteps:
            outputs['binary'] = binary_logits
        else:
            # Use only the last valid (non-padded) prediction
            last_valid = (~padding_mask).sum(dim=1) - 1  # [B]
            last_valid = last_valid.clamp(min=0, max=T-1)
            
            batch_idx = torch.arange(B, device=device)
            final_logits = binary_logits[batch_idx, last_valid]  # [B, 2]
            outputs['binary'] = final_logits
            
            if self.use_multi_head:
                outputs['anomaly'] = outputs['anomaly'][batch_idx, last_valid]
                outputs['caustic'] = outputs['caustic'][batch_idx, last_valid]
        
        return outputs
    
    def streaming_forward(
        self,
        x_buffer: torch.Tensor,
        timestep: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process single timestep for streaming inference.
        
        Args:
            x_buffer: Circular buffer of past observations [1, buffer_size]
            timestep: Current timestep index
            
        Returns:
            Predictions for current timestep
        """
        # Use only observations up to current timestep
        x_current = x_buffer[:, :timestep+1]
        
        # Get predictions
        outputs = self.forward(x_current, return_all_timesteps=True)
        
        # Return only the current timestep predictions
        return {
            'binary': outputs['binary'][:, timestep],
            'anomaly': outputs.get('anomaly', [None])[:, timestep] if 'anomaly' in outputs else None,
            'caustic': outputs.get('caustic', [None])[:, timestep] if 'caustic' in outputs else None
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*60)
    print("STREAMING TRANSFORMER TEST")
    print("="*60)
    
    # Test model creation
    model = StreamingTransformer()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, seq_len)
    
    # Add some padding
    x[:, -50:] = -1.0
    
    # Test streaming mode
    outputs = model(x, return_all_timesteps=True)
    print(f"\nStreaming output shapes:")
    for key, val in outputs.items():
        if val is not None:
            print(f"  {key}: {val.shape}")
    
    # Test final prediction mode
    outputs_final = model(x, return_all_timesteps=False)
    print(f"\nFinal output shapes:")
    for key, val in outputs_final.items():
        if val is not None:
            print(f"  {key}: {val.shape}")
    
    # Test causality
    print("\nTesting causality...")
    x1 = torch.randn(1, 100)
    x2 = x1.clone()
    x2[0, 50:] = torch.randn(50)  # Change future
    
    out1 = model(x1, return_all_timesteps=True)['binary']
    out2 = model(x2, return_all_timesteps=True)['binary']
    
    # First 50 timesteps should be identical
    first_half_same = torch.allclose(out1[:, :50], out2[:, :50], atol=1e-5)
    print(f"  First half identical: {first_half_same}")
    
    if first_half_same:
        print("✅ Causal attention verified!")
    else:
        print("❌ Causality violation detected!")
    
    print("\n✅ Model test complete!")
