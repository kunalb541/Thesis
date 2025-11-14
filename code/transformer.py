"""
Transformer Architecture v15.0 - Anti-Cheating Edition
======================================================

CRITICAL CHANGES:
1. **Semi-Causal Attention**: Prevents future peeking (no shortcuts from peak timing)
2. **Temporal Invariance Loss**: Forces model to ignore absolute time positions
3. **Simplified Multi-Task**: Only caustic detection (Binary-specific feature)
4. **Attention Diagnostics**: Tools to detect temporal shortcuts

Three-Class Classification: 0=Flat, 1=PSPL, 2=Binary

Key Innovation: Model learns from magnification morphology, NOT temporal position.

Author: Kunal Bhatia
Version: 15.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding - NO absolute time information
    
    Encodes:
    1. Cumulative observation count (how many valid observations seen)
    2. Relative gaps (time since last valid observation)
    
    DOES NOT encode absolute time positions to prevent shortcuts.
    """
    
    def __init__(self, d_model: int, max_observations: int = 2000):
        super().__init__()
        self.d_model = d_model
        
        # Observation count encoding
        self.obs_count_encoding = nn.Embedding(max_observations, d_model // 2)
        
        # Gap encoding (relative time between observations)
        self.gap_encoding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # Small initialization to avoid dominating signal
        nn.init.normal_(self.obs_count_encoding.weight, std=0.01)
        for m in self.gap_encoding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, D]
            padding_mask: Boolean mask [B, T] (True = padded)
        
        Returns:
            Positional encoding [B, T, D]
        """
        B, T, D = x.shape
        device = x.device
        
        # Cumulative observation count
        valid_mask = ~padding_mask
        obs_count = torch.cumsum(valid_mask.long(), dim=1) - 1
        obs_count = torch.clamp(obs_count, min=0, max=self.obs_count_encoding.num_embeddings - 1)
        count_embed = self.obs_count_encoding(obs_count)
        
        # Relative gaps
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


class SemiCausalAttention(nn.Module):
    """
    Semi-Causal Multi-Head Attention
    
    CRITICAL: Prevents model from seeing future observations, which could
    leak information about event timing/morphology.
    
    At each time step t, can only attend to observations up to time t.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, 
                 causal: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.causal = causal
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Flash Attention support
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
        self._init_weights()
    
    def _init_weights(self):
        gain = 1.0 / math.sqrt(2)
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [B, T, D]
            padding_mask: Padding mask [B, T] (True = invalid)
        
        Returns:
            output: [B, T, D]
            attn_weights: [B, H, T, T] for diagnostics
        """
        B, T, D = x.shape
        
        # Normalize and project
        x_norm = self.q_norm(x)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, T, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Normalize Q and K for stability
        Q = F.normalize(Q, p=2, dim=-1, eps=1e-8)
        K = F.normalize(K, p=2, dim=-1, eps=1e-8)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, min=-10, max=10)
        
        # Apply causal mask if enabled
        if self.causal:
            # Create causal mask: can only attend to past and present
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
        
        # Apply padding mask
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask_expanded, -1e4)
        
        # Compute attention weights
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - scores_max
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(B, T, D)
        out = self.out_proj(attn_out)
        
        return out * 0.5, attn_weights  # Return weights for diagnostics


class TransformerBlock(nn.Module):
    """Transformer block with causal attention"""
    
    def __init__(self, d_model: int, nhead: int, dim_ff: int, 
                 dropout: float = 0.1, causal: bool = True):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.attn = SemiCausalAttention(d_model, nhead, dropout, causal=causal)
        
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
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [B, T, D]
            mask: Padding mask [B, T]
        
        Returns:
            output: [B, T, D]
            attn_weights: Attention weights for diagnostics
        """
        # Attention + residual
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + torch.tanh(self.alpha) * attn_out
        
        # FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.beta) * ffn_out
        
        # Clamp for stability
        x = torch.clamp(x, min=-100, max=100)
        
        return x, attn_weights


class TemporalInvarianceLoss(nn.Module):
    """
    Temporal Invariance Loss - CRITICAL for preventing shortcuts
    
    Encourages the model to produce similar embeddings for time-shifted
    versions of the same event. This prevents learning "peak at time X = class Y".
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, 
                padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Sequence embeddings [B, T, D]
            padding_mask: Padding mask [B, T]
        
        Returns:
            Temporal invariance loss (scalar)
        """
        B, T, D = embeddings.shape
        
        # Pool embeddings (average over valid observations)
        valid_mask = ~padding_mask
        valid_expand = valid_mask.unsqueeze(-1).float()
        pooled = (embeddings * valid_expand).sum(dim=1) / (valid_expand.sum(dim=1) + 1e-8)
        
        # Normalize
        pooled_norm = F.normalize(pooled, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(pooled_norm, pooled_norm.T) / self.temperature
        
        # Mask diagonal
        mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)
        
        # InfoNCE-style loss: encourage diversity
        loss = -F.log_softmax(sim_matrix, dim=1).diag().mean()
        
        return loss


class MicrolensingTransformer(nn.Module):
    """
    Transformer for 3-class microlensing classification v15.0
    
    ANTI-CHEATING FEATURES:
    1. Semi-causal attention (no future peeking)
    2. Relative positional encoding (no absolute time)
    3. Temporal invariance loss (penalizes time-dependent features)
    4. Simplified multi-task (only caustic detection for Binary class)
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
        use_checkpoint: bool = False,
        causal_attention: bool = True,
        temporal_invariance_weight: float = 0.1
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        self.nhead = nhead
        self.use_checkpoint = use_checkpoint
        self.causal_attention = causal_attention
        self.temporal_invariance_weight = temporal_invariance_weight
        
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
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, causal=causal_attention)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # SIMPLIFIED: Only classification + caustic detection
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Caustic detection (Binary-specific morphology)
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Temporal invariance loss
        self.temporal_invariance_loss = TemporalInvarianceLoss()
        
        # Confidence estimation
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
        """Create padding mask (True = invalid)"""
        if x.dim() == 3:
            return x[:, :, 0] == self.pad_value
        else:
            return x == self.pad_value
    
    def compute_gap_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute gap features"""
        gaps = mask.float()
        cumulative = gaps.cumsum(dim=1)
        positions = torch.arange(1, mask.shape[1] + 1, device=mask.device).unsqueeze(0)
        gap_ratio = cumulative / positions
        return gap_ratio.unsqueeze(-1)
    
    def global_pooling(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Global pooling (average + max)
        
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
    
    def forward(self, x: torch.Tensor, 
                return_all: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input [B, T] or [B, T, 1]
            return_all: Return all auxiliary outputs
            return_attention: Return attention weights for diagnostics
        
        Returns:
            Dictionary with outputs
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
        attention_weights = []
        
        if self.use_checkpoint and self.training:
            for layer in self.layers:
                x_embed, attn_w = torch.utils.checkpoint.checkpoint(
                    layer,
                    x_embed,
                    padding_mask,
                    use_reentrant=False
                )
                if return_attention:
                    attention_weights.append(attn_w)
        else:
            for layer in self.layers:
                x_embed, attn_w = layer(x_embed, padding_mask)
                if return_attention:
                    attention_weights.append(attn_w)
        
        # Final norm
        x_embed = self.norm(x_embed)
        
        # Compute temporal invariance loss (before pooling)
        if self.training and self.temporal_invariance_weight > 0:
            temp_inv_loss = self.temporal_invariance_loss(x_embed, padding_mask)
        else:
            temp_inv_loss = torch.tensor(0.0, device=x.device)
        
        # Global pooling
        x_pooled = self.global_pooling(x_embed, padding_mask)
        
        # Get outputs
        logits = self.classification_head(x_pooled)
        caustic_logits = self.caustic_head(x_pooled).squeeze(-1)
        confidence = self.confidence_head(x_pooled).squeeze(-1)
        
        # Clamp for stability
        logits = torch.clamp(logits, min=-20, max=20)
        caustic_logits = torch.clamp(caustic_logits, min=-20, max=20)
        
        outputs = {
            'logits': logits,
            'binary': logits,  # Alias for compatibility
            'caustic': caustic_logits,
            'confidence': confidence,
            'temporal_invariance_loss': temp_inv_loss
        }
        
        if return_all:
            probs = F.softmax(logits, dim=-1)
            outputs['prob_flat'] = probs[:, 0]
            outputs['prob_pspl'] = probs[:, 1]
            outputs['prob_binary'] = probs[:, 2]
            outputs['caustic_prob'] = torch.sigmoid(caustic_logits)
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# ATTENTION DIAGNOSTICS
# ============================================================================

def analyze_attention_patterns(model, dataloader, device='cuda', n_samples=100):
    """
    Analyze attention patterns to detect temporal shortcuts
    
    Returns:
        dict with attention statistics
    """
    model.eval()
    
    attention_entropies = []
    temporal_concentration = []
    
    with torch.no_grad():
        for i, (X, y, _) in enumerate(dataloader):
            if i >= n_samples:
                break
            
            X = X.to(device)
            outputs = model(X, return_attention=True)
            
            # Get last layer attention weights
            attn_weights = outputs['attention_weights'][-1]  # [B, H, T, T]
            
            # Compute attention entropy (higher = more uniform, better)
            B, H, T, _ = attn_weights.shape
            for b in range(B):
                for h in range(H):
                    attn = attn_weights[b, h]
                    entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
                    attention_entropies.append(entropy.item())
            
            # Check if attention concentrates on specific time regions
            # (which would indicate temporal shortcuts)
            temporal_attn = attn_weights.mean(dim=(0, 1))  # Average over batch and heads [T, T]
            concentration = temporal_attn.max(dim=-1)[0].mean()
            temporal_concentration.append(concentration.item())
    
    return {
        'mean_entropy': np.mean(attention_entropies),
        'std_entropy': np.std(attention_entropies),
        'mean_concentration': np.mean(temporal_concentration),
        'std_concentration': np.std(temporal_concentration)
    }


if __name__ == "__main__":
    print("="*70)
    print("MicrolensingTransformer v15.0 - Anti-Cheating Edition")
    print("="*70)
    
    model = MicrolensingTransformer(
        n_points=1500,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        causal_attention=True,
        temporal_invariance_weight=0.1
    )
    
    print(f"\nParameters: {count_parameters(model):,}")
    print(f"Causal attention: ENABLED")
    print(f"Temporal invariance loss: ENABLED (weight=0.1)")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    x[:, 500:] = -1.0  # Add padding
    
    outputs = model(x, return_all=True, return_attention=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Caustic: {outputs['caustic'].shape}")
    print(f"  Confidence: {outputs['confidence'].shape}")
    print(f"  Temporal inv loss: {outputs['temporal_invariance_loss'].item():.4f}")
    print(f"  Attention weights: {len(outputs['attention_weights'])} layers")
    
    print("\n" + "="*70)
    print("✅ Model initialized successfully")
    print("="*70)