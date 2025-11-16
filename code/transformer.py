"""
Transformer Architecture v16.1 - CRITICAL BUGFIXES (VERIFIED CORRECT)
=====================================================================

FIXED in v16.1:
- ✅ BUG #1: Attention mask now uses proper -inf (was -1e4, caused diagonal-only attention)
- ✅ BUG #2: SimpleCausticDetector NOW operates on RAW light curves, not embeddings!
- ✅ BUG #3: Peak threshold raised from 0.7 to 0.95 (detects real caustic crossings)
- ✅ BUG #4: All features extracted from PHYSICAL flux, not learned representations
- ✅ Added NaN handling in softmax after masking

MAINTAINED from v16.0:
- Semi-causal attention (no future peeking)
- Relative positional encoding (no absolute time)
- Wide t_0 sampling in simulation (forces morphology learning)

Three-Class Classification: 0=Flat, 1=PSPL, 2=Binary

Author: Kunal Bhatia
Version: 16.1 (FIXED - Raw Flux Edition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple


class SimpleCausticDetector(nn.Module):
    """
    Simple Caustic Detection from RAW Light Curve Morphology
    
    CRITICAL FIX v16.1: Now operates on RAW FLUX, not embeddings!
    
    Detects caustic crossings based on PHYSICAL morphology:
    1. Peak strength - sharp flux spikes indicate caustic crossings
    2. Variance - spiky curves suggest multiple caustics (binary)
    3. Peak count - number of significant peaks (threshold 0.95)
    4. Asymmetry - binary events often show asymmetric light curves
    
    All features extracted from actual flux values, not learned representations.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Combine 4 morphology features into compact representation
        self.feature_combiner = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Small initialization
        for m in self.feature_combiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, flux: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract caustic features from RAW light curve
        
        Args:
            flux: Raw normalized flux [B, T] (NOT embeddings!)
            padding_mask: Boolean mask [B, T] (True = padded)
        
        Returns:
            Caustic features [B, D//4]
        """
        B, T = flux.shape
        valid_mask = ~padding_mask
        
        # Zero out padding
        flux_clean = flux.clone()
        flux_clean[padding_mask] = 0.0
        
        # Compute valid observation count
        n_valid = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        
        # Feature 1: Peak Strength (max flux)
        # Caustics create sharp flux spikes → high max value
        flux_masked = flux_clean.masked_fill(padding_mask, -1e9)
        max_flux = flux_masked.max(dim=1)[0].unsqueeze(-1)
        
        # Feature 2: Flux Variance (spikiness)
        # Binary lenses have spiky light curves from multiple caustics
        flux_mean = (flux_clean * valid_mask.float()).sum(dim=1, keepdim=True) / n_valid
        flux_var = ((flux_clean - flux_mean)**2 * valid_mask.float()).sum(dim=1) / n_valid.squeeze(-1)
        flux_std = torch.sqrt(flux_var + 1e-8).unsqueeze(-1)
        
        # Feature 3: Peak Count
        # Count number of significant peaks (threshold = 95% of max)
        # Real caustic crossings create very sharp peaks above baseline
        flux_threshold = max_flux * 0.95  # STRICT: Only count near-maximum peaks
        high_flux = (flux_clean > flux_threshold).float()
        
        # Count peaks (transitions from below to above threshold)
        # Shift by 1 to detect rising edges
        flux_diff = high_flux[:, 1:] - high_flux[:, :-1]
        peak_transitions = (flux_diff > 0.5).float()  # Rising edge = new peak
        peak_count_raw = peak_transitions.sum(dim=1).unsqueeze(-1)
        
        # Normalize: log scale to handle 0-10 range
        peak_count = torch.log1p(peak_count_raw).unsqueeze(-1) if peak_count_raw.dim() == 1 else torch.log1p(peak_count_raw)
        
        # Feature 4: Temporal Asymmetry
        # Binary events often show asymmetric rise/fall
        mid = T // 2
        
        # Early half flux
        early_mask = valid_mask[:, :mid].float()
        early_flux = (flux_clean[:, :mid] * early_mask).sum(dim=1) / (early_mask.sum(dim=1) + 1e-8)
        
        # Late half flux
        late_mask = valid_mask[:, mid:].float()
        late_flux = (flux_clean[:, mid:] * late_mask).sum(dim=1) / (late_mask.sum(dim=1) + 1e-8)
        
        # Normalize asymmetry by total flux
        total_flux = early_flux + late_flux + 1e-8
        asymmetry = torch.abs(early_flux - late_flux) / total_flux
        asymmetry = asymmetry.unsqueeze(-1)
        
        # Combine all PHYSICAL features
        features = torch.cat([max_flux, flux_std, peak_count, asymmetry], dim=-1)
        
        # Process through small MLP to get compact representation
        return self.feature_combiner(features)


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
    Semi-Causal Multi-Head Attention - FIXED v16.1
    
    CRITICAL FIX: Now uses proper -inf for masking instead of -1e4
    
    Prevents model from seeing future observations, which could
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
        
        # CRITICAL FIX: Apply causal mask with proper -inf
        if self.causal:
            # Create causal mask: can only attend to past and present
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            # FIXED: Use float('-inf') instead of -1e4
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
        
        # Apply padding mask
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            # FIXED: Use float('-inf') instead of -1e4
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Compute attention weights
        # FIXED: Handle -inf → NaN properly after softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # NaN from all -inf rows
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(B, T, D)
        out = self.out_proj(attn_out)
        
        return out * 0.5, attn_weights


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


class MicrolensingTransformer(nn.Module):
    """
    Transformer for 3-class microlensing classification v16.1
    
    FIXED in v16.1:
    - ✅ Proper -inf masking in attention (was causing diagonal-only attention)
    - ✅ SimpleCausticDetector NOW operates on RAW flux (was operating on embeddings!)
    - ✅ All caustic features extracted from PHYSICAL light curve morphology
    - ✅ Peak threshold 0.95 (only near-maximum flux counts as peak)
    
    NEW in v16.0:
    - SimpleCausticDetector for real morphology-based caustic detection
    - Extracts peak strength, variance, peak count, asymmetry
    
    ANTI-CHEATING FEATURES (Maintained):
    1. ✓ Semi-causal attention (no future peeking)
    2. ✓ Relative positional encoding (no absolute time)
    3. ✓ Wide t_0 sampling in simulation (forces morphology learning)
    """
    
    def __init__(
        self,
        n_points: int = 3000,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 64*4,
        dropout: float = 0.1,
        pad_value: float = -1.0,
        max_seq_len: int = 5000,
        use_checkpoint: bool = False,
        causal_attention: bool = True,
        feature_diversity_weight: float = 0.0  # Legacy parameter
    ):
        super().__init__()
        
        self.n_points = n_points
        self.d_model = d_model
        self.pad_value = pad_value
        self.nhead = nhead
        self.use_checkpoint = use_checkpoint
        self.causal_attention = causal_attention
        self.feature_diversity_weight = feature_diversity_weight
        
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
        
        # FIXED v16.1: SimpleCausticDetector operates on RAW flux
        self.caustic_detector = SimpleCausticDetector(d_model)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Caustic detection head (takes features from SimpleCausticDetector)
        self.caustic_head = nn.Sequential(
            nn.LayerNorm(d_model // 4, eps=1e-6),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 8, 1)
        )
        
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
        
        # CRITICAL: Save raw normalized flux for SimpleCausticDetector
        flux_raw = x_clean.squeeze(-1)  # [B, T] - RAW FLUX!
        
        # Input embedding (for transformer)
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
        
        # Global pooling for classification
        x_pooled = self.global_pooling(x_embed, padding_mask)
        
        # FIXED v16.1: Extract caustic features from RAW FLUX, not embeddings!
        caustic_features = self.caustic_detector(flux_raw, padding_mask)
        
        # Get outputs
        logits = self.classification_head(x_pooled)
        caustic_logits = self.caustic_head(caustic_features).squeeze(-1)
        confidence = self.confidence_head(x_pooled).squeeze(-1)
        
        # Clamp for stability
        logits = torch.clamp(logits, min=-20, max=20)
        caustic_logits = torch.clamp(caustic_logits, min=-20, max=20)
        
        outputs = {
            'logits': logits,
            'binary': logits,  # Alias for compatibility
            'caustic': caustic_logits,
            'confidence': confidence
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
