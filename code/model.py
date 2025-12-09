import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Setup Logger
logger = logging.getLogger("causal_model")

@dataclass
class CausalConfig:
    d_model: int = 512
    n_heads: int = 8
    n_transformer_layers: int = 6
    kernel_size: int = 3
    n_conv_layers: int = 4
    dilation_growth: int = 2
    dropout: float = 0.1
    max_seq_len: int = 4096
    n_classes: int = 3
    classifier_hidden_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

# =============================================================================
# 1. STRICT CAUSAL CONVOLUTIONS
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=0, 
            dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0)) # Left padding only
        out = self.conv(x)
        return self.dropout(out)

class ResidualCausalBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(d_model)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm1(x)
        z = z.transpose(1, 2)
        z = self.conv1(z)
        z = z.transpose(1, 2)
        z = self.act(z)
        z = self.norm2(z)
        z = z.transpose(1, 2)
        z = self.conv2(z)
        z = z.transpose(1, 2)
        return residual + z

# =============================================================================
# 2. OPTIMIZED TRANSFORMER (Flash Attention)
# =============================================================================

class ContinuousSinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1) + 1e-6
        scaled = torch.log1p(dt) * self.div_term
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe

class FlashCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True)
        else:
            # 1. Causal Mask
            attn_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            # GOD-MODE FIX: Use -1e9 instead of -inf to avoid NaN in 0/0 edge cases
            attn_mask = torch.zeros(T, T, device=x.device, dtype=q.dtype).masked_fill(~attn_mask, -1e9)
            
            # 2. Merge Padding Mask
            pad_mask = key_padding_mask.view(B, 1, 1, T).expand(-1, self.n_heads, T, -1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) 
            
            # GOD-MODE FIX: Ensure we don't mask everything if a row is full padding (though unlikely)
            final_mask = attn_mask.masked_fill(pad_mask.bool(), -1e9)

            out = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask, dropout_p=self.dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# =============================================================================
# 3. MAIN MODEL
# =============================================================================

class CausalHybridModel(nn.Module):
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        self.flux_proj = nn.Linear(1, config.d_model)
        self.time_enc = ContinuousSinusoidalEncoding(config.d_model)
        self.emb_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.feature_extractor = nn.ModuleList()
        curr_dilation = 1
        for _ in range(config.n_conv_layers):
            self.feature_extractor.append(
                ResidualCausalBlock(config.d_model, config.kernel_size, curr_dilation, config.dropout)
            )
            curr_dilation *= config.dilation_growth
            
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': FlashCausalAttention(config.d_model, config.n_heads, config.dropout),
                'norm1': nn.LayerNorm(config.d_model),
                'ffn': nn.Sequential(
                    nn.Linear(config.d_model, 4 * config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, config.d_model),
                    nn.Dropout(config.dropout)
                ),
                'norm2': nn.LayerNorm(config.d_model)
            }) for _ in range(config.n_transformer_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Linear(config.classifier_hidden_dim, config.n_classes)
        )
        self.log_temperature = nn.Parameter(torch.tensor([0.0]))

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None,
                return_all_timesteps: bool = True) -> Dict[str, torch.Tensor]:
        
        B, T = flux.shape
        key_padding_mask = None
        mask_float = torch.ones(B, T, 1, device=flux.device)
        
        if lengths is not None:
            range_tensor = torch.arange(T, device=flux.device).unsqueeze(0)
            key_padding_mask = range_tensor >= lengths.unsqueeze(1)
            mask_float = (~key_padding_mask).float().unsqueeze(-1)
            # Hard zeroing of padded inputs
            flux = torch.nan_to_num(flux, 0.0)
            flux = flux.unsqueeze(-1) * mask_float
        else:
            flux = flux.unsqueeze(-1)

        x = self.flux_proj(flux) 
        t_emb = self.time_enc(delta_t)
        x = x + t_emb
        x = self.emb_norm(x)
        x = self.dropout(x)
        
        if lengths is not None: x = x * mask_float
        for block in self.feature_extractor:
            x = block(x)
            if lengths is not None: x = x * mask_float

        for layer in self.transformer_layers:
            residual = x
            x = layer['norm1'](x)
            x = layer['attn'](x, key_padding_mask=key_padding_mask)
            x = residual + x
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['ffn'](x)
        
        x = self.final_norm(x)
        logits = self.classifier(x)
        
        temp = torch.exp(self.log_temperature).clamp(min=0.1, max=5.0)
        scaled_logits = logits / temp
        
        if not return_all_timesteps and lengths is not None:
            last_idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, self.config.n_classes)
            final_logits = scaled_logits.gather(1, last_idx).squeeze(1)
            return {'logits': final_logits, 'probs': F.softmax(final_logits, dim=-1)}
            
        return {'logits': scaled_logits, 'probs': F.softmax(scaled_logits, dim=-1)}
