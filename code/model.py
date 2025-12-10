import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import logging
import warnings
import random
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, List, Union

# =============================================================================
# GOD MODE V5: TURBO ARCHITECTURE (CuDNN OPTIMIZED)
# =============================================================================
logging.basicConfig(
    format="%(asctime)s - [GOD_MODE_TURBO] - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ROMAN_MODEL")
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class GRUConfig:
    d_model: int = 128
    n_layers: int = 3
    dropout: float = 0.2
    window_size: int = 7
    max_seq_len: int = 2000
    n_classes: int = 3
    
    hierarchical: bool = True
    use_residual: bool = True
    use_layer_norm: bool = True
    feature_extraction: Literal["conv", "mlp"] = "mlp"
    use_attention_pooling: bool = True
    
    use_amp: bool = True
    use_gradient_checkpointing: bool = False 
    compile_model: bool = False 

# =============================================================================
# COMPONENTS
# =============================================================================

class NanSafeSinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # Protect against log(0)
        dt = delta_t.abs().unsqueeze(-1) + 1e-6 
        scaled_time = torch.log1p(dt) 
        args = scaled_time * self.div_term
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pe

class StableAttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1) # (B, T)
        
        if mask is not None:
            min_val = -1e4 if x.dtype == torch.float32 else -1e3
            scores = scores.masked_fill(~mask.bool(), min_val)
            
        weights = F.softmax(scores, dim=-1)
        
        if mask is not None:
            # Re-normalize to handle potential underflow
            weights = weights * mask.float()
            sum_w = weights.sum(dim=-1, keepdim=True) + 1e-8
            weights = weights / sum_w

        weights = self.dropout(weights)
        # Weighted sum: (B, 1, T) x (B, T, D) -> (B, 1, D)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

class MLPFeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CausalConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        x_p = F.pad(x, (2, 0)) # Causal padding
        x = self.conv1(x_p)
        x = x.permute(0, 2, 1) # Back to (B, T, C) for Norm
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        x_p2 = F.pad(x, (2, 0))
        x = self.conv2(x_p2)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        return x

class CausalWindowProcessor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, window_size: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(
            input_size, hidden_size, 
            kernel_size=window_size, 
            padding=0,
            groups=1
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)
        pad = max(0, self.window_size - 1)
        x_padded = F.pad(x_perm, (pad, 0))
        out = self.conv(x_padded)
        out = out.permute(0, 2, 1)
        return self.proj(out)

# =============================================================================
# OPTIMIZED RECURRENT BLOCK
# =============================================================================

class FastRNNBlock(nn.Module):
    """
    Hardware-Accelerated RNN Block.
    Uses native CuDNN GRU for maximum throughput.
    Applies LayerNorm + Residual *between* layers.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float, use_residual: bool):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
        # Projection if input dim != hidden dim (for first layer residual)
        if use_residual and input_size != hidden_size:
            self.res_proj = nn.Linear(input_size, hidden_size)
        else:
            self.res_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, I)
        out, _ = self.gru(x)
        out = self.norm(out)
        out = self.dropout(out)
        
        if self.use_residual:
            res = x if self.res_proj is None else self.res_proj(x)
            out = out + res
            
        return out

class TurboGRU(nn.Module):
    """Stack of FastRNNBlocks."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, use_residual: bool):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(
                FastRNNBlock(in_dim, hidden_size, dropout, use_residual)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        for layer in self.layers:
            x = layer(x)
        return x, None # Match signature of custom GRU

# =============================================================================
# MAIN MODEL
# =============================================================================

class GodModeCausalGRU(nn.Module):
    def __init__(self, config: GRUConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.amp_dtype = dtype
        
        # 1. Inputs
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        self.time_enc = NanSafeSinusoidalEncoding(config.d_model // 2)
        
        self.input_mix = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 2. Features
        if config.feature_extraction == "conv":
            self.feature_extractor = CausalConvFeatureExtractor(config.d_model, config.d_model, config.dropout)
        else:
            self.feature_extractor = MLPFeatureExtractor(config.d_model, config.d_model, config.dropout)
            
        self.window_processor = CausalWindowProcessor(
            config.d_model, config.d_model, config.window_size, config.dropout
        )
        
        # 3. Recurrent Core (TURBO MODE)
        rnn_input_dim = config.d_model * 2
        self.gru = TurboGRU(
            input_size=rnn_input_dim,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout,
            use_residual=config.use_residual
        )
        
        self.norm_final = nn.LayerNorm(config.d_model)
        
        # 4. Pooling
        if config.use_attention_pooling:
            self.pool = StableAttentionPooling(config.d_model, config.dropout)
        else:
            self.pool = None
            
        self.raw_temperature = nn.Parameter(torch.tensor([0.0]))
        
        # 5. Heads
        if config.hierarchical:
            self.head_deviation = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model // 2),
                nn.Dropout(config.dropout), 
                nn.Linear(config.d_model // 2, 2)
            )
            self.head_type = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 2)
            )
        else:
            self.classifier = nn.Linear(config.d_model, config.n_classes)

        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self, 
        flux: torch.Tensor, 
        delta_t: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        device = flux.device
        use_amp = self.config.use_amp and device.type == 'cuda'
        
        with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=use_amp):
            
            B, T = flux.shape
            
            mask = None
            if lengths is not None:
                mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                flux = flux * mask.float()
                delta_t = delta_t * mask.float()

            f_emb = self.flux_proj(flux.unsqueeze(-1))
            t_emb = self.time_enc(delta_t)
            x = torch.cat([f_emb, t_emb], dim=-1)
            x = self.input_mix(x)
            
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()

            if self.config.use_gradient_checkpointing and self.training:
                x_feat = checkpoint.checkpoint(self.feature_extractor, x, use_reentrant=False)
                x_window = checkpoint.checkpoint(self.window_processor, x_feat, use_reentrant=False)
            else:
                x_feat = self.feature_extractor(x)
                x_window = self.window_processor(x_feat)
                
            combined = torch.cat([x_feat, x_window], dim=-1)
            
            # FAST GRU (CuDNN)
            gru_out, _ = self.gru(combined)
            gru_out = self.norm_final(gru_out)
            
            # POOLING
            if self.pool is not None:
                features_pooled = self.pool(gru_out, mask)
            elif lengths is not None:
                idx = (lengths - 1).clamp(min=0).long()
                idx = idx.view(-1, 1, 1).expand(-1, 1, gru_out.size(-1))
                features_pooled = gru_out.gather(1, idx).squeeze(1)
            else:
                features_pooled = gru_out[:, -1, :]
                
            temp = F.softplus(self.raw_temperature).clamp(min=0.1, max=10.0)
            
            result = {}

            if self.config.hierarchical:
                hier_res = self._hierarchical_inference(features_pooled, temp)
                result.update(hier_res)
            else:
                logits = self.classifier(features_pooled) / temp
                result['logits'] = logits
                result['probs'] = F.softmax(logits, dim=-1)

            if return_all_timesteps:
                if self.config.hierarchical:
                    hier_res_seq = self._hierarchical_inference(gru_out, temp)
                    result['logits_seq'] = hier_res_seq['logits']
                    result['probs_seq'] = hier_res_seq['probs']
                else:
                    logits_seq = self.classifier(gru_out) / temp
                    result['logits_seq'] = logits_seq
                    result['probs_seq'] = F.softmax(logits_seq, dim=-1)

            return result

    def _hierarchical_inference(self, features: torch.Tensor, temp: torch.Tensor) -> Dict[str, torch.Tensor]:
        dev_logits = self.head_deviation(features)
        type_logits = self.head_type(features)
        
        dev_logits = dev_logits / temp
        type_logits = type_logits / temp
        
        dev_log_probs = F.log_softmax(dev_logits, dim=-1)
        type_log_probs = F.log_softmax(type_logits, dim=-1)
        
        log_p_flat = dev_log_probs[..., 0:1] 
        log_p_dev_yes = dev_log_probs[..., 1:2]
        
        log_p_classA = log_p_dev_yes + type_log_probs[..., 0:1]
        log_p_classB = log_p_dev_yes + type_log_probs[..., 1:2]
        
        final_log_probs = torch.cat([log_p_flat, log_p_classA, log_p_classB], dim=-1)
        
        return {
            'logits': final_log_probs,
            'probs': torch.exp(final_log_probs),
            'aux_dev': dev_logits,
            'aux_type': type_logits
        }

if __name__ == '__main__':
    # Diagnostic
    model = GodModeCausalGRU(GRUConfig())
    print(f"Turbo Model Params: {sum(p.numel() for p in model.parameters())}")
