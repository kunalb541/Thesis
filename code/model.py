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
# ULTIMATE GOD MODE DEBUG - ALL ISSUES FIXED
# =============================================================================
logging.basicConfig(
    format="%(asctime)s - [GOD_MODE_V4_JIT] - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CAUSAL_GRU_V4_FINAL")
warnings.filterwarnings("ignore", category=UserWarning)

def configure_hardware_optimization():
    """Configures GPU for maximum throughput with proper precision."""
    device_type = 'cpu'
    precision = torch.float32
    
    if torch.cuda.is_available():
        device_type = 'cuda'
        if torch.cuda.is_bf16_supported():
            precision = torch.bfloat16
            logger.info("⚡ HARDWARE: BF16 ACCELERATION ACTIVE")
        else:
            precision = torch.float16
            logger.info("⚡ HARDWARE: FP16 ACCELERATION ACTIVE")

        # TensorFloat-32 for A100/H100 speedup
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        logger.warning("⚠️  HARDWARE: CPU MODE - EXPECT HIGH LATENCY")
        
    return device_type, precision

def set_global_seed(seed: int = 42):
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"🌱 SEED PLANTED: {seed}")

# =============================================================================
# CONFIGURATION
# =============================================================================

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
    use_gradient_checkpointing: bool = False # Disabled by default for JIT speed
    compile_model: bool = False # Disabled default: JIT is faster to start
    
    def __post_init__(self):
        assert self.window_size >= 1, "Window size must be positive."
        assert self.d_model % 2 == 0, "d_model must be divisible by 2."

# =============================================================================
# ENCODING & UTILITIES
# =============================================================================

class NanSafeSinusoidalEncoding(nn.Module):
    """Continuous Time Encoding with NaN protection."""
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.abs().unsqueeze(-1) + 1e-6 
        scaled_time = torch.log1p(dt) 
        args = scaled_time * self.div_term
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pe

class StableAttentionPooling(nn.Module):
    """Attention pooling with dtype-aware masking."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        
        if mask is not None:
            if x.dtype == torch.float16:
                min_val = torch.finfo(torch.float16).min / 10
            elif x.dtype == torch.bfloat16:
                min_val = torch.finfo(torch.bfloat16).min / 10
            else:
                min_val = -1e9
            scores = scores.masked_fill(~mask.bool(), min_val)
            
        weights = F.softmax(scores, dim=-1)
        
        if mask is not None:
            weights = weights * mask.float()
            sum_w = weights.sum(dim=-1, keepdim=True) + 1e-8
            weights = weights / sum_w

        weights = self.dropout(weights)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

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
        x = x.permute(0, 2, 1)
        x_p = F.pad(x, (2, 0))
        x = self.conv1(x_p)
        x = x.permute(0, 2, 1)
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
# JIT OPTIMIZED GRU ENGINE (THE FIX)
# =============================================================================

@torch.jit.script
def jit_fused_gru_layer(
    input_seq: torch.Tensor,
    hidden_state: torch.Tensor,
    w_ih: torch.Tensor, b_ih: torch.Tensor,
    w_hh: torch.Tensor, b_hh: torch.Tensor,
    ln_r_w: torch.Tensor, ln_r_b: torch.Tensor,
    ln_z_w: torch.Tensor, ln_z_b: torch.Tensor,
    ln_n_w: torch.Tensor, ln_n_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized JIT kernel that processes the ENTIRE sequence in C++.
    Eliminates Python loop overhead.
    """
    output_seq = []
    h = hidden_state
    
    # Pre-compute input projections (MatMul hoisting)
    # Shape: (B, T, 3*H)
    x_gates_all = torch.matmul(input_seq, w_ih.t()) + b_ih
    
    # Iterate over time
    for t in range(x_gates_all.size(1)):
        x_gates = x_gates_all[:, t]
        h_gates = torch.matmul(h, w_hh.t()) + b_hh
        
        x_r, x_z, x_n = x_gates.chunk(3, 1)
        h_r, h_z, h_n = h_gates.chunk(3, 1)
        
        # Fused LayerNorm + Activation
        r = torch.sigmoid(F.layer_norm(x_r + h_r, ln_r_w.shape, ln_r_w, ln_r_b))
        z = torch.sigmoid(F.layer_norm(x_z + h_z, ln_z_w.shape, ln_z_w, ln_z_b))
        n = torch.tanh(F.layer_norm(x_n + r * h_n, ln_n_w.shape, ln_n_w, ln_n_b))
        
        h = (1 - z) * n + z * h
        output_seq.append(h)
        
    stacked_output = torch.stack(output_seq, dim=1)
    return stacked_output, h

class JITLayerNormGRU(nn.Module):
    """Wrapper that holds parameters and calls the JIT kernel."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Parameters
        self.i2h_w = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.i2h_b = nn.Parameter(torch.empty(3 * hidden_size))
        self.h2h_w = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.h2h_b = nn.Parameter(torch.empty(3 * hidden_size))
        
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_n = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.i2h_w)
        nn.init.zeros_(self.i2h_b)
        nn.init.orthogonal_(self.h2h_w)
        nn.init.zeros_(self.h2h_b)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return jit_fused_gru_layer(
            x, h,
            self.i2h_w, self.i2h_b,
            self.h2h_w, self.h2h_b,
            self.ln_r.weight, self.ln_r.bias,
            self.ln_z.weight, self.ln_z.bias,
            self.ln_n.weight, self.ln_n.bias
        )

class FusedLayerNormGRU(nn.Module):
    """Multi-layer GRU stacking the JIT kernels."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, use_residual: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = dropout
        
        self.layers = nn.ModuleList([
            JITLayerNormGRU(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.res_proj = None
        if use_residual and input_size != hidden_size:
            self.res_proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=x.dtype)
            
        final_h_list = []
        layer_output = x
        x_residual = x # Keep original for residual
        
        for i, layer in enumerate(self.layers):
            h_prev = h_0[i]
            
            # Call JIT Kernel
            layer_seq, h_final = layer(layer_output, h_prev)
            
            # Residual Connection
            if self.use_residual:
                if i == 0:
                    res = self.res_proj(x_residual) if self.res_proj is not None else x_residual
                    layer_seq = layer_seq + res
                else:
                    layer_seq = layer_seq + layer_output # Simple add for subsequent layers
            
            # Dropout (except last layer)
            if i < self.num_layers - 1:
                if self.dropout > 0:
                    layer_seq = F.dropout(layer_seq, p=self.dropout, training=self.training)
            
            layer_output = layer_seq
            final_h_list.append(h_final)
            
        h_final_stack = torch.stack(final_h_list, dim=0)
        return layer_output, h_final_stack

# =============================================================================
# MAIN MODEL
# =============================================================================

class GodModeCausalGRU(nn.Module):
    def __init__(self, config: GRUConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.amp_dtype = dtype
        
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        self.time_enc = NanSafeSinusoidalEncoding(config.d_model // 2)
        
        self.input_mix = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        if config.feature_extraction == "conv":
            self.feature_extractor = CausalConvFeatureExtractor(config.d_model, config.d_model, config.dropout)
        else:
            self.feature_extractor = MLPFeatureExtractor(config.d_model, config.d_model, config.dropout)
            
        self.window_processor = CausalWindowProcessor(
            config.d_model, config.d_model, config.window_size, config.dropout
        )
        
        rnn_input_dim = config.d_model * 2
        
        self.gru = FusedLayerNormGRU(
            input_size=rnn_input_dim,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout,
            use_residual=config.use_residual
        )
        
        self.norm_final = nn.LayerNorm(config.d_model)
        
        if config.use_attention_pooling:
            self.pool = StableAttentionPooling(config.d_model, config.dropout)
        else:
            self.pool = None
            
        self.raw_temperature = nn.Parameter(torch.tensor([0.0]))
        
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
            
            # GRU Forward (JIT Enabled internally)
            gru_out, _ = self.gru(combined)
            gru_out = self.norm_final(gru_out)
            
            # --- POOLED/FINAL PREDICTION ---
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
    # Simple diagnostic check
    model = GodModeCausalGRU(GRUConfig(compile_model=False))
    print("JIT GodMode Model Initialized. Params:", sum(p.numel() for p in model.parameters()))
