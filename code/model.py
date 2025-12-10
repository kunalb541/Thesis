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
    format="%(asctime)s - [GOD_MODE_V3] - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CAUSAL_GRU_V3_FINAL")
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
    use_gradient_checkpointing: bool = True 
    compile_model: bool = True
    
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
            # FIX: Dtype-aware minimum values
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

class MLPFeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        # FIX: Explicit dim for GLU
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.GLU(dim=-1),  # Critical: Specify dimension
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# CORE PROCESSORS
# =============================================================================

class CausalWindowProcessor(nn.Module):
    """Efficient sliding window via Causal Conv1d."""
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

class LayerNormGRUCell(nn.Module):
    """GRU cell with Layer Normalization."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # FIX: Add bias=True for proper learning
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_n = nn.LayerNorm(hidden_size)
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_gates = self.i2h(x)
        h_gates = self.h2h(h)
        
        x_r, x_z, x_n = x_gates.chunk(3, 1)
        h_r, h_z, h_n = h_gates.chunk(3, 1)
        
        r = torch.sigmoid(self.ln_r(x_r + h_r))
        z = torch.sigmoid(self.ln_z(x_z + h_z))
        n = torch.tanh(self.ln_n(x_n + r * h_n))
        
        h_new = (1 - z) * n + z * h
        return h_new

class FusedLayerNormGRU(nn.Module):
    """Python-loop GRU optimized for torch.compile()."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, use_residual: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # FIX: Proper conditional projection
        self.res_proj = None
        if use_residual and input_size != hidden_size:
            self.res_proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=x.dtype)
            
        h_states = [h_0[i] for i in range(self.num_layers)]
        output_seq = torch.empty(B, T, self.hidden_size, device=device, dtype=x.dtype)
        
        # FIX: Store original input for residual
        x_original = x
        
        for t in range(T):
            x_step = x[:, t, :]
            x_step_original = x_original[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_states[layer_idx]
                h_new = cell(x_step, h_prev)
                
                # FIX: Correct residual application
                if self.use_residual:
                    if layer_idx == 0:
                        if self.res_proj is not None:
                            residual = self.res_proj(x_step_original)
                        else:
                            residual = x_step_original
                        h_new = h_new + residual
                    else:
                        h_new = h_new + x_step
                
                h_states[layer_idx] = h_new
                
                if layer_idx < self.num_layers - 1:
                    if self.dropout > 0 and self.training:
                        x_step = F.dropout(h_new, p=self.dropout, training=True)
                    else:
                        x_step = h_new
                else:
                    output_seq[:, t, :] = h_new
            
        h_final = torch.stack(h_states, dim=0)
        return output_seq, h_final

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
            
        # FIX: Better temperature initialization
        self.raw_temperature = nn.Parameter(torch.tensor([0.0]))
        
        if config.hierarchical:
            self.head_deviation = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model // 2),
                nn.Dropout(config.dropout),  # FIX: Add dropout
                nn.Linear(config.d_model // 2, 2)
            )
            self.head_type = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),  # FIX: Add dropout
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

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # FIX: Device-aware AMP
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
            
            gru_out, _ = self.gru(combined)
            gru_out = self.norm_final(gru_out)
            
            if self.pool is not None:
                features = self.pool(gru_out, mask)
            elif lengths is not None:
                # FIX: Type-safe indexing
                idx = (lengths - 1).clamp(min=0).long()
                idx = idx.view(-1, 1, 1).expand(-1, 1, gru_out.size(-1))
                features = gru_out.gather(1, idx).squeeze(1)
            else:
                features = gru_out[:, -1, :]
                
            # FIX: Bounded temperature
            temp = F.softplus(self.raw_temperature).clamp(min=0.1, max=10.0)
            
            if self.config.hierarchical:
                return self._hierarchical_inference(features, temp)
            else:
                logits = self.classifier(features) / temp
                return {'logits': logits, 'probs': F.softmax(logits, dim=-1)}

    def _hierarchical_inference(self, features, temp):
        dev_logits = self.head_deviation(features)
        type_logits = self.head_type(features)
        
        dev_logits = dev_logits / temp
        type_logits = type_logits / temp
        
        dev_log_probs = F.log_softmax(dev_logits, dim=-1)
        type_log_probs = F.log_softmax(type_logits, dim=-1)
        
        log_p_flat = dev_log_probs[:, 0:1]
        log_p_dev_yes = dev_log_probs[:, 1:2]
        
        log_p_classA = log_p_dev_yes + type_log_probs[:, 0:1]
        log_p_classB = log_p_dev_yes + type_log_probs[:, 1:2]
        
        final_log_probs = torch.cat([log_p_flat, log_p_classA, log_p_classB], dim=-1)
        
        return {
            'logits': final_log_probs,
            'probs': torch.exp(final_log_probs),
            'aux_dev': dev_logits,
            'aux_type': type_logits
        }

# =============================================================================
# DIAGNOSTICS
# =============================================================================

def create_model(d_model=128, n_layers=2, **kwargs) -> GodModeCausalGRU:
    device_type, precision = configure_hardware_optimization()
    config = GRUConfig(d_model=d_model, n_layers=n_layers, **kwargs)
    model = GodModeCausalGRU(config, dtype=precision)
    
    if config.compile_model and torch.cuda.is_available():
        logger.info("🔧 COMPILING MODEL WITH INDUCTOR...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("✅ COMPILATION SUCCESSFUL")
        except Exception as e:
            logger.warning(f"⚠️  Compilation failed: {e}")
            
    return model

if __name__ == '__main__':
    set_global_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*60)
    logger.info("INITIALIZING GOD MODE V3 DIAGNOSTICS")
    logger.info("="*60)
    
    model = create_model(d_model=128, n_layers=3, window_size=10, compile_model=False)
    model.to(device)
    
    B, T = 8, 200
    x = torch.randn(B, T, device=device)
    t = torch.abs(torch.randn(B, T, device=device))
    lengths = torch.tensor([200, 150, 100, 50, 10, 5, 200, 200], device=device)
    
    try:
        # FORWARD PASS
        model.eval()
        with torch.no_grad():
            logger.info(f"🔍 Running Inference (Batch={B}, Seq={T})...")
            out = model(x, t, lengths)
        
        probs = out['probs']
        logger.info(f"✅ FORWARD PASS: SUCCESS | Shape: {probs.shape}")
        
        # PROBABILITY CHECK
        sums = probs.sum(dim=-1)
        is_valid = torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
        
        if is_valid:
            logger.info("✅ PROBABILITY MATH: PASSED (Σ=1.0)")
        else:
            logger.error(f"❌ PROBABILITY MATH: FAILED | Sums: {sums}")

        # BACKWARD PASS
        logger.info("🔍 Running Backward Pass...")
        model.train()
        targets = torch.randint(0, 3, (B,), device=device)
        
        out_train = model(x, t, lengths)
        loss = F.nll_loss(out_train['logits'], targets)
        loss.backward()
        
        # GRADIENT ANALYSIS
        has_grads = True
        grad_stats = []
        nan_grads = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    logger.warning(f"⚠️  MISSING GRADIENT: {name}")
                    has_grads = False
                elif torch.isnan(param.grad).any():
                    nan_grads.append(name)
                    has_grads = False
                elif torch.isinf(param.grad).any():
                    logger.error(f"❌ Inf GRADIENT: {name}")
                    has_grads = False
                else:
                    grad_norm = param.grad.norm().item()
                    grad_stats.append((name, grad_norm))
                
        if has_grads:
            logger.info(f"✅ BACKWARD PASS: PASSED | Loss: {loss.item():.6f}")
            grad_stats.sort(key=lambda x: x[1], reverse=True)
            logger.info("📊 Top 5 Gradient Norms:")
            for name, norm in grad_stats[:5]:
                logger.info(f"   {name}: {norm:.6f}")
        else:
            logger.error("❌ BACKWARD PASS: FAILED")
            if nan_grads:
                logger.error(f"   NaN Gradients in: {nan_grads}")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    logger.info("="*60)
    logger.info("GOD MODE V3 DIAGNOSTICS COMPLETE")
    logger.info("="*60)
