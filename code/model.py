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
from typing import Dict, Optional, Tuple, Literal, List

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRU_GOD_MODE")
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# HARDWARE OPTIMIZATION
# =============================================================================

def configure_cuda_for_speed():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("CUDA OPTIMIZATION: ENGAGED.")
    else:
        logger.warning("RUNNING ON CPU. PREPARE FOR SLOWNESS.")

def set_global_seed(seed: int = 42, rank: int = 0):
    adjusted_seed = seed + rank
    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    os.environ['PYTHONHASHSEED'] = str(adjusted_seed)
    torch.manual_seed(adjusted_seed)
    torch.cuda.manual_seed(adjusted_seed)
    torch.cuda.manual_seed_all(adjusted_seed)
    logger.info(f"SEED PLANTED: {adjusted_seed}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GRUConfig:
    d_model: int = 128
    n_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.2
    window_size: int = 7
    max_seq_len: int = 1500
    n_classes: int = 3
    classifier_hidden_dim: Optional[int] = None
    use_layer_norm: bool = True
    use_residual_connection: bool = True
    hierarchical: bool = True
    feature_extraction: Literal["none", "conv", "mlp"] = "conv"
    use_attention_pooling: bool = True
    
    # Advanced Tech
    use_amp: bool = True
    use_gradient_checkpointing: bool = False 
    compile_model: bool = False
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        assert not self.bidirectional, "CAUSALITY VIOLATION DETECTED."
        assert self.window_size >= 1, "Window size must be positive."

# =============================================================================
# UTILITY MODULES
# =============================================================================

class ContinuousSinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1) + 1e-6 
        scaled_time = torch.log1p(dt) 
        # FIX: Removed explicit .to() call. Buffers move with model automatically.
        args = scaled_time * self.div_term
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pe

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.attention(x).squeeze(-1) # (B, T)
        
        if mask is not None:
            bool_mask = mask.bool()
            # FIX: Used -1e9 instead of -inf to prevent NaN in Softmax
            scores = scores.masked_fill(~bool_mask, -1e9)
            
        weights = F.softmax(scores, dim=-1)
        
        if mask is not None:
            # FIX: Explicit zeroing and re-normalization for numerical stability
            weights = weights * mask.float()
            sum_w = weights.sum(dim=-1, keepdim=True) + 1e-6
            weights = weights / sum_w

        weights = self.dropout(weights)
        # (B, 1, T) x (B, T, D) -> (B, 1, D) -> (B, D)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pad = 2 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()
        x_p = F.pad(x, (self.pad, 0)) 
        x_out = self.conv1(x_p)
        x_out = x_out.permute(0, 2, 1)
        x_out = self.norm1(x_out)
        x_out = self.act(x_out)
        x_out = self.dropout(x_out)
        
        x_out = x_out.permute(0, 2, 1)
        x_p2 = F.pad(x_out, (self.pad, 0))
        x_final = self.conv2(x_p2)
        x_final = x_final.permute(0, 2, 1)
        x_final = self.norm2(x_final)
        x_final = self.act(x_final)
        return x_final

class MLPFeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.GLU(),
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# RECURRENT MODULES
# =============================================================================

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_n = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

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

class LayerNormGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, use_residual: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout_layer = nn.Dropout(dropout)
        
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # FIX: Added projections for residual connections where dims mismatch
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            if use_residual and in_dim != hidden_size:
                self.residual_projs.append(nn.Linear(in_dim, hidden_size))
            else:
                self.residual_projs.append(nn.Identity())

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        
        for t in range(T):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_states[layer_idx]
                h_new = cell(x_t, h_prev)
                
                # FIX: Logic for residual connection with dimension matching
                if self.use_residual and layer_idx > 0:
                    res_input = x_t
                    # Apply projection if dimensions differ
                    if isinstance(self.residual_projs[layer_idx], nn.Linear):
                        res_input = self.residual_projs[layer_idx](res_input)
                    
                    if res_input.shape == h_new.shape:
                        h_new = h_new + res_input
                
                h_states[layer_idx] = h_new
                x_t = self.dropout_layer(h_new)
            outputs.append(h_states[-1])
            
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h_states, dim=0)
        return output, h_n

class SlidingWindowProcessor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, window_size: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.window_gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.window_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        pad_len = max(0, self.window_size - 1)
        padding = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([padding, x], dim=1)
        
        # (B, T, C, Window)
        windows = x_padded.unfold(1, self.window_size, 1)
        
        # Permute to (B, T, Window, C)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        
        # Truncate if unfold produces extra (usually matches T exactly if padded right, but safety first)
        if windows.shape[1] > T:
            windows = windows[:, :T, :, :]
            
        # Flatten for batch processing: (B*T, window_size, C)
        windows_flat = windows.view(B * T, self.window_size, C)
        
        self.window_gru.flatten_parameters()
        _, h_n = self.window_gru(windows_flat) # h_n: (1, B*T, H)
        
        features = h_n.squeeze(0).view(B, T, -1)
        return self.window_proj(features)

# =============================================================================
# HEADS
# =============================================================================

class DeviationDetector(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EventTypeClassifier(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# MAIN MODEL
# =============================================================================

class CausalGRUModel(nn.Module):
    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        self.amp_dtype = torch.float16 if config.use_amp else torch.float32
        
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        self.time_enc = ContinuousSinusoidalEncoding(config.d_model // 2)
        
        self.input_combine = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        if config.feature_extraction == "conv":
            self.feature_extractor = ConvFeatureExtractor(config.d_model, config.d_model, config.dropout)
        elif config.feature_extraction == "mlp":
            self.feature_extractor = MLPFeatureExtractor(config.d_model, config.d_model, config.dropout)
        else:
            self.feature_extractor = nn.Identity()
            
        self.window_processor = SlidingWindowProcessor(
            config.d_model, config.d_model, config.window_size, config.dropout
        )
        
        # RNN input: Feature (d) + Window (d) = 2d
        rnn_input_dim = config.d_model * 2
        
        self.fast_gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0
        )
        
        self.stable_gru = LayerNormGRU(
            input_size=rnn_input_dim,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout,
            use_residual=config.use_residual_connection
        )
        
        self.use_fast_gru = True 
        self.final_norm = nn.LayerNorm(config.d_model)
        
        if config.use_attention_pooling:
            self.attention_pool = AttentionPooling(config.d_model, config.dropout)
        else:
            self.attention_pool = None
            
        if config.hierarchical:
            self.deviation_detector = DeviationDetector(config.d_model, config.dropout)
            self.event_classifier = EventTypeClassifier(config.d_model, config.dropout)
            self.classifier = None
        else:
            self.deviation_detector = None
            self.event_classifier = None
            self.classifier = nn.Sequential(
                nn.Linear(config.d_model, config.classifier_hidden_dim),
                nn.LayerNorm(config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_hidden_dim, config.n_classes)
            )
            
        self.log_temperature = nn.Parameter(torch.tensor([0.0]))
        self._init_weights()
        
        if config.compile_model and hasattr(torch, 'compile'):
            self._compile_submodules()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gru' in name.lower() and 'custom' not in name.lower():
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _compile_submodules(self):
        try:
            self.feature_extractor = torch.compile(self.feature_extractor)
            self.window_processor = torch.compile(self.window_processor)
            logger.info("TORCH.COMPILE: SUCCESSFUL.")
        except Exception as e:
            logger.warning(f"Compilation failed: {e}. IGNORING.")

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None, 
                return_all_timesteps: bool = True) -> Dict[str, torch.Tensor]:
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.config.use_amp:
            with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
                return self._forward_impl(flux, delta_t, lengths, return_all_timesteps)
        else:
            return self._forward_impl(flux, delta_t, lengths, return_all_timesteps)

    def _forward_impl(self, flux, delta_t, lengths, return_all_timesteps):
        B, T = flux.shape
        device = flux.device
        
        flux = torch.nan_to_num(flux, 0.0)
        mask = None
        if lengths is not None:
            range_tensor = torch.arange(T, device=device).unsqueeze(0)
            mask = range_tensor < lengths.unsqueeze(1)
            flux = flux * mask.float()
            delta_t = delta_t * mask.float()

        flux_emb = self.flux_proj(flux.unsqueeze(-1))
        time_emb = self.time_enc(delta_t)
        x = torch.cat([flux_emb, time_emb], dim=-1)
        x = self.input_combine(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        if self.config.use_gradient_checkpointing and self.training:
             x_feat = checkpoint.checkpoint(self.feature_extractor, x, use_reentrant=False)
             x_window = checkpoint.checkpoint(self.window_processor, x_feat, use_reentrant=False)
        else:
             x_feat = self.feature_extractor(x)
             x_window = self.window_processor(x_feat)
        
        combined = torch.cat([x_feat, x_window], dim=-1)
        
        if self.use_fast_gru:
            if self.training: 
                self.fast_gru.flatten_parameters()
            
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    combined, lengths.cpu().clamp(min=1), 
                    batch_first=True, enforce_sorted=False
                )
                gru_out, _ = self.fast_gru(packed)
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                    gru_out, batch_first=True, total_length=T
                )
            else:
                gru_out, _ = self.fast_gru(combined)
        else:
            gru_out, _ = self.stable_gru(combined)
            
        gru_out = self.final_norm(gru_out)
        
        if return_all_timesteps:
            features = gru_out
        else:
            if self.attention_pool is not None:
                features = self.attention_pool(gru_out, mask)
            elif lengths is not None:
                idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, gru_out.size(-1)).to(device)
                features = gru_out.gather(1, idx).squeeze(1)
            else:
                features = gru_out[:, -1, :]

        temp = torch.exp(self.log_temperature)
        if self.config.hierarchical:
            return self._hierarchical_forward_stable(features, temp)
        else:
            return self._direct_forward(features, temp)

    def _direct_forward(self, features, temp):
        logits = self.classifier(features)
        scaled_logits = logits / (temp + 1e-6) 
        return {'logits': scaled_logits, 'probs': F.softmax(scaled_logits, dim=-1)}

    def _hierarchical_forward_stable(self, features, temp):
        dev_logits = self.deviation_detector(features)
        type_logits = self.event_classifier(features)
        
        dev_logits = dev_logits / (temp + 1e-6)
        type_logits = type_logits / (temp + 1e-6)

        dev_log_probs = F.log_softmax(dev_logits, dim=-1)   
        type_log_probs = F.log_softmax(type_logits, dim=-1) 
        
        log_p_flat = dev_log_probs[..., 0:1]
        log_p_dev = dev_log_probs[..., 1:2]
        
        log_p_typeA = log_p_dev + type_log_probs[..., 0:1]
        log_p_typeB = log_p_dev + type_log_probs[..., 1:2]
        
        combined_logits = torch.cat([log_p_flat, log_p_typeA, log_p_typeB], dim=-1)
        
        return {
            'logits': combined_logits, 
            'probs': torch.exp(combined_logits),
            'deviation_logits': dev_logits,
            'type_logits': type_logits
        }
        
    def enable_layernorm_gru(self):
        self.use_fast_gru = False
        logger.info("MODE: STABLE (LayerNormGRU).")

    def enable_fast_gru(self):
        self.use_fast_gru = True
        logger.info("MODE: FAST (cuDNN).")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_model(d_model: int = 128, n_layers: int = 2, hierarchical: bool = True, **kwargs) -> CausalGRUModel:
    configure_cuda_for_speed()
    config = GRUConfig(d_model=d_model, n_layers=n_layers, hierarchical=hierarchical, **kwargs)
    model = CausalGRUModel(config)
    return model

if __name__ == '__main__':
    set_global_seed(42, rank=0)
    
    # Enable Stable GRU to test the custom LayerNorm cell fixes
    model = create_model(d_model=64, compile_model=False, use_gradient_checkpointing=False)
    model.enable_layernorm_gru() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info("--- STARTING GOD MODE DIAGNOSTICS ---")
    
    # 1. Test Tiny Sequence (Fixing the window size crash)
    B_tiny = 2
    T_tiny = 3 # Smaller than default window_size 7
    x_tiny = torch.randn(B_tiny, T_tiny).to(device)
    t_tiny = torch.abs(torch.randn(B_tiny, T_tiny)).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            out_tiny = model(x_tiny, t_tiny, return_all_timesteps=False)
        logger.info(f"✅ TINY INPUT TEST PASSED. Output shape: {out_tiny['logits'].shape}")
    except Exception as e:
        logger.error(f"❌ TINY INPUT FAILED: {e}")

    # 2. Test Masking/NaN Protection
    B_mask = 2
    T_mask = 10
    x_mask = torch.randn(B_mask, T_mask).to(device)
    t_mask = torch.abs(torch.randn(B_mask, T_mask)).to(device)
    # Second sequence has length 0 (simulate padding issue or heavy masking)
    lengths = torch.tensor([10, 1]).to(device) 
    
    try:
        model.train() # Enable dropout to test random NANs
        out_mask = model(x_mask, t_mask, lengths=lengths, return_all_timesteps=False)
        
        if torch.isnan(out_mask['logits']).any():
             logger.error("❌ NAN DETECTED IN OUTPUTS.")
        else:
             logger.info(f"✅ NAN PROTECTION PASSED. No NaNs detected.")
             
    except Exception as e:
        logger.error(f"❌ MASKING TEST FAILED: {e}")

    logger.info("--- DIAGNOSTICS COMPLETE ---")
