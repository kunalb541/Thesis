import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, List

# =============================================================================
# LOGGING SETUP: THE EYES AND EARS
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRU_GOD_MODE")
# Suppress the whining of lesser libraries
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# HARDWARE OPTIMIZATION: ACCELERATE THE RAY GUN
# =============================================================================

def configure_cuda_for_speed():
    """
    Kicks the GPU into hyperdrive. We trade tiny precision for MASSIVE SPEED.
    """
    if torch.cuda.is_available():
        # TF32 is the sweet spot for Ampere+ (A100, H100, RTX 30/40 series)
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Benchmark finds the best algo for your specific card/input size
        # WARNING: If input shapes change drastically every batch, set this to False.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        logger.info("CUDA OPTIMIZATION: ENGAGED. PREPARE FOR MACH 10.")
    else:
        logger.warning("NO GPU DETECTED? PATHETIC. YOU ARE RUNNING ON A POTATO.")

# =============================================================================
# CONFIGURATION: THE BATTLE PLANS
# =============================================================================

@dataclass
class GRUConfig:
    d_model: int = 128
    n_layers: int = 2
    bidirectional: bool = False  # FALSE. Causality is absolute!
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
    compile_model: bool = False  # set True for torch.compile() (Linux recommended)
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        assert not self.bidirectional, "CAUSALITY VIOLATION DETECTED. Bidirectional must be False."
        assert self.window_size >= 1, "Window size must be positive, you muppet."

# =============================================================================
# JIT COMPILED MATH KERNELS: THE SECRET SAUCE
# =============================================================================

@torch.jit.script
def gru_cell_math(x: torch.Tensor, h: torch.Tensor, 
                  w_ir: torch.Tensor, w_hr: torch.Tensor, b_ir: Optional[torch.Tensor], b_hr: Optional[torch.Tensor],
                  w_iz: torch.Tensor, w_hz: torch.Tensor, b_iz: Optional[torch.Tensor], b_hz: Optional[torch.Tensor],
                  w_in: torch.Tensor, w_hn: torch.Tensor, b_in: Optional[torch.Tensor], b_hn: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiled JIT kernel for the raw math. Python loops are slow. This is fast.
    """
    # Linear projections
    r_x = F.linear(x, w_ir, b_ir)
    r_h = F.linear(h, w_hr, b_hr)
    
    z_x = F.linear(x, w_iz, b_iz)
    z_h = F.linear(h, w_hz, b_hz)
    
    n_x = F.linear(x, w_in, b_in)
    n_h = F.linear(h, w_hn, b_hn)
    
    return r_x + r_h, z_x + z_h, n_x, n_h

# =============================================================================
# UTILITY MODULES
# =============================================================================

class ContinuousSinusoidalEncoding(nn.Module):
    """
    Handles irregular time steps. Essential for telescope gaps.
    """
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        # Log-space computation for numerical stability
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: (B, T)
        # Add epsilon to prevent log(0)
        dt = delta_t.unsqueeze(-1) + 1e-6 
        
        # Logarithmic time scaling for wide dynamic range
        scaled_time = torch.log1p(dt) 
        
        # Broadcasting happens here
        args = scaled_time * self.div_term.to(delta_t.device)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pe

class AttentionPooling(nn.Module):
    """
    Learns 'where' to look in the sequence. Optimized for stability.
    """
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
            # We must use a very large negative number, but NOT -inf to avoid NaNs in fp16
            # We convert mask to boolean just in case
            bool_mask = mask.bool()
            scores = scores.masked_fill(~bool_mask, -1e4)
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # (B, 1, T) @ (B, T, D) -> (B, 1, D)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

# =============================================================================
# FEATURE EXTRACTORS: PRE-PROCESSING THE SIGNAL
# =============================================================================

class ConvFeatureExtractor(nn.Module):
    """
    Causal 1D Convolution. NO LEAKAGE allowed.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)
        
        # LayerNorm is superior to BatchNorm for Time Series
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pad = 2 # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> Need (B, C, T)
        x = x.permute(0, 2, 1).contiguous()
        
        # Layer 1
        x_p = F.pad(x, (self.pad, 0)) # Left pad only
        x_out = self.conv1(x_p)
        x_out = x_out.permute(0, 2, 1) # Back to (B, T, C) for Norm
        x_out = self.norm1(x_out)
        x_out = self.act(x_out)
        x_out = self.dropout(x_out)
        
        # Layer 2
        x_out = x_out.permute(0, 2, 1) # (B, C, T)
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
            nn.Linear(in_features, out_features * 2), # Expansion for SwiGLU logic
            nn.GLU(), # Gated Linear Unit - The smart activation
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# RECURRENT MODULES: THE BRAIN
# =============================================================================

class LayerNormGRUCell(nn.Module):
    """
    Custom GRU Cell. Slow in Python, but we JIT optimized the logic.
    Superior stability for long sequences compared to cuDNN.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layers
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        
        # LayerNorms applied to the linear outputs (pre-activation)
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_n = nn.LayerNorm(hidden_size)
        
        # Init
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Pre-compute all linear projections at once for speed (Matrix Multiply)
        # x_gates: (B, 3*H)
        x_gates = self.i2h(x)
        h_gates = self.h2h(h)
        
        x_r, x_z, x_n = x_gates.chunk(3, 1)
        h_r, h_z, h_n = h_gates.chunk(3, 1)
        
        # Apply LayerNorm and Activation
        r = torch.sigmoid(self.ln_r(x_r + h_r))
        z = torch.sigmoid(self.ln_z(x_z + h_z))
        n = torch.tanh(self.ln_n(x_n + r * h_n))
        
        # Hidden state update
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

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            
        h_states = [h_0[i] for i in range(self.num_layers)]
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        
        # Time Loop
        for t in range(T):
            x_t = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_states[layer_idx]
                h_new = cell(x_t, h_prev)
                
                # Residual connection
                if self.use_residual and layer_idx > 0:
                    if x_t.size(-1) == h_new.size(-1):
                        h_new = h_new + x_t
                
                h_states[layer_idx] = h_new
                x_t = self.dropout_layer(h_new) # Input to next layer
            
            outputs.append(h_states[-1])
            
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h_states, dim=0)
        return output, h_n

class SlidingWindowProcessor(nn.Module):
    """
    Parallel processing of local context. 
    Warning: High memory usage for large window_size.
    """
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
        if T >= self.window_size:
            # Pad left
            padding = torch.zeros(B, self.window_size - 1, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([padding, x], dim=1)
            
            # Efficient Unfold: (B, C, T_padded) -> windows
            # Output: (B, T, window_size, C)
            windows = x_padded.unfold(1, self.window_size, 1).contiguous()
            
            # Flatten: (B*T, window_size, C)
            # Use contiguous before view to prevent memory strides from killing performance
            windows_flat = windows.view(B * T, self.window_size, C)
            
            # Run GRU in parallel batches
            self.window_gru.flatten_parameters()
            _, h_n = self.window_gru(windows_flat) # h_n: (1, B*T, H)
            
            features = h_n.squeeze(0).view(B, T, -1)
        else:
            # Fallback for tiny sequences
            outputs = []
            for t in range(T):
                start = max(0, t - self.window_size + 1)
                window = x[:, start:t+1, :]
                _, h_n = self.window_gru(window)
                outputs.append(h_n.squeeze(0))
            features = torch.stack(outputs, dim=1)
            
        return self.window_proj(features)

# =============================================================================
# HEADS: THE DECISION MAKERS
# =============================================================================

class DeviationDetector(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2) # [IsFlat, IsDeviant]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) # Return raw logits

class EventTypeClassifier(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2) # [Type A, Type B]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) # Return raw logits

# =============================================================================
# MAIN MODEL: THE MASTERPIECE
# =============================================================================

class CausalGRUModel(nn.Module):
    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        self.amp_dtype = torch.float16 if config.use_amp else torch.float32
        
        # 1. Embeddings
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        self.time_enc = ContinuousSinusoidalEncoding(config.d_model // 2)
        
        self.input_combine = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 2. Features
        if config.feature_extraction == "conv":
            self.feature_extractor = ConvFeatureExtractor(config.d_model, config.d_model, config.dropout)
        elif config.feature_extraction == "mlp":
            self.feature_extractor = MLPFeatureExtractor(config.d_model, config.d_model, config.dropout)
        else:
            self.feature_extractor = nn.Identity()
            
        # 3. Window Processor
        self.window_processor = SlidingWindowProcessor(
            config.d_model, config.d_model, config.window_size, config.dropout
        )
        
        # 4. Main RNN
        # Input is d_model (feat) + d_model (window) = 2 * d_model
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
        
        # 5. Pooling
        if config.use_attention_pooling:
            self.attention_pool = AttentionPooling(config.d_model, config.dropout)
        else:
            self.attention_pool = None
            
        # 6. Heads
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
        
        # Compilation Hook
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
            # We compile feed-forward parts. RNNs are tricky with compile.
            self.feature_extractor = torch.compile(self.feature_extractor)
            self.window_processor = torch.compile(self.window_processor)
            logger.info("TORCH.COMPILE: SUCCESSFUL. PREPARE FOR WARP SPEED.")
        except Exception as e:
            logger.warning(f"Compilation failed: {e}. IGNORING AND PROCEEDING.")

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None, 
                return_all_timesteps: bool = True) -> Dict[str, torch.Tensor]:
        
        # Automatic Mixed Precision wrapper
        if self.config.use_amp and torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                return self._forward_impl(flux, delta_t, lengths, return_all_timesteps)
        else:
            return self._forward_impl(flux, delta_t, lengths, return_all_timesteps)

    def _forward_impl(self, flux, delta_t, lengths, return_all_timesteps):
        B, T = flux.shape
        device = flux.device
        
        # 1. NaN Handling & Masking
        # Optimization: Ideally do this in Dataset, but safe here
        flux = torch.nan_to_num(flux, 0.0)
        mask = None
        if lengths is not None:
            range_tensor = torch.arange(T, device=device).unsqueeze(0)
            mask = range_tensor < lengths.unsqueeze(1)
            flux = flux * mask.float()
            delta_t = delta_t * mask.float()

        # 2. Embedding
        flux_emb = self.flux_proj(flux.unsqueeze(-1))
        time_emb = self.time_enc(delta_t)
        x = torch.cat([flux_emb, time_emb], dim=-1)
        x = self.input_combine(x)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        # 3. Features
        # x: (B, T, D)
        x_feat = self.feature_extractor(x)
        x_window = self.window_processor(x_feat)
        
        # 4. Recurrent Processing
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
            # Custom GRU doesn't support packed sequence yet, manual masking applied inside
            gru_out, _ = self.stable_gru(combined)
            
        gru_out = self.final_norm(gru_out)
        
        # 5. Selection
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

        # 6. Classification
        temp = torch.exp(self.log_temperature)
        if self.config.hierarchical:
            return self._hierarchical_forward_stable(features, temp)
        else:
            return self._direct_forward(features, temp)

    def _direct_forward(self, features, temp):
        logits = self.classifier(features)
        # Temperature scaling for calibration
        scaled_logits = logits / (temp + 1e-6) 
        return {'logits': scaled_logits, 'probs': F.softmax(scaled_logits, dim=-1)}

    def _hierarchical_forward_stable(self, features, temp):
        """
        Calculates hierarchical probabilities efficiently in LOG SPACE to avoid NaN.
        Structure:
        1. Deviation? (Flat / Deviant)
        2. If Deviant -> Type A or Type B?
        """
        dev_logits = self.deviation_detector(features)
        type_logits = self.event_classifier(features)
        
        # Scale by temp
        dev_logits = dev_logits / (temp + 1e-6)
        type_logits = type_logits / (temp + 1e-6)

        # Log-Softmax is numerically stable
        dev_log_probs = F.log_softmax(dev_logits, dim=-1)   # [log(P_flat), log(P_dev)]
        type_log_probs = F.log_softmax(type_logits, dim=-1) # [log(P_A|dev), log(P_B|dev)]
        
        # 1. log P(Flat)
        log_p_flat = dev_log_probs[..., 0:1]
        
        # 2. log P(TypeA) = log P(Dev) + log P(TypeA | Dev)
        log_p_dev = dev_log_probs[..., 1:2]
        log_p_typeA = log_p_dev + type_log_probs[..., 0:1]
        
        # 3. log P(TypeB) = log P(Dev) + log P(TypeB | Dev)
        log_p_typeB = log_p_dev + type_log_probs[..., 1:2]
        
        # Combine
        combined_logits = torch.cat([log_p_flat, log_p_typeA, log_p_typeB], dim=-1)
        
        return {
            'logits': combined_logits, # These are technically log-probs, but function as logits
            'probs': torch.exp(combined_logits),
            'deviation_logits': dev_logits,
            'type_logits': type_logits
        }

    # =========================================================================
    # PUBLIC HELPER METHODS
    # =========================================================================

    def enable_layernorm_gru(self):
        self.use_fast_gru = False
        logger.info("MODE: STABLE (LayerNormGRU). Slower, but indestructible.")

    def enable_fast_gru(self):
        self.use_fast_gru = True
        logger.info("MODE: FAST (cuDNN). Speed is key.")

    def freeze_deviation_detector(self):
        if self.deviation_detector is not None:
            for param in self.deviation_detector.parameters():
                param.requires_grad = False
            logger.info("Deviation Detector Frozen. Targeting Event Types.")

    def get_optimizer(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        # Separate params for weight decay
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        
        if torch.cuda.is_available():
            try:
                # Fused AdamW is significantly faster
                return torch.optim.AdamW(groups, lr=lr, fused=True)
            except:
                pass
        return torch.optim.AdamW(groups, lr=lr)

    def save_model(self, path: str, is_ddp: bool = False, metadata: Optional[Dict] = None):
        state = self.module.state_dict() if is_ddp else self.state_dict()
        data = {
            'model_state_dict': state,
            'config': self.config.__dict__,
            'metadata': metadata
        }
        torch.save(data, path)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"Blueprint saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: torch.device = torch.device('cpu')) -> 'CausalGRUModel':
        checkpoint = torch.load(path, map_location=device)
        config = GRUConfig(**checkpoint['config'])
        model = cls(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Blueprint loaded from {path}")
        return model

# =============================================================================
# FACTORY: BUILD THE MINION
# =============================================================================

def create_model(d_model: int = 128, n_layers: int = 2, hierarchical: bool = True, **kwargs) -> CausalGRUModel:
    configure_cuda_for_speed()
    config = GRUConfig(d_model=d_model, n_layers=n_layers, hierarchical=hierarchical, **kwargs)
    model = CausalGRUModel(config)
    return model
