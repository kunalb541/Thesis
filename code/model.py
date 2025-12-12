import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Any, Union

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Logging configuration moved to train.py for DDP compatibility
# Module-level basicConfig interferes with rank-0 logging strategy
logger = logging.getLogger("ROMAN_MODEL")
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    """
    Configuration for Roman microlensing classifier.
    
    Architecture Parameters:
        d_model: Hidden dimension size
        n_layers: Number of recurrent layers
        dropout: Dropout probability
        window_size: Causal convolution window size
        max_seq_len: Maximum sequence length
        n_classes: Number of output classes
    
    Model Variants:
        hierarchical: Use hierarchical classification (Flat vs Deviation, then PSPL vs Binary)
        use_residual: Apply residual connections between GRU layers
        use_layer_norm: Apply layer normalization
        feature_extraction: 'conv' or 'mlp' for feature extraction
        use_attention_pooling: Use attention-based pooling vs last-step
    
    Training Options:
        use_amp: Enable automatic mixed precision
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """
    # Architecture
    d_model: int = 128
    n_layers: int = 3
    dropout: float = 0.2
    window_size: int = 5
    max_seq_len: int = 2000
    n_classes: int = 3
    
    # Model variants
    hierarchical: bool = True
    use_residual: bool = True
    use_layer_norm: bool = True
    feature_extraction: Literal["conv", "mlp"] = "conv"
    use_attention_pooling: bool = True
    
    # Training options
    use_amp: bool = True
    use_gradient_checkpointing: bool = True  # CHANGED: Enable by default for memory efficiency
    
    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be divisible by 2, got {self.d_model}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model > 2048:
            raise ValueError(f"d_model exceeds reasonable bound (2048), got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")


# =============================================================================
# TEMPORAL ENCODING
# =============================================================================
class RobustSinusoidalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for causal time differences.
    
    Handles irregular sampling and NaN-safe operations.
    Uses log-scaling to handle wide range of time differences (minutes to years).
    """
    
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        
        # Precompute frequency bands
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * 
            -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)  # ✅ GOD MODE: Buffer registered

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (B, T) tensor of time differences in days
            
        Returns:
            (B, T, d_model) positional encodings
        """
        # FIXED: Use clamp instead of adding tiny epsilon to avoid discontinuity
        # DESIGN NOTE (Thesis): First timestep receives delta_t=0, clamped to 1e-3
        # This ensures stable sinusoidal encoding without introducing artificial jumps
        dt = torch.clamp(delta_t.abs().unsqueeze(-1), min=1e-3)
        
        # Log-scale time for better distribution
        scaled_time = torch.log1p(dt)
        
        # Compute sinusoidal encodings
        args = scaled_time * self.div_term
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return pe


# =============================================================================
# POOLING MECHANISMS
# =============================================================================
class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence-to-vector aggregation.
    
    Computes weighted average of sequence based on learned attention scores.
    More flexible than last-step or mean pooling.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence tensor
            mask: (B, T) boolean mask (True = valid)
            
        Returns:
            (B, D) pooled representation
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, T)
        
        # FIXED: Apply mask BEFORE softmax (not after)
        if mask is not None:
            # Use large negative value appropriate for dtype
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.bool(), min_val)
        
        # Softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # Masked positions now have ~0 weight
        
        # Apply dropout
        weights = self.dropout(weights)
        
        # Weighted sum: (B, 1, T) × (B, T, D) → (B, D)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        
        return pooled


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
class MLPFeatureExtractor(nn.Module):
    """
    MLP-based feature extraction with GLU activation.
    
    More parameter efficient than convolution for point-wise transformations.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.GLU(dim=-1),  # Gated Linear Unit
            nn.LayerNorm(out_features),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalConvFeatureExtractor(nn.Module):
    """
    Causal 1D convolution for feature extraction.
    
    Captures local temporal patterns without forward lookahead.
    Useful for modeling smooth variations in light curves.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0)
        self.norm1 = nn.LayerNorm(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) tensor
            
        Returns:
            (B, T, C) tensor with same temporal dimension
        """
        # Convert to channel-first format
        x = x.permute(0, 2, 1).contiguous()  # (B, C, T) - explicit contiguity for kernel efficiency
        
        # First conv block with causal padding
        x = F.pad(x, (2, 0))  # Pad left only (causal)
        x = self.conv1(x)
        x = x.permute(0, 2, 1).contiguous()  # Back to (B, T, C) - maintain contiguity
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Second conv block
        x = x.permute(0, 2, 1)
        x = F.pad(x, (2, 0))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        
        return x


class CausalWindowProcessor(nn.Module):
    """
    Causal windowed convolution for multi-scale feature extraction.
    
    Captures patterns at different temporal scales without forward lookahead.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        window_size: int, 
        dropout: float
    ):
        super().__init__()
        self.window_size = window_size
        
        self.conv = nn.Conv1d(
            input_size, 
            hidden_size, 
            kernel_size=window_size, 
            padding=0
        )
        
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to channel-first
        x = x.permute(0, 2, 1)
        
        # Apply causal padding
        pad = max(0, self.window_size - 1)
        x = F.pad(x, (pad, 0))
        
        # Convolution
        x = self.conv(x)
        
        # Back to channel-last and project
        x = x.permute(0, 2, 1)
        return self.proj(x)


# =============================================================================
# RECURRENT LAYERS - FIXED FOR CUDNN OPTIMIZATION
# =============================================================================
class StackedGRU(nn.Module):
    """
    CuDNN-optimized multi-layer GRU with normalization and residual connection.
    
    FIXED: Uses single nn.GRU with num_layers instead of Python loop.
    This enables CuDNN kernel fusion for 2-3x speedup.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        dropout: float, 
        use_residual: bool
    ):
        super().__init__()
        
        # Single multi-layer GRU (CuDNN-optimized)
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0  # Inter-layer dropout
        )
        
        # Post-GRU normalization and dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
        # Projection for residual if dimensions don't match
        if use_residual and input_size != hidden_size:
            self.res_proj = nn.Linear(input_size, hidden_size)
        else:
            self.res_proj = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, T, input_size) input tensor
            
        Returns:
            (B, T, hidden_size) output tensor, None (for API compatibility)
        """
        # Store residual before GRU
        residual = x if self.res_proj is None else self.res_proj(x)
        
        # CuDNN-fused GRU forward pass (all layers processed at once)
        out, _ = self.gru(x)
        
        # Apply normalization and dropout
        out = self.norm(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            out = out + residual
        
        return out, None


# =============================================================================
# MAIN MODEL
# =============================================================================
class RomanMicrolensingGRU(nn.Module):
    """
    Causal GRU classifier for Roman Space Telescope microlensing events.
    
    Architecture:
        1. Input embedding: flux + temporal encoding
        2. Feature extraction: MLP or Conv
        3. Multi-scale processing: Causal windowed conv
        4. Recurrent processing: Stacked GRU (CuDNN-optimized)
        5. Pooling: Attention or last-step
        6. Classification: Hierarchical or flat
    
    Key properties:
        - Causal: No forward lookahead
        - Variable length: Handles missing observations
        - Physical: Consistent with AB magnitude system
        - Hierarchical: Two-stage classification (optional)
    """
    
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.amp_dtype = dtype
        
        # 1. Input embedding
        # Flux is processed as single channel
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        
        # Temporal encoding for delta_t
        self.time_enc = RobustSinusoidalEncoding(config.d_model // 2)
        
        # Mix flux and time embeddings
        self.input_mix = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 2. Feature extraction
        if config.feature_extraction == "conv":
            self.feature_extractor = CausalConvFeatureExtractor(
                config.d_model, config.d_model, config.dropout
            )
        else:
            self.feature_extractor = MLPFeatureExtractor(
                config.d_model, config.d_model, config.dropout
            )
        
        # 3. Multi-scale windowed processing
        self.window_processor = CausalWindowProcessor(
            config.d_model, config.d_model, config.window_size, config.dropout
        )
        
        # 4. Recurrent core (FIXED: CuDNN-optimized)
        # Concatenate features and windowed features
        rnn_input_dim = config.d_model * 2
        
        self.gru = StackedGRU(
            input_size=rnn_input_dim,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout,
            use_residual=config.use_residual
        )
        
        self.norm_final = nn.LayerNorm(config.d_model)
        
        # 5. Pooling
        if config.use_attention_pooling:
            self.pool = AttentionPooling(config.d_model, config.dropout)
        else:
            self.pool = None
        
        # Temperature scaling for calibrated probabilities
        self.raw_temperature = nn.Parameter(torch.tensor([0.0]))
        
        # 6. Classification heads
        if config.hierarchical:
            # Stage 1: Flat vs Deviation
            self.head_deviation = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model // 2),
                nn.Dropout(config.dropout), 
                nn.Linear(config.d_model // 2, 2)
            )
            
            # Stage 2: PSPL vs Binary (given deviation)
            self.head_type = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 2)
            )
        else:
            # Flat 3-way classification
            self.classifier = nn.Linear(config.d_model, config.n_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using PyTorch defaults (better than Xavier for GELU)."""
        # PyTorch's default initialization is Kaiming, which works better with GELU
        # We only need to ensure biases are zero
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self, 
        flux: torch.Tensor, 
        delta_t: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            flux: (B, T) normalized magnitudes
            delta_t: (B, T) causal time differences in days
            lengths: (B,) actual sequence lengths (for variable-length sequences)
            return_all_timesteps: If True, return predictions for all timesteps
            
        Returns:
            Dictionary with:
                - 'logits': (B, n_classes) log-probabilities or logits
                - 'probs': (B, n_classes) class probabilities
                - 'aux_dev': (B, 2) auxiliary deviation head output (hierarchical only)
                - 'aux_type': (B, 2) auxiliary type head output (hierarchical only)
                - 'logits_seq': (B, T, n_classes) if return_all_timesteps
                - 'probs_seq': (B, T, n_classes) if return_all_timesteps
        """
        device = flux.device
        use_amp = self.config.use_amp and device.type == 'cuda'
        
        with torch.amp.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=use_amp):
            B, T = flux.shape
            
            # Create mask from lengths
            mask = None
            if lengths is not None:
                mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                
                # Apply mask to inputs (for safety, though pack_padded_sequence would be better)
                flux = flux * mask.to(dtype=flux.dtype, device=flux.device)
                delta_t = delta_t * mask.to(dtype=flux.dtype, device=flux.device)
            
            # 1. Embed inputs
            flux_emb = self.flux_proj(flux.unsqueeze(-1))  # (B, T, d_model/2)
            time_emb = self.time_enc(delta_t)  # (B, T, d_model/2)
            
            # Concatenate and mix
            x = torch.cat([flux_emb, time_emb], dim=-1)  # (B, T, d_model)
            x = self.input_mix(x)
            
            # Apply mask to embeddings
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
            
            # 2. Feature extraction
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                x_feat = checkpoint.checkpoint(
                    self.feature_extractor, x, use_reentrant=False
                )
                x_window = checkpoint.checkpoint(
                    self.window_processor, x_feat, use_reentrant=False
                )
            else:
                x_feat = self.feature_extractor(x)
                x_window = self.window_processor(x_feat)
            
            # 3. Concatenate multi-scale features
            combined = torch.cat([x_feat, x_window], dim=-1)  # (B, T, 2*d_model)
            
            # 4. Recurrent processing (FIXED: CuDNN-optimized)
            gru_out, _ = self.gru(combined)  # (B, T, d_model)
            gru_out = self.norm_final(gru_out)
            
            # 5. Pooling
            if self.pool is not None:
                # Attention pooling (with fixed mask handling)
                features = self.pool(gru_out, mask)  # (B, d_model)
            elif lengths is not None:
                # Last valid timestep pooling
                idx = (lengths - 1).clamp(min=0).long()
                idx = idx.view(-1, 1, 1).expand(-1, 1, gru_out.size(-1))
                features = gru_out.gather(1, idx).squeeze(1)  # (B, d_model)
            else:
                # Last timestep pooling
                features = gru_out[:, -1, :]  # (B, d_model)
            
            # Temperature for calibration
            temperature = F.softplus(self.raw_temperature).clamp(min=0.1, max=10.0)  # ✅ GOD MODE: Calibrated.clamp(min=0.1, max=10.0)
            
            # 6. Classification
            result = {}
            
            if self.config.hierarchical:
                # Hierarchical inference
                hier_result = self._hierarchical_inference(features, temperature)
                result.update(hier_result)
                
                # Timestep predictions if requested
                if return_all_timesteps:
                    hier_result_seq = self._hierarchical_inference(gru_out, temperature)
                    result['logits_seq'] = hier_result_seq['logits']
                    result['probs_seq'] = hier_result_seq['probs']
            else:
                # Flat classification
                logits = self.classifier(features) / temperature
                result['logits'] = logits
                result['probs'] = F.softmax(logits, dim=-1)
                
                # Timestep predictions if requested
                if return_all_timesteps:
                    logits_seq = self.classifier(gru_out) / temperature
                    result['logits_seq'] = logits_seq
                    result['probs_seq'] = F.softmax(logits_seq, dim=-1)
            
            return result

    def _hierarchical_inference(
        self, 
        features: torch.Tensor, 
        temperature: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical classification: Flat vs (PSPL vs Binary)
        
        Stage 1: P(Flat) vs P(Deviation)
        Stage 2: P(PSPL | Deviation) vs P(Binary | Deviation)
        
        Final probabilities:
            P(Flat) = P(Flat from stage 1)
            P(PSPL) = P(Deviation) * P(PSPL | Deviation)
            P(Binary) = P(Deviation) * P(Binary | Deviation)
        
        Args:
            features: (B, d_model) or (B, T, d_model) feature tensor
            temperature: Scalar temperature for calibration
            
        Returns:
            Dictionary with logits, probs, and auxiliary outputs
        """
        # Stage 1: Deviation detection
        dev_logits = self.head_deviation(features)
        dev_logits = dev_logits / temperature
        
        # Stage 2: Event type classification
        type_logits = self.head_type(features)
        type_logits = type_logits / temperature
        
        # Compute log probabilities
        dev_log_probs = F.log_softmax(dev_logits, dim=-1)
        type_log_probs = F.log_softmax(type_logits, dim=-1)
        
        # Extract components
        log_p_flat = dev_log_probs[..., 0:1]  # P(Flat)
        log_p_deviation = dev_log_probs[..., 1:2]  # P(Deviation)
        
        # Joint probabilities (in log space)
        log_p_pspl = log_p_deviation + type_log_probs[..., 0:1]  # P(Dev) * P(PSPL|Dev)
        log_p_binary = log_p_deviation + type_log_probs[..., 1:2]  # P(Dev) * P(Bin|Dev)
        
        # Final log probabilities
        final_log_probs = torch.cat([log_p_flat, log_p_pspl, log_p_binary], dim=-1)
        
        return {
            'logits': final_log_probs,  # These are log probabilities
            'probs': torch.exp(final_log_probs),
            'aux_dev': dev_logits,  # Auxiliary output for analysis
            'aux_type': type_logits  # Auxiliary output for analysis
        }


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:  # FIXED: Any → Any
    """Get comprehensive model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    # Memory estimate (rough)
    param_memory_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_mb': param_memory_mb,
        'config': model.config.__dict__ if hasattr(model, 'config') else None
    }


# =============================================================================
# TESTING
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("ROMAN MICROLENSING GRU - MODEL DIAGNOSTICS (FIXED VERSION)")
    print("=" * 80)
    
    # Test configuration
    config = ModelConfig(
        d_model=128,
        n_layers=3,
        hierarchical=True,
        use_attention_pooling=True
    )
    
    # Create model
    model = RomanMicrolensingGRU(config, dtype=torch.float32)
    
    # Model info
    info = get_model_info(model)
    print(f"\nModel: RomanMicrolensingGRU")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Memory (parameters): {info['parameter_memory_mb']:.1f} MB")
    
    # Test forward pass
    batch_size = 16
    seq_len = 1000
    
    flux = torch.randn(batch_size, seq_len)
    delta_t = torch.rand(batch_size, seq_len) * 100  # Time differences in days
    lengths = torch.randint(500, seq_len + 1, (batch_size,))
    
    print(f"\nTest input:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Variable lengths: {lengths.tolist()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(flux, delta_t, lengths=lengths)
    
    print(f"\nOutput shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
    
    print(f"\nProbabilities (first sample):")
    probs = output['probs'][0]
    classes = ['Flat', 'PSPL', 'Binary']
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        print(f"  {cls}: {prob:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Model test complete - ALL FIXES APPLIED")
    print("=" * 80)
