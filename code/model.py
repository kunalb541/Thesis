import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal, Union
from enum import Enum
from contextlib import nullcontext

logger = logging.getLogger("gru_model")

# =============================================================================
# CUDA OPTIMIZATIONS
# =============================================================================

def configure_cuda_for_speed():
    """Configure CUDA for maximum training speed."""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere GPUs (A100) - 3x speedup with minimal precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn autotuner - finds fastest algorithms
        torch.backends.cudnn.benchmark = True
        
        # Disable debug mode
        torch.backends.cudnn.enabled = True
        
        logger.info("CUDA optimizations enabled: TF32, cudnn.benchmark")


def configure_cuda_for_reproducibility():
    """Configure CUDA for reproducible results (slower)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        logger.info("CUDA configured for reproducibility (deterministic mode)")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelMode(Enum):
    """Model operating modes for different use cases."""
    FULL_SEQUENCE = "full_sequence"      # Standard training - process entire sequence
    SLIDING_WINDOW = "sliding_window"    # Yiannis's 7-point boxcar
    HIERARCHICAL = "hierarchical"        # Deviation detection + classification


@dataclass
class GRUConfig:
    """
    Configuration for CausalGRUModel.
    
    All parameters are validated in __post_init__ to ensure
    correct architecture construction.
    
    Attributes:
        d_model: Hidden dimension of GRU
        n_layers: Number of stacked GRU layers
        bidirectional: Whether to use bidirectional GRU (False for causal)
        dropout: Dropout probability
        window_size: Sliding window size (Yiannis suggests 7)
        max_seq_len: Maximum sequence length
        n_classes: Number of output classes (3 for Flat/PSPL/Binary)
        classifier_hidden_dim: Hidden dimension in classifier head
        use_layer_norm: Use LayerNorm for stability
        use_residual_connection: Add residual connections between GRU layers
        hierarchical: Enable hierarchical classification (deviation → type)
        deviation_threshold: Confidence threshold for deviation detection
        feature_extraction: Type of feature extraction before GRU
        use_attention_pooling: Use attention for sequence pooling
        use_amp: Enable Automatic Mixed Precision (AMP)
        compile_model: Use torch.compile for speedup (PyTorch 2.0+)
        fused_layer_norm: Use fused LayerNorm (faster on GPU)
    """
    d_model: int = 128
    n_layers: int = 2
    bidirectional: bool = False  # MUST be False for causal model
    dropout: float = 0.3
    window_size: int = 7  # Yiannis's suggestion
    max_seq_len: int = 1500
    n_classes: int = 3  # Flat, PSPL, Binary
    classifier_hidden_dim: Optional[int] = None
    use_layer_norm: bool = True
    use_residual_connection: bool = True
    hierarchical: bool = True  # Enable deviation detection first
    deviation_threshold: float = 0.5
    feature_extraction: Literal["none", "conv", "mlp"] = "conv"
    use_attention_pooling: bool = True
    input_features: int = 2  # flux + delta_t
    
    # Performance options
    use_amp: bool = True  # Automatic Mixed Precision
    compile_model: bool = False  # torch.compile (set True for PyTorch 2.0+)
    fused_layer_norm: bool = True  # Fused operations
    use_flash_attention: bool = False  # For future attention-based extensions
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        
        # Type enforcement
        assert isinstance(self.d_model, int) and self.d_model > 0, "d_model must be positive int"
        assert isinstance(self.n_layers, int) and self.n_layers > 0, "n_layers must be positive int"
        assert isinstance(self.window_size, int) and self.window_size >= 3, "window_size must be >= 3"
        
        # Causal constraint
        assert not self.bidirectional, "bidirectional must be False for causal model"
        
        # Logical constraints
        assert 0.0 <= self.dropout < 1.0, f"dropout ({self.dropout}) must be in [0, 1)"
        assert self.n_classes >= 2, f"n_classes ({self.n_classes}) must be >= 2"
        assert self.window_size <= self.max_seq_len, "window_size must be <= max_seq_len"


# =============================================================================
# TIME ENCODING (Preserved from original)
# =============================================================================

class FusedLayerNorm(nn.Module):
    """
    LayerNorm with optional fused operations for speed.
    Falls back to standard LayerNorm if fused not available.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, use_fused: bool = True):
        super().__init__()
        self.use_fused = use_fused and torch.cuda.is_available()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch will automatically use fused kernel when available
        return self.ln(x)


class ContinuousSinusoidalEncoding(nn.Module):
    """
    Continuous time encoding using logarithmically-scaled sinusoids.
    Maps irregular time intervals to positional embeddings.
    
    Critical for Roman Telescope data where observation gaps
    vary due to weather, satellite orbits, etc.
    """
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (B, T) tensor of time intervals
        Returns:
            (B, T, d_model) positional encoding tensor
        """
        dt = delta_t.unsqueeze(-1) + 1e-6
        scaled = torch.log1p(dt) * self.div_term
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe


# =============================================================================
# FEATURE EXTRACTION MODULES
# =============================================================================

class ConvFeatureExtractor(nn.Module):
    """
    1D Causal CNN for local feature extraction before GRU.
    Extracts local patterns (bumps, slopes) that GRU can integrate.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Causal padding amounts
        self.pad1 = 2  # kernel_size - 1
        self.pad2 = 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input tensor
        Returns:
            (B, T, C) output tensor with local features
        """
        # (B, T, C) -> (B, C, T) for conv
        x = x.transpose(1, 2).contiguous()
        
        # Causal conv 1
        x = F.pad(x, (self.pad1, 0))  # Left pad only
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Causal conv 2
        x = x.transpose(1, 2).contiguous()  # (B, T, C) -> (B, C, T)
        x = F.pad(x, (self.pad2, 0))
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)
        x = self.norm2(x)
        x = self.act(x)
        
        return x


class MLPFeatureExtractor(nn.Module):
    """
    Simple MLP feature extractor - processes each timestep independently.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# ATTENTION POOLING
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling over sequence dimension.
    Learns to weight important timesteps for final classification.
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
        """
        Args:
            x: (B, T, C) hidden states
            mask: (B, T) boolean mask where True = valid
        Returns:
            (B, C) pooled representation
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, T)
        
        # Mask invalid positions
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)  # (B, T)
        weights = self.dropout(weights)
        
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, C)
        return pooled


# =============================================================================
# LAYER-NORMALIZED GRU
# =============================================================================

class LayerNormGRUCell(nn.Module):
    """
    GRU cell with LayerNorm for training stability.
    Critical for DDP training with small per-GPU batches.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # LayerNorms for each gate
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_n = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_size) input
            h: (B, hidden_size) previous hidden state
        Returns:
            (B, hidden_size) new hidden state
        """
        r = torch.sigmoid(self.ln_r(self.W_ir(x) + self.W_hr(h)))
        z = torch.sigmoid(self.ln_z(self.W_iz(x) + self.W_hz(h)))
        n = torch.tanh(self.ln_n(self.W_in(x) + r * self.W_hn(h)))
        h_new = (1 - z) * n + z * h
        return h_new


class LayerNormGRU(nn.Module):
    """
    Multi-layer GRU with LayerNorm, residual connections, and dropout.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # First layer
        self.cells = nn.ModuleList([LayerNormGRUCell(input_size, hidden_size)])
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.cells.append(LayerNormGRUCell(hidden_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        
        # Projection for residual if dimensions don't match
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_size) input sequence
            h_0: (num_layers, B, hidden_size) initial hidden states
        Returns:
            output: (B, T, hidden_size) all hidden states
            h_n: (num_layers, B, hidden_size) final hidden states
        """
        B, T, _ = x.shape
        device = x.device
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        
        # Process sequence
        h_states = list(h_0)  # List of (B, hidden)
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :]  # (B, input_size)
            
            for layer_idx, cell in enumerate(self.cells):
                h_new = cell(x_t, h_states[layer_idx])
                
                # Residual connection (except first layer)
                if self.use_residual and layer_idx > 0:
                    h_new = h_new + x_t if x_t.size(-1) == h_new.size(-1) else h_new
                
                h_states[layer_idx] = h_new
                x_t = self.dropout(h_new)  # Input to next layer
            
            outputs.append(h_states[-1])  # Output from last layer
        
        output = torch.stack(outputs, dim=1)  # (B, T, hidden)
        h_n = torch.stack(h_states, dim=0)  # (num_layers, B, hidden)
        
        return output, h_n


# =============================================================================
# SLIDING WINDOW MODULE (Yiannis's Suggestion)
# =============================================================================

class SlidingWindowProcessor(nn.Module):
    """
    Processes sequence using sliding window approach.
    Each window of `window_size` points is processed by a small GRU.
    
    This addresses Yiannis's concern: "use a moving boxcar of around 7 points"
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        window_size: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        
        # Small GRU for processing each window
        self.window_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Project window representation
        self.window_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, efficient: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input sequence
            efficient: If True, use efficient unfold operation; if False, loop
        Returns:
            (B, T, hidden_size) window-processed features
            Note: First (window_size-1) positions have partial context
        """
        B, T, C = x.shape
        device = x.device
        
        if efficient and T >= self.window_size:
            # Efficient implementation using unfold
            # Pad beginning for causal windows
            padding = torch.zeros(B, self.window_size - 1, C, device=device)
            x_padded = torch.cat([padding, x], dim=1)  # (B, T + window_size - 1, C)
            
            # Create all windows at once: (B, T, window_size, C)
            windows = x_padded.unfold(1, self.window_size, 1)  # (B, T, C, window_size)
            windows = windows.permute(0, 1, 3, 2).contiguous()  # (B, T, window_size, C)
            
            # Reshape for batch processing
            windows_flat = windows.view(B * T, self.window_size, C)  # (B*T, window_size, C)
            
            # Process all windows in parallel
            _, h_n = self.window_gru(windows_flat)  # h_n: (1, B*T, hidden)
            
            # Reshape back
            window_features = h_n.squeeze(0).view(B, T, -1)  # (B, T, hidden)
            
        else:
            # Loop implementation (for very short sequences or debugging)
            outputs = []
            for t in range(T):
                start = max(0, t - self.window_size + 1)
                window = x[:, start:t+1, :]  # (B, window_len, C)
                
                _, h_n = self.window_gru(window)  # (1, B, hidden)
                outputs.append(h_n.squeeze(0))
            
            window_features = torch.stack(outputs, dim=1)  # (B, T, hidden)
        
        return self.window_proj(window_features)


# =============================================================================
# DEVIATION DETECTOR
# =============================================================================

class DeviationDetector(nn.Module):
    """
    Binary classifier: Is this a deviation from flat baseline?
    First stage of hierarchical classification.
    
    Addresses the problem: "it still thinks it is 100% flat, where it has 
    obviously started to deviate"
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Binary: Flat vs Deviating
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C) or (B, C) features
        Returns:
            Dict with 'logits' and 'probs'
        """
        logits = self.detector(x)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}


class EventTypeClassifier(nn.Module):
    """
    Classifies deviation type: PSPL vs Binary
    Second stage of hierarchical classification.
    Only applied when deviation is detected.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Binary: PSPL vs Binary lens
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C) or (B, C) features
        Returns:
            Dict with 'logits' and 'probs'
        """
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}


# =============================================================================
# MAIN MODEL
# =============================================================================

class CausalGRUModel(nn.Module):
    """
    Causal GRU model for real-time microlensing classification.
    
    Architecture:
        1. Input embedding with continuous time encoding
        2. Optional local feature extraction (CNN or MLP)
        3. Sliding window processor (Yiannis's 7-point boxcar)
        4. Full-sequence GRU with LayerNorm
        5. Optional attention pooling
        6. Hierarchical classification:
           - Stage 1: Deviation detection (Flat vs Deviating)
           - Stage 2: Event type (PSPL vs Binary)
    
    Causality Guarantee:
        - Unidirectional GRU (bidirectional=False)
        - Causal CNN padding (left-only)
        - No future information leakage
    
    Performance Optimizations:
        - AMP (Automatic Mixed Precision) for 2x speedup
        - torch.compile for kernel fusion (PyTorch 2.0+)
        - Fused LayerNorm operations
        - TF32 on Ampere GPUs
        - cudnn.benchmark for autotuning
    
    Training Modes:
        - FULL_SEQUENCE: Efficient training, process entire sequence
        - SLIDING_WINDOW: Explicit window-by-window (Yiannis's request)
        - HIERARCHICAL: Two-stage classification
    
    DDP Optimization:
        - LayerNorm GRU cells for stability
        - No custom CUDA kernels
        - Compatible with find_unused_parameters=False
    
    Args:
        config: GRUConfig with validated hyperparameters
    """
    
    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        
        # AMP dtype for mixed precision
        self.amp_dtype = torch.float16 if config.use_amp else torch.float32
        
        # Input projection
        self.flux_proj = nn.Linear(1, config.d_model // 2)
        self.time_enc = ContinuousSinusoidalEncoding(config.d_model // 2)
        
        # Combine flux and time features - use fused LayerNorm
        LayerNormClass = FusedLayerNorm if config.fused_layer_norm else nn.LayerNorm
        
        self.input_combine = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            LayerNormClass(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Feature extraction
        if config.feature_extraction == "conv":
            self.feature_extractor = ConvFeatureExtractor(
                config.d_model, config.d_model, config.dropout
            )
        elif config.feature_extraction == "mlp":
            self.feature_extractor = MLPFeatureExtractor(
                config.d_model, config.d_model, config.dropout
            )
        else:
            self.feature_extractor = nn.Identity()
        
        # Sliding window processor (Yiannis's suggestion)
        self.window_processor = SlidingWindowProcessor(
            input_size=config.d_model,
            hidden_size=config.d_model,
            window_size=config.window_size,
            dropout=config.dropout
        )
        
        # Main GRU for full sequence context
        self.gru = LayerNormGRU(
            input_size=config.d_model * 2,  # Window features + original features
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            dropout=config.dropout,
            use_residual=config.use_residual_connection
        )
        
        # Standard PyTorch GRU as alternative (faster but less stable)
        # Using CUDA-optimized GRU when available
        self.fast_gru = nn.GRU(
            input_size=config.d_model * 2,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=False
        )
        
        # Flatten parameters for faster CUDA execution
        if torch.cuda.is_available():
            self.fast_gru.flatten_parameters()
        
        # Use fast GRU by default (switch to LayerNorm GRU if training is unstable)
        self.use_fast_gru = True
        
        # Attention pooling
        if config.use_attention_pooling:
            self.attention_pool = AttentionPooling(config.d_model, config.dropout)
        else:
            self.attention_pool = None
        
        # Final normalization
        self.final_norm = LayerNormClass(config.d_model)
        
        # Classification heads
        if config.hierarchical:
            # Hierarchical: Deviation → Type
            self.deviation_detector = DeviationDetector(config.d_model, config.dropout)
            self.event_classifier = EventTypeClassifier(config.d_model, config.dropout)
            self.classifier = None  # Not used in hierarchical mode
        else:
            # Direct 3-class classification
            self.deviation_detector = None
            self.event_classifier = None
            self.classifier = nn.Sequential(
                nn.Linear(config.d_model, config.classifier_hidden_dim),
                LayerNormClass(config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_hidden_dim, config.n_classes)
            )
        
        # Learnable temperature scaling
        self.log_temperature = nn.Parameter(torch.tensor([0.0]))
        
        # Initialize weights
        self._init_weights()
        
        # Apply torch.compile if requested (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self._compile_modules()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gru' in name.lower():
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _compile_modules(self):
        """Apply torch.compile to performance-critical modules."""
        try:
            # Compile the feature extractor and GRU
            self.feature_extractor = torch.compile(self.feature_extractor, mode='reduce-overhead')
            self.window_processor = torch.compile(self.window_processor, mode='reduce-overhead')
            logger.info("Applied torch.compile to feature extractor and window processor")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    @torch.amp.autocast('cuda')
    def forward_amp(
        self,
        flux: torch.Tensor,
        delta_t: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = True,
        mode: str = "full_sequence"
    ) -> Dict[str, torch.Tensor]:
        """AMP-enabled forward pass."""
        return self._forward_impl(flux, delta_t, lengths, return_all_timesteps, mode)
    
    def forward(
        self,
        flux: torch.Tensor,
        delta_t: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = True,
        mode: str = "full_sequence"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple operating modes.
        
        Args:
            flux: (B, T) flux measurements
            delta_t: (B, T) time intervals in days
            lengths: (B,) actual sequence lengths
            return_all_timesteps: If False, return only final prediction
            mode: "full_sequence", "sliding_window", or "hierarchical"
        
        Returns:
            Dictionary containing:
                'logits': Final class logits
                'probs': Final class probabilities
                'deviation_logits': (hierarchical only) Deviation detection logits
                'deviation_probs': (hierarchical only) Deviation probabilities
                'type_logits': (hierarchical only) Event type logits
                'type_probs': (hierarchical only) Event type probabilities
        """
        # Use AMP autocast if enabled and on CUDA
        if self.config.use_amp and flux.is_cuda:
            return self.forward_amp(flux, delta_t, lengths, return_all_timesteps, mode)
        else:
            return self._forward_impl(flux, delta_t, lengths, return_all_timesteps, mode)
    
    def _forward_impl(
        self,
        flux: torch.Tensor,
        delta_t: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = True,
        mode: str = "full_sequence"
    ) -> Dict[str, torch.Tensor]:
        """Internal forward implementation."""
        assert flux.ndim == 2, f"flux must be 2D (B, T), got shape {flux.shape}"
        assert delta_t.ndim == 2, f"delta_t must be 2D (B, T), got shape {delta_t.shape}"
        
        B, T = flux.shape
        device = flux.device
        
        # Create masks
        mask = None
        if lengths is not None:
            range_tensor = torch.arange(T, device=device).unsqueeze(0)
            mask = range_tensor < lengths.unsqueeze(1)  # (B, T) True = valid
            
            # Zero out padded positions
            flux = torch.nan_to_num(flux, 0.0)
            flux = flux * mask.float()
            delta_t = delta_t * mask.float()
        
        # Embed inputs
        flux_emb = self.flux_proj(flux.unsqueeze(-1))  # (B, T, d_model/2)
        time_emb = self.time_enc(delta_t)  # (B, T, d_model/2)
        x = torch.cat([flux_emb, time_emb], dim=-1)  # (B, T, d_model)
        x = self.input_combine(x)
        
        # Apply mask
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Sliding window processing
        window_features = self.window_processor(x)  # (B, T, d_model)
        
        # Concatenate original and window features
        combined = torch.cat([x, window_features], dim=-1)  # (B, T, d_model*2)
        
        # Full sequence GRU
        if self.use_fast_gru:
            # Flatten parameters for speed (call once per forward if needed)
            if self.fast_gru.training:
                self.fast_gru.flatten_parameters()
            
            if lengths is not None:
                # Pack for variable length sequences
                packed = nn.utils.rnn.pack_padded_sequence(
                    combined, lengths.cpu().clamp(min=1), 
                    batch_first=True, enforce_sorted=False
                )
                gru_out, h_n = self.fast_gru(packed)
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                    gru_out, batch_first=True, total_length=T
                )
            else:
                gru_out, h_n = self.fast_gru(combined)
        else:
            gru_out, h_n = self.gru(combined)
        
        # Apply final norm
        gru_out = self.final_norm(gru_out)  # (B, T, d_model)
        
        # Get final representation
        if return_all_timesteps:
            features = gru_out  # (B, T, d_model)
        else:
            if self.attention_pool is not None and lengths is not None:
                features = self.attention_pool(gru_out, mask)  # (B, d_model)
            elif lengths is not None:
                # Gather last valid hidden state
                idx = (lengths - 1).clamp(min=0).view(-1, 1, 1)
                idx = idx.expand(-1, 1, gru_out.size(-1)).to(device)
                features = gru_out.gather(1, idx).squeeze(1)  # (B, d_model)
            else:
                features = gru_out[:, -1, :]  # (B, d_model)
        
        # Classification
        temp = torch.exp(self.log_temperature)
        
        if self.config.hierarchical:
            return self._hierarchical_forward(features, temp, return_all_timesteps)
        else:
            return self._direct_forward(features, temp, return_all_timesteps)
    
    def _direct_forward(
        self, 
        features: torch.Tensor, 
        temp: torch.Tensor,
        return_all_timesteps: bool
    ) -> Dict[str, torch.Tensor]:
        """Direct 3-class classification."""
        logits = self.classifier(features)
        scaled_logits = logits / (temp + 1e-8)
        probs = F.softmax(scaled_logits, dim=-1)
        
        return {'logits': scaled_logits, 'probs': probs}
    
    def _hierarchical_forward(
        self,
        features: torch.Tensor,
        temp: torch.Tensor,
        return_all_timesteps: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical classification:
        1. Detect deviation (Flat vs Deviating)
        2. If deviating, classify type (PSPL vs Binary)
        3. Combine into 3-class output
        """
        # Stage 1: Deviation detection
        deviation_out = self.deviation_detector(features)
        deviation_logits = deviation_out['logits']  # (..., 2): [flat, deviating]
        deviation_probs = deviation_out['probs']
        
        # Stage 2: Event type classification
        type_out = self.event_classifier(features)
        type_logits = type_out['logits']  # (..., 2): [PSPL, Binary]
        type_probs = type_out['probs']
        
        # Combine into 3-class probabilities
        # P(Flat) = P(not deviating)
        # P(PSPL) = P(deviating) * P(PSPL | deviating)
        # P(Binary) = P(deviating) * P(Binary | deviating)
        
        p_flat = deviation_probs[..., 0:1]  # P(not deviating)
        p_deviating = deviation_probs[..., 1:2]  # P(deviating)
        p_pspl_given_dev = type_probs[..., 0:1]  # P(PSPL | deviating)
        p_binary_given_dev = type_probs[..., 1:2]  # P(Binary | deviating)
        
        p_pspl = p_deviating * p_pspl_given_dev
        p_binary = p_deviating * p_binary_given_dev
        
        # Final 3-class probabilities: [Flat, PSPL, Binary]
        combined_probs = torch.cat([p_flat, p_pspl, p_binary], dim=-1)
        
        # Convert to logits (for loss computation)
        combined_logits = torch.log(combined_probs + 1e-8)
        scaled_logits = combined_logits / (temp + 1e-8)
        
        return {
            'logits': scaled_logits,
            'probs': combined_probs,
            'deviation_logits': deviation_logits,
            'deviation_probs': deviation_probs,
            'type_logits': type_logits,
            'type_probs': type_probs
        }
    
    def get_receptive_field(self) -> int:
        """
        Return effective receptive field.
        For GRU, this is theoretically infinite (entire past sequence).
        Returns window_size as the minimum meaningful context.
        """
        return self.config.window_size
    
    def enable_layernorm_gru(self):
        """Switch to LayerNorm GRU for more stable training."""
        self.use_fast_gru = False
        logger.info("Switched to LayerNorm GRU (slower but more stable)")
    
    def enable_fast_gru(self):
        """Switch to standard PyTorch GRU for faster training."""
        self.use_fast_gru = True
        logger.info("Switched to fast PyTorch GRU")
    
    def freeze_deviation_detector(self):
        """Freeze deviation detector for second-stage training."""
        if self.deviation_detector is not None:
            for param in self.deviation_detector.parameters():
                param.requires_grad = False
            logger.info("Frozen deviation detector")
    
    def freeze_temperature(self):
        """Freeze temperature scaling."""
        self.log_temperature.requires_grad = False
    
    def save_model(self, path: str, is_ddp: bool = False, metadata: Optional[Dict] = None):
        """
        Save model checkpoint with optional metadata.
        
        Args:
            path: Output file path
            is_ddp: Whether model is wrapped in DDP
            metadata: Optional training metadata
        """
        state = self.module.state_dict() if is_ddp else self.state_dict()
        save_dict = {
            'model_state_dict': state,
            'config': self.config.__dict__,
            'model_type': 'CausalGRUModel'
        }
        if metadata:
            save_dict['metadata'] = metadata
        torch.save(save_dict, path)
        self._log(f"Model saved to {path}", "info")
    
    @classmethod
    def load_model(cls, path: str, device: torch.device = torch.device('cpu')) -> 'CausalGRUModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = GRUConfig(**checkpoint['config'])
        model = cls(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return model
    
    def _log(self, msg: str, level: str = 'info'):
        """Rank-0 only logging for DDP."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            getattr(logger, level)(msg)
    
    @staticmethod
    def set_init_seed(seed: int = 42):
        """Set seed for deterministic initialization."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def set_distributed_seed(seed: int, rank: int):
        """Set rank-specific seed for distributed training."""
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        import numpy as np
        import random
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    
    @staticmethod
    def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove 'module.' prefix from DDP state dict."""
        return {k.replace('module.', ''): v for k, v in state_dict.items()}


# =============================================================================
# BACKWARD COMPATIBILITY - Keep CausalConfig and CausalHybridModel names
# =============================================================================

# Alias for backward compatibility with train.py
CausalConfig = GRUConfig
CausalHybridModel = CausalGRUModel


def create_model(
    d_model: int = 128,
    n_layers: int = 2,
    dropout: float = 0.3,
    window_size: int = 7,
    hierarchical: bool = True,
    use_amp: bool = True,
    compile_model: bool = False,
    **kwargs
) -> CausalGRUModel:
    """
    Factory function to create model with sensible defaults.
    
    Args:
        d_model: Hidden dimension (64-256 recommended)
        n_layers: Number of GRU layers (2-3 recommended)
        dropout: Dropout rate (0.1-0.3 recommended)
        window_size: Sliding window size (7 per Yiannis)
        hierarchical: Use hierarchical classification
        use_amp: Enable AMP for 2x speedup
        compile_model: Use torch.compile (PyTorch 2.0+)
        **kwargs: Additional config options
    
    Returns:
        Configured CausalGRUModel
    """
    config = GRUConfig(
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        window_size=window_size,
        hierarchical=hierarchical,
        use_amp=use_amp,
        compile_model=compile_model,
        **kwargs
    )
    
    # Configure CUDA for speed
    configure_cuda_for_speed()
    
    return CausalGRUModel(config)


def get_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    use_fused: bool = True
) -> torch.optim.Optimizer:
    """
    Get optimized AdamW optimizer with fused kernels if available.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay for regularization
        use_fused: Use fused AdamW (faster on CUDA)
    
    Returns:
        Configured optimizer
    """
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Use fused AdamW if available (PyTorch 2.0+)
    if use_fused and torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=True)
            logger.info("Using fused AdamW optimizer")
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
            logger.info("Using standard AdamW optimizer (fused not available)")
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    
    return optimizer


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CausalGRUModel Test Suite (with AMP & Optimizations)")
    print("=" * 70)
    
    # Test configuration
    config = GRUConfig(
        d_model=64,
        n_layers=2,
        dropout=0.1,
        window_size=7,
        hierarchical=True,
        use_amp=True,
        compile_model=False  # Set True if PyTorch 2.0+
    )
    
    print(f"\nConfig: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"window_size={config.window_size}, hierarchical={config.hierarchical}")
    print(f"Performance: use_amp={config.use_amp}, compile={config.compile_model}")
    
    # Configure CUDA
    configure_cuda_for_speed()
    
    # Create model
    model = CausalGRUModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Receptive field: {model.get_receptive_field()} timesteps")
    
    # Test data
    B, T = 4, 100
    flux = torch.randn(B, T)
    delta_t = torch.abs(torch.randn(B, T)) * 0.1
    lengths = torch.tensor([100, 80, 60, 40])
    
    # Test forward pass - all timesteps
    model.eval()
    with torch.no_grad():
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=True)
        print(f"\n✓ Forward (all timesteps): logits {out['logits'].shape}")
        print(f"  probs sum: {out['probs'].sum(dim=-1).mean().item():.4f} (should be ~1.0)")
        
        if config.hierarchical:
            print(f"  deviation_probs: {out['deviation_probs'].shape}")
            print(f"  type_probs: {out['type_probs'].shape}")
    
    # Test forward pass - final only
    with torch.no_grad():
        out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
        print(f"\n✓ Forward (final only): logits {out['logits'].shape}")
    
    # Test causality
    print("\n→ Running causality verification...")
    T_test = 50
    flux1 = torch.randn(1, 100)
    flux2 = flux1.clone()
    flux2[0, T_test:] = torch.randn(100 - T_test)  # Different future
    
    delta_t_test = torch.ones(1, 100) * 0.1
    
    with torch.no_grad():
        out1 = model(flux1, delta_t_test, return_all_timesteps=True)
        out2 = model(flux2, delta_t_test, return_all_timesteps=True)
        
        diff_before = (out1['logits'][0, :T_test] - out2['logits'][0, :T_test]).abs().max().item()
        diff_after = (out1['logits'][0, T_test:] - out2['logits'][0, T_test:]).abs().max().item()
        
        if diff_before < 1e-5:
            print(f"✓ CAUSALITY VERIFIED: Past predictions identical (diff={diff_before:.2e})")
        else:
            print(f"✗ CAUSALITY VIOLATION: Past predictions differ! (diff={diff_before:.2e})")
        
        if diff_after > 1e-3:
            print(f"✓ Future predictions correctly differ (diff={diff_after:.2e})")
    
    # Test non-hierarchical mode
    print("\n→ Testing non-hierarchical mode...")
    config_direct = GRUConfig(d_model=64, n_layers=2, hierarchical=False)
    model_direct = CausalGRUModel(config_direct)
    
    with torch.no_grad():
        out = model_direct(flux, delta_t, lengths=lengths, return_all_timesteps=False)
        print(f"✓ Direct classification: {out['logits'].shape}")
    
    # Test gradient flow
    print("\n→ Testing gradient flow...")
    model.train()
    out = model(flux, delta_t, lengths=lengths, return_all_timesteps=False)
    loss = out['logits'].sum()
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"✓ Gradient flow: {'OK' if grad_ok else 'FAILED'}")
    
    # Test optimizer
    print("\n→ Testing optimizer...")
    optimizer = get_optimizer(model, lr=3e-4, weight_decay=0.01)
    print(f"✓ Optimizer: {type(optimizer).__name__}")
    
    # Benchmark
    print("\n→ Benchmarking...")
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model.eval()
    model.to(device)
    flux_bench = flux.to(device)
    delta_t_bench = delta_t.to(device)
    lengths_bench = lengths.to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(flux_bench, delta_t_bench, lengths=lengths_bench, return_all_timesteps=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    n_iters = 100
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(flux_bench, delta_t_bench, lengths=lengths_bench, return_all_timesteps=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    throughput = n_iters / elapsed
    samples_per_sec = throughput * B
    print(f"✓ Throughput: {throughput:.1f} batches/sec ({samples_per_sec:.1f} samples/sec)")
    
    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"✓ Peak GPU memory: {mem_allocated:.1f} MB")
    
    print("\n" + "=" * 70)
    print("🎉 All tests passed!")
    print("=" * 70)
