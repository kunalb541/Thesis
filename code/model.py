#!/usr/bin/env python3
"""
Roman Microlensing Event Classifier - Neural Network Architecture
=================================================================

V7.0.1 - LSTM UPGRADE 
----------------------

FIXES:
    - Replaced BatchNorm with GroupNorm (stable with variable-length padding)
    - Applied locked dropout properly (post-LSTM variational dropout)
    - Fixed stage2_temperature application (now correctly applied at inference only)
    - Fixed GroupNorm validation (num_groups <= d_model constraint)
    - Fixed log-prob clamping (removed max clamp)
    - Added probability normalization for numerical stability
    - Disabled lstm_proj_size (incompatible with current architecture)
    - Renamed logits → log_probs for semantic clarity

Author: Kunal Bhatia
Institution: University of Heidelberg
Version: 7.0.1
Date: January 2025
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Final, Optional, Tuple, Union, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__version__: Final[str] = "7.0.1"

__all__ = [
    "ModelConfig",
    "RomanMicrolensingClassifier",
    "HierarchicalOutput",
    "create_model",
    "load_checkpoint",
    "validate_inputs",
    "validate_batch_size_for_ddp",
    "get_mask_value",
    "CausalConv1d",
    "DepthwiseSeparableConv1d",
    "CausalFeatureExtractor",
    "FlashAttentionPooling",
]

# =============================================================================
# CONSTANTS
# =============================================================================

MASK_VALUE_FP32: Final[float] = -1e9
MASK_VALUE_FP16: Final[float] = -6e4
MASK_VALUE_BF16: Final[float] = -1e9

MIN_VALID_SEQ_LEN: Final[int] = 1
EPS: Final[float] = 1e-8

DEFAULT_D_MODEL: Final[int] = 32
DEFAULT_N_LAYERS: Final[int] = 4
DEFAULT_DROPOUT: Final[float] = 0.3
DEFAULT_N_CLASSES: Final[int] = 3

# GroupNorm default (replaces BN_MOMENTUM)
DEFAULT_NUM_GROUPS: Final[int] = 8
HEAD_INIT_STD: Final[float] = 0.15

# Locked dropout safety threshold
MAX_SAFE_LOCKED_DROPOUT: Final[float] = 0.95

# =============================================================================
# OUTPUT TYPES
# =============================================================================

class HierarchicalOutput(NamedTuple):
    """
    Output container for hierarchical classification.
    
    Note: p_deviation and p_pspl_given_deviation are raw probabilities
    WITHOUT temperature scaling. Temperature is only applied during
    inference via the predict() method.
    
    IMPORTANT: Field naming changed from 'logits' to 'log_probs' in v7.0.1
    to prevent confusion. These are log-probabilities (normalized), NOT
    raw logits. Use with NLLLoss, not CrossEntropyLoss.
    """
    log_probs: Tensor  # Log-probabilities (NOT raw logits) for NLLLoss
    stage1_logit: Tensor  # Raw logit for Stage 1 (deviation vs flat)
    stage2_logit: Tensor  # Raw logit for Stage 2 (PSPL vs binary), NO temperature
    aux_logits: Optional[Tensor]  # Auxiliary 3-class logits (if enabled)
    p_deviation: Tensor  # P(deviation) without temperature
    p_pspl_given_deviation: Tensor  # P(PSPL|deviation) without temperature

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Model configuration with LSTM architecture.
    
    CRITICAL FIXES:
        - lstm_proj_size disabled (dimension mismatch with downstream layers)
        - num_groups added for GroupNorm (replaces bn_momentum)
        - locked_dropout now properly applied (variational dropout across time)
    
    Attributes
    ----------
    d_model : int
        Hidden dimension (must be even, ≥8).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout probability.
    window_size : int
        Convolution kernel size.
    max_seq_len : int
        Maximum sequence length.
    n_classes : int
        Number of output classes (3).
    hierarchical : bool
        Use hierarchical classification.
    use_aux_head : bool
        Add auxiliary 3-class head.
    stage2_temperature : float
        Temperature for Stage 2 (applied at inference only).
    use_residual : bool
        Add residual connections.
    use_layer_norm : bool
        Use layer norm after LSTM.
    feature_extraction : str
        Feature extraction method ('conv').
    use_attention_pooling : bool
        Use attention pooling vs mean pooling.
    use_amp : bool
        Enable automatic mixed precision.
    use_gradient_checkpointing : bool
        Enable gradient checkpointing for LSTM.
    use_flash_attention : bool
        Use flash attention (PyTorch 2.0+).
    num_attention_heads : int
        Number of attention heads.
    lstm_dropout : float
        Dropout between LSTM layers.
    locked_dropout : float
        Locked (variational) dropout on LSTM outputs (shared across time).
    num_groups : int
        Number of groups for GroupNorm (replaces BN).
    init_scale : float
        Weight initialization scale.
    """
    
    # Core architecture
    d_model: int = DEFAULT_D_MODEL
    n_layers: int = DEFAULT_N_LAYERS
    dropout: float = DEFAULT_DROPOUT
    window_size: int = 5
    max_seq_len: int = 7000
    n_classes: int = DEFAULT_N_CLASSES
    
    # Architecture options
    hierarchical: bool = True
    use_aux_head: bool = True
    stage2_temperature: float = 1.0
    use_residual: bool = True
    use_layer_norm: bool = True
    feature_extraction: str = 'conv'
    use_attention_pooling: bool = True
    
    # Performance options
    use_amp: bool = False
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    
    # Advanced options
    num_attention_heads: int = 2
    lstm_dropout: float = 0.1
    locked_dropout: float = 0.1
    num_groups: int = DEFAULT_NUM_GROUPS
    init_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.d_model, int) or self.d_model <= 0:
            raise ValueError(f"d_model must be positive int, got {self.d_model}")
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {self.d_model}")
        if self.d_model < 8:
            raise ValueError(f"d_model must be >= 8, got {self.d_model}")
        
        if not isinstance(self.n_layers, int) or self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive int, got {self.n_layers}")
        
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        if self.lstm_dropout == 0.0 and self.n_layers > 1:
            object.__setattr__(self, 'lstm_dropout', self.dropout)
        
        if not isinstance(self.n_classes, int) or self.n_classes <= 0:
            raise ValueError(f"n_classes must be positive int, got {self.n_classes}")
        
        if self.feature_extraction != 'conv':
            raise ValueError(f"feature_extraction must be 'conv', got '{self.feature_extraction}'")
        
        if self.use_attention_pooling:
            if not isinstance(self.num_attention_heads, int) or self.num_attention_heads <= 0:
                raise ValueError(f"num_attention_heads must be positive int")
            if self.d_model % self.num_attention_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
        
        if self.stage2_temperature <= 0:
            raise ValueError(f"stage2_temperature must be positive, got {self.stage2_temperature}")
        
        if not (0.0 <= self.locked_dropout < 1.0):
            raise ValueError(f"locked_dropout must be in [0, 1), got {self.locked_dropout}")
        
        if self.locked_dropout > 0.5:
            warnings.warn(
                f"locked_dropout={self.locked_dropout} is unusually high (>0.5). "
                f"This may severely limit gradient flow through time. "
                f"Typical values are 0.1-0.3.",
                UserWarning
            )
        
        # FIXED: Added constraint that num_groups <= d_model
        if self.num_groups <= 0 or self.num_groups > self.d_model or self.d_model % self.num_groups != 0:
            raise ValueError(
                f"num_groups must be positive, <= d_model, and divide d_model evenly. "
                f"Got d_model={self.d_model}, num_groups={self.num_groups}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Handle backward compatibility: recurrent_dropout -> locked_dropout
        if 'recurrent_dropout' in config_dict and 'locked_dropout' not in filtered_dict:
            filtered_dict['locked_dropout'] = config_dict['recurrent_dropout']
        
        return cls(**filtered_dict)

# =============================================================================
# UTILITIES
# =============================================================================

def get_mask_value(dtype: torch.dtype) -> float:
    """Get appropriate mask value for dtype."""
    if dtype == torch.float16:
        return MASK_VALUE_FP16
    elif dtype == torch.bfloat16:
        return MASK_VALUE_BF16
    else:
        return MASK_VALUE_FP32

def validate_batch_size_for_ddp(batch_size: int, world_size: int) -> None:
    """Validate batch size for distributed training."""
    effective_batch = batch_size * world_size
    if effective_batch < 16:
        warnings.warn(
            f"Effective batch size ({effective_batch}) is small for DDP. "
            f"Consider increasing batch_size (currently {batch_size}).",
            UserWarning
        )

def locked_dropout(x: Tensor, p: float, training: bool) -> Tensor:
    """
    Apply locked (variational) dropout to RNN outputs.
    
    Locked dropout uses a single dropout mask shared across the time dimension,
    which helps prevent overfitting in recurrent networks by maintaining
    consistent dropout patterns through time.
    
    Parameters
    ----------
    x : Tensor
        Input tensor of shape [T, B, D] (time, batch, features).
    p : float
        Dropout probability. Values > 0.5 will trigger a warning as they
        severely limit gradient flow. Values >= 0.95 are capped for safety.
    training : bool
        Whether model is in training mode.
    
    Returns
    -------
    Tensor
        Tensor with locked dropout applied.
    
    Raises
    ------
    ValueError
        If p >= 1.0 (completely drops all information) or if p is non-finite.
    """
    if not training or p <= 0.0:
        return x
    
    # Safety checks for extreme dropout values
    if p >= 1.0:
        raise ValueError(
            f"locked_dropout probability must be < 1.0, got {p}. "
            f"p=1.0 would drop all information."
        )
    
    # Ensure p is a float and check for non-finite values
    p = float(p)
    if not math.isfinite(p):
        raise ValueError(
            f"locked_dropout must be finite, got {p}. "
            f"Check for NaN or Inf values in dropout probability."
        )
    
    # Cap at safe maximum to prevent numerical instability
    if p > MAX_SAFE_LOCKED_DROPOUT:
        warnings.warn(
            f"locked_dropout={p} exceeds safe maximum ({MAX_SAFE_LOCKED_DROPOUT}). "
            f"Capping to {MAX_SAFE_LOCKED_DROPOUT} to prevent numerical instability.",
            UserWarning
        )
        p = MAX_SAFE_LOCKED_DROPOUT
    
    # Create mask with shape [1, B, D] - shared across time dimension
    mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - p)
    mask = mask.div_(1 - p)
    return x * mask

# =============================================================================
# CAUSAL CONVOLUTION LAYERS
# =============================================================================

class CausalConv1d(nn.Module):
    """Strictly causal 1D convolution with left-padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.receptive_field = self.padding + 1
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal left-padding."""
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution with GroupNorm.
    
    CRITICAL FIX: Replaced BatchNorm with GroupNorm for stability
    with variable-length padded sequences.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        num_groups: int = DEFAULT_NUM_GROUPS
    ) -> None:
        super().__init__()
        
        # FIXED: Validate GroupNorm divisibility for both in_channels and out_channels
        if in_channels % num_groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        if out_channels % num_groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        
        self.depthwise = CausalConv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        
        # GroupNorm instead of BatchNorm (stable with variable-length sequences)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.receptive_field = self.depthwise.receptive_field

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.gn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.gn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class CausalFeatureExtractor(nn.Module):
    """Multi-scale causal feature extraction."""

    def __init__(
        self,
        d_model: int,
        window_size: int,
        dropout: float = 0.0,
        num_groups: int = DEFAULT_NUM_GROUPS
    ) -> None:
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv1d(
            d_model, d_model, window_size,
            dilation=1, dropout=dropout, num_groups=num_groups
        )
        self.conv2 = DepthwiseSeparableConv1d(
            d_model, d_model, window_size,
            dilation=2, dropout=dropout, num_groups=num_groups
        )
        
        rf1 = (window_size - 1) * 1 + 1
        rf2 = (window_size - 1) * 2 + 1
        self.receptive_field = rf1 + rf2 - 1

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# =============================================================================
# ATTENTION POOLING
# =============================================================================

class FlashAttentionPooling(nn.Module):
    """Multi-head attention pooling with flash attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_flash: bool = True
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.register_buffer('_mask_zero', torch.tensor(0.0), persistent=False)
        self.register_buffer('_mask_neg_inf_fp32', torch.tensor(MASK_VALUE_FP32), persistent=False)
        self.register_buffer('_mask_neg_inf_fp16', torch.tensor(MASK_VALUE_FP16), persistent=False)
        self.register_buffer('_mask_neg_inf_bf16', torch.tensor(MASK_VALUE_BF16), persistent=False)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.query, std=0.02)

    def _get_mask_values(self, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Get mask values for dtype."""
        zero = self._mask_zero.to(dtype=dtype)
        
        if dtype == torch.float16:
            neg_inf = self._mask_neg_inf_fp16.to(dtype=dtype)
        elif dtype == torch.bfloat16:
            neg_inf = self._mask_neg_inf_bf16.to(dtype=dtype)
        else:
            neg_inf = self._mask_neg_inf_fp32.to(dtype=dtype)
        
        return zero, neg_inf

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """Forward pass."""
        B, T, D = x.shape
        
        query = self.query.expand(B, 1, D)
        
        q = self.q_proj(query)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_mask = None
        if lengths is not None:
            indices = torch.arange(T, device=x.device)[None, :]
            valid_mask = indices < lengths[:, None]
            
            zero, neg_inf = self._get_mask_values(q.dtype)
            attn_mask = torch.where(
                valid_mask[:, None, None, :],
                zero,
                neg_inf
            )
        
        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out.squeeze(1)

# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

class RomanMicrolensingClassifier(nn.Module):
    """
    Strictly causal CNN-LSTM classifier.
    
    CRITICAL FIXES:
        - GroupNorm replaces BatchNorm (stable with variable-length padding)
        - Locked dropout properly applied (variational dropout on LSTM outputs)
        - stage2_temperature correctly applied at inference only
        - Fixed log-prob clamping (removed max clamp)
        - Added probability normalization for numerical stability
        - Output renamed to log_probs (not logits) for clarity
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(2, config.d_model)
        
        self.feature_extractor = CausalFeatureExtractor(
            config.d_model,
            config.window_size,
            config.dropout,
            config.num_groups
        )
        self._receptive_field = self.feature_extractor.receptive_field
        
        # LSTM (no projection support - would break downstream layers)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=False,
            dropout=config.lstm_dropout if config.n_layers > 1 else 0.0,
            bidirectional=False
        )
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.d_model)
        else:
            self.layer_norm = nn.Identity()
        
        if config.use_attention_pooling:
            self.pooling = FlashAttentionPooling(
                config.d_model,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                use_flash=config.use_flash_attention
            )
        else:
            self.pooling = None
        
        self.head_shared = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
            nn.Dropout(config.dropout)
        )
        
        if config.hierarchical:
            self.head_stage1 = nn.Linear(config.d_model, 1)
            self.head_stage2 = nn.Linear(config.d_model, 1)
            
            if config.use_aux_head:
                self.head_aux = nn.Linear(config.d_model, config.n_classes)
            else:
                self.head_aux = None
        else:
            self.head_flat = nn.Linear(config.d_model, config.n_classes)
        
        self._init_weights()

    @property
    def receptive_field(self) -> int:
        """Total CNN receptive field in timesteps."""
        return self._receptive_field

    def _init_weights(self) -> None:
        """Initialize weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name or 'proj' in name:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1 (encourage remembering)
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
            
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if self.config.hierarchical:
            nn.init.zeros_(self.head_stage1.bias)
            nn.init.normal_(self.head_stage1.weight, mean=0.0, std=HEAD_INIT_STD)
            
            nn.init.zeros_(self.head_stage2.bias)
            nn.init.normal_(self.head_stage2.weight, mean=0.0, std=HEAD_INIT_STD)
            
            if self.head_aux is not None:
                nn.init.zeros_(self.head_aux.bias)
                nn.init.normal_(self.head_aux.weight, mean=0.0, std=HEAD_INIT_STD)

    def _apply_lstm_with_checkpointing(self, x: Tensor) -> Tensor:
        """Apply LSTM with optional gradient checkpointing."""
        if self.config.use_gradient_checkpointing and self.training:
            def lstm_forward(x_in: Tensor) -> Tensor:
                out, _ = self.lstm(x_in)
                return out
            
            x = torch.utils.checkpoint.checkpoint(
                lstm_forward,
                x,
                use_reentrant=False
            )
        else:
            x, _ = self.lstm(x)
        
        return x

    def forward(
        self,
        flux: Tensor,
        delta_t: Tensor,
        lengths: Optional[Tensor] = None,
        return_intermediates: bool = False
    ) -> Union[Tensor, HierarchicalOutput]:
        """
        Forward pass.
        
        CRITICAL: stage2_temperature is NOT applied during forward pass.
        It should only be applied at inference via predict() method.
        
        Returns
        -------
        Union[Tensor, HierarchicalOutput]
            If hierarchical: log-probabilities (NOT raw logits) suitable for NLLLoss.
            If return_intermediates: HierarchicalOutput with raw stage logits and
            probabilities (without temperature scaling).
        """
        B, T = flux.shape
        device = flux.device
        
        x = torch.stack([flux, delta_t], dim=-1)
        x = self.input_proj(x)
        
        x = x.transpose(1, 2)
        
        if self.config.use_residual:
            residual = x
            x = self.feature_extractor(x)
            x = x + residual
        else:
            x = self.feature_extractor(x)
        
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        
        x = self._apply_lstm_with_checkpointing(x)
        # FIXED: Apply locked dropout (variational dropout across time)
        x = locked_dropout(x, self.config.locked_dropout, self.training)
        
        x = x.transpose(0, 1)
        x = self.layer_norm(x)
        
        if self.pooling is not None:
            x = self.pooling(x, lengths)
        else:
            if lengths is not None:
                mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
                mask = mask.float().unsqueeze(-1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(EPS)
            else:
                x = x.mean(dim=1)
        
        x = self.head_shared(x)
        
        if self.config.hierarchical:
            stage1_logit = self.head_stage1(x)
            p_deviation = torch.sigmoid(stage1_logit)
            
            # Stage 2: NO temperature in forward (applied in predict() at inference)
            stage2_logit = self.head_stage2(x)
            p_pspl_given_deviation = torch.sigmoid(stage2_logit)
            
            p_flat = 1.0 - p_deviation
            p_pspl = p_deviation * p_pspl_given_deviation
            p_binary = p_deviation * (1.0 - p_pspl_given_deviation)
            
            probs = torch.cat([p_flat, p_pspl, p_binary], dim=1)
            
            # FIXED: Normalize probabilities for numerical stability
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
            
            # FIXED: Only clamp min (not max)
            log_probs = torch.log(probs.clamp_min(EPS))
            
            aux_logits = None
            if self.head_aux is not None:
                aux_logits = self.head_aux(x)
            
            if return_intermediates:
                return HierarchicalOutput(
                    log_probs=log_probs,
                    stage1_logit=stage1_logit,
                    stage2_logit=stage2_logit,
                    aux_logits=aux_logits,
                    p_deviation=p_deviation,
                    p_pspl_given_deviation=p_pspl_given_deviation
                )
            else:
                return log_probs
        else:
            logits = self.head_flat(x)
            return logits

    @torch.no_grad()
    def predict(
        self,
        flux: Tensor,
        delta_t: Tensor,
        lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions without gradients.
        
        FIXED: stage2_temperature is now correctly applied at inference only.
        
        Returns
        -------
        Tuple[Tensor, Tensor]
            (predictions, probabilities) where predictions are class indices
            and probabilities are normalized class probabilities.
        """
        was_training = self.training
        self.eval()
        
        try:
            if self.config.hierarchical:
                # Get intermediate outputs to apply temperature correctly
                out = self.forward(flux, delta_t, lengths, return_intermediates=True)
                
                # Stage 1: P(deviation) - no temperature
                p_deviation = torch.sigmoid(out.stage1_logit)
                
                # Stage 2: P(PSPL | deviation) - FIXED: temperature applied here
                p_pspl_given_deviation = torch.sigmoid(
                    out.stage2_logit / self.config.stage2_temperature
                )
                
                # Compute final probabilities
                p_flat = 1.0 - p_deviation
                p_pspl = p_deviation * p_pspl_given_deviation
                p_binary = p_deviation * (1.0 - p_pspl_given_deviation)
                
                probabilities = torch.cat([p_flat, p_pspl, p_binary], dim=1)
                
                # FIXED: Normalize for numerical stability using clamp_min
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(EPS)
                
                predictions = probabilities.argmax(dim=1)
                return predictions, probabilities
                
            else:
                logits = self.forward(flux, delta_t, lengths, return_intermediates=False)
                probabilities = F.softmax(logits, dim=-1)
                predictions = probabilities.argmax(dim=-1)
                return predictions, probabilities
                
        finally:
            if was_training:
                self.train()

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_complexity_info(self) -> Dict[str, Any]:
        """Get model complexity information."""
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        
        return {
            'version': __version__,
            'architecture': 'CNN-LSTM',
            'normalization': 'GroupNorm',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_classes': self.config.n_classes,
            'dropout': self.config.dropout,
            'lstm_dropout': self.config.lstm_dropout,
            'locked_dropout': self.config.locked_dropout,
            'num_groups': self.config.num_groups,
            'hierarchical': self.config.hierarchical,
            'use_aux_head': self.config.use_aux_head if self.config.hierarchical else False,
            'stage2_temperature': self.config.stage2_temperature if self.config.hierarchical else None,
            'attention_pooling': self.config.use_attention_pooling,
            'num_attention_heads': self.config.num_attention_heads,
            'residual_connections': self.config.use_residual,
            'layer_normalization': self.config.use_layer_norm,
            'receptive_field': self._receptive_field,
            'flash_attention': hasattr(F, 'scaled_dot_product_attention'),
            'gradient_checkpointing': self.config.use_gradient_checkpointing,
            'head_init_std': HEAD_INIT_STD,
        }

# =============================================================================
# VALIDATION
# =============================================================================

@torch._dynamo.disable
def validate_inputs(
    flux: Tensor,
    delta_t: Tensor,
    lengths: Optional[Tensor],
    receptive_field: int
) -> None:
    """Validate inputs before training."""
    B, T = flux.shape
    
    if delta_t.shape != (B, T):
        raise ValueError(f"delta_t shape {delta_t.shape} must match flux {flux.shape}")
    
    if lengths is not None:
        if lengths.shape != (B,):
            raise ValueError(f"lengths shape {lengths.shape} must be ({B},)")
        
        min_len = lengths.min().item()
        if min_len < receptive_field:
            raise ValueError(
                f"Minimum length ({min_len}) must be >= receptive field ({receptive_field})"
            )
        
        if min_len < MIN_VALID_SEQ_LEN:
            raise ValueError(f"Minimum length ({min_len}) must be >= {MIN_VALID_SEQ_LEN}")

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_model(
    config: Optional[ModelConfig] = None,
    **kwargs: Any
) -> RomanMicrolensingClassifier:
    """Create model instance."""
    if config is None:
        config = ModelConfig(**kwargs)
    elif kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = ModelConfig.from_dict(config_dict)
    
    return RomanMicrolensingClassifier(config)

def load_checkpoint(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    map_location: Union[str, torch.device] = 'cpu',
    strict: bool = True
) -> RomanMicrolensingClassifier:
    """Load model from checkpoint."""
    import os
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )
    
    if config is None:
        if 'model_config' not in checkpoint:
            raise KeyError(
                f"Checkpoint missing 'model_config'. "
                f"Available: {list(checkpoint.keys())}"
            )
        
        config_data = checkpoint['model_config']
        if isinstance(config_data, dict):
            config = ModelConfig.from_dict(config_data)
        elif isinstance(config_data, ModelConfig):
            config = config_data
        else:
            raise TypeError(f"Unknown config type: {type(config_data).__name__}")
    
    model = RomanMicrolensingClassifier(config)
    
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'")
    
    state_dict = checkpoint['model_state_dict']
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    has_gru_weights = any('gru.' in k for k in state_dict.keys())
    has_lstm_weights = any('lstm.' in k for k in state_dict.keys())
    
    if has_gru_weights and not has_lstm_weights:
        warnings.warn(
            "Cannot migrate GRU checkpoint to LSTM architecture. "
            "Retraining required.",
            UserWarning
        )
        strict = False
    
    if config.hierarchical:
        migrated = False
        for key in ['head_stage1.weight', 'head_stage1.bias',
                    'head_stage2.weight', 'head_stage2.bias']:
            if key in state_dict:
                old_shape = state_dict[key].shape
                if 'weight' in key and old_shape[0] == 2:
                    state_dict[key] = state_dict[key][0:1, :]
                    migrated = True
                elif 'bias' in key and old_shape[0] == 2:
                    state_dict[key] = state_dict[key][0:1]
                    migrated = True
        
        if migrated:
            warnings.warn(
                "Migrated hierarchical heads from 2-output to 1-output format.",
                UserWarning
            )
        
        if config.use_aux_head and 'head_aux.weight' not in state_dict:
            warnings.warn(
                "Checkpoint missing auxiliary head. Initializing randomly.",
                UserWarning
            )
            strict = False
    
    model.load_state_dict(state_dict, strict=strict)
    return model
