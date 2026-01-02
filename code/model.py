from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Final, Optional, Tuple, Union, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__version__: Final[str] = "7.1.0"

__all__ = [
    "ModelConfig",
    "RomanMicrolensingClassifier",
    "HierarchicalOutput",
    "create_model",
    "load_checkpoint",
    "validate_inputs",
]

# =============================================================================
# CONSTANTS
# =============================================================================

MASK_VALUE_FP32: Final[float] = -1e9
MASK_VALUE_FP16: Final[float] = -6e4
EPS: Final[float] = 1e-8
HEAD_INIT_STD: Final[float] = 0.15
MAX_LOCKED_DROPOUT: Final[float] = 0.95


# =============================================================================
# OUTPUT TYPES
# =============================================================================

class HierarchicalOutput(NamedTuple):
    """
    Output container for hierarchical classification.
    
    Attributes
    ----------
    log_probs : Tensor
        Log-probabilities for NLLLoss (NOT raw logits).
    stage1_logit : Tensor
        Raw logit for Stage 1 (deviation vs flat).
    stage2_logit : Tensor
        Raw logit for Stage 2 (PSPL vs binary), without temperature.
    aux_logits : Optional[Tensor]
        Auxiliary 3-class logits if enabled.
    p_deviation : Tensor
        P(deviation) without temperature.
    p_pspl_given_deviation : Tensor
        P(PSPL|deviation) without temperature.
    """
    log_probs: Tensor
    stage1_logit: Tensor
    stage2_logit: Tensor
    aux_logits: Optional[Tensor]
    p_deviation: Tensor
    p_pspl_given_deviation: Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Model configuration for CNN-LSTM classifier.
    
    Attributes
    ----------
    d_model : int
        Hidden dimension (must be even, ≥8, divisible by num_groups and num_attention_heads).
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout probability for conv layers and classification head.
    window_size : int
        Convolution kernel size.
    n_classes : int
        Number of output classes (default 3: flat, PSPL, binary).
    hierarchical : bool
        Use hierarchical classification (flat vs deviation, then PSPL vs binary).
    use_aux_head : bool
        Add auxiliary 3-class classification head.
    stage2_temperature : float
        Temperature for Stage 2 softmax (applied at inference only).
    use_residual : bool
        Add residual connections around feature extractor.
    use_layer_norm : bool
        Use layer norm after LSTM.
    use_attention_pooling : bool
        Use attention pooling (True) or masked mean pooling (False).
    use_flash_attention : bool
        Use flash attention if available (PyTorch 2.0+).
    num_attention_heads : int
        Number of attention heads for pooling.
    lstm_dropout : float
        Dropout between LSTM layers.
    locked_dropout : float
        Variational dropout on LSTM outputs (shared mask across time).
    num_groups : int
        Number of groups for GroupNorm in conv layers.
    use_gradient_checkpointing : bool
        Enable gradient checkpointing for LSTM (saves memory).
    """
    
    d_model: int = 32
    n_layers: int = 4
    dropout: float = 0.3
    window_size: int = 5
    n_classes: int = 3
    
    hierarchical: bool = True
    use_aux_head: bool = True
    stage2_temperature: float = 1.0
    use_residual: bool = True
    use_layer_norm: bool = True
    use_attention_pooling: bool = True
    use_flash_attention: bool = True
    
    num_attention_heads: int = 2
    lstm_dropout: float = 0.1
    locked_dropout: float = 0.1
    num_groups: int = 8
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.d_model < 8 or self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even and ≥8, got {self.d_model}")
        
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥1, got {self.n_layers}")
        
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        if not (0.0 <= self.locked_dropout < 1.0):
            raise ValueError(f"locked_dropout must be in [0, 1), got {self.locked_dropout}")
        
        if self.locked_dropout > 0.5:
            warnings.warn(f"locked_dropout={self.locked_dropout} is high (>0.5), may limit gradient flow")
        
        if self.d_model % self.num_groups != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_groups ({self.num_groups})")
        
        if self.use_attention_pooling and self.d_model % self.num_attention_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.stage2_temperature <= 0:
            raise ValueError(f"stage2_temperature must be positive, got {self.stage2_temperature}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# =============================================================================
# LAYERS
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-padding."""

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
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, groups=groups, bias=bias, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution with GroupNorm."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        num_groups: int = 8
    ) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(channels, channels, kernel_size, dilation, groups=channels, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.gn1(self.depthwise(x)))
        x = self.dropout(self.act(self.gn2(self.pointwise(x))))
        return x


class CausalFeatureExtractor(nn.Module):
    """Multi-scale causal feature extraction."""

    def __init__(self, d_model: int, window_size: int, dropout: float, num_groups: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(d_model, window_size, dilation=1, dropout=dropout, num_groups=num_groups)
        self.conv2 = DepthwiseSeparableConv1d(d_model, window_size, dilation=2, dropout=dropout, num_groups=num_groups)
        self.receptive_field = (window_size - 1) * 1 + (window_size - 1) * 2 + 1

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class AttentionPooling(nn.Module):
    """Multi-head attention pooling with per-position masking support."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, use_flash: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input [B, T, D].
        mask : Tensor
            Boolean mask [B, T] where True = valid observation.
        """
        B, T, D = x.shape
        
        q = self.q_proj(self.query.expand(B, 1, D))
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Build attention mask: [B, 1, 1, T]
        mask_value = MASK_VALUE_FP16 if q.dtype == torch.float16 else MASK_VALUE_FP32
        attn_mask = torch.zeros((B, 1, 1, T), device=x.device, dtype=q.dtype)
        attn_mask.masked_fill_(~mask[:, None, None, :], mask_value)
        
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + attn_mask
            out = torch.matmul(F.softmax(scores, dim=-1), v)
        
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, 1, D))
        return self.dropout(out).squeeze(1)


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

class RomanMicrolensingClassifier(nn.Module):
    """
    Causal CNN-LSTM classifier for Roman microlensing events.
    
    Input: 3 channels [flux, delta_t, is_observed]
    - flux: Magnification values (valid observations ≥ 1.0, missing = 0)
    - delta_t: Time since previous valid observation (0 for missing/first)
    - is_observed: Binary mask (1 = valid, 0 = missing)
    
    The observation mask is inferred from flux > 0.5 if not provided explicitly.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        # Input: [flux, delta_t, is_observed]
        self.input_proj = nn.Linear(3, config.d_model)
        
        self.feature_extractor = CausalFeatureExtractor(
            config.d_model, config.window_size, config.dropout, config.num_groups
        )
        self.receptive_field = self.feature_extractor.receptive_field
        
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=False,
            dropout=config.lstm_dropout if config.n_layers > 1 else 0.0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        
        if config.use_attention_pooling:
            self.pooling = AttentionPooling(
                config.d_model, config.num_attention_heads, config.dropout, config.use_flash_attention
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
            self.head_aux = nn.Linear(config.d_model, config.n_classes) if config.use_aux_head else None
        else:
            self.head_flat = nn.Linear(config.d_model, config.n_classes)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head_stage' in name or 'head_aux' in name or 'head_flat' in name:
                    nn.init.normal_(module.weight, std=HEAD_INIT_STD)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param)
                    elif 'bias' in pname:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1
                        hidden = param.size(0) // 4
                        param.data[hidden:2*hidden].fill_(1.0)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _locked_dropout(self, x: Tensor) -> Tensor:
        """Variational dropout: same mask across time dimension. Input: [T, B, D]."""
        if not self.training or self.config.locked_dropout <= 0:
            return x
        p = min(self.config.locked_dropout, MAX_LOCKED_DROPOUT)
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - p).div_(1 - p)
        return x * mask

    def _masked_mean_pool(self, x: Tensor, mask: Tensor) -> Tensor:
        """Masked mean pooling. x: [B, T, D], mask: [B, T] boolean."""
        mask_f = mask.float().unsqueeze(-1)  # [B, T, 1]
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(EPS)

    def forward(
        self,
        flux: Tensor,
        delta_t: Tensor,
        observation_mask: Optional[Tensor] = None,
        return_intermediates: bool = False
    ) -> Union[Tensor, HierarchicalOutput]:
        """
        Forward pass.
        
        Parameters
        ----------
        flux : Tensor
            Magnification values [B, T]. Missing observations = 0.
        delta_t : Tensor
            Time differences [B, T].
        observation_mask : Optional[Tensor]
            Boolean mask [B, T] where True = valid. Inferred from flux != 0.0 if None.
        return_intermediates : bool
            Return HierarchicalOutput with all intermediate values.
        
        Returns
        -------
        Union[Tensor, HierarchicalOutput]
            Log-probabilities [B, n_classes] for NLLLoss, or HierarchicalOutput.
        """
        # Infer observation mask if not provided
        mask = observation_mask if observation_mask is not None else (flux != 0.0)
        mask = mask.bool()
        
        # Input: [flux, delta_t, is_observed]
        x = torch.stack([flux, delta_t, mask.float()], dim=-1)  # [B, T, 3]
        x = self.input_proj(x)  # [B, T, D]
        
        # CNN feature extraction
        x = x.transpose(1, 2)  # [B, D, T]
        if self.config.use_residual:
            x = self.feature_extractor(x) + x
        else:
            x = self.feature_extractor(x)
        
        # LSTM
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # [T, B, D]
        
        if self.config.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                lambda inp: self.lstm(inp)[0], x, use_reentrant=False
            )
        else:
            x, _ = self.lstm(x)
        
        x = self._locked_dropout(x)
        x = self.layer_norm(x.transpose(0, 1))  # [B, T, D]
        
        # Pooling
        if self.pooling is not None:
            x = self.pooling(x, mask)
        else:
            x = self._masked_mean_pool(x, mask)
        
        # Classification
        x = self.head_shared(x)
        
        if self.config.hierarchical:
            stage1_logit = self.head_stage1(x)
            stage2_logit = self.head_stage2(x)
            
            p_deviation = torch.sigmoid(stage1_logit)
            p_pspl_given_dev = torch.sigmoid(stage2_logit)
            
            p_flat = 1.0 - p_deviation
            p_pspl = p_deviation * p_pspl_given_dev
            p_binary = p_deviation * (1.0 - p_pspl_given_dev)
            
            probs = torch.cat([p_flat, p_pspl, p_binary], dim=1)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
            log_probs = torch.log(probs.clamp_min(EPS))
            
            aux_logits = self.head_aux(x) if self.head_aux is not None else None
            
            if return_intermediates:
                return HierarchicalOutput(
                    log_probs=log_probs,
                    stage1_logit=stage1_logit,
                    stage2_logit=stage2_logit,
                    aux_logits=aux_logits,
                    p_deviation=p_deviation,
                    p_pspl_given_deviation=p_pspl_given_dev
                )
            return log_probs
        else:
            return self.head_flat(x)

    @torch.no_grad()
    def predict(
        self,
        flux: Tensor,
        delta_t: Tensor,
        observation_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions with temperature scaling applied.
        
        Returns
        -------
        Tuple[Tensor, Tensor]
            (class_predictions [B], class_probabilities [B, n_classes])
        """
        was_training = self.training
        self.eval()
        
        try:
            if self.config.hierarchical:
                out = self.forward(flux, delta_t, observation_mask, return_intermediates=True)
                
                p_deviation = torch.sigmoid(out.stage1_logit)
                p_pspl_given_dev = torch.sigmoid(out.stage2_logit / self.config.stage2_temperature)
                
                p_flat = 1.0 - p_deviation
                p_pspl = p_deviation * p_pspl_given_dev
                p_binary = p_deviation * (1.0 - p_pspl_given_dev)
                
                probs = torch.cat([p_flat, p_pspl, p_binary], dim=1)
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)
                
                return probs.argmax(dim=1), probs
            else:
                logits = self.forward(flux, delta_t, observation_mask)
                probs = F.softmax(logits, dim=-1)
                return probs.argmax(dim=1), probs
        finally:
            if was_training:
                self.train()

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_model(config: Optional[ModelConfig] = None, **kwargs: Any) -> RomanMicrolensingClassifier:
    """Create model from config or keyword arguments."""
    if config is None:
        config = ModelConfig(**kwargs)
    elif kwargs:
        d = config.to_dict()
        d.update(kwargs)
        config = ModelConfig.from_dict(d)
    return RomanMicrolensingClassifier(config)


def load_checkpoint(
    path: str,
    map_location: Union[str, torch.device] = 'cpu',
    strict: bool = True
) -> RomanMicrolensingClassifier:
    """
    Load model from checkpoint.
    
    Expected checkpoint format:
        {
            'model_config': ModelConfig or dict,
            'model_state_dict': state_dict
        }
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    config_data = checkpoint['model_config']
    config = config_data if isinstance(config_data, ModelConfig) else ModelConfig.from_dict(config_data)
    
    model = RomanMicrolensingClassifier(config)
    
    state_dict = checkpoint['model_state_dict']
    
    # Handle DDP and torch.compile prefixes
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)
    return model


def validate_inputs(flux: Tensor, delta_t: Tensor, observation_mask: Optional[Tensor] = None) -> None:
    """Validate input tensor shapes."""
    B, T = flux.shape
    if delta_t.shape != (B, T):
        raise ValueError(f"delta_t shape {delta_t.shape} must match flux {flux.shape}")
    if observation_mask is not None and observation_mask.shape != (B, T):
        raise ValueError(f"observation_mask shape {observation_mask.shape} must match flux {flux.shape}")
