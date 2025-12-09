import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
        assert self.kernel_size > 0, "kernel_size must be positive"
        assert self.n_conv_layers > 0, "n_conv_layers must be positive"
        assert self.dilation_growth > 0, "dilation_growth must be positive"

# =============================================================================
# CAUSAL CONVOLUTIONS
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Strictly causal 1D convolution with left-padding only.
    Ensures output at time t depends only on inputs at times <= t.
    """
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) tensor
        """
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        out = self.conv(x)
        return self.dropout(out)

class ResidualCausalBlock(nn.Module):
    """
    Residual block with two causal convolutions and layer normalization.
    Operates entirely in (B, C, T) format to minimize transpose operations.
    Uses LayerNorm instead of BatchNorm for DDP stability with small per-GPU batches.
    """
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(d_model)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) tensor
        """
        residual = x
        # LayerNorm expects (B, T, C), so transpose before and after
        z = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        z = self.norm1(z)
        z = z.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        z = self.conv1(z)
        z = self.act(z)
        
        z = z.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        z = self.norm2(z)
        z = z.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        z = self.conv2(z)
        return residual + z

# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class ContinuousSinusoidalEncoding(nn.Module):
    """
    Continuous time encoding using logarithmically-scaled sinusoids.
    Maps irregular time intervals to positional embeddings.
    """
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(torch.arange(0, half_dim * 2, 2).float() * -(math.log(max_timescale) / half_dim))
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

class FlashCausalAttention(nn.Module):
    """
    Multi-head causal attention using PyTorch's scaled_dot_product_attention.
    Supports optional padding masks merged with causal masking.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input tensor
            key_padding_mask: (B, T) boolean mask where True indicates padding
        Returns:
            (B, T, C) output tensor
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout_p if self.training else 0.0, 
                is_causal=True
            )
        else:
            # Create causal mask
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            attn_mask = torch.zeros(T, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(~causal_mask, float('-inf'))
            
            # Expand padding mask and merge with causal mask
            pad_mask_expanded = key_padding_mask.view(B, 1, 1, T).expand(B, self.n_heads, T, T)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, T, T).clone()
            attn_mask.masked_fill_(pad_mask_expanded, float('-inf'))

            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout_p if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# =============================================================================
# MAIN MODEL
# =============================================================================

class CausalHybridModel(nn.Module):
    """
    Hybrid causal model combining:
    1. Causal CNN feature extraction with exponentially growing dilations
    2. Causal transformer layers with flash attention
    3. Per-timestep classification head
    
    Maintains strict causality: output at time t depends only on inputs <= t.
    Designed for real-time streaming inference on Roman Space Telescope data.
    
    NaN Handling Policy:
        Input NaNs are replaced with 0.0 via torch.nan_to_num. This treats
        missing observations as zero flux. Time encodings for padded positions
        are also zeroed to ensure complete neutralization of padding information.
    
    DDP Considerations:
        Uses LayerNorm instead of BatchNorm for stability with small per-GPU batches
        in multi-GPU training scenarios (e.g., 40 GPUs with global batch size 64-128).
    """
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Input projection and time encoding
        self.flux_proj = nn.Linear(1, config.d_model)
        self.time_enc = ContinuousSinusoidalEncoding(config.d_model)
        self.emb_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal CNN feature extractor with exponential dilation
        self.feature_extractor = nn.ModuleList()
        curr_dilation = 1
        for _ in range(config.n_conv_layers):
            self.feature_extractor.append(
                ResidualCausalBlock(config.d_model, config.kernel_size, curr_dilation, config.dropout)
            )
            curr_dilation *= config.dilation_growth
        
        # Calculate total receptive field for streaming inference
        self.receptive_field = self._calculate_receptive_field()
            
        # Causal transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': FlashCausalAttention(config.d_model, config.n_heads, config.dropout),
                'norm1': nn.LayerNorm(config.d_model),
                'ffn': nn.Sequential(
                    nn.Linear(config.d_model, 4 * config.d_model),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(4 * config.d_model, config.d_model),
                    nn.Dropout(config.dropout)
                ),
                'norm2': nn.LayerNorm(config.d_model)
            }) for _ in range(config.n_transformer_layers)
        ])
        
        # Classification head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.n_classes)
        )
        
        # Learnable temperature for logit scaling
        self.log_temperature = nn.Parameter(torch.tensor([0.0]))
    
    def _calculate_receptive_field(self) -> int:
        """
        Calculate total receptive field of causal CNN layers.
        For each residual block with two convolutions:
        RF_new = RF_old + 2 * (kernel_size - 1) * dilation
        """
        rf = 1
        dilation = 1
        for _ in range(self.config.n_conv_layers):
            # Each residual block has 2 causal convolutions
            rf += 2 * (self.config.kernel_size - 1) * dilation
            dilation *= self.config.dilation_growth
        return rf

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None,
                return_all_timesteps: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional sequence length masking.
        
        Args:
            flux: (B, T) flux measurements
            delta_t: (B, T) time intervals in days
            lengths: (B,) actual sequence lengths for variable-length inputs
            return_all_timesteps: If False, return only final timestep predictions
            
        Returns:
            Dictionary containing:
                'logits': (B, n_classes) or (B, T, n_classes) logits
                'probs': (B, n_classes) or (B, T, n_classes) probabilities
        """
        assert flux.ndim == 2, f"flux must be 2D (B, T), got shape {flux.shape}"
        assert delta_t.ndim == 2, f"delta_t must be 2D (B, T), got shape {delta_t.shape}"
        
        B, T = flux.shape
        device = flux.device
        
        # Create padding mask if lengths provided
        key_padding_mask = None
        mask_float = torch.ones(B, T, 1, device=device, dtype=flux.dtype)
        
        if lengths is not None:
            range_tensor = torch.arange(T, device=device).unsqueeze(0)
            key_padding_mask = range_tensor >= lengths.unsqueeze(1)
            mask_float = (~key_padding_mask).float().unsqueeze(-1)
            
            # Zero out padded positions in input (NaN handling policy)
            flux = torch.nan_to_num(flux, 0.0)
            flux = flux * mask_float.squeeze(-1)
            
            # Zero out time encoding for padded positions
            delta_t = delta_t * mask_float.squeeze(-1)
        
        # Embed flux and add time encoding
        flux_embedded = self.flux_proj(flux.unsqueeze(-1))
        t_emb = self.time_enc(delta_t)
        x = flux_embedded + t_emb
        x = self.emb_norm(x)
        x = self.dropout(x)
        
        # Apply mask after embedding
        if lengths is not None:
            x = x * mask_float
        
        # Causal CNN feature extraction - single transpose in/out
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        if lengths is not None:
            # Prepare mask for CNN: (B, T, 1) -> (B, 1, T) -> broadcast to (B, C, T)
            mask_cnn = mask_float.transpose(1, 2)  # (B, T, 1) -> (B, 1, T)
        
        for block in self.feature_extractor:
            x = block(x)
            if lengths is not None:
                x = x * mask_cnn  # Broadcasts (B, 1, T) to (B, C, T)
        
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        
        # Causal transformer layers
        for layer in self.transformer_layers:
            # Self-attention with residual
            residual = x
            x = layer['norm1'](x)
            x = layer['attn'](x, key_padding_mask=key_padding_mask)
            x = residual + x
            
            # Feed-forward with residual
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['ffn'](x)
        
        # Final classification
        x = self.final_norm(x)
        logits = self.classifier(x)
        
        # Apply learnable temperature scaling
        temp = torch.exp(self.log_temperature)
        scaled_logits = logits / (temp + 1e-8)
        
        # Return only final timestep if requested
        if not return_all_timesteps and lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=device)
            final_logits = scaled_logits[batch_indices, last_idx]
            return {
                'logits': final_logits, 
                'probs': F.softmax(final_logits, dim=-1)
            }
        
        return {
            'logits': scaled_logits, 
            'probs': F.softmax(scaled_logits, dim=-1)
        }

    def get_receptive_field(self) -> int:
        """Return the receptive field size for streaming inference buffer management."""
        return self.receptive_field
    
    def get_module_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return state dict suitable for saving in DDP training.
        If wrapped in DDP, strips the 'module.' prefix.
        """
        if hasattr(self, 'module'):
            return self.module.state_dict()
        return self.state_dict()
