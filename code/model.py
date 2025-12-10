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
    use_gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        
        # Type enforcement
        assert isinstance(self.d_model, int) and self.d_model > 0, "d_model must be positive int"
        assert isinstance(self.n_heads, int) and self.n_heads > 0, "n_heads must be positive int"
        assert isinstance(self.n_transformer_layers, int) and self.n_transformer_layers > 0
        assert isinstance(self.kernel_size, int) and self.kernel_size > 0
        assert isinstance(self.n_conv_layers, int) and self.n_conv_layers > 0
        assert isinstance(self.dilation_growth, int) and self.dilation_growth > 0
        
        # Logical constraints
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_model >= self.n_heads, \
            f"d_model ({self.d_model}) must be >= n_heads ({self.n_heads})"
        assert self.max_seq_len > self.kernel_size, \
            f"max_seq_len ({self.max_seq_len}) must exceed kernel_size ({self.kernel_size})"
        assert 0.0 <= self.dropout < 1.0, \
            f"dropout ({self.dropout}) must be in [0, 1)"
        assert self.n_classes >= 2, \
            f"n_classes ({self.n_classes}) must be >= 2"
        
        # Warn about extreme dilations
        max_dilation = self.dilation_growth ** (self.n_conv_layers - 1)
        max_rf = 2 * (self.kernel_size - 1) * max_dilation
        if max_rf > self.max_seq_len:
            logger.warning(
                f"Max receptive field per block ({max_rf}) exceeds max_seq_len ({self.max_seq_len}). "
                "Consider reducing n_conv_layers or dilation_growth."
            )

# =============================================================================
# CAUSAL CONVOLUTIONS
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Strictly causal 1D convolution with left-padding only.
    Ensures output at time t depends only on inputs at times <= t.
    
    Boundary Conditions (Roman Telescope Streaming):
        The first (kernel_size-1)*dilation timesteps receive PARTIAL context
        due to zero-padding. For streaming inference:
        1. Pre-fill buffer with receptive_field-1 warmup observations
        2. Alternatively, discard predictions for first receptive_field-1 steps
        3. Zero-padding is the scientifically correct approach (no lookahead)
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
        
        # GELU-appropriate initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) tensor with causal padding applied
        """
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))  # Left-pad with zeros (no future info)
        out = self.conv(x)
        return self.dropout(out)

class ChannelLayerNorm(nn.Module):
    """
    LayerNorm that operates on (B, C, T) format by normalizing over C dimension.
    Eliminates expensive transpose operations in CNN blocks.
    Critical for DDP stability with small per-GPU batches.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) normalized tensor
        """
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

class ResidualCausalBlock(nn.Module):
    """
    Residual block with two causal convolutions and channel layer normalization.
    Operates entirely in (B, C, T) format - NO TRANSPOSES.
    Uses ChannelLayerNorm for DDP stability with small per-GPU batches.
    """
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = ChannelLayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)
        self.act = nn.GELU()
        self.norm2 = ChannelLayerNorm(d_model)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size, dilation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) tensor
        """
        residual = x
        z = self.norm1(x)
        z = self.conv1(z)
        z = self.act(z)
        z = self.norm2(z)
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
    Caches causal masks for efficiency on 40-GPU training.
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
        
        # Cache for causal mask (HPC optimization) - FIX: Use None for efficiency
        self.register_buffer('_cached_mask', None, persistent=False)
        self._cached_size = 0
    
    def _get_causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Cache and reuse causal masks for repeated sequence lengths."""
        if T != self._cached_size or self._cached_mask is None or self._cached_mask.device != device:
            mask = torch.zeros(T, T, device=device, dtype=dtype)
            causal_mask = torch.ones(T, T, device=device, dtype=torch.bool).tril()
            mask.masked_fill_(~causal_mask, float('-inf'))
            self._cached_mask = mask
            self._cached_size = T
        return self._cached_mask

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
            # Use cached causal mask
            attn_mask = self._get_causal_mask(T, x.device, q.dtype)
            
            # Expand padding mask and merge with causal mask
            pad_mask_expanded = key_padding_mask.view(B, 1, 1, T).expand(B, self.n_heads, T, T)
            # FIX: Remove unnecessary .clone() - expand creates view, masked_fill_ operates in-place safely
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, T, T)
            attn_mask = attn_mask.masked_fill(pad_mask_expanded, float('-inf'))

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
    Hybrid causal model for real-time microlensing classification.
    
    Architecture:
        1. Causal CNN (exponential dilation): Local feature extraction
        2. Causal Transformer (flash attention): Long-range dependencies
        3. Per-timestep classification: Enables real-time streaming inference
    
    Causality Guarantee:
        Output at time t depends ONLY on inputs at times <= t. Verified via:
        - CausalConv1d: Left-padding only (zero-pad, no future info)
        - FlashCausalAttention: is_causal=True flag
        - No global pooling operations
    
    Boundary Conditions (Roman Telescope Warm-Up):
        First `receptive_field - 1` timesteps receive partial context due to
        zero-padding in causal convolutions. For streaming inference:
        1. Pre-fill buffer with `receptive_field - 1` warmup observations OR
        2. Discard predictions for first `receptive_field - 1` timesteps OR
        3. Accept reduced accuracy for early timesteps (scientifically valid)
    
    Streaming Inference (Roman Telescope):
        For real-time classification, maintain a rolling buffer of size
        `model.get_receptive_field()` to preserve causal history. Example:
```python
        rf = model.get_receptive_field()
        buffer_flux = torch.zeros(1, rf, device=device)
        buffer_time = torch.zeros(1, rf, device=device)
        
        for new_flux, new_time in roman_stream:
            # Roll buffer and insert new observation
            buffer_flux = torch.cat([buffer_flux[:, 1:], new_flux.view(1, 1)], dim=1)
            buffer_time = torch.cat([buffer_time[:, 1:], new_time.view(1, 1)], dim=1)
            
            # Predict on full buffer (only last timestep matters)
            pred = model(buffer_flux, buffer_time, return_all_timesteps=False)
```
    
    NaN Handling:
        - Training: NaNs replaced with 0.0 via torch.nan_to_num
        - Padding: Zero-masking ensures padded positions don't contribute
        - Inference: User must handle NaNs in preprocessing
    
    DDP Training (40-GPU Optimization):
        - Uses ChannelLayerNorm for stability with small per-GPU batches
        - No custom CUDA kernels requiring special DDP configuration
        - CRITICAL: Wrap with `find_unused_parameters=False` for efficiency
        - Save via: `model.save_model(path, is_ddp=True)`
        
        Example DDP initialization:
```python
        model = CausalHybridModel(config).to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
```
    
    Args:
        config: CausalConfig with validated hyperparameters
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
        
        # Calculate and validate receptive field
        self.receptive_field = self._calculate_receptive_field()
        actual_rf = self._verify_receptive_field()
        assert self.receptive_field == actual_rf, \
            f"RF calculation mismatch: calculated={self.receptive_field}, actual={actual_rf}"
            
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
        
        # Apply gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()
        
        # Classification head with proper initialization
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.n_classes)
        )
        
        # Initialize classifier weights
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Learnable temperature for logit scaling
        self.log_temperature = nn.Parameter(torch.tensor([0.0]))
    
    def _calculate_receptive_field(self) -> int:
        """
        Calculate total receptive field of causal CNN layers.
        For each residual block with two convolutions at the same dilation:
        RF_new = RF_old + 2 * (kernel_size - 1) * dilation
        """
        rf = 1
        dilation = 1
        for _ in range(self.config.n_conv_layers):
            # Each residual block has 2 causal convolutions at same dilation
            rf += 2 * (self.config.kernel_size - 1) * dilation
            dilation *= self.config.dilation_growth  # Update for NEXT block
        return rf
    
    def _verify_receptive_field(self) -> int:
        """Empirically verify receptive field by tracing actual architecture."""
        rf = 1
        for block in self.feature_extractor:
            # Each CausalConv1d contributes (k-1)*d to RF
            conv1_rf = (block.conv1.kernel_size - 1) * block.conv1.dilation
            conv2_rf = (block.conv2.kernel_size - 1) * block.conv2.dilation
            rf += conv1_rf + conv2_rf
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
        key_padding_mask: Optional[torch.Tensor] = None
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
        
        # Causal CNN feature extraction - operates in (B, C, T) format
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        if lengths is not None:
            # Prepare mask for CNN: (B, T, 1) -> (B, 1, T) -> broadcast to (B, C, T)
            mask_cnn: torch.Tensor = mask_float.transpose(1, 2)  # (B, T, 1) -> (B, 1, T)
        
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
        temp: torch.Tensor = torch.exp(self.log_temperature)
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
    
    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory efficiency.
        Reduces activation memory by ~40% at cost of ~20% compute overhead.
        Essential for 40-GPU training with limited per-node memory.
        """
        self._log("Enabling gradient checkpointing for transformer layers", "info")
        for layer in self.transformer_layers:
            # Wrap attention and FFN in checkpoint wrappers
            layer['attn'] = torch.utils.checkpoint.checkpoint_wrapper(
                layer['attn'], preserve_rng_state=True
            )
            layer['ffn'] = torch.utils.checkpoint.checkpoint_wrapper(
                layer['ffn'], preserve_rng_state=True
            )
    
    def freeze_temperature(self):
        """Freeze temperature scaling for controlled experiments."""
        self.log_temperature.requires_grad = False
        self._log("Frozen temperature parameter", "info")
    
    def freeze_embeddings(self):
        """Freeze input projection and time encoding."""
        self.flux_proj.weight.requires_grad = False
        self.flux_proj.bias.requires_grad = False
        for param in self.time_enc.parameters():
            param.requires_grad = False
        self._log("Frozen embedding layers", "info")
    
    def save_model(self, path: str, is_ddp: bool = False, metadata: Optional[Dict] = None):
        """
        Save model state dict with optional metadata.
        Handles DDP prefix stripping automatically.
        
        Args:
            path: Output file path (.pth or .pt)
            is_ddp: Whether model is wrapped in DistributedDataParallel
            metadata: Optional dict with training info (epoch, loss, optimizer state, etc.)
        
        Example:
            >>> model.save_model('checkpoint.pth', is_ddp=True, 
            ...                  metadata={'epoch': 10, 'val_acc': 0.94})
        """
        state = self.module.state_dict() if is_ddp else self.state_dict()
        save_dict = {
            'model_state_dict': state, 
            'config': self.config.__dict__,
            'receptive_field': self.receptive_field
        }
        if metadata:
            save_dict['metadata'] = metadata
        torch.save(save_dict, path)
        self._log(f"Model saved to {path}", "info")
    
    @classmethod
    def load_model(cls, path: str, device: torch.device = torch.device('cpu')) -> 'CausalHybridModel':
        """
        Load model from checkpoint with automatic config reconstruction.
        
        Args:
            path: Checkpoint file path
            device: Device to load model onto
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        config = CausalConfig(**checkpoint['config'])
        model = cls(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path} (RF={checkpoint.get('receptive_field', 'unknown')})")
        return model
    
    def _log(self, msg: str, level: str = 'info'):
        """
        Rank-0 only logging for DDP training.
        Prevents log spam on 40-GPU training runs.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            getattr(logger, level)(msg)
    
    @staticmethod
    def set_init_seed(seed: int = 42):
        """
        Set seed for deterministic weight initialization (thesis reproducibility).
        Must be called BEFORE model instantiation.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def set_distributed_seed(seed: int, rank: int):
        """
        Set rank-specific seed for distributed training reproducibility.
        Ensures different data augmentation per GPU while maintaining reproducibility.
        
        Args:
            seed: Base random seed
            rank: GPU rank in distributed training
        
        Usage:
            >>> CausalHybridModel.set_distributed_seed(42, rank)  # Call before training loop
        """
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        import numpy as np
        import random
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    
    @staticmethod
    def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove 'module.' prefix from DDP-wrapped model state dict."""
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
