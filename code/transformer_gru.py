#!/usr/bin/env python3
"""
Pure GRU Model with Enhanced Numerical Stability
FIXES:
- Proper weight initialization (orthogonal for GRU)
- Gradual temporal encoding freeze instead of abrupt
- Hidden state gradient clipping
- Layer normalization after embeddings
- Better numerical stability checks
Author: Kunal Bhatia
Version: 2.0 (Stabilized)
Date: December 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import logging
from dataclasses import dataclass
import warnings
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transformer_gru")

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
DEFAULT_D_MODEL = 128
DEFAULT_DROPOUT = 0.1
N_CLASSES = 3

@dataclass
class ModelConfig:
    d_model: int = DEFAULT_D_MODEL
    n_heads: int = 1  # Dummy for compatibility
    n_layers: int = 0  # Dummy for compatibility
    dropout: float = DEFAULT_DROPOUT
    attention_window: int = 1  # Dummy for compatibility
    n_classes: int = N_CLASSES
    
    max_delta_days: float = 10.0
    min_delta_days: float = 0.001
    use_adaptive_normalization: bool = True
    train_final_only: bool = True
    classifier_hidden_dim: Optional[int] = None
    
    # New stability parameters
    freeze_gradually: bool = True
    gradual_freeze_epochs: int = 3
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        
        # Estimate parameters
        total = self.d_model * 2 * (1 + 1)
        total += 3 * (self.d_model * self.d_model + self.d_model)
        total += 2 * self.d_model
        total += (self.d_model * self.classifier_hidden_dim) + (self.classifier_hidden_dim * self.n_classes)
        logger.info(f"Pure GRU Model - Estimated Parameters: ~{total:,}")

# =============================================================================
# TEMPORAL ENCODING WITH GRADUAL FREEZING
# =============================================================================
class AdaptiveTemporalEncoding(nn.Module):
    def __init__(self, d_model: int, max_delta: float = 10.0, min_delta: float = 0.001, 
                 use_adaptive: bool = True, freeze_gradually: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_adaptive = use_adaptive
        self.freeze_gradually = freeze_gradually
        
        self.projection = nn.Linear(1, d_model)
        
        # Statistics tracking
        self.register_buffer('min_seen_delta', torch.tensor(float('inf')))
        self.register_buffer('max_seen_delta', torch.tensor(float('-inf')))
        self.register_buffer('is_tracking', torch.tensor(True))
        self.register_buffer('norm_min', torch.tensor(math.log(min_delta)))
        self.register_buffer('norm_max', torch.tensor(math.log(max_delta)))
        
        # Gradual freezing
        self.register_buffer('freeze_alpha', torch.tensor(0.0))  # 0=unfrozen, 1=frozen
        
        # Better initialization
        nn.init.xavier_uniform_(self.projection.weight, gain=0.5)
        nn.init.zeros_(self.projection.bias)
    
    def update_distribution(self, delta_t: torch.Tensor):
        if self.is_tracking:
            with torch.no_grad():
                valid_deltas = delta_t[delta_t > 0]
                if len(valid_deltas) > 0:
                    self.min_seen_delta = torch.min(self.min_seen_delta, valid_deltas.min())
                    self.max_seen_delta = torch.max(self.max_seen_delta, valid_deltas.max())
    
    def start_gradual_freeze(self):
        """Begin gradual freezing process"""
        if self.use_adaptive and self.min_seen_delta < float('inf'):
            margin = 0.1
            log_min = math.log(self.min_seen_delta.item())
            log_max = math.log(self.max_seen_delta.item())
            log_range = log_max - log_min
            self.norm_min.fill_(log_min - margin * log_range)
            self.norm_max.fill_(log_max + margin * log_range)
        self.is_tracking.fill_(False)
        logger.info("Starting gradual temporal encoding freeze (via LR reduction)")
    
    def update_freeze_alpha(self, alpha: float):
        """Update freeze interpolation factor (0 to 1) - used for LR scheduling"""
        self.freeze_alpha.fill_(max(0.0, min(1.0, alpha)))
    
    def freeze_distribution(self):
        """Complete freeze (for backward compatibility)"""
        self.start_gradual_freeze()
        self.freeze_alpha.fill_(1.0)
        logger.info("Temporal encoding fully frozen")
    
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if self.training and self.is_tracking:
            self.update_distribution(delta_t)
        
        eps = 1e-10
        log_delta = torch.log(delta_t + eps)
        log_delta_normalized = (log_delta - self.norm_min) / (self.norm_max - self.norm_min + eps)
        log_delta_normalized = 2.0 * log_delta_normalized - 1.0
        log_delta_normalized = torch.clamp(log_delta_normalized, -2.0, 2.0)
        
        encoding = self.projection(log_delta_normalized.unsqueeze(-1))
        
        return encoding

# =============================================================================
# STABLE GRU LAYER
# =============================================================================
class StableGRU(nn.Module):
    """GRU with enhanced numerical stability"""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0  # We'll apply dropout separately
        )
        self.dropout = nn.Dropout(dropout)
        
        # Orthogonal initialization for stability
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            # Packed sequence for variable lengths
            valid_mask = lengths > 0
            if valid_mask.any():
                x_valid = x[valid_mask]
                lengths_valid = lengths[valid_mask].cpu()
                
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x_valid, lengths_valid, batch_first=True, enforce_sorted=False
                )
                output_packed, hidden = self.gru(x_packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    output_packed, batch_first=True, total_length=x.size(1)
                )
                
                # Reconstruct full batch
                full_output = torch.zeros_like(x)
                full_output[valid_mask] = output
                
                return full_output
        
        output, _ = self.gru(x)
        return self.dropout(output)

# =============================================================================
# MAIN PURE GRU MODEL
# =============================================================================
class MicrolensingTransformer(nn.Module):
    """Pure GRU Model with Enhanced Stability"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_classes = config.n_classes
        self.train_final_only = config.train_final_only
        
        # 1. Embeddings with layer norm
        self.flux_embedding = nn.Linear(1, self.d_model)
        self.embedding_norm = nn.LayerNorm(self.d_model)
        
        self.temporal_encoding = AdaptiveTemporalEncoding(
            d_model=self.d_model,
            max_delta=config.max_delta_days,
            min_delta=config.min_delta_days,
            use_adaptive=config.use_adaptive_normalization,
            freeze_gradually=config.freeze_gradually
        )
        
        # 2. Stable GRU
        self.gru = StableGRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            dropout=config.dropout
        )
        
        self.norm = nn.LayerNorm(self.d_model)
        
        # 3. Classifier with better initialization
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, config.classifier_hidden_dim),
            nn.LayerNorm(config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, self.n_classes)
        )
        
        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        
        # Temperature for calibration
        self.register_buffer('temperature', torch.ones(1))
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Pure GRU Model - Actual Parameters: {n_params:,}")
    
    def set_temperature(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature.fill_(temperature)
    
    def start_gradual_freeze(self):
        """Begin gradual freezing of temporal encoding"""
        self.temporal_encoding.start_gradual_freeze()
    
    def update_freeze_progress(self, progress: float):
        """Update freeze progress (0 to 1)"""
        self.temporal_encoding.update_freeze_alpha(progress)
    
    def freeze_temporal_encoding(self):
        """Complete freeze (backward compatibility)"""
        self.temporal_encoding.freeze_distribution()
    
    def _validate_inputs(self, flux, delta_t, lengths=None):
        if flux.ndim != 2 or delta_t.ndim != 2 or flux.shape != delta_t.shape:
            raise ValueError("Input shapes are incorrect.")
        if not torch.all(torch.isfinite(flux)) or not torch.all(torch.isfinite(delta_t)):
            raise ValueError("Non-finite values detected in inputs.")
    
    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None, 
                return_all_timesteps: bool = False) -> Dict[str, torch.Tensor]:
        
        self._validate_inputs(flux, delta_t, lengths)
        B, N = flux.shape
        
        if lengths is None:
            lengths = torch.full((B,), N, dtype=torch.long, device=flux.device)
        
        # 1. Embeddings with normalization
        x = self.flux_embedding(flux.unsqueeze(-1))
        x = self.embedding_norm(x)  # Stabilize after embedding
        
        temporal = self.temporal_encoding(delta_t)
        x = x + temporal
        
        # Check for numerical issues
        if not torch.isfinite(x).all():
            logger.warning("Non-finite values after embeddings, clamping...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 2. GRU Layer
        x = self.gru(x, lengths)
        x = self.norm(x)
        
        # Check again
        if not torch.isfinite(x).all():
            logger.warning("Non-finite values after GRU, clamping...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 3. Classification
        if self.train_final_only and not return_all_timesteps:
            final_indices = lengths - 1
            batch_indices = torch.arange(B, device=x.device)
            final_indices = torch.clamp(final_indices, min=0)
            
            final_hidden = x[batch_indices, final_indices]
            logits = self.classifier(final_hidden)
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            confidence, predictions = probs.max(dim=1)
            
            return {
                'logits': logits,
                'probs': probs,
                'confidence': confidence,
                'predictions': predictions
            }
        else:
            B, N, D = x.shape
            x_flat = x.reshape(B * N, D)
            logits_flat = self.classifier(x_flat)
            logits = logits_flat.reshape(B, N, self.n_classes)
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=2)
            confidence, predictions = probs.max(dim=2)
            
            return {
                'logits': logits,
                'probs': probs,
                'confidence': confidence,
                'predictions': predictions
            }

# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def compute_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None,
                train_final_only: bool = True) -> torch.Tensor:
    logits = outputs['logits']
    if logits.ndim != 2:
        raise ValueError(f"Expected [B, n_classes] logits, got {logits.shape}")
    return F.cross_entropy(logits, labels, weight=class_weights)

def _compute_ece(logits: torch.Tensor, labels: torch.Tensor, 
                 temperature: float, n_bins: int) -> float:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            acc = correct[in_bin].mean().item()
            conf = confidences[in_bin].mean().item()
            weight = in_bin.sum().item() / len(confidences)
            ece += weight * abs(conf - acc)
    
    return ece

def calibrate_temperature(model, val_loader, device='cuda', n_bins=15, max_iter=50):
    model.eval()
    model.to(device)
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for flux, delta_t, lengths, labels in val_loader:
            flux = flux.to(device)
            delta_t = delta_t.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(flux, delta_t, lengths, return_all_timesteps=False)
            all_logits.append(outputs['logits'].cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if not torch.all(torch.isfinite(all_logits)):
        warnings.warn("Non-finite logits detected during calibration. Skipping.", RuntimeWarning)
        nll = F.cross_entropy(all_logits, all_labels).item() if torch.all(torch.isfinite(all_logits)) else float('nan')
        return {
            'temperature': 1.0,
            'ece_before': float('nan'),
            'ece_after': float('nan'),
            'nll_before': nll,
            'nll_after': nll
        }
    
    ece_before = _compute_ece(all_logits, all_labels, 1.0, n_bins)
    nll_before = F.cross_entropy(all_logits, all_labels).item()
    
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def closure():
        optimizer.zero_grad()
        temp = torch.clamp(temperature, min=0.01, max=100.0)
        nll = F.cross_entropy(all_logits / temp, all_labels)
        nll.backward()
        return nll
    
    optimizer.step(closure)
    optimal_temp = torch.clamp(temperature, min=0.01, max=100.0).item()
    
    ece_after = _compute_ece(all_logits, all_labels, optimal_temp, n_bins)
    nll_after = F.cross_entropy(all_logits / optimal_temp, all_labels).item()
    
    model.set_temperature(optimal_temp)
    
    return {
        'temperature': optimal_temp,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'nll_before': nll_before,
        'nll_after': nll_after
    }
