#!/usr/bin/env python3
"""
Hybrid GRU-Transformer Architecture

Defines a PyTorch module combining a Gated Recurrent Unit (GRU) for local 
temporal processing and a Transformer Decoder for long-range dependency modeling.
Enforces strict causality via attention masking.

Key Implementation Details:
- Input Embedding: Projects flux and delta_t into d_model dimensions.
- Recurrent Layer: GRU processes sequential inputs to form a hidden state.
- Attention Mechanism: Causal Transformer Decoder with look-ahead masking.
- Padding Logic: Post-LayerNorm masking ensures zero-contribution from padded indices.
- Inference Caching: Supports incremental state updates (KV-caching) for step-by-step inference.
- Stabilization: Temperature scaling applied to final logits.

Author: Kunal Bhatia
Version: 1.0
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, Optional, Tuple, List
import logging
import math
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kunal_bhatia_v2_hardened")

@dataclass
class CausalConfig:
    d_model: int = 128
    n_heads: int = 8
    n_transformer_layers: int = 2
    n_gru_layers: int = 1
    dropout: float = 0.1
    attention_window: int = 64
    n_classes: int = 3
    
    use_incremental_state: bool = True
    classifier_hidden_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.classifier_hidden_dim is None:
            self.classifier_hidden_dim = self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

class ContinuousSinusoidalEncoding(nn.Module):
    """
    Anti-Cheating Temporal Encoding.
    """
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_timescale = max_timescale
        
        num_frequencies = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(max_timescale) / num_frequencies)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # Strict Causal Check: Clip negative time (impossible in reality, possible in dirty data)
        delta_t = torch.clamp(delta_t, min=0.0)
        time_val = torch.nan_to_num(delta_t, nan=eps, posinf=self.max_timescale, neginf=eps)
        time_val = torch.clamp(time_val, min=eps, max=self.max_timescale)
        
        time_val = time_val.unsqueeze(-1)
        scaled_time = time_val * self.div_term
        pe_sin = torch.sin(scaled_time)
        pe_cos = torch.cos(scaled_time)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        
        if self.d_model % 2 == 1:
            pe = F.pad(pe, (0, 1))
            
        return pe

class StrictCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.window_size = window_size
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _create_strict_causal_mask(self, n_q: int, n_k: int, device, query_offset: int = 0) -> torch.Tensor:
        # Row idx (queries) vs Col idx (keys)
        row_idx = torch.arange(n_q, device=device).unsqueeze(1) + query_offset 
        col_idx = torch.arange(n_k, device=device).unsqueeze(0) 
        
        # Strict Causal: Key must exist before or at the same time as Query
        causal_mask = col_idx <= row_idx
        
        # Window attention (Local)
        window_mask = (row_idx - col_idx) <= self.window_size
        
        return causal_mask & window_mask
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                incremental_state: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, N, D = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        query_offset = 0
        
        # --- FIXED: Handle Incremental State & Cached Masks Correctly ---
        if incremental_state is not None:
            if 'prev_k' in incremental_state:
                prev_k = incremental_state['prev_k']
                prev_v = incremental_state['prev_v']
                query_offset = prev_k.size(2)
                
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            
            incremental_state['prev_k'] = k
            incremental_state['prev_v'] = v
        
        # Attention Calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        n_k = k.size(2)
        causal_mask = self._create_strict_causal_mask(N, n_k, x.device, query_offset)
        
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # --- FIXED: Logic for Padding Mask in Incremental Mode ---
        if padding_mask is not None:
            full_padding_mask = padding_mask
            
            # If we are in incremental mode, we need to reconcile the padding mask
            # with the history. If padding_mask is only for the *current* step (N=1),
            # we need to recover the mask for the cached keys.
            if incremental_state is not None:
                # Retrieve cached mask or create one if this is the first step
                # NOTE: In a robust system, you must cache the mask too, 
                # otherwise you assume history was valid.
                prev_mask = incremental_state.get('prev_key_padding_mask', None)
                
                if prev_mask is not None:
                    full_padding_mask = torch.cat([prev_mask, padding_mask], dim=1)
                
                # Update cache
                incremental_state['prev_key_padding_mask'] = full_padding_mask
            
            # Safety align: If mask is wider than keys (rare), crop it.
            if full_padding_mask.size(1) > n_k:
                 full_padding_mask = full_padding_mask[:, :n_k]
            
            # Safety align: If mask is narrower than keys (e.g. initial cache miss), pad it
            if full_padding_mask.size(1) < n_k:
                diff = n_k - full_padding_mask.size(1)
                # Assume valid if unknown (or handle stricter)
                ones_pad = torch.ones(B, diff, device=full_padding_mask.device, dtype=torch.bool)
                full_padding_mask = torch.cat([ones_pad, full_padding_mask], dim=1)

            # Apply mask
            key_mask = full_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, D)
        
        output = self.out_proj(attn_out)
        return output, incremental_state

class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 window_size: int = 64):
        super().__init__()
        self.attention = StrictCausalAttention(d_model, n_heads, dropout, window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                incremental_state: Optional[Dict] = None):
        
        # Residual Connection 1
        resid = x
        x = self.norm1(x)
        attn_out, incremental_state = self.attention(x, padding_mask, incremental_state)
        x = resid + attn_out
        
        # Residual Connection 2
        resid = x
        x = self.norm2(x)
        x = resid + self.ffn(x)
        
        return x, incremental_state

class CausalHybridModel(nn.Module):
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.flux_embedding = nn.Linear(1, self.d_model)
        self.embedding_norm = nn.LayerNorm(self.d_model)
        
        self.temporal_encoding = ContinuousSinusoidalEncoding(d_model=self.d_model)
        
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=config.n_gru_layers,
            batch_first=True,
            dropout=config.dropout if config.n_gru_layers > 1 else 0.0
        )
        
        self.gru_norm = nn.LayerNorm(self.d_model)
        self.gru_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=self.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                window_size=config.attention_window
            )
            for _ in range(config.n_transformer_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, config.classifier_hidden_dim),
            nn.LayerNorm(config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.n_classes)
        )
        
        # FIXED: Initialize to 1.0, but clamp logic applied in forward
        self.temperature = nn.Parameter(torch.ones(1))

    def create_padding_mask_from_lengths(self, lengths: torch.Tensor, 
                                     max_len: int) -> torch.Tensor:
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask
    
    def init_incremental_state(self, batch_size: int, device: torch.device) -> List[Dict]:
        return [{} for _ in range(len(self.layers) + 1)] 

    def forward(self, flux: torch.Tensor, delta_t: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                return_all_timesteps: bool = True,
                incremental_state: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        
        B, N = flux.shape
        if lengths is None:
            lengths = torch.full((B,), N, dtype=torch.long, device=flux.device)
        
        # Create mask for current sequence [B, N]
        padding_mask = self.create_padding_mask_from_lengths(lengths, N)
        
        # 1. Embeddings
        flux_safe = torch.nan_to_num(flux, nan=0.0, posinf=1e5, neginf=-1e5)
        flux_safe = flux_safe * padding_mask.float()
        flux_norm = torch.sign(flux_safe) * torch.log1p(torch.abs(flux_safe))
        
        x = self.flux_embedding(flux_norm.unsqueeze(-1))
        
        # --- FIXED: Apply mask after Norm/Embedding to kill "LayerNorm Shift" artifacts ---
        # LayerNorm shifts 0s to Beta. We must re-zero them to prevent bias.
        x = self.embedding_norm(x)
        x = x * padding_mask.unsqueeze(-1).float() 
        
        temporal = self.temporal_encoding(delta_t)
        x = x + temporal
        x = x * padding_mask.unsqueeze(-1).float() # Re-apply mask after add
        
        # 2. GRU
        if incremental_state is not None:
            gru_state = incremental_state[0].get('gru_hidden', None)
            
            # --- FIXED: Handle Variable Batch Sizes in Streaming ---
            if gru_state is not None and gru_state.size(1) != B:
                 gru_state = gru_state[:, :B, :].contiguous()

            gru_out, gru_hidden = self.gru(x, gru_state)
            
            # --- FIXED: Prevent GRU state update if the sample is padding ---
            # In streaming, if a sample is "done", we shouldn't update its state with garbage
            if gru_state is not None:
                # Identify which samples in batch are valid (length > 0)
                is_valid = (lengths > 0).view(1, B, 1).float()
                # Interpolate: Keep old state if invalid, use new state if valid
                gru_hidden = is_valid * gru_hidden + (1.0 - is_valid) * gru_state
                
            incremental_state[0]['gru_hidden'] = gru_hidden
        else:
            cpu_lengths = lengths.cpu()
            packed_x = pack_padded_sequence(x, cpu_lengths, batch_first=True, enforce_sorted=False)
            packed_gru_out, _ = self.gru(packed_x)
            gru_out, _ = pad_packed_sequence(packed_gru_out, batch_first=True, total_length=N)
            
        x = self.gru_norm(gru_out)
        
        # --- FIXED: Re-apply Zero Mask after GRU Norm ---
        # Crucial: pad_packed_sequence returns 0s, but LayerNorm turns 0s into noise.
        x = x * padding_mask.unsqueeze(-1).float()
        x = self.gru_dropout(x)
        
        # 3. Transformer
        for i, layer in enumerate(self.layers):
            layer_dict = incremental_state[i+1] if incremental_state is not None else None
            x, _ = layer(x, padding_mask, layer_dict)
            
            # Mask after every block to keep the "void" clean
            x = x * padding_mask.unsqueeze(-1).float()
            
        x = self.final_norm(x)
        x = x * padding_mask.unsqueeze(-1).float()
        
        # 4. Classification
        if return_all_timesteps:
            logits = self.classifier(x)
        else:
            final_indices = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.d_model)
            final_hidden = x.gather(1, final_indices).squeeze(1)
            logits = self.classifier(final_hidden)
            
        # --- FIXED: Clamp temperature to prevent numerical cheating ---
        clamped_temp = torch.clamp(self.temperature, min=0.5, max=5.0)
        scaled_logits = logits / clamped_temp
        
        if return_all_timesteps:
            probs = F.softmax(scaled_logits, dim=2)
            confidence, predictions = probs.max(dim=2)
        else:
            probs = F.softmax(scaled_logits, dim=1)
            confidence, predictions = probs.max(dim=1)
            
        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'confidence': confidence
        }

def verify_anti_cheating(model):
    logger.info("\n--- STARTING HARDENED ANTI-CHEATING VERIFICATION ---")
    model.eval()
    
    B, N = 1, 15
    flux = torch.randn(B, N)
    delta_t = torch.abs(torch.randn(B, N)) + 0.1
    
    # Run 1: Original
    with torch.no_grad():
        out1 = model(flux, delta_t, return_all_timesteps=True)
        preds1 = out1['probs']
        
    # Run 2: Modify future (Test strict causality)
    flux_mod = flux.clone()
    delta_mod = delta_t.clone()
    flux_mod[0, -1] = 99999.0 
    delta_mod[0, -1] = 10000.0 
    
    with torch.no_grad():
        out2 = model(flux_mod, delta_mod, return_all_timesteps=True)
        preds2 = out2['probs']
        
    diff = (preds1[:, :-1, :] - preds2[:, :-1, :]).abs().max().item()
    logger.info(f"Max difference in past predictions (Causality Leak): {diff:.9f}")
    
    if diff < 1e-7:
        logger.info("✅ SUCCESS: Model is STRICTLY CAUSAL.")
    else:
        logger.error("❌ FAILURE: Future leaked into past!")

    # --- NEW TEST: Padding Artifact Verification ---
    logger.info("--- VERIFYING PADDING ARTIFACTS ---")
    # Feed a batch where the last 5 tokens are padding
    flux_pad = flux.clone()
    lengths = torch.tensor([10]) # Only first 10 valid
    
    with torch.no_grad():
        out_pad = model(flux_pad, delta_t, lengths=lengths, return_all_timesteps=True)
        # Check if output at index 11 (padding) is strictly distinct or zero?
        # Actually, if we mask post-layer-norm, the vector should be close to 0
        pad_vector_norm = torch.norm(out_pad['logits'][0, 11]).item()
        
    logger.info(f"Norm of output vector at padding location: {pad_vector_norm:.9f}")
    if pad_vector_norm < 1e-2: # Allow small bias term from final Linear layer
        logger.info("✅ SUCCESS: Padding tokens contain no ghost signal.")
    else:
        logger.warning(f"⚠️ WARNING: Padding tokens have high norm ({pad_vector_norm}). Model 'sees' padding.")

    # Verify Online Inference
    logger.info("--- VERIFYING ONLINE vs BATCH CONSISTENCY ---")
    incremental_state = model.init_incremental_state(1, flux.device)
    online_preds = []
    
    with torch.no_grad():
        for t in range(N):
            obs_f = flux[:, t:t+1]
            obs_d = delta_t[:, t:t+1]
            step_len = torch.tensor([1], device=flux.device)
            
            out_step = model(obs_f, obs_d, lengths=step_len, 
                             return_all_timesteps=True, 
                             incremental_state=incremental_state)
            online_preds.append(out_step['probs'][0, 0])
            
    online_preds = torch.stack(online_preds)
    batch_preds = preds1[0]
    
    diff_online = (online_preds - batch_preds).abs().max().item()
    logger.info(f"Max difference Batch vs Online: {diff_online:.9f}")
    
    if diff_online < 1e-5:
        logger.info("✅ SUCCESS: Online inference matches Batch.")
    else:
        logger.error("❌ FAILURE: Online inference calculation is wrong.")

if __name__ == '__main__':
    config = CausalConfig(d_model=64, n_heads=4, n_transformer_layers=2)
    model = CausalHybridModel(config)
    verify_anti_cheating(model)
