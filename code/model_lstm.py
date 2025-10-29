"""
LSTM-based model for microlensing classification
"""

import torch
import torch.nn as nn


class BidirectionalLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary microlensing classification
    
    Architecture:
    1. Embedding layer (projects 1D input to higher dimension)
    2. Multi-layer Bidirectional LSTM
    3. Attention mechanism (weighted pooling over time)
    4. Classification head
    """
    
    def __init__(self, 
                 sequence_length=1500,
                 input_dim=1,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3,
                 use_attention=True):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Optional: Project input to higher dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        # Output: (batch, seq, hidden_dim * 2) because bidirectional
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional but recommended)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length) or (batch_size, 1, sequence_length)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq, 1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2)  # (batch, 1, seq) -> (batch, seq, 1)
        
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden_dim//4)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq, hidden_dim*2)
        
        # Aggregation
        if self.use_attention:
            # Attention-weighted pooling
            attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # Weighted sum
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        else:
            # Simple mean pooling
            context = lstm_out.mean(dim=1)  # (batch, hidden_dim*2)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits


class SimpleLSTMClassifier(nn.Module):
    """
    Simpler LSTM variant for comparison
    """
    
    def __init__(self,
                 sequence_length=1500,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq, 1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2)
        
        # LSTM - use last hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits


class GRUClassifier(nn.Module):
    """
    GRU variant (often faster than LSTM with similar performance)
    """
    
    def __init__(self,
                 sequence_length=1500,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2)
        
        gru_out, h_n = self.gru(x)
        
        # Use final output (mean over sequence)
        context = gru_out.mean(dim=1)
        
        logits = self.classifier(context)
        
        return logits
