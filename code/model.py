#!/usr/bin/env python3
"""
Unified TimeDistributedCNN model definition
"""
import torch
import torch.nn as nn
import config as CFG

class TimeDistributedCNN(nn.Module):
    """
    CNN that outputs predictions at each timestep.
    Aggregated via mean/max pooling at inference time.
    """
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super().__init__()
        c1, c2, c3 = CFG.CONV1_FILTERS, CFG.CONV2_FILTERS, CFG.CONV3_FILTERS
        
        # Feature extraction
        self.conv1 = nn.Conv1d(num_channels, c1, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(c3)
        
        self.dropout = nn.Dropout(CFG.DROPOUT_RATE)
        self.relu = nn.ReLU()
        
        # Classification head (applied to each timestep)
        self.fc1 = nn.Linear(c3, CFG.FC1_UNITS)
        self.fc2 = nn.Linear(CFG.FC1_UNITS, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, L] input sequences
        Returns:
            [B, L, num_classes] predictions at each timestep
        """
        # x: [B, 1, L]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, c1, L]
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))  # [B, c2, L]
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))  # [B, c3, L]
        x = self.dropout(x)
        
        # Transpose to [B, L, c3] for FC layers
        x = x.transpose(1, 2)  # [B, L, c3]
        
        # Apply classifier at each timestep
        x = self.relu(self.fc1(x))  # [B, L, FC1_UNITS]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, L, num_classes]
        
        return x