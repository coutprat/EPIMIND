"""
Lightweight EEG models for seizure detection.
Designed to be Jetson-friendly with minimal parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEEGCNN(nn.Module):
    """
    Tiny CNN for EEG seizure detection.
    
    Input: (batch, channels, timesteps)
    Output: (batch, 1) - single logit for BCE with logits loss
    
    Architecture:
    - 3 conv layers with small kernels
    - Adaptive pooling to handle variable input sizes
    - 2 fully connected layers
    - ~50K parameters (Jetson-friendly)
    """
    
    def __init__(
        self,
        num_channels: int = 23,
        num_timesteps: int = 512,
        dropout: float = 0.3,
    ):
        """
        Initialize TinyEEGCNN.
        
        Args:
            num_channels: Number of EEG channels (e.g., 23)
            num_timesteps: Length of each EEG window (e.g., 512)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        # Conv block 1: (batch, 23, 512) -> (batch, 32, 256)
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv block 2: (batch, 32, 128) -> (batch, 64, 64)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv block 3: (batch, 64, 32) -> (batch, 128, 16)
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling -> (batch, 128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, timesteps)
            
        Returns:
            Output logits (batch, 1)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout_layer(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout_layer(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EEGResBlock(nn.Module):
    """Simple residual block for EEG processing."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            residual = self.skip(x)
        
        out = out + residual
        out = F.relu(out)
        
        return out


class TinyEEGResNet(nn.Module):
    """
    Tiny ResNet for EEG seizure detection.
    Alternative to TinyEEGCNN with residual connections.
    
    Input: (batch, channels, timesteps)
    Output: (batch, 1) - single logit
    
    Parameters: ~40K (even more Jetson-friendly)
    """
    
    def __init__(
        self,
        num_channels: int = 23,
        num_timesteps: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Initial conv: (batch, 23, 512) -> (batch, 32, 512)
        self.conv_in = nn.Conv1d(
            num_channels, 32, kernel_size=7, padding=3, bias=False
        )
        self.bn_in = nn.BatchNorm1d(32)
        
        # Residual blocks with downsampling
        self.layer1 = self._make_layer(32, 32, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.layer2 = self._make_layer(32, 64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.layer3 = self._make_layer(64, 128, kernel_size=3, stride=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> nn.Module:
        """Create residual block."""
        return EEGResBlock(in_channels, out_channels, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.pool2(x)
        
        x = self.layer3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation and parameter count
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Test TinyEEGCNN
    print("=" * 60)
    print("TinyEEGCNN")
    print("=" * 60)
    model1 = TinyEEGCNN(num_channels=23, num_timesteps=512)
    model1.to(device)
    print(f"Parameters: {model1.count_parameters():,}")
    
    # Forward pass
    x = torch.randn(4, 23, 512).to(device)
    y = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test TinyEEGResNet
    print("\n" + "=" * 60)
    print("TinyEEGResNet")
    print("=" * 60)
    model2 = TinyEEGResNet(num_channels=23, num_timesteps=512)
    model2.to(device)
    print(f"Parameters: {model2.count_parameters():,}")
    
    # Forward pass
    y = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
