import torch
import torch.nn as nn
import math
from modules._03_layerNormalisation import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        norm (LayerNormalization): Layer normalization.
    """
    
    def __init__(self, dropout: float, features: int):
        """
        Initializes the ResidualConnection module.

        Args:
            dropout (float): The dropout rate to be applied.
            features (int): The number of features for layer normalization.
        """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass for the residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, Features).
            sublayer (nn.Module): The sublayer to apply the residual connection to.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection and layer normalization.
        """
        return x + self.dropout(sublayer(self.norm(x)))
