
import torch
import torch.nn as nn
import math
from modules._03_layerNormalisation import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Makes a residual connection

    Args:
        dropout (float): The dropout rate in the Dropout layer.
    """
    def __init__(self,dropout):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x))) + x
