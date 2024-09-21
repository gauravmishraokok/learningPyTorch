import torch
import torch.nn as nn
from modules._03_layerNormalisation import LayerNormalization
from modules._04_feedForwardNetwork import FeedForwardNetwork
from modules._05_multiHeadAttention import MultiHeadAttention
from modules._06_residualConnection import ResidualConnection

class EncoderBlock(nn.Module):
    """
    Implements a single encoder block consisting of a multi-head self-attention mechanism,
    followed by a feed-forward network, each with residual connections and layer normalization.

    Attributes:
        self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward_block (FeedForwardNetwork): Feed-forward network.
        residual_connections (nn.ModuleList): List of residual connections.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float) -> None:
        """
        Initializes the EncoderBlock.

        Args:
            features (int): The number of features for layer normalization.
            self_attention_block (MultiHeadAttention): Multi-head self-attention mechanism.
            feed_forward_block (FeedForwardNetwork): Feed-forward network.
            dropout (float): The dropout rate to be applied.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout, features=features) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, Features).
            src_mask (torch.Tensor): Source mask tensor of shape (Batch, 1, 1, Seq_len) or None.

        Returns:
            torch.Tensor: Output tensor after applying the encoder block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Implements the encoder consisting of multiple encoder blocks and a final layer normalization.

    Attributes:
        layers (nn.ModuleList): List of encoder blocks.
        norm (LayerNormalization): Layer normalization.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initializes the Encoder.

        Args:
            features (int): The number of features for layer normalization.
            layers (nn.ModuleList): List of encoder blocks.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, Features).
            mask (torch.Tensor): Mask tensor of shape (Batch, 1, 1, Seq_len) or None.

        Returns:
            torch.Tensor: Output tensor after applying the encoder.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
