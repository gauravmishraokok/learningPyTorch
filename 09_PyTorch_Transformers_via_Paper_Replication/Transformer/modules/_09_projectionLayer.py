import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """
    A projection layer that projects the input tensor to the vocabulary size and applies log softmax.

    Attributes:
        proj (nn.Linear): Linear layer to project the input tensor to the vocabulary size.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the ProjectionLayer.

        Args:
            d_model (int): The dimension of the input tensor.
            vocab_size (int): The size of the vocabulary.

        """
        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_len, vocab_size) after applying log softmax.
        """
        return torch.log_softmax(self.proj(x), dim=-1)
        # Dim -1 is the last dimension, Log Softmax is used for numerical stability
