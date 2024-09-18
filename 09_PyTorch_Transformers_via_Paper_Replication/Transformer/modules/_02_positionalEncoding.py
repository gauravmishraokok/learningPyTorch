import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Initializes the positional encoding module.

    Args:
        d_model (int): The dimension of the model.
        seq_len (int): The length of the sequence.
        dropout (float): The dropout rate.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Creating a Matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Creating a position vector of length seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [0,1,2,...,n]
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # This comes from the denominator of the function.
        
        # Applying sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(dim=0)  # [1, seq_len, d_model]
        
        # Register Buffer 
        # --> Is used for saving positional encoding to model's state_dict as it is not updated during any backward propagation step but is needed for reliability and reusability.
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Input tokens  --> Input tokens  + Position of the respective tokens
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        
        # Dropout layer 
        return self.dropout(x)
