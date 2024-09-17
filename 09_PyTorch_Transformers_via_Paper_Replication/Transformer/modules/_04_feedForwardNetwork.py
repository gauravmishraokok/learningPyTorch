
import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    Initializes the FeedForward network.

    Args:
        d_model (int): The dimensionality of the input and output features.
        d_ff (int): The dimensionality of the hidden layer.
        dropout (float): The dropout probability to be applied after the first linear layer.

    Attributes:
        linear_1 (nn.Linear): The first linear transformation layer (input to hidden).
        dropout (nn.Dropout): The dropout layer applied after the first linear layer.
        linear_2 (nn.Linear): The second linear transformation layer (hidden to output).
    """
    def __init__(self, d_model: int , d_ff:int , dropout:float):
        super().__init__()
        
        self.linear_1=nn.Linear(in_features=d_model,out_features=d_ff) #W1,B1
        self.dropout = nn.Dropout(p=dropout) 
        
        self.linear_2 = nn.Linear(d_ff,d_model) #W2,B2
        
    def forward(self, x): 
        #(Batch_len , Seq_len , d_model) -> (Batch_len , Seq_len , d_ff) ->(Batch_len , Seq_len , d_model) 
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
