import torch
import torch.nn as nn
class ProjectionLayer(nn.Module):
    
    def __init__(self , d_model:int , vocab_size : int):
        super().__init__()
        
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
        #Dim -1 is the last dimension, Log Softmax is used for numerical stability
