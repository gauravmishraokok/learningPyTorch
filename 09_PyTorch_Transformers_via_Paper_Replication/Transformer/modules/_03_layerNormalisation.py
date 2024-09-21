import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Initializes the class which creates default values for epsilon, alpha, and bias.

    Args:
        eps (float): A small value added for numerical stability and to avoid division by zero. Default is 10**-6.
    """
    
    def __init__(self, eps:float = 10**-6):
        
        super().__init__()
        
        #Epsilon is a small value added for numerical stability and also to avoid division by 0
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones(1))    #This is multiplied
        self.bias = nn.Parameter(torch.zeros(1))    #This is added
        
    
    def forward(self,x):
        mean = x.mean(dim = -1 , keepdim=True) #Usually mean doesnt keep dimension.
        std = x.std(dim = -1 , keepdim=True) #Usually std doesnt keep dimension.
        
        return self.alpha * (x-mean)/torch.sqrt(std + self.eps)   + self.bias
