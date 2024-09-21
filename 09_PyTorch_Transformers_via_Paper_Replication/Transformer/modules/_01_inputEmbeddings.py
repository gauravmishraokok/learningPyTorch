import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int ):
        
        """
    Initialize the embedding layer.

    Args:
        d_model (int): Dimensionality of the embedding vectors. For example, d_model = 512 means each word is represented by a 512-dimensional vector.
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.

    """
        
        super().__init__()
        self.d_model = d_model  #Dimensionality -> d_model = 512: You choose to represent each word by a 512-dimensional vector.
        self.vocab_size = vocab_size #Number of Tokens 
        self.embedding = nn.Embedding(vocab_size,d_model)
        
        
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
        
