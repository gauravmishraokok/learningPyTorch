

import torch
import torch.nn as nn
from modules._03_layerNormalisation import LayerNormalization
from modules._04_feedForwardNetwork import FeedForwardNetwork
from modules._05_multiHeadAttention import MultiHeadAttention
from modules._06_residualConnection import ResidualConnection

class EncoderBlock(nn.Module):
    
    def __init__(self,self_attention_block : MultiHeadAttention , feed_forward_block : FeedForwardNetwork, dropout:float):
        super().__init__()
        #Initialising the self attention and the feed forward block
        self.self_attention_block = self_attention_block 
        self.feed_forward_block = feed_forward_block
        
        # Initialising the residual connections, 2 because we have 2 sublayers
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)]) 
        
        
    def forward(self, x , src_mask):
        #src_mask -> Source mask -> Used to mask out the padding tokens
        x = ResidualConnection[0](x,lambda x : self.self_attention_block(x,x,x,src_mask))
        x = ResidualConnection[1](x,self.feed_forward_block(x))
            
        return x
        
class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x, mask):
        
        for layer in self.layers:
            x=layer(x,mask)
