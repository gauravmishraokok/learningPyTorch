import torch
import torch.nn as nn

from modules._03_layerNormalisation import LayerNormalization
from modules._04_feedForwardNetwork import FeedForwardNetwork
from modules._05_multiHeadAttention import MultiHeadAttention
from modules._06_residualConnection import ResidualConnection


class DecoderBlock(nn.Module):
    
    def __init__(self,self_attention_block : MultiHeadAttention , cross_attention_block : MultiHeadAttention, feed_forward_block : FeedForwardNetwork, dropout: float):
        
        super().__init__()
        # Initialising the self attention , feed forward and the cross attention blocks
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        
        # Initialising the residual connections, 3 because we have 3 sublayers
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

        
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        #Movement is as per the diagram
        x= self.residual_connections[0](x , lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x= self.residual_connections[1](x , lambda x : self.self_attention_block(x,encoder_output,encoder_output,src_mask))
        x= self.residual_connections[1](x , self.feed_forward_block)
        
        
        


class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask , tgt_mask):
        
        #For each layer we pass the input through the layer & use the output & encoder output
        for layer in self.layers:
            x= layer(x,encoder_output,src_mask, tgt_mask)
            
        return self.norm(x)
            
