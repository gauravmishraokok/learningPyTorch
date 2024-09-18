import torch
import torch.nn as nn

from modules._01_inputEmbeddings import InputEmbeddings
from modules._02_positionalEncoding import PositionalEncoding
from modules._03_layerNormalisation import LayerNormalization
from modules._04_feedForwardNetwork import FeedForwardNetwork
from modules._05_multiHeadAttention import MultiHeadAttention
from modules._06_residualConnection import ResidualConnection
from modules._07_encoder import Encoder
from modules._08_decoder import Decoder
from modules._09_projectionLayer import ProjectionLayer


class Transformer(nn.Module):
    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : InputEmbeddings,
                 tgt_embed : InputEmbeddings,
                 src_pos : PositionalEncoding,
                 tgt_pos : PositionalEncoding, 
                 projection_layer : ProjectionLayer):
        super().__init__()
        
        #Initialising encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos & projection_layer
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
        
    def encode(self, src, src_mask):
        src = self.src_embed(src_mask)
        src = self.src_pos(src)
        
        return self.encoder(src, src_mask) #Parameter order is same as per forward() of Encoder
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        
        return self.decoder(tgt, encoder_output,src_mask, tgt_mask) #Parameter order is same as per forward() of Decoder
    
    def project(self,x): 
        return self.projection_layer(x)
        
