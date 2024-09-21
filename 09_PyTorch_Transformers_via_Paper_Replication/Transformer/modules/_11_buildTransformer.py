import torch
import torch.nn as nn
from modules._01_inputEmbeddings import InputEmbeddings
from modules._02_positionalEncoding import PositionalEncoding
from modules._03_layerNormalisation import LayerNormalization
from modules._04_feedForwardNetwork import FeedForwardNetwork
from modules._05_multiHeadAttention import MultiHeadAttention
from modules._06_residualConnection import ResidualConnection
from modules._07_encoder import Encoder, EncoderBlock
from modules._08_decoder import Decoder, DecoderBlock
from modules._09_projectionLayer import ProjectionLayer
from modules._10_transformer import Transformer

def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,  # According to paper, The number of encoder and decoder blocks is 6.
                      h: int = 8,  # According to paper, The number of heads is 8.
                      dropout: float = 0.1,
                      d_ff: int = 2048  # Dimensions of feed forward network is 2048
                      ) -> Transformer:
    """
    Builds a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequence.
        tgt_seq_len (int): Length of the target sequence.
        d_model (int, optional): Dimension of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimension of the feed forward network. Defaults to 2048.

    Returns:
        Transformer: The constructed Transformer model.
    """
    
    # Creating the embedding layers -> 
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)
    
    # Creating the positional encoding layers -> 
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout) 
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)
    
    # Creating the encoder blocks -> 
    encoder_blocks = []
    for _ in range(N):
        
        encoder_self_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        encoder_feed_forward_block = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention_block, feed_forward_block=encoder_feed_forward_block, dropout=dropout, features=d_model) 
        encoder_blocks.append(encoder_block)
        
    # Creating the decoder blocks -> 
    decoder_blocks = []
    for _ in range(N):
        
        decoder_self_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        decoder_feed_forward_block = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block, cross_attention_block=decoder_cross_attention_block, feed_forward_block=decoder_feed_forward_block, dropout=dropout, features=d_model)
        decoder_blocks.append(decoder_block)
        
    # Creating the encoder and decoder -> 
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks), features=d_model)
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks), features=d_model)
    
    # Creating the projection layer -> 
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)
    
    # CREATING THE TRANSFORMER ----> 
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed,
                              tgt_embed=tgt_embed,
                              src_pos=src_pos,
                              tgt_pos=tgt_pos,
                              projection_layer=projection_layer)
    
    # Initializing the parameters using Xavier uniform distribution ->
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer
