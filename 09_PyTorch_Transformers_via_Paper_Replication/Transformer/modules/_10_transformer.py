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
    """
    Transformer model that consists of an encoder, decoder, and necessary embedding and positional encoding layers.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): The source input embeddings.
        tgt_embed (InputEmbeddings): The target input embeddings.
        src_pos (PositionalEncoding): The positional encoding for the source.
        tgt_pos (PositionalEncoding): The positional encoding for the target.
        projection_layer (ProjectionLayer): The projection layer to map the output to the desired shape.
    """
    
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer):
        """
        Initializes the Transformer model with the given components.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            src_embed (InputEmbeddings): The source input embeddings.
            tgt_embed (InputEmbeddings): The target input embeddings.
            src_pos (PositionalEncoding): The positional encoding for the source.
            tgt_pos (PositionalEncoding): The positional encoding for the target.
            projection_layer (ProjectionLayer): The projection layer to map the output to the desired shape.
        """
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): The source sequence tensor of shape (Batch, Seq_len).
            src_mask (torch.Tensor): The source mask tensor of shape (Batch, 1, 1, Seq_len).

        Returns:
            torch.Tensor: The encoded source sequence of shape (Batch, Seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence using the encoder output.

        Args:
            encoder_output (torch.Tensor): The output from the encoder of shape (Batch, Seq_len, d_model).
            src_mask (torch.Tensor): The source mask tensor of shape (Batch, 1, 1, Seq_len).
            tgt (torch.Tensor): The target sequence tensor of shape (Batch, Seq_len).
            tgt_mask (torch.Tensor): The target mask tensor of shape (Batch, 1, Seq_len, Seq_len).

        Returns:
            torch.Tensor: The decoded target sequence of shape (Batch, Seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the output tensor to the desired shape using the projection layer.

        Args:
            x (torch.Tensor): The input tensor of shape (Batch, Seq_len, d_model).

        Returns:
            torch.Tensor: The projected tensor of shape (Batch, Seq_len, vocab_size).
        """
        return self.projection_layer(x)
