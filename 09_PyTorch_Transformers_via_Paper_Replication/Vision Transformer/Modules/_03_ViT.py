import torch
import torch.nn as nn
from Modules._01_config import *
from Modules._02_patchEmbedding import PatchEmbedding

class ViT(nn.Module):
    def __init__(self,
                 num_classes=NUM_CLASSES,
                 patch_size=PATCH_SIZE,
                 num_patches=NUM_PATCHES,
                 in_channels=IN_CHANNELS,
                 embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS,
                 hidden_dim=HIDDEN_DIMENSION,
                 num_encoders=NUM_ENCODERS,
                 activation=ACTIVATION,
                 dropout=DROPOUT):
        
        super().__init__()
        
        # Embedding Block
        self.embedding_block = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, num_patches=num_patches, dropout=dropout)
        
        # Encoder Layer and Encoder Block
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=False)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_encoders, enable_nested_tensor=False)
        
        # MLP (Multi Layer Perceptron) Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embedding_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Taking the CLS Token only
        
        return x
