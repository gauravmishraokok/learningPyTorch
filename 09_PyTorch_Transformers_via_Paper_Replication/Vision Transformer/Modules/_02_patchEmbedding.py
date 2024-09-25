import torch
import torch.nn as nn
from Modules._01_config import * 

class PatchEmbedding(nn.Module):
    
    def __init__(self,
                 in_channels:int = IN_CHANNELS,
                 embed_dim:int = EMBED_DIM,
                 patch_size:int = PATCH_SIZE,
                 num_patches = NUM_PATCHES,
                 dropout = DROPOUT):
        super().__init__()
        
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size), 
            nn.Flatten(2)
        ) #This moves over the image exactly the number of times patches are and in the shape of patches.
        
        self.cls_token = nn.Parameter(torch.randn(size=(1,in_channels,embed_dim)),requires_grad=True) #Shape -> [1,3,48]
        
        self.positional_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)),requires_grad=True) #+1 for CLS Token
        
        self.dropout = nn.Dropout(p=dropout)
        
        
        
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0],-1,-1) #Shape -> [1,3,48] -> [Input_shape,3,48]
        
        x= self.patcher(x).permute(0,2,1) #Shape Change -> [Input_shape,3,48] -> [Input_shape,48,3]
        
        x = torch.cat([cls_token, x], dim = 1) 

        x += self.positional_embeddings 
        
        x=self.dropout(x)
        
        return x
        
