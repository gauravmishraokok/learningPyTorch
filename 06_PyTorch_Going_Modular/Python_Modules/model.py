import torch
import torch.nn as nn

INPUT_FEATURES=224*224
HIDDEN_FEATURES=10
OUTPUT_FEATURES=3

class TVGGModel(nn.Module):
    """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  
    def __init__(self,input_features=INPUT_FEATURES, hidden_features=HIDDEN_FEATURES,output_features=OUTPUT_FEATURES):
        super().__init__()
        
        self.convBlock1 = nn.Sequential(
            
            nn.Conv2d(in_channels=input_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            
        )
        
        self.convBlock2 = nn.Sequential(
            
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            
        )
        
        self.classifier  = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*59*59,out_features=OUTPUT_FEATURES)
            
        )
        
    def forward(self,x):
        
        # x=self.convBlock1(x)
        # print(f"conv1 {x.shape}")
        # x=self.convBlock2(x)
        # print(f"conv2 {x.shape}")
        # x=self.classifier(x)
        # print(f"classifier {x.shape}")
        # return x
        
        return self.classifier(self.convBlock2(self.convBlock1(x)))
