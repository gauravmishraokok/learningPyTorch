import torch
import torch.nn as nn
from modules.LSTM_Architecture import *
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


model = LSTM()

#Before Training -> 
print("Before Training ===>\n\n\n")
print("Comparing Observed and Predicted Value ==>\n\nCompany A => Observed -> 0 & Predicted ->",model(torch.tensor([0.0,0.5,0.25,1.0])).detach(),"\n\nCompany B => Observed -> 0 & Predicted ->",model(torch.tensor([1.0,0.5,0.25,1.0])).detach())


#Training -> 
inputs = torch.tensor([[0.0,0.5,0.25,1.0],[1.0,0.5,0.25,1.0]])
labels = torch.tensor([0.0,1.0])

dataset = TensorDataset(inputs,labels)
dataloader = DataLoader(dataset=dataset)

trainer = pl.Trainer(max_epochs=20001)

trainer.fit(model=model, train_dataloaders=dataloader)
print("After Training ===>\n\n\n")
print("Comparing Observed and Predicted Value ==>\n\nCompany A => Observed -> 0 & Predicted ->",model(torch.tensor([0.0,0.5,0.25,1.0])).detach(),"\n\nCompany B => Observed -> 0 & Predicted ->",model(torch.tensor([1.0,0.5,0.25,1.0])).detach())
