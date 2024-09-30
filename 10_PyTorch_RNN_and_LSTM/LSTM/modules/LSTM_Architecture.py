import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTM(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        
        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        
        
    def lstm_unit(self, input_value, long_memory, short_memory):
        
        long_remember_percent = torch.sigmoid((short_memory*self.wlr1)+(input_value*self.wlr2)+self.blr1)
        
        potential_memory_remember_percent = torch.sigmoid((short_memory*self.wpr1) + (input_value*self.wpr2) + self.bpr1)
        
        potential_long_term_memory = torch.tanh((short_memory*self.wp1) + (input_value*self.wp2) + self.bp1)
        
        updated_long_term_memory = ((long_memory*long_remember_percent) + (potential_long_term_memory*potential_memory_remember_percent))
        
        output_percent = torch.sigmoid((short_memory*self.wo1) + (input_value*self.wo2) + self.bo1)
        
        updated_short_term_memory = torch.tanh(updated_long_term_memory) * output_percent
        
        return ([updated_long_term_memory, updated_short_term_memory])
    
    
    
    #This LSTM Example code takes the input values of a share for 4 days and predicts the 5th day by considering all the day values. 
    #Predefined values -> First Company : [0,0.5,0.25,1] (Expected Prediction -> 0)
    #                   Second Company : [1,0.5,0.25,1] (Expected Prediction -> 1)
    def forward(self, input):
        long_memory = 0
        short_memory = 0
    
        for day in input:
            long_memory, short_memory = self.lstm_unit(day, long_memory, short_memory)
    
        return short_memory
    
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    
    def training_step(self, batch, batch_idx):
        input_i , label_i = batch
        output_i = self.forward(input_i[0])
        
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output_i, label_i)
        
        self.log("Training Loss", loss)
        
        if label_i==0:
            self.log("out_0", output_i)  #Very bruteforce way to check if we predicted for company first or second
        else:
            self.log("out_1", output_i)
            
        return loss
