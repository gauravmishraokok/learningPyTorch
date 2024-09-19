
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        
        self.d_model = d_model
        self.h=h
        assert d_model%h==0 , 'Embedding dimension must be divisible by number of heads'
        
        self.d_k = d_model//h  #d_k = d_model/h
        
        self.w_q = nn.Linear(d_model,d_model) #Wq
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv
        
        self.w_o = nn.Linear(d_model,d_model) #Wo as d_v is same as d_k and d_k*h = d_model

        self.dropout = nn.Dropout(p=dropout)
        
        
    @staticmethod    
    def attention(query,key,value,mask,dropout:nn.Dropout):
        #Calculating the attention score
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k) # @ --> Matrix Multiplication
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0 , -1e9) #-1e9 --> -Infinity. These values later become 0 after softmax
            
        attention_scores = attention_scores.softmax(dim = -1) #(Batch , h , Seq_len , Seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value) , attention_scores 
        
        
        
        
    def forward(self,q,k,v,mask):
        
        #Getting the Q',K' & V'
        
        query = self.w_q(q) # (Batch , Seq_len , d_model) -> (Batch , Seq_len , d_model)
        key = self.w_k(k) # (Batch , Seq_len , d_model) -> (Batch , Seq_len , d_model)
        value = self.w_v(v) # (Batch , Seq_len , d_model) -> (Batch , Seq_len , d_model)
        
        #Splitting the d_model dimension into h heads
        
        #Here query.shape[0] -> Batch & query.shape[1] -> Seq_len
        #Final shape of query after transpose -> (Batch , h , Seq_len , d_k) 
        query = query.reshape(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.reshape(key.shape[0] , key.shape[1],self.h,self.d_k).transpose(1,2)
        value=key.reshape(value.shape[0] , value.shape[1],self.h,self.d_k).transpose(1,2) 
        
        x,self.attention_score = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        #Shape of x -> (Batch , h , Seq_len , d_k)
        #Shape of attention_score -> (Batch , h , Seq_len , Seq_len)
        
        x= x.transpose(1,2).contiguous().view(x.shape[0], -1 , self.d_k * self.h) #(Batch , Seq_len , d_model)
        
        
        return self.w_o(x) #(Batch , Seq_len , d_model) -> (Batch , Seq_len , d_model)
                   
