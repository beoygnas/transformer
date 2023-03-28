import torch
from torch import nn

class FeedForward(nn.Module) :
    
    def __init__(self, d_model, hidden, drop_prob=0.1) :
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLu()
        self.dropout = nn.Dropout(n=drop_prob)
        
    def forward(self, x) : 
        x = self.w1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        
        return x