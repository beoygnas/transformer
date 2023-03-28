import torch
from torch import nn

class FeedForwardNetwork(nn.Module) : 
    def __init__(self, d_model, hidden, drop_prob=0.1) : 
        super(FeedForwardNetwork, self).__init__()
        self.W1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p = drop_prob)
        
    def forward(self, x) : 
        x = self.W1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        return x 