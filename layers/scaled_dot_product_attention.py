import torch
from torch import nn

class ScaledDotProductAttention(nn.Module) :
    
    def __init__(self) :
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask=None, e=1e-12) :
        
        #input은 4차원 tensor [256, ]
        batch_size, head, length, d_tensor = k.size() 
        
        # 1. Matmul
        score = q @ k.transpose