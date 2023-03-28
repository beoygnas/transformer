import torch
import math
from torch import nn

class ScaledDotProductAttention(nn.Module) :
    
    def __init__(self) :
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask=None, e=1e-12) :
        
        #input은 4차원 tensor
        batch_size, head, length, d_tensor = k.size() 
        
        # 1. Matmul
        score = (q @ k.transpose(2, 3)) / math.sqrt(d_tensor)
        
        # 2. mask
        if mask is not None : 
            score = score.masked_fill(mask==0, -10000)
        
        # 3. attention.shape = (batch_size, head, length, length)
        attention =  self.softmax(score) 
        
        # 4. attetnion_value.shape = (batch_size, head, length, d_tensor)
        attention_value = attention @ v 
        
        return attention_value

        