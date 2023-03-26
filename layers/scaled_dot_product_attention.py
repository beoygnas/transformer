import torch
from torch import nn
import math

class ScaledDotProductAttention(nn.Module) :
    
    def __init__(self) :
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask=None, e=1e-12) :
        
        # input은 4차원 tensor || k : (batch_size x head x sequence_length x d_tensor)
        batch_size, head, length, d_tensor = k.size() 
        
        # 1. score 구하기 + scale = Matmul QK_t / root(d_k)
        # q      : (batch_size x head x sequence_length x d_tensor)
        # k_t    : (batch_size x head x d_tensor        x sequence_length)
        # score  : (batch_size x head x sequence_length x sequence_length)
        
        score = q @ k.transpose(2, 3) / math.sqrt(d_tensor)
        
        # 2. mask option -> mask는 score와 크기가 같은 행렬이며, True, False로 이루어짐.
        if mask is not None : 
            score = score.masked_fill(mask == 0, -10000)
            
        # 3. softmax
        attention = self.softmax(score)
        
        # 4. score (matmul) Value QK_t 
        # attention : (batch_size x head x sequence_length x sequence_length)
        # v         : (batch_size x head x sequence_length x d_tensor)
        # attention_value : (batch_size x head x sequence_length x d_tensor)
        attention_value = attention @ v
        
        return attention_value, attention