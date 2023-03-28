import torch
from torch import nn
from models.layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, n_head) :
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask) : 
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        attention = self.attention(q, k, v, mask=mask)
        attention = self.concat(attention)
        attention = self.W_concat(attention)
        return attention
    
    # input  : (batch_size, length, d_model)
    # output : (batch_size, head, length, d_tensor)
    def split(self, x) : 
        batch_size, length, d_model = x.size()
        d_tensor = d_model // self.n_head
        x = x.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return x
    
    
    # input  : (batch_size, head, length, d_tensor)
    # output : (batch_size, length, d_model)
    def concat(self, x) :
        batch_size, head, length, d_tensor = x.size()
        d_model = head * d_tensor
        x = x.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return x