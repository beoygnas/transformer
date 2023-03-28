import torch
from torch import nn
import os, sys

from layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module) : 
    
    def __init__(self, d_model, n_head) :
        
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None) :
        # q, k, v => (batch_size x sequence_length x d_model)
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        
        # split by head => (batch_size x head x sequence_length x d_tensor)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        attention_value, attention = self.attention(q, k, v, mask=mask)
        
        attention_value.concat(attention_value)
        attention_value = self.w_concat(attention_value)
        # attention_value -> (batch_size x sequence_length x d_model)
        # 이후 linear layer에 연결
        # split의 역과정이라고 보면됨.
        
        return attention_value
        
    def split(self, w) :
        # 각 배치의 데이터들을 head 단위로 쪼갬 => d
        batch_size, length, d_model = w.size()
        d_tensor = d_model / self.n_head
        split_w = w.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return split_w
        
    
    def concat(self, w) : 
        # head 단위로 쪼개진 attention_value들을 다시 묶음.
        batch_size, head, length, d_tensor = w.size()
        d_model = head * d_tensor
        concat_w = w.transpose(2, 3).contiguous().view(batch_size, length, d_model)
        return concat_w