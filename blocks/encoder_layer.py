import torch
from torch import nn
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward_network import FeedForward

class EncoderLayer(nn.Module) : 
    def __init__(self, d_model, n_head, hidden, drop_prob=0.1) : 
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.feed_forward = FeedForward(d_model, hidden, drop_prob=drop_prob)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x, s_mask) :
        
        # attention
        x_attention = self.multi_head_attention(x, x, x, mask=s_mask)
        x_attention = self.dropout1(x_attention)
        
        # add & norm
        x_attention = self.layer_norm1(x + x_attention)
        
        # feedforward
        x_ffn = self.feed_forward(x_attention)
        x_ffn = self.dropout2(x_ffn)
        
        # add & norm
        x = self.layer_norm2(x_attention + x_ffn)
        
        return x
         