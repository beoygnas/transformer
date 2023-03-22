import torch
from torch import nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork
from models.layers.layer_norm import LayerNorm

class EncoderLayer(nn.Module) :
    def __init__(self, d_model, n_head, hidden, drop_prob) :
        super(EncoderLayer, self).__init()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.attention_norm = LayerNorm(d_model)    
        self.ffn = FeedForwardNetwork(d_model, hidden)
        self.ffn_norm = LayerNorm(d_model)
        self.dropout_att = nn.Dropout(p=drop_prob)
        self.dropout_ffn = nn.Dropout(p=drop_prob)
        
    def forward(self, input, mask) :
        
        input_attention = self.attention(q=input , k=input, v=input, mask=mask)
        input_attention = self.dropout_att(input_attention)
        input_attention = self.attention_norm(input + input_attention)
        
        output = self.ffn(input_attention)
        output = self.dropout_ffn(output)
        output = self.ffn_norm(input_attention + output)
        
        return output 
