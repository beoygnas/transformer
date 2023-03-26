import torch
from torch import nn
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward_network import FeedForward

class DecoderLayer(nn.Module) : 
    
    def __init__(self, d_model, n_head, hidden, drop_prob=0.1) : 
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.feed_forward = FeedForward(d_model, hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, dec, enc, s_mask1, s_mask2) :
        
        # attention
        self_attention = self.self_attention(dec, dec, dec, mask=s_mask1)
        self_attention = self.dropout1(self_attention)
        # add & norm1
        self_attention = self.layer_norm1(dec + self_attention)
        
        
        if enc is not None :
            # attention2
            cross_attention = self.cross_attention(q = self_attention, k = enc, v = enc, mask=s_mask2)
            cross_attention = self.dropout2(cross_attention)
            # add & norm2
            cross_attention = self.layer_norm2(cross_attention + self_attention)
        
        
        # feedforward
        x_ffn = self.feed_forward(cross_attention)
        x_ffn = self.dropout3(x_ffn)
        
        # add & norm
        x = self.layer_norm3(x_ffn + cross_attention)
        
        return x
         