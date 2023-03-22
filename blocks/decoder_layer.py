import torch
from torch import nn
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward_network import FeedForwardNetwork
from layers.layer_norm import LayerNorm

class DecoderLayer(nn.Module) :
    def __init__(self, d_model, n_head, hidden, drop_prob) :
        super(DecoderLayer, self).__init()
        
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.self_attention_norm = LayerNorm(d_model)    
        
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.cross_attention_norm = LayerNorm(d_model)    
        
        self.ffn = FeedForwardNetwork(d_model, hidden)
        self.ffn_norm = LayerNorm(d_model)
        
        self.dropout_self_att = nn.Dropout(p=drop_prob)
        self.dropout_cross_att = nn.Dropout(p=drop_prob)
        self.dropout_ffn = nn.Dropout(p=drop_prob)
        
        
    def forward(self, dec_input, enc_output, mask_self, mask_cross) :
        
        self_attention = self.self_attention(q=dec_input, k=dec_input, v=dec_input, mask=mask_self)
        self_attention = self.dropout_self_att(self_attention)
        self_attention = self.self_attention_norm(dec_input, self_attention)
        
        if enc_output is not None : 
            cross_attention = self.cross_attention(q=self_attention, k=enc_output, v=enc_output, mask=mask_cross)
            cross_attention = self.dropout_cross_att(cross_attention)
            cross_attention = self.cross_attention_norm(self_attention + cross_attention)
        else : 
            cross_attention = self_attention 
            
        output = self.ffn(cross_attention)
        output = self.dropout_ffn(output)
        output = self.ffn_norm(output + cross_attention) 
    
        return output 
