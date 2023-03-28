import torch
from torch import nn
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformerEmbedding import TransformerEmbedding

class Decoder(nn.Module) :
    def __init__(self, vocab_size, d_model, max_len, device, drop_prob, n_head, n_layers, hidden) : 
        
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size = vocab_size, 
                                              d_model = d_model, 
                                              max_len = max_len,
                                              device = device, 
                                              drop_prob = drop_prob)
              
        # nn.Sequential로도 할 수 있는데, layer 개수 지정을 할 수 없음.
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model = d_model, 
                                                        n_head = n_head,
                                                        hidden = hidden, 
                                                        drop_prob = drop_prob) 
                                            for _ in range(n_layers)])
        
        self.linear = nn.Linear(d_model, vocab_size)
                    
    def forward(self, dec_input, enc_output, mask_self, mask_cross) :
        
        output = self.embedding(dec_input)
        
        for layer in self.decoder_layers : 
            output = layer(output, enc_output, mask_self, mask_cross)
        
        output = self.linear(output)
        return output