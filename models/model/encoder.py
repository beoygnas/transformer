import torch
from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformerEmbedding import TransformerEmbedding

class Encoder(nn.Module) :
    def __init__(self, vocab_size, d_model, max_len, device, drop_prob, n_head, n_layers, hidden) : 
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size = vocab_size, 
                                              d_model = d_model, 
                                              max_len = max_len,
                                              device = device, 
                                              drop_prob = drop_prob)
              
        # nn.Sequential로도 할 수 있는데, layer 개수 지정을 할 수 없음.
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model = d_model, 
                                        n_head = n_head,
                                        hidden = hidden, 
                                        drop_prob = drop_prob) for _ in range(n_layers)])
        
        
        # self.encoder = nn.Sequential(
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob),
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob),
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob),
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob),
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob),
        #                 EncoderLayer(d_model, n_head, hidden, drop_prob)
        #                 )
        
    def forward(self, input, mask) :
        output = self.embedding(input)
        
        for layer in self.encoder_layers : 
            output = layer(output, mask)
            
        return output