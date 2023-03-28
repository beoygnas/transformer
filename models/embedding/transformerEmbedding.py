import torch
from torch import nn

from models.embedding.positionalEncoding import PositionalEncoding
from models.embedding.tokenEmbedding import TokenEmbedding 


class TransformerEmbedding(nn.Module) :
    def __init__(self, vocab_size, d_model, max_len, device, drop_prob) :
        
        ## 상속받은 nn.Module 초기화
        ## 토큰임베딩, 포지셔널인코딩 초기화
        super(TransformerEmbedding, self).__init__()
        self.tokenEmbedding = TokenEmbedding(vocab_size, d_model)
        self.positionalEncoding = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x) :
        tokenEmbedding = self.tokenEmbedding(x)
        positionalEncoding = self.positionalEncoding(x)
        
        # print(f'tokenembedding : \n\t{x_emb.tokenEmbedding}')
        # print(f'positionalEncoding : \n\t{x_emb.positionalEncoding}')
        
        return self.drop_out(tokenEmbedding + positionalEncoding)


# if __name__ == '__main__' : 
#     emb = TransformerEmbedding(vocab_size = 26, 
#                                 d_model = 8,
#                                 max_len = 3,
#                                 drop_prob = 0.1,
#                                 device = 'cpu')
    
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#     x_emb = emb(x)
    
#     print(f'input : \n\t{x}')
#     print(f'output : \n\t{x_emb}')
