from torch import nn

class TokenEmbedding(nn.Embedding) :
    
    ## torch.nn에서 제공하는 Embedding class를 사용
    ## vocab_size,d_model이 input
        
    def __init__(self, vocab_size, d_model) :
        ## nn.Embedding을 상속하고, super로 이를 init하여 메소드를 사용가능.
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx = 1)