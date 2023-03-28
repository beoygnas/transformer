import torch
from torch import nn


class LayerNorm(nn.Module) : 
    def __init__(self, d_model, eps=1e-12) : 
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x) : 
        # (batch_size x sequence_length x d_model)
        # d_model를 축으로(샘플 단위로) normalization 실행 -> layer normalization
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, unbiased=False, keepdim = True)
        
        # eps는 엡실론으로, division by zero를 방지하기 위함.
        x = (x - mean)/torch.sqrt(var + self.eps)
        
        # r * x + b
        x = self.gamma * x + self.beta
        return x
        

        