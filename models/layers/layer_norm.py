import torch
from torch import nn


class LayerNorm(nn.Module) : 
    def __init__(self, d_model, eps=1e-12) :
        super(LayerNorm, self).__init__()
        
        ## 얘네는 layer가 아닌 하나의 tensor이지만, gradient도 계산해야하고 value도 update해야됨.
        ## 따라서 nn.Parameter로 정의
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    # input x : (batch_size, length, d_model)
    def forward(self, x) :
    
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, unbiased = False, keepdim =True)
        x = (x-mean) / torch.sqrt(var + self.eps) 
        
        x = self.gamma * x + self.beta
        return x
        