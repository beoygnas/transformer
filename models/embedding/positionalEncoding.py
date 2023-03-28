
import torch
from torch import nn

class PositionalEncoding(nn.Module) :
    
    def __init__(self, d_model, max_len, device):
        super().__init__()
        
        self.positionalEncoding = torch.zeros(max_len, d_model, device=device)
        self.positionalEncoding.requires_grad = False
    
        pos = torch.arange(0, max_len, device=device) # pos means position    image.png
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.positionalEncoding[:, 0::2] = torch.sin(pos/ (10000 ** (_2i/d_model)))
        self.positionalEncoding[:, 1::2] = torch.cos(pos/ (10000 ** (_2i/d_model)))
        # print(self.positionalEncoding)
        # print(self.positionalEncoding.shape)
        # print(self.positionalEncoding[:7, :])
        
        
    def forward(self, x) :
        batch_size, seq_len = x.size()
        return self.positionalEncoding[:seq_len, :]
        
    

    
    