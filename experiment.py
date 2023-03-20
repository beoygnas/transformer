
import torch
from torch import nn

# 2 x 3 x 4
a = torch.tensor([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
])
print(a)
print(a.shape)


## 여러개의 헤드로 나누고, 각자 self attention

# ex) head가 두 개인 경우
def split(tensor) :
    batch_size, length, d_model = a.size()
    d_tensor = d_model // 2

    print(tensor.view(batch_size, length, 2, d_tensor))
    print(tensor.view(batch_size, length, 2, d_tensor).transpose(1, 2))
    
    return tensor

# a_split = torchsplit(a)