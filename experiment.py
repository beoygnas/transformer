
import torch
from torch import nn

# 2 x 3 x 4
a = torch.tensor([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
])

a = torch.tensor(
    [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]
)

b = torch.tensor([[13, 14, 15, 16], [17, 18, 19, -1], [-1,-1, -1, -1]])

pad_idx = 5

print(a)
print(a.shape)

a = a.ne(-1)
b = b.ne(-1)

print(a)
print(b)
print(a.shape)
print(b.shape)

a = a.unsqueeze(1).unsqueeze(2)
b = b.unsqueeze(1).unsqueeze(3)

print(a)
print(a.shape)

a = a.repeat(1, 1, 4, 1)
b = b.repeat(1, 1, 1, 4)

mask = b & a

print(mask)
print(mask.shape)


a = torch.tensor(
    [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]
)

b = torch.tensor([[13, 14, 15, 16], [17, 18, 19, -1], [-1,-1, -1, -1]])


def make_no_peak_mask(q, k):
    len_q, len_k = q.size(1), k.size(1)

    # len_q x len_k
    mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

    return mask

print(make_no_peak_mask(a, b))


## 여러개의 헤드로 나누고, 각자 self attention

# ex) head가 두 개인 경우
def split(tensor) :
    batch_size, length, d_model = a.size()
    d_tensor = d_model // 2

    print(tensor.view(batch_size, length, 2, d_tensor))
    print(tensor.view(batch_size, length, 2, d_tensor).transpose(1, 2))
    
    return tensor

# a_split = torchsplit(a)
