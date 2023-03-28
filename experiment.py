
import torch
from torch import nn
import math


# q = torch.tensor(
# [    [[[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]],
#      [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]],
#     [[[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]],
#      [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]]]
# )

# print(q.size())

# k = torch.tensor(
# [    [[[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]],
#      [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]],
#     [[[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]],
#      [[1, 2, -1, -1], [5, 6, 7, -1], [9, -1, -1, -1]]]]
# )

# print(k.size())

# print(k.transpose(2, 3).size())


# print(q@k.transpose(2, 3) / math.sqrt(4))
# print((q@k.transpose(2, 3)) / math.sqrt(4))


a = 1

if a is None :
    b = 1
else : b = 2


_2i = 2 
d_model = 4
print(10000 ** (_2i/d_model))
print(10000 ** _2i/d_model)

