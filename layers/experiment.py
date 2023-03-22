  
import torch
  
x = torch.tensor([[
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
    ],
    [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
    ]
]).float()

print(x.shape)

y = torch.tensor([1, 2, 3, 4])

print(y, y.shape)

print(x * y)