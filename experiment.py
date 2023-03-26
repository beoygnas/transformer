
import torch
from torch import nn

from layers.scaled_dot_product_attention import ScaledDotProductAttention
from layers.multi_head_attention import MultiHeadAttention
from embedding.transformerEmbedding import TransformerEmbedding

a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]   


a = torch.tensor(a).float()
print(a.shape)
# print(a.mean(0, True))
print(a.mean(0, True))
print(a.mean(-1, True))
# print(a.mean(1, False))