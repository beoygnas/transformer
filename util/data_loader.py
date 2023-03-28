from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

class Dataloader :
    source : Field = None
    target : Field = None