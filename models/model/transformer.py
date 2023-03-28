import torch
from torch import nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module) :
    def __init__(self, source_pad_idx, target_pad_idx, target_sos_idx, enc_vocab_size, dec_vocab_size, d_model, max_len, device, drop_prob, n_head, n_layers, hidden) : 
        super(Transformer, self).__init__()
        
        ## 패딩마스크
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.target_sos_idx = target_sos_idx
        self.device =device
        self.encoder = Encoder(vocab_size=enc_vocab_size,
                               d_model=d_model,
                               max_len=max_len,
                               device=device,
                               drop_prob=drop_prob,
                               n_head=n_head,
                               n_layers=n_layers,
                               hidden=hidden
                               )
        
        self.decoder = Decoder(vocab_size=dec_vocab_size,
                               d_model=d_model,
                               max_len=max_len,
                               device=device,
                               drop_prob=drop_prob,
                               n_head=n_head,
                               n_layers=n_layers,
                               hidden=hidden
                               )
        
    def forward(self, source, target) :
        
        source_mask = self.make_pad_mask(source, source, self.source_pad_idx, self.source_pad_idx)
        cross_mask = self.make_pad_mask(target, source, self.target_pad_idx, self.source_pad_idx)
        target_mask = self.make_pad_mask(target, target, self.target_pad_idx, self.target_pad_idx) 
        target_mask = target_mask * self.make_tgt_mask(target, target)
        
        source_encoded = self.encoder(source, source_mask)
        output = self.decoder(target, source_encoded, target_mask, cross_mask)
        return output
        

    ## 패딩 mask 만들기.
    ## 길이 맞춰주기 위해 사용된 패딩을 mask
    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        
        # k : (batch_size x sequence_length), 아직 임베딩하기전임
        len_q, len_k = q.size(1), k.size(1)
        k = k.ne(k_pad_idx) # pad_idx 제외하고 모두 True
        
        # k : (batch_size x 1 x 1 x sequence_length)
        k = k.unsqueeze(1).unsqueeze(2)
        
        # k : (batch_size x 1 x sequence_length x sequence_length)
        k = k.repeat(1, 1, len_q, 1)
        
        # q는 transpose를 고려하여 unsqueeze 위치가 다름.
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
         
        # query와 key의 연산에서 pad mask가 필요함
        mask = k & q 
        return mask
    
    def make_tgt_mask(self, q, k) :
        # 아직 만들어지지 않은 tgt의 뒤에올 text를 읽는 것을 방지. 
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        # output (sequence_length x sequence_length)
        return mask