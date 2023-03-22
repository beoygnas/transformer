import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#model params
batch_size = 128
max_len = 256
d_model = 514
n_layers = 6 
n_heads = 8
hidden = 2048
drop_prob = 0.1

# optimizer parameter

## Adam
init_lr = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9 # 

## ReduceLROnPlateau
factor = 0.9 # learning rate 앞에 붙는 계수, 감소시킬 비율을 의미함.
patience = 10 # metric이 향상이 안될 때, 10번의 epoch는 일단보고 그래도 향상이 안되면 학습 종료




 


warmup = 100 # 초기 100번의 epoch 동안, learning rate warm up, 초기의 랜덤 value params는 학습을불안정하게 함 => 따라서 learning rate를 작게 시작해서 서서히 웜업시키고, 이후 큰 값을 다시넣어줌.
epoch = 1000 # 전체 데이터셋을 몇 번 학습할거냐
clip = 1.0

inf = float('inf')