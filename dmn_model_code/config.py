from torch import nn
import os
import torch
from networks import DMNLstm
from networks import LossSharpe

# dataset path
root = 'dataset'
features_path = os.path.join(root, 'features.csv')

# train parameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# device = 'cpu'
max_iter = 250
batch_size = 64
num_workers = 8
num_epochs = 10
log_interval = 10
eval_interval = 80
seq_length = 63     # sequence length for LSTM input
num_y_dim = 2       # use to split X and Y
train_ratio = 0.8

# Model Parameter
in_dim = 34
out_dim = 1
hid_dim = 80
num_lstm_layer = 2
is_dropout = True
lstm_dropout = 0.2

model_use = DMNLstm(in_dim, out_dim, hid_dim, num_lstm_layer, is_dropout, lstm_dropout)

# optimization
loss_criterion = LossSharpe
lr = 0.001
weight_decay = 0.0

# save path
load_model = True
model_load_name = 'best_save.pt'
model_best_save = 'best_save.pt'
model_save_path = 'model'
tensorboard_path = 'tb_log/'

# select parameter
choose_year = 2021   ## ！！！！！！！！！通过这个控制目前训练的年份
bt_start_year = choose_year // 5 * 5    # autocalculate
bt_period = f'{bt_start_year}to{bt_start_year+5}'    # autocalculate