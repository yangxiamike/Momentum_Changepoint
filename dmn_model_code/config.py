from torch import nn
import os
import torch
from networks import DMNLstm, LossReturn
from networks import LossSharpe

# dataset path
root = 'data/feature_data/feature_data_nocpd'
features_path = os.path.join(root, 'features.csv')

# train parameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# device = 'cpu'
batch_size = 32
num_workers = 4
log_interval = 50
eval_interval = 300
num_y_dim = 2       # use to split X and Y
train_ratio = 0.8
num_epochs = 200
early_stop_epochs = 25

# Model Parameter
seq_length = 63     # sequence length for LSTM input
in_dim = 9
out_dim = 1
hid_dim = 10
num_lstm_layer = 2
is_dropout = True
lstm_dropout = 0.15

model_use = DMNLstm(in_dim, out_dim, hid_dim, num_lstm_layer, lstm_dropout)

# optimization
loss_criterion = LossReturn
lr = 0.0001
weight_decay = 0.0

# save path
is_load_model = False
model_load_name = 'best_save.pt'
model_best_save = 'best_save.pt'
model_save_path = 'model'
model_load_path = 'model/2020to2025/2020to2025'
tensorboard_path = 'tb_log/'

# select parameter
choose_year = 2020   ## ！！！！！！！！！通过这个控制目前训练的年份
bt_start_year = choose_year // 5 * 5    # autocalculate
bt_period = f'{bt_start_year}to{bt_start_year+5}'    # autocalculate