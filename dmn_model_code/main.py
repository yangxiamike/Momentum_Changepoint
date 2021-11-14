import numpy as np
from numpy.core.defchararray import count, index
import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List

import config
from data_util import get_train_loader
from data_util import get_test_loader
from dmn_model_code.routine import train
from model_utils import split_by_category
from routine import inference
from routine import evaluate
from routine import train

def run_year(datasets: List[pd.DataFrame], year_range: str = '1995to2000'):
    tb_path = os.path.join(config.tensorboard_path, year_range)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

    save_dir = os.path.join(config.model_save_path, year_range)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = config.model_use
    if config.is_load_model:
        model.load_state_dict(os.path.join(save_dir, config.model_load_name))
    
    train_loader, valid_loader = get_train_loader(datasets, train_ratio = config.train_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)

    best_loss = np.inf
    count_valid_stuck = 0    # count for validataion loss for early stopping

    for epoch in range(config.num_epochs):
        epoch += 1
        train(model, config.loss_criterion, train_loader, valid_loader, optimizer, epoch, writer)
        valid_loss = evaluate(model, config.loss_criterion, valid_loader)

        print('Epoch {}  Validation Loss: {:3.6f}\t'.format(epoch, valid_loss))

        file_name = f'Epoch{str(epoch)}Loss{str(round(valid_loss, 3))}'.replace('.', '_') + '.pt'
        file_path = os.path.join(save_dir, file_name)
        best_path = os.path.join(save_dir, config.model_best_save)

        torch.save(model.state_dict(), file_path)
        if best_loss > valid_loss:
            best_loss = valid_loss
            count_valid_stuck = 0
            torch.save(model.state_dict(), best_path)
        else:
            count_valid_stuck += 1
        writer.add_scalar('Total Negative Valid loss', valid_loss, epoch)

        # Add early stopping
        if count_valid_stuck > config.early_stop_epochs:
            print('Early Stopping Triggered.')
            break

def run_train():
    # load data features
    data_features = pd.read_csv(config.features_path, parse_dates = True)
    data_features.date = pd.to_datetime(data_features.date)
    # create windowed subset of the data
    date_start = datetime(config.bt_start_year, 1, 1)
    date_end = datetime(config.bt_start_year + 5, 1, 1)

    train_sets, _ = split_by_category(data_features, date_start, date_end)
    run_year(train_sets, config.bt_period)

def run_test():
    # load data features
    data_features = pd.read_csv(config.features_path, parse_dates = True)
    data_features['date'] = pd.to_datetime(data_features['date'])
    # create windowed subset of the data
    date_start = datetime(config.bt_start_year, 1, 1)
    date_end = datetime(config.bt_start_year + 5, 1, 1)

    _, test_sets = split_by_category(data_features, date_start, date_end)
    dataloader_test, index_test = get_test_loader(test_sets)

    model = config.model_use
    model.load_state_dict(torch.load(os.path.join(config.model_load_path, config.model_best_save)))

    signals, ys = inference(model, dataloader_test)

    # Calculate Sharpe Loss
    ret = ys[:, 0]
    signals = signals[:, 0]
    R = signals * ret
    R_expect = torch.mean(R)
    R_std = torch.std(R)
    sharpe = np.sqrt(252) * R_expect / R_std
    sharpe = sharpe.detach().cpu().numpy()

    print(f'Sharpe ratio during {date_start} to {date_end} is {sharpe}')

    data = torch.cat([signals, ys], axis = 1)
    data = data.detach().cpu().numpy()

    df = pd.DataFrame(data = data, columns = ['signal', 'ret', 'sigma'], index = index_test)
    df.to_csv(os.path.join(config.model_save_path, 'run_time.csv'))
    daily_ret = df['ret'] * df['signal']
    daily_ret = daily_ret.groupby('date').mean()
    daily_ret.to_csv(os.path.join(config.model_save_path, 'run_time_ret.csv'))


if __name__ == '__main__':
    run_test()