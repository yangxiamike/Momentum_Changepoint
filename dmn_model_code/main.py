import numpy as np
from numpy.core.defchararray import index
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
    if config.load_model:
        model.load_state_dict(os.path.join(save_dir, config.model_load_name))
    
    train_loader, valid_loader = get_train_loader(datasets, train_ratio = config.train_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)

    best_loss = np.inf

    for epoch in range(config.num_epochs):
        epoch += 1

        train(model, config.loss_criterion, train_loader, valid_loader, optimizer, epoch, writer)
        valid_loss = evaluate(model, config.loss_criterion, valid_loader)

        print('Epoch {}  Validation Loss: {:3.2f}\t'.format(epoch, valid_loss))

        file_name = f'Epoch{str(epoch)}Loss{str(round(valid_loss, 3))}'.replace('.', '_') + '.pt'
        file_path = os.path.join(save_dir, file_name)
        best_path = os.path.join(save_dir, config.model_best_save)

        torch.save(model.state_dict(), file_path)
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), best_path)
        writer.add_scalar('Total Negative Valid loss', -valid_loss, epoch)

def run_train():
    # load data features
    data_features = pd.read_csv(config.features_path, index = False, parse_dates = True)
    # create windowed subset of the data
    date_start = datetime(config.bt_start_year, 1, 1)
    date_end = datetime(config.bt_start_year + 5, 1, 1)

    train_sets, _ = split_by_category(data_features, date_start, date_end)
    run_year(train_sets, config.bt_period)

def run_test():
    # load data features
    data_features = pd.read_csv(config.features_path, index = False, parse_dates = True)
    # create windowed subset of the data
    date_start = datetime(config.bt_start_year, 1, 1)
    date_end = datetime(config.bt_start_year + 5, 1, 1)

    _, test_sets = split_by_category(data_features, date_start, date_end)
    dataloader_test, index_test = get_test_loader(test_sets)

    model = config.model_use()
    model.load_state_dict(os.path.join(config.model_save_path, config.model_best_save))

    signals, ys = inference(model, dataloader_test)
    loss_criterion = config.loss_criterion
    loss = loss_criterion(signals, ys)
    loss = loss.item()

    print(f'Sharpe ratio during {date_start} to {date_end} is {loss}')

    data = torch.cat([signals, ys], axis = 1)
    data = torch.detach().numpy()

    df = pd.DataFrame(data = data, columns = ['signal', 'ret', 'sigma'], index = index_test)
    df.to_csv(os.path.join(config.model_save_path, 'run_time.csv'))

    