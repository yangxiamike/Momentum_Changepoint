from numpy.core.defchararray import split
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
import config
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import routine
import pandas as pd
from typing import Tuple
from typing import List

def combine_seq(x: np.array, y: np.array, window_size: int) -> Tuple:
    """
    Split financial data x and y based on window_size for RNN training
    Non-overlapping version
    Parameter:
        x: (np.array), N x H
        y: (np.array), N x H
    """
    N, H = x.shape
    _, H_y = y.shape
    new_N = N // window_size * window_size
    x = sliding_window_view(x, window_shape = (config.seq_length, H)) # TxH -> Nx1xLxH
    x = x[:,0]                                              # Nx1xLxH -> NxLxH
    y = y[config.seq_length - 1:]   # NxH
    # y = sliding_window_view(y, window_shape = (config.seq_length, H_y)) # TxH -> Nx1xLxH
    # y = y[:,0]                                              # Nx1xLxH -> NxLxH
    # x = x[-new_N:]
    # y = y[-new_N:]
    # x -> window_size samples y -> window_size+1 index
    # # BxTxH
    # x = x.reshape((-1, window_size, H))
    # y = y.reshape((-1, window_size, H_y))
    
    return x, y

def make_train_data(datas: List[Tuple[np.array]]) -> Tuple:
    X, Y = [], []
    for x, y in datas:
        x, y = combine_seq(x, y, config.seq_length)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis = 0)
    Y = np.concatenate(Y, axis = 0)
    return X, Y

def make_test_data(datas: List[Tuple[np.array]]) -> Tuple:
    X, Y = [], []
    seq_len = config.seq_length
    for x, y in datas:
        T, H = x.shape
        x = sliding_window_view(x, window_shape = (seq_len, H)) # TxH -> Nx1xLxH
        x = x[:,0]                                              # Nx1xLxH -> NxLxH
        y = y[seq_len-1:]                                       # Nxh

        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis = 0)
    Y = np.concatenate(Y, axis = 0)
    return X, Y

def split_train_valid(datasets: List[pd.DataFrame], train_ratio):
    Ns = [X.shape[0] for X in datasets]
    N_trains = [int(N * train_ratio) for N in Ns]
    y_dim = config.num_y_dim

    dataset_train = [(data.values[:N, :-y_dim], data.values[:N, -y_dim:]) for (data, N) in zip(datasets, N_trains)]
    dataset_valid = [(data.values[N:, :-y_dim], data.values[N:, -y_dim:]) for (data, N) in zip(datasets, N_trains)]

    X_train, y_train = make_train_data(dataset_train)
    X_valid, y_valid = make_train_data(dataset_valid)

    return X_train, X_valid, y_train, y_valid

def split_test(datasets: List[pd.DataFrame]):
    """
    Split with overlapping moving window to predict one by one
    """
    y_dim = config.num_y_dim
    seq_len = config.seq_length

    indexs = [data.index for data in datasets]
    indexs = [index[seq_len - 1:] for index in indexs]
    datasets = [(data.values[:, :-y_dim], data.values[:, -y_dim:]) for data in datasets]
    
    X, Y = make_test_data(datasets)
    return X, Y, indexs

    
class DMN_Dataset(Dataset):
    """
    Only for backtest purpose, assume future information is available all time
    """
    def __init__(self, alphas: np.array, future_info: np.array):
        self.alphas = alphas
        self.future_info = future_info
    
    def __len__(self):
        return self.alphas.shape[0]
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.alphas[idx]).float(), torch.from_numpy(self.future_info[idx]).float()

def get_train_loader(datasets, train_ratio = 0.7):

    X_train, X_valid, y_train, y_valid = split_train_valid(datasets, train_ratio)

    dataset_train = DMN_Dataset(X_train, y_train)
    dataset_valid = DMN_Dataset(X_valid, y_valid)
    dataloader_train = DataLoader(dataset_train, batch_size = config.batch_size, 
                                            num_workers = config.num_workers,
                                            pin_memory = True, shuffle = True)
    dataloader_valid = DataLoader(dataset_valid, batch_size = config.batch_size, 
                                            num_workers = config.num_workers,
                                            pin_memory = True, shuffle = True)

    print(f'Train data loaded successfully!')
    return dataloader_train, dataloader_valid

def get_test_loader(datasets):

    X, Y, indexs = split_test(datasets)

    index_total = None
    for index in indexs:
        if index_total is None:
            index_total = index
        else:
            index_total = index_total.append(index)
    
    dataset_test = DMN_Dataset(X, Y)
    dataloader_test = DataLoader(dataset_test, batch_size = config.batch_size, 
                                            num_workers = config.num_workers,
                                            pin_memory = True, shuffle = False)
    return dataloader_test, index_total
