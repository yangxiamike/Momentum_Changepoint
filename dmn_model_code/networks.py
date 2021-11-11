from tokenize import Decnumber
import numpy as np
import torch
from torch import nn



class DMNLstm(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, num_lstm_layer, lstm_dropout = 0.1):
        """
        LSTM strucuture for DMN
        """
        super(DMNLstm, self).__init__()
        self.linear = nn.Linear(in_dim, hid_dim)
        self.bn = nn.BatchNorm1d(self.hid_dim)
        self.tanh_linear = nn.Tanh()
        self.lstm = nn.LSTM(input_size = hid_dim, hidden_size = hid_dim, 
                            num_layers = num_lstm_layer, batch_first = False,
                            dropout = lstm_dropout)
        self.linear_out = nn.Linear(hid_dim, out_dim)
        self.tanh_out = nn.Tanh()

    def forward(self, x, is_online = False):
        B, T, H = x.shape


        x = x.permute((1, 2, 0)).contiguous() # TxBxH -> BxHxT
        x = self.linear(x)
        x = self.tanh_linear(x)
        x = self.bn(x)
        x = x.permute((2, 0, 1)).contiguous() # BxHxT -> TxBxH
        out, (h, c) = self.lstm(x)

        if is_online:
            # out: TxBxH
            x = out[-1]              # BxH
        else:
            x = out
            x = x.permute((1, 2, 0)) # TxBxH -> BxHxT
        # Time distributed layer
        x = self.linear_out(x)
        x = self.tanh_out(x)

        if not is_online:
            x = x.permute((2, 0, 1)).contiguous() # BxHxT -> TxBxH
        return x

class LossSharpe(nn.Module):
    """
    Loss function as annual sharpe ratio
    """
    def __init__(self):
        super(LossSharpe, self).__init__()
    
    def forward(self, signals: torch.tensor, ys: torch.tensor) -> torch.tensor:
        """
        Attension!! ys shape is [return, sigma]
        """
        ret = ys[:, 0]
        sigma = ys[:, 1]
        R = signals * ret
        R_expect = torch.mean(R)
        R_std = torch.std(R)
        scale = torch.tensor(252)
        sharpe = -torch.sqrt(scale) * R_expect / R_std
        return sharpe
