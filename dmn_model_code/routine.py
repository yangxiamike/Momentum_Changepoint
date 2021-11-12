import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import networks


def train(model, model_loss, train_loader, valid_loader, optimizer, epoch, tb_writer):
    model.to(config.device)
    model.train()
    optimizer.zero_grad()

    writer = tb_writer   # tensorboard

    total_loss = []
    
    for idx, (x, y) in tqdm(enumerate(train_loader), total = len(train_loader)):
        # with torch.autograd.set_detect_anomaly(True):
        # x -> BxTxH, y -> BxTxh
        # todo: change dim to TxBxH
        x = x.to(config.device)
        y = y.to(config.device)

        predictions = model(x)

        loss_criterion = model_loss()
        loss_criterion.train()
        loss_criterion.to(config.device)

        loss = loss_criterion(predictions, y)
        loss.backward()
        
        total_loss.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()


        idx += 1
        if idx % config.log_interval == 0:
            avg_loss = np.mean(total_loss)
            print('Epoch {} Training Loss: {:3.6f}'.format(epoch, avg_loss))
            global_step = epoch * config.batch_size + idx 
            total_loss = []
            global_step = epoch * config.batch_size + idx 
            writer.add_scalar('Training Loss', -avg_loss, global_step)
        
        if idx % config.eval_interval == 0:
            loss = evaluate(model, model_loss, valid_loader)
            writer.add_scalar('Validation Loss', loss, global_step)
            print('Validation loss:{:3.6f}'.format(loss))

def evaluate(model, model_loss, data_loader):
    model.eval()
    model.to(config.device)

    total_loss = 0.0

    for i, (x, y) in enumerate(data_loader):
        x = x.to(config.device)
        y = y.to(config.device)

        predictions = model(x)

        loss_criterion = model_loss()
        loss = loss_criterion(predictions, y)

        total_loss += loss.item()

    total_loss /= len(data_loader)
    model.train()
    
    return total_loss

def inference(model, dataloader):
    model.eval()
    model.to(config.device)

    signals = []
    ys = []
    for x, y in enumerate(dataloader):
        x = x.to(config.device)
        y = y.to(config.device)
        preds = model(x)
        signals.append(preds)
        ys.append(y)
    signals = torch.cat(signals)
    ys = torch.cat(signals)

    return signals, ys

