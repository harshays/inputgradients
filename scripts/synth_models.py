import sys, copy
import torch, torchvision
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import utils
import numpy as np
import gpu_utils as gu

def get_linear(input_dim, num_classes):
    return nn.Sequential(nn.Linear(input_dim, num_classes))

def get_fcn(idim, hdim, odim, hl=1, activation=nn.ReLU, use_activation=True, use_bn=False, input_dropout=0, dropout=0):
    use_dropout = dropout > 0
    layers = []
    if input_dropout > 0: layers.append(nn.Dropout(input_dropout))
    layers.append(nn.Linear(idim, hdim))
    if use_activation: layers.append(activation())
    if use_dropout: layers.append(nn.Dropout(dropout))
    if use_bn: layers.append(nn.BatchNorm1d(hdim))
    for _ in range(hl-1):
        l = [nn.Linear(hdim, hdim)]
        if use_activation: l.append(activation())
        if use_dropout: l.append(nn.Dropout(dropout))
        if use_bn: l.append(nn.BatchNorm1d(hdim))
        layers.extend(l)
    layers.append(nn.Linear(hdim, odim))
    model = nn.Sequential(*layers)
    return model