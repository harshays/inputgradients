import sys

import random, os, copy, pickle, time, random, argparse, itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
import torchvision
from torch import optim, nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

import gpu_utils as gu
import synth_models
import utils
import matplotlib.pyplot as plt

def _get_dataloaders(trd, ted,  bs, pm=True, shuffle=True):
    train_dl = DataLoader(trd, batch_size=bs, shuffle=shuffle, pin_memory=pm)
    test_dl = DataLoader(ted, batch_size=bs, pin_memory=pm)
    return train_dl, test_dl

def get_cifar(fpath='/data/t-hashah/pytorch_datasets/', use_cifar10=False, flatten_data=False, transform_type='none',
              means=None, std=None, img_size=32, use_grayscale=False, binarize=False, normalize=True, y0={0,1,2,3,4}):
    """get preprocessed cifar torch.Dataset class"""

    if transform_type == 'none':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        tensorize = torchvision.transforms.ToTensor()
        to_grayscale = torchvision.transforms.Grayscale()
        flatten = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())

        transforms = [tensorize]
        if use_grayscale: transforms = [to_grayscale] + transforms
        if normalize: transforms.append(normalize_cifar())
        if flatten_data: transforms.append(flatten)
        tr_transforms = te_transforms = torchvision.transforms.Compose(transforms)

    if transform_type == 'basic_resize':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms= [
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    if transform_type == 'basic':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms= [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    if transform_type == 'robustness_repo':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms= [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(.25,.25,.25),
            torchvision.transforms.RandomRotation(2),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)
    target_transforms = to_binary if binarize else None
    dset = 'cifar10' if use_cifar10 else 'cifar100'
    func = torchvision.datasets.CIFAR10 if use_cifar10 else torchvision.datasets.CIFAR100

    X_tr = func(fpath, download=True, transform=tr_transforms, target_transform=target_transforms)
    X_te = func(fpath, download=True, train=False, transform=te_transforms, target_transform=target_transforms)

    return X_tr, X_te

def get_cifar_dl(fpath='/data/t-hashah/pytorch_datasets/', use_cifar10=False, bs=128, shuffle=True, transform_type='none',
                 means=None, std=None, normalize=True, flatten_data=False, use_grayscale=False, nw=3, pm=False, 
                 img_size=32, binarize=False, y0={0,1,2,3,4}):
    """data in dataloaders have has shape (B, C, W, H)"""
    d_tr, d_te = get_cifar(fpath, use_cifar10=use_cifar10, use_grayscale=use_grayscale, transform_type=transform_type, img_size=img_size, normalize=normalize, means=means, std=std, flatten_data=flatten_data, binarize=binarize, y0=y0)

    tr_dl = DataLoader(d_tr, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pm)
    te_dl = DataLoader(d_te, batch_size=bs, num_workers=nw, pin_memory=pm)
    return tr_dl, te_dl