import numpy as np
from scipy.linalg import qr
import pandas as pd
import pickle
import copy
from collections import defaultdict, Counter, OrderedDict
import time, datetime
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate

import torch
import torchvision
from torch import optim, nn
import torch.nn.functional as F

import misc_utils as mu

class SaliencyMap(object):

    def __init__(self, device):
        self.device = device 

    def get_attributions(self):
        pass 

    def _get_attributions_given_batch(self):
        pass

    def get_attributions_given_dl(self, dl, model):
        model = model.to(self.device)
        grads = []

        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            gb = self._get_attributions_given_batch(xb, yb, model)
            grads.append(gb)
            
            model.zero_grad()
            #xb.grad.zero_()
            xb.cpu()
            yb.cpu()
            del xb 
        
        grads = torch.cat(grads)
        return grads.numpy() 

    def get_attribution_indices(self, G, topk=True):
        """return top/bottom-k attribution A given saliency map G"""
        Gf = np.abs(G.reshape(G.shape[0], -1))
        indices = np.argsort(Gf, axis=1)
        if topk: indices = np.flip(indices, 1)
        return indices 

    def get_attribution_masked_data(self, X, A, mask_fraction, mask_val):
        """return masked data given data + attribution indices"""
        return mu.get_masked_data(X, A, mask_fraction, mask_val)

# random attributions
class RandomAttribution(SaliencyMap):

    def __init__(self, device, batch_size=100):
        super().__init__(device)
        self.bs = batch_size

    def get_attributions(self, X, Y, model):
        model = model.to(self.device)
        grads = []

        sampler = torch.utils.data.SequentialSampler(X)
        sampler = torch.utils.data.BatchSampler(sampler, self.bs, False)

        for idx in sampler:
            xb = X[idx].to(self.device)
            yb = Y[idx].to(self.device)
            
            gb = self._get_attributions_given_batch(xb, yb, model)
            grads.append(gb)
            
            model.zero_grad()
            yb.cpu()
            del xb 

        grads = torch.cat(grads)
        return grads.numpy() 

    def _get_attributions_given_batch(self, xb, yb, model):
        # helper; assume all same device
        return torch.randn_like(xb).cpu()

# loss and logit gradients
class LossGrad(SaliencyMap):

    def __init__(self, device, loss_fn=F.cross_entropy, loss_reduction='sum', batch_size=100):
        super().__init__(device)
        self.loss_fn = loss_fn
        self.loss_reduction = loss_reduction
        self.bs = batch_size
        
    def get_attributions(self, X, Y, model):
        model = model.to(self.device)
        grads = []

        sampler = torch.utils.data.SequentialSampler(X)
        sampler = torch.utils.data.BatchSampler(sampler, self.bs, False)

        for idx in sampler:
            xb = X[idx].to(self.device)
            yb = Y[idx].to(self.device)
            
            gb = self._get_attributions_given_batch(xb, yb, model)
            grads.append(gb)
            
            model.zero_grad()
            yb.cpu()
            del xb 

        grads = torch.cat(grads)
        return grads.numpy() 

    def _get_attributions_given_batch(self, xb, yb, model):
        # helper; assume all same device
        xb = torch.autograd.Variable(xb)
        xb.requires_grad = True

        out = model(xb)
        loss = self.loss_fn(out, yb, reduction=self.loss_reduction)
        loss.backward()

        gb = xb.grad.detach().cpu()
        return gb

class LogitGrad(SaliencyMap):

    def __init__(self, device, batch_size=100, apply_softmax=False):
        super().__init__(device)
        self.bs = batch_size
        self.apply_softmax = apply_softmax
    
    def get_attributions(self, X, Y, model):
        model = model.to(self.device)
        grads = []

        sampler = torch.utils.data.SequentialSampler(X)
        sampler = torch.utils.data.BatchSampler(sampler, self.bs, False)

        for idx in sampler:
            xb = X[idx].to(self.device)
            yb = Y[idx].to(self.device)
            
            gb = self._get_attributions_given_batch(xb, yb, model)
            grads.append(gb)
            
            model.zero_grad()
            yb.cpu()
            del xb 

        grads = torch.cat(grads)
        return grads.numpy() 

    def _get_attributions_given_batch(self, xb, yb, model):
        # helper; assume all same device
        xb = torch.autograd.Variable(xb)
        xb.requires_grad = True

        out = model(xb)
        if self.apply_softmax: out = F.softmax(out, dim=1)

        indexed_out = out.gather(1, yb.reshape(-1, 1))
        indexed_out_sum = indexed_out.sum()
        indexed_out_sum.backward()
        gb = xb.grad.detach().cpu()
        return gb

# loss + logit/softmax gradient times input
class LossGradientTimesInput(LossGrad):

    def __init__(self, device, loss_fn=F.cross_entropy, loss_reduction='sum', batch_size=100):
        super().__init__(device, loss_fn=F.cross_entropy, loss_reduction=loss_reduction, batch_size=batch_size)

    def _get_attributions_given_batch(self, xb, yb, model):
        # helper; assume all same device
        gb = super()._get_attributions_given_batch(xb, yb, model)
        return np.multiply(xb.detach().cpu().numpy(), gb)

class LogitGradientTimesInput(LogitGrad):

    def __init__(self, device, batch_size=100, apply_softmax=True):
        super().__init__(device, batch_size=batch_size, apply_softmax=apply_softmax)

    def _get_attributions_given_batch(self, xb, yb, model):
        # helper; assume all same device
        gb = super()._get_attributions_given_batch(xb, yb, model)
        return np.multiply(xb.detach().cpu().numpy(), gb)