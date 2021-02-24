import torch
from torch import optim, nn
import torch.nn.functional as F
import random 
import utils 
import gpu_utils as gu
import numpy as np 
from collections import defaultdict

class RandomSignalCoordinate(object):

    def __init__(self, input_dim, num_signal, min_val=1.0, max_val=1.0):
        self.d = input_dim
        self.dims = np.array(list(range(self.d)))
        self.num_sig = num_signal
        self.min_val = min_val 
        self.max_val = max_val 
        self.gen_signal = lambda: min_val + (max_val-min_val)*random.random()
        self.dataset_gen = False  
        self.bounds = None
    
    def _get_dataset(self, N):
        # diff objects for train and test..
        if self.dataset_gen: assert False, "make new obj or use .X"
        
        # generate labels
        self.N = N
        Y = torch.randint(low=0, high=2, size=(N,)).float()
        
        # generate data
        X = torch.zeros(N, self.d) 
        
        for idx in range(N):
            indices = np.random.choice(self.dims, size=self.num_sig, replace=False)
            label = 2.*Y[idx]-1.
            for i in indices: X[idx, i] = self.gen_signal()*label
            
        self.X, self.Y = X.float(), Y.long()
        self.dataset_gen = True        
        self.bounds = (X.min().item(), X.max().item())
        return self.X, self.Y
    
    def get_dataloader(self, N, bs):
        if self.dataset_gen: return self.dl
        self.N = N
        self.X, self.Y = self._get_dataset(N)
        self.dl = utils._to_dl(self.X, self.Y, bs)
        return self.dl
    
    def get_input_gradients(self, model, device, bs=100, output='loss'):
        if output=='loss':
            G = utils.get_input_loss_gradients(self.X, self.Y, model, device=device, 
                                               loss_reduction='sum', bs=100, flatten=False)
        elif output=='logit':
            G = utils.get_input_logit_gradients(self.X, self.Y, model, device=device, bs=100)
        return G
        
    def get_model_predictions(self, model, device, bs=100):
        Yh = utils.get_predictions_given_tensor(self.X, model, device=device, bs=bs).cpu()
        return Yh
    
class SemiRandomSignalCoordinate(RandomSignalCoordinate):

    def __init__(self, input_dim, num_signal, bias_prob, min_val=1.0, max_val=1.0):
        super().__init__(input_dim, num_signal, min_val=min_val, max_val=max_val)
        self.bias_prob = bias_prob
        self.rem_dims = self.dims[1:]

    def _get_dataset(self, N):
        # diff objects for train and test..
        if self.dataset_gen: assert False, "make new obj or use .X"
        
        # generate labels
        self.N = N
        Y = torch.randint(low=0, high=2, size=(N,)).float()
        
        # generate data
        X = torch.zeros(N, self.d) 
        
        for idx in range(N):
            label = 2.*Y[idx]-1.
            if random.random() < self.bias_prob:
                indices = [0]
            else:
                indices = np.random.choice(self.rem_dims, size=self.num_sig, replace=False)
            for i in indices: X[idx, i] = self.gen_signal()*label
            
        self.X, self.Y = X.float(), Y.long()
        self.dataset_gen = True        
        self.bounds = (X.min().item(), X.max().item())
        return self.X, self.Y
