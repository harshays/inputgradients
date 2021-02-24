import os
import sys
import json 
import copy
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
import numpy as np
import torch


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})

def plot_gradient_quality_real_data(all_data, dataset_name, ax):
    
    def get_marker(n):
        if 'l2' in n: return '^'
        if 'linf' in n: return 's'
        return 'o'

    title_map = {
        'cifar': r'Input gradients of \textbf{ResNet50} on \textbf{CIFAR-10} Data',
        'imagenet': r'Input gradients of \textbf{ResNet18} on \textbf{ImageNet-10} Data'
    }

    legend_map = {
        'cifar': {
            'standard': r'Standard, $\epsilon=0.00$',
            'linf-8': r'$\ell_{\infty}$ Robust, $\epsilon=8/255$',
            'l2-0.50': r'$\ell_2$ Robust, $\epsilon=0.50$',
            'l2-0.25': r'$\ell_2$ Robust $\epsilon=0.25$'        
        },
        'imagenet': {
            'standard': r'Standard, $\epsilon=0.00$',
            'linf-4': r'$\ell_{\infty}$ Robust, $\epsilon=4/255$',
            'linf-2': r'$\ell_{\infty}$ Robust, $\epsilon=2/255$',
            'l2-1.0': r'$\ell_{2}$ Robust, $\epsilon=1.00$',
        }
    }

    xlabel = r'Level $k$: fraction of unmasked pixels'
    ylabel = r'Attribution quality: $\textrm{AQ}(A_{G}, k)$'
    
    data = all_data[dataset_name]
    
    for mtype, label in legend_map[dataset_name].items():
        x, m, s = map(np.array, data[mtype])
        ax.plot(x, m, label=label, marker=get_marker(mtype), ms=7, mfc='w', ls='--', lw=2, mew=2)
        ax.fill_between(x, m-s, m+s, alpha=0.1)

    ax.fill_between([-10,10], -.2, 0, color='black', alpha=0.1, label=r'Feature Inversion')

    if dataset_name == 'cifar':
        ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
        ax.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], minor=True)
        ax.set_yticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax.set_yticks([-0.025, 0.025, 0.075, 0.125, 0.175, 0.225], minor=True)
        ax.set_ylim(bottom=-0.09, top=0.28)
        ax.set_xlim(left=0, right=1.03)

    elif dataset_name == 'imagenet':
        ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
        ax.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], minor=True)
        ax.set_xlim(left=0, right=1.03)
        ax.set_yticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticks([-0.025, 0.025, 0.075, 0.125, 0.175, 0.225], minor=True)
        ax.set_ylim(bottom=-0.09, top=0.175)
        
    ax.axhline(0, color='black', lw=3, ls=':', alpha=0.5, label=r'Random Attribution')
    
    utils.update_ax(ax, title_map[dataset_name], xlabel, ylabel,
                    ticks_fs=15, label_fs=20, title_fs=21, legend_loc=False)

    ax.legend(loc='best', fontsize=17, ncol=2, frameon=True, fancybox=True, 
              handletextpad=0.5, borderpad=0.25, columnspacing=0.9, handlelength=0.75)

    ax.grid(lw=0.5, alpha=0.8)
    return ax


def get_masked_data(X, A, mask_fraction, mask_val):
    """
    X: data / images
    A: attributions (higher -> more important)
    mask_fraction: Masks k% of the least important pixels in X using A
    mask_val: value to be replaced by (usually mean)
    """
    assert X.shape[0] == A.shape[0]
    n = X.shape[0]
    start_idx = int(round(mask_fraction*A.shape[1]))
    
    Xm = copy.deepcopy(X).numpy()
    Xm = Xm.reshape(n, -1)
    
    for idx in range(n):
        mask_idx = A[idx, :start_idx]
        Xm[idx, mask_idx] = mask_val
        
    Xm = torch.FloatTensor(Xm.reshape(X.shape))
    return Xm

def evaluate_robustness(models, loader, pgd_func, eps_vals, device, runs=1, print_info=True):
    """evaluate individual model adversarial robustness"""
    accs = defaultdict(dict)

    for idx, model in enumerate(models):
        model = model.to(device)
        if print_info: print ('Model #{}'.format(idx))
            
        for eps in eps_vals:
            ind_accs = []
            for _ in range(runs):
                attack = pgd_func(eps)
                ev = attack.evaluate_attack(loader, model)
                ind_accs.append(ev['acc'])

            accs[idx][eps] = np.min(ind_accs)

            if print_info:
                print ('eps {}'.format(eps), '{:.3f}'.format(accs[idx][eps]))

        model = model.cpu()

    return accs
