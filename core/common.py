import os
import math
import typing as t

import numpy as np
import scipy

import torch.nn as nn
from torch.optim import Adam

import torch.nn.functional as F

import numpy as np
import scipy

Model = t.Literal['GraphDAU', 'NestDAU'],
Task = t.Literal['denoising', 'interpolation']
Suffix = t.Literal['TV-E', 'TV-C', 'EN-E', 'EN-C']


class Eval:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def mean_squared_error(self, x1, x2):
        mse = self.criterion(x1.float(), x2.float())
        return mse

    def root_mean_squared_error(self, x1, x2):
        mse = self.criterion(x1.float(), x2.float())
        rmse = math.sqrt(mse.item())
        return mse, rmse

def soft_threshold(x, thr):
    '''Soft-thresholding function
    :difinition of proximal operator

    Parameters
    ----------
    x : tensor
    thr : beta

    Returns
    -------
    F.softshrink(input_, l) : tensor
    '''
    return F.relu(x - thr) - F.relu(-x - thr)

def compute_norm_adj(graph, self_loop=True):
    adj = graph.W
    sci_I = scipy.sparse.identity(graph.N, dtype='int8', format='dia')
    if self_loop:
        adj = adj + sci_I
        deg = adj.toarray().sum(axis=1)
    else:  # non-self-loop
        deg = adj.toarray().sum(axis=1)
    deg_sloop_mat = np.diag(deg**(-0.5))
    norm_adj = deg_sloop_mat @ adj @ deg_sloop_mat
    return norm_adj

def eval_loss(sig1, sig2):
    '''
    Parameters
    ----------
    sig1 : tensor
        input like origin signal
    sig2 : tensor
        input like noisy or reconst signal
    '''
    criterion = nn.MSELoss(reduction='mean')
    mse = criterion(sig1.float(), sig2.float())
    rmse = math.sqrt(mse)
    return rmse
