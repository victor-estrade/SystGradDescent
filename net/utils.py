#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np
import pandas as pd


def to_torch(arr, cuda=True):
    """
    Transform given numpy array to a torch.autograd.Variable
    """
    tensor = torch.from_numpy(arr)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def to_numpy(X):
    if isinstance(X, pd.core.generic.NDFrame):
        X = X.values
    return X


def classwise_balance_weight(sample_weight, y):
    """Balance the weights between positive (1) and negative (0) class."""
    w = sample_weight.copy()
    neg_mask = (y == 0)
    pos_mask = (y == 1)
    
    neg_sum_weight = np.sum(w[neg_mask])
    pos_sum_weight = np.sum(w[pos_mask])

    w[neg_mask] = w[neg_mask] / neg_sum_weight
    w[pos_mask] = w[pos_mask] / pos_sum_weight
    return w
