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
    tensor = arr if torch.is_tensor(arr) else torch.from_numpy(arr)
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
    categories   = np.unique(y)
    n_samples    = y.shape[0]
    n_categories = len(categories)
    for c in categories:
        mask = (y == c)
        w_sum = np.sum(w[mask])
        w[mask] = w[mask] / w_sum
    w = w * n_samples / n_categories
    return w
