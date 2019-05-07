# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

def to_numpy(X):
    if isinstance(X, pd.core.generic.NDFrame):
        X = X.values
    return X


def balance_training_weight(w, y):
    """Balance the weights between positive and negative class."""
    sample_weight = w.copy()
    neg_mask = (y == 0)
    pos_mask = (y == 1)
    
    bkg_sum_weight = np.sum(sample_weight[neg_mask])
    sig_sum_weight = np.sum(sample_weight[pos_mask])

    sample_weight[pos_mask] = sample_weight[pos_mask] / sig_sum_weight
    sample_weight[neg_mask] = sample_weight[neg_mask] / bkg_sum_weight
    return sample_weight

