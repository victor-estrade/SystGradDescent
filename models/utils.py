# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import pandas as pd

def to_numpy(X):
    if isinstance(X, pd.core.generic.NDFrame):
        X = X.values
    return X
