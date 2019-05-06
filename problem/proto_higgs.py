# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit

SEED = 42


def transform(data, param):
    """Dummy transform function"""
    transformed_data = data.copy()
    transformed_data['label'] = transformed_data['label'] + 2
    return transformed_data


class ProtoHiggs():
    def __init__(self, data, test_size=0.5, seed=SEED):
        cv_train_test = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_train, idx_test = next(cv_train_test.split(data, data['label']))
        self.train_data = data.iloc[idx_train]
        self.test_data = data.iloc[idx_test]
        self._train_sampler = CircularSampler(self.train_data, seed=seed)
        
    def train_sample(self, param, n_samples=None):
        if n_samples is None:
            return self.train_data
        data = self._train_sampler.sample(n_samples)
        data = transform(data, param)
        return data
    
    def test_sample(self):
        return self.test_data


class CircularSampler():
    def __init__(self, data, seed=SEED):
        self.data = data
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.size = data.shape[0]
        self.indexes = np.arange(self.size)
        self.random.shuffle(self.indexes)
        self.i = 0

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)
        self.indexes = np.arange(self.size)
        self._restart()

    def _restart(self):
        self.random.shuffle(self.indexes)
        self.i = 0
    
    def sample(self, n_samples):
        assert n_samples > 0, 'n_samples must be > 0'
        blocs = []
        remains = self.size - self.i
        while n_samples > remains:
            excerpt = self.data.iloc[self.indexes[self.i:]]
            blocs.append(excerpt)
            n_samples -= remains
            self._restart()
            remains = self.size - self.i
        if n_samples > 0:
            excerpt = self.data.iloc[self.indexes[self.i:self.i+n_samples]]
            blocs.append(excerpt)            
            self.i += n_samples
        res = blocs[0] if len(blocs) == 1 else pd.concat(blocs, axis=0)
        return res

