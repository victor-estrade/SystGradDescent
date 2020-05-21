# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from .higgs_geant import split_data_label_weights
from .higgs_geant import load_data
from .higgs_geant import normalize_weight
from .higgs_4v_pandas import mu_reweighting
from .higgs_4v_pandas import syst_effect
from .higgs_4v_pandas import nasty_background

from sklearn.model_selection import ShuffleSplit


def get_generators(seed, train_size=0.5, test_size=0.1):
    data = load_data()
    # print("data.shape", data.shape)
    data['origWeight'] = data['Weight'].copy()
    cv_train_other = ShuffleSplit(n_splits=1, train_size=train_size, test_size=1-train_size, random_state=seed)
    idx_train, idx_other = next(cv_train_other.split(data, data['Label']))
    train_data = data.iloc[idx_train]
    train_generator = Generator(train_data, seed=seed)
    other_data = data.iloc[idx_other]
    # print("train_data.shape", train_data.shape)

    cv_valid_test = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    idx_valid, idx_test = next(cv_valid_test.split(other_data, other_data['Label']))
    valid_data = other_data.iloc[idx_valid]
    test_data = other_data.iloc[idx_test]
    valid_generator = Generator(valid_data, seed=seed+1)
    test_generator = Generator(test_data, seed=seed+2)
    # print("valid_data.shape", valid_data.shape)
    # print("test_data.shape", test_data.shape)

    return train_generator, valid_generator, test_generator



class Generator():
    def __init__(self, data, seed=None):
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
        data_sample = blocs[0] if len(blocs) == 1 else pd.concat(blocs, axis=0)
        return data_sample


    def generate(self, tau_es, jet_es, lep_es, nasty_bkg, sigma_soft, mu, n_samples=None):
        if n_samples is None:
            data = self.data.copy()
        else:
            data = self.sample(n_samples).copy()
        syst_effect(data, tes=tau_es, jes=jet_es, les=lep_es, sigma_met=sigma_soft, missing_value=0.0)
        normalize_weight(data)
        nasty_background(data, nasty_bkg)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data)
        return X.values, y.values, w.values
