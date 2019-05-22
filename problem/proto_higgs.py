# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

from .higgs.higgs_geant import load_data
from .higgs.higgs_geant import normalize_weight
from .higgs.higgs_4v_pandas import mu_reweighting
from .higgs.higgs_4v_pandas import tau_energy_scale
from .higgs.higgs_4v_pandas import jet_energy_scale
from .higgs.higgs_4v_pandas import lep_energy_scale
from .higgs.higgs_4v_pandas import soft_term
from .higgs.higgs_4v_pandas import nasty_background

from sklearn.model_selection import ShuffleSplit

SEED = 42


class Higgs():
    def __init__(self, seed=SEED):
        data = load_data()
        data['origWeight'] = data['Weight'].copy()
        cv_train_other = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        idx_train, idx_other = next(cv_train_other.split(data, data['Label']))
        self.train_data = data.iloc[idx_train]
        self._train_sampler = CircularSampler(self.train_data, seed=seed)
        other_data = data.iloc[idx_other]

        cv_test_final = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed+1)
        idx_test, idx_final = next(cv_test_final.split(other_data, other_data['Label']))
        self.test_data = other_data.iloc[idx_test]
        self.final_data = other_data.iloc[idx_final]

    def train_sample(self, mu, tau_es, jet_es, lep_es, sigma_soft, nasty_bkg, n_samples=None):
        if n_samples is None:
            data = self.train_data.copy()
        else:
            data = self._train_sampler.sample(n_samples).copy()
        tau_energy_scale(data, scale=tau_es)
        jet_energy_scale(data, scale=jet_es)
        lep_energy_scale(data, scale=lep_es)
        soft_term(data, sigma_soft)
        normalize_weight(data)
        nasty_background(data, nasty_bkg)
        mu_reweighting(data, mu)
        return data
    
    def test_sample(self, mu, tau_es, jet_es, lep_es, sigma_soft, nasty_bkg):
        data = self.test_data.copy()
        tau_energy_scale(data, scale=tau_es)
        jet_energy_scale(data, scale=jet_es)
        lep_energy_scale(data, scale=lep_es)
        soft_term(data, sigma_soft)
        normalize_weight(data)
        nasty_background(data, nasty_bkg)
        mu_reweighting(data, mu)
        return data

    def final_sample(self, mu, tau_es, jet_es, lep_es, sigma_soft, nasty_bkg):
        data = self.final_data.copy()
        tau_energy_scale(data, scale=tau_es)
        jet_energy_scale(data, scale=jet_es)
        lep_energy_scale(data, scale=lep_es)
        soft_term(data, sigma_soft)
        normalize_weight(data)
        nasty_background(data, nasty_bkg)
        mu_reweighting(data, mu)
        return data


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

