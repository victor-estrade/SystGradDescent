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
from .higgs_4v_pandas import tau_energy_scale
from .higgs_4v_pandas import nasty_background

from sklearn.model_selection import ShuffleSplit


def get_generators(seed, train_size=0.5, test_size=0.1, GeneratorClass=Generator):
    data = load_data()
    data['origWeight'] = data['Weight'].copy()

    cv_train_other = ShuffleSplit(n_splits=1, train_size=train_size, test_size=1-train_size, random_state=seed)
    idx_train, idx_other = next(cv_train_other.split(data, data['Label']))
    train_data = data.iloc[idx_train]
    train_generator = GeneratorClass(train_data, seed=seed)
    other_data = data.iloc[idx_other]

    cv_valid_test = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    idx_valid, idx_test = next(cv_valid_test.split(other_data, other_data['Label']))
    valid_data = other_data.iloc[idx_valid]
    test_data = other_data.iloc[idx_test]
    valid_generator = GeneratorClass(valid_data, seed=seed+1)
    test_generator = GeneratorClass(test_data, seed=seed+2)

    return train_generator, valid_generator, test_generator


def get_balanced_generators(seed, train_size=0.5, test_size=0.1, GeneratorClass=Generator):
    train_generator, valid_generator, test_generator = get_generators(seed, train_size=train_size, test_size=test_size, GeneratorClass=GeneratorClass)
    train_generator.background_luminosity = 1
    train_generator.signal_luminosity = 1

    valid_generator.background_luminosity = 1
    valid_generator.signal_luminosity = 1

    test_generator.background_luminosity = 1
    test_generator.signal_luminosity = 1

    return train_generator, valid_generator, test_generator


def get_easy_generators(seed, train_size=0.5, test_size=0.1, GeneratorClass=Generator):
    train_generator, valid_generator, test_generator = get_generators(seed, train_size=train_size, test_size=test_size, GeneratorClass=GeneratorClass)
    train_generator.background_luminosity = 95
    train_generator.signal_luminosity = 5

    valid_generator.background_luminosity = 95
    valid_generator.signal_luminosity = 5

    test_generator.background_luminosity = 95
    test_generator.signal_luminosity = 5

    return train_generator, valid_generator, test_generator


class GeneratorCPU:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.n_samples = data_generator.size

    def generate(self, *params, n_samples=None, no_grad=False):
            X, y, w = self.data_generator.generate(*params, n_samples=n_samples, no_grad=no_grad)
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            return X, y, w

    def reset(self):
        self.data_generator.reset()



class Generator():
    def __init__(self, data, seed=None, background_luminosity=410999.84732187376,
                              signal_luminosity=691.9886077135781):
        self.data = data
        self.feature_names = data.columns[:-2] if len(data.columns) == 31 else data.columns[:-3]
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.size = data.shape[0]
        self.indexes = np.arange(self.size)
        self.random.shuffle(self.indexes)
        self.i = 0
        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity

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


    def generate(self, tau_es, jet_es, lep_es, mu, n_samples=None, no_grad=None):
        if n_samples is None:
            data = self.data.copy()
        else:
            data = self.sample(n_samples).copy()
        syst_effect(data, tes=tau_es, jes=jet_es, les=lep_es, missing_value=0.0)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data)
        return X.values, y.values, w.values



class MonoGenerator(Generator):
    def generate(self, tau_es, mu, n_samples=None, no_grad=None):
        if n_samples is None:
            data = self.data.copy()
        else:
            data = self.sample(n_samples).copy()
        tau_energy_scale(data, scale=tau_es)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data)
        return X.values, y.values, w.values



class FuturGenerator(Generator):
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
