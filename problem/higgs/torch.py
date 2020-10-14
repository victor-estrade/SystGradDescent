# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
# from collections import OrderedDict

from .higgs_geant import load_data
from .higgs_4v_torch import split_data_label_weights
from .higgs_4v_torch import normalize_weight
from .higgs_4v_torch import mu_reweighting
from .higgs_4v_torch import syst_effect
from .higgs_4v_torch import nasty_background

from sklearn.model_selection import ShuffleSplit


def get_generators_torch(seed, train_size=0.5, test_size=0.5, cuda=False):
    data = load_data()
    # print("data.shape", data.shape)
    data['origWeight'] = data['Weight'].copy()
    cv_train_other = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx_train, idx_other = next(cv_train_other.split(data, data['Label']))
    train_data = data.iloc[idx_train]
    train_generator = GeneratorTorch(train_data, seed=seed, cuda=cuda)
    other_data = data.iloc[idx_other]
    # print("train_data.shape", train_data.shape)

    cv_valid_test = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    idx_valid, idx_test = next(cv_valid_test.split(other_data, other_data['Label']))
    valid_data = other_data.iloc[idx_valid]
    test_data = other_data.iloc[idx_test]
    valid_generator = GeneratorTorch(valid_data, seed=seed+1, cuda=cuda)
    test_generator = GeneratorTorch(test_data, seed=seed+2, cuda=cuda)
    # print("valid_data.shape", valid_data.shape)
    # print("test_data.shape", test_data.shape)

    return train_generator, valid_generator, test_generator

def get_balanced_generators_torch(seed, train_size=0.5, test_size=0.1, cuda=False):
    train_generator, valid_generator, test_generator = get_generators_torch(seed, train_size=train_size, test_size=test_size, cuda=cuda)
    train_generator.background_luminosity = 1
    train_generator.signal_luminosity = 1

    valid_generator.background_luminosity = 1
    valid_generator.signal_luminosity = 1

    test_generator.background_luminosity = 1
    test_generator.signal_luminosity = 1

    return train_generator, valid_generator, test_generator


def get_easy_generators_torch(seed, train_size=0.5, test_size=0.1):
    train_generator, valid_generator, test_generator = get_generators_torch(seed, train_size=train_size, test_size=test_size)
    train_generator.background_luminosity = 95
    train_generator.signal_luminosity = 5

    valid_generator.background_luminosity = 95
    valid_generator.signal_luminosity = 5

    test_generator.background_luminosity = 95
    test_generator.signal_luminosity = 5

    return train_generator, valid_generator, test_generator




class GeneratorTorch():
    def __init__(self, data, seed=None, cuda=False,
                background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781):

        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity
        self.seed = seed
        if cuda:
            self.cuda()
        else:
            self.cpu()

        self.feature_names = data.columns[:-2] if len(data.columns) == 31 else data.columns[:-3]
        dtypes = {name : "float32" for name in self.feature_names}
        dtypes.update({"Label": "int32", "Weight": "float32"})
        data = data.astype(dtypes)
        self.data_dict = {col: self.tensor(data[col].values) for col in data.columns}

        self.size = data.shape[0]
        self.i = 0
        self.reset()

    def cpu(self):
        self.cuda_flag = False
        self.device = 'cpu'

    def cuda(self):
        self.cuda_flag = True
        self.device = 'cuda'

    def tensor(self, data, requires_grad=False, dtype=None):
        return torch.tensor(data, requires_grad=requires_grad, device=self.device, dtype=dtype)

    def reset(self):
        torch.manual_seed(self.seed)
        self._restart()

    def _restart(self):
        self.indexes = torch.randperm(self.size, device=self.device)
        self.i = 0

    def __call__(self, n_samples):
        return self.generate(n_samples)

    def sample(self, n_samples):
        assert n_samples > 0, 'n_samples must be > 0'
        blocs = []
        remains = self.size - self.i
        while n_samples > remains:
            excerpt = {k: v[self.indexes[self.i:]] for k, v in self.data_dict.items()}
            blocs.append(excerpt)
            n_samples -= remains
            self._restart()
            remains = self.size
        if n_samples > 0:
            excerpt = {k: v[self.indexes[self.i:self.i+n_samples]] for k, v in self.data_dict.items()}
            blocs.append(excerpt)
            self.i += n_samples
        data_sample = self._concat_bloc(blocs)
        return data_sample

    def _concat_bloc(self, blocs):
        if len(blocs) == 1:
            return blocs[0]
        else:
            col_names = blocs[0].keys()
            data_sample = {col: torch.cat([bloc[col] for bloc in blocs], axis=0)  for col in col_names}
            return data_sample

    def _deep_copy_data(self, data):
        data = {k: v.clone().detach() for k, v in data.items()}
        return data

    def generate(self, tau_es, jet_es, lep_es, mu, n_samples=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                X, y, w = self._generate(tau_es, jet_es, lep_es, mu, n_samples=n_samples)
        else:
            X, y, w = self._generate(tau_es, jet_es, lep_es, mu, n_samples=n_samples)
        return X, y, w

    def _generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        tau_es = self.tensor(tau_es, requires_grad=True, dtype=torch.float32)
        jet_es = self.tensor(jet_es, requires_grad=True, dtype=torch.float32)
        lep_es = self.tensor(lep_es, requires_grad=True, dtype=torch.float32)
        mu = self.tensor(mu, requires_grad=True, dtype=torch.float32)
        missing_value = self.tensor(0.0, dtype=torch.float32)

        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)

        syst_effect(data, tes=tau_es, jes=jet_es, les=lep_es, missing_value=missing_value)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w
