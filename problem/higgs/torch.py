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
        # dtypes = {name : "float32" for name in self.feature_names}
        # dtypes.update({"Label": "int32", "Weight": "float32"})
        # data = data.astype(dtypes)
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
        return torch.tensor(data, requires_grad=requires_grad, device=self.device, dtype=None)

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

    def generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)
        syst_effect(data, tes=tau_es, jes=jet_es, les=lep_es, missing_value=0.0)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w
