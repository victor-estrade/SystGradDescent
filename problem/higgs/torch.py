# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn
# from collections import OrderedDict

from .higgs_geant import load_data
from .higgs_4v_torch import split_data_label_weights
from .higgs_4v_torch import normalize_weight
from .higgs_4v_torch import mu_reweighting
from .higgs_4v_torch import tau_energy_scale
from .higgs_4v_torch import jet_energy_scale
from .higgs_4v_torch import lep_energy_scale
from .higgs_4v_torch import syst_effect
from .higgs_4v_torch import nasty_background

from sklearn.model_selection import ShuffleSplit

from hessian import hessian

from .config import HiggsConfigTes
from .config import HiggsConfigJes
from .config import HiggsConfigLes
from .config import HiggsConfigTesJes
from .config import HiggsConfigTesLes
from .config import HiggsConfigTesJesLes


class BaseGeneratorTorch():
    def __init__(self, data, seed=None, cuda=False, float32=False,
                background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781):

        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity
        self.seed = seed
        if cuda:
            self.cuda()
        else:
            self.cpu()
        self.float32 = float32

        self.feature_names = data.columns[:-2] if len(data.columns) == 31 else data.columns[:-3]
        if float32 :
            dtypes = {name : "float32" for name in self.feature_names}
            dtypes["Label"] = "int64"
            dtypes["Weight"] = "float32"
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
        raise NotImplemented("This is an abstract class. Use other generators !")

    def _generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        raise NotImplemented("This is an abstract class. Use other generators !")

    def _skew(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        raise NotImplemented("This is an abstract class. Use other generators !")

    def diff_generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        """Generator for Tangent Propagation"""
        raise NotImplemented("This is an abstract class. Use other generators !")

    def split_generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        """Generator for INFERNO"""
        raise NotImplemented("This is an abstract class. Use other generators !")


class GeneratorTorchTes(BaseGeneratorTorch):
    def generate(self, tau_es, mu, n_samples=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                X, y, w = self._generate(tau_es, mu, n_samples=n_samples)
        else:
            X, y, w = self._generate(tau_es, mu, n_samples=n_samples)
        return X, y, w

    def _generate(self, tau_es, mu, n_samples=None):
        tau_es = self.tensor(tau_es, requires_grad=True, dtype=torch.float32)
        mu = self.tensor(mu, requires_grad=True, dtype=torch.float32)
        X, y, w = self._skew(tau_es, mu, n_samples=n_samples)
        return X, y, w

    def _skew(self, tau_es, mu, n_samples=None):
        missing_value = self.tensor(0.0, dtype=torch.float32)
        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)

        data = tau_energy_scale(data, scale=tau_es, missing_value=missing_value)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w

    def diff_generate(self, tau_es, mu, n_samples=None):
        """Generator for Tangent Propagation"""
        X, y, w = self._skew(tau_es, mu, n_samples=n_samples)
        return X, y, w.view(-1, 1)

    def split_generate(self, tau_es, mu, n_samples=None):
        """Generator for INFERNO"""
        # torch.autograd.set_detect_anomaly(True)
        X, y, w = self._skew(tau_es, mu, n_samples=n_samples)
        X_s = X[y==1]
        X_b = X[y==0]
        w_s = w[y==1]
        w_b = w[y==0]
        return X_s, w_s.view(-1, 1), X_b, w_b.view(-1, 1), y


class GeneratorTorchJes(BaseGeneratorTorch):
    def generate(self, jet_es, mu, n_samples=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                X, y, w = self._generate(jet_es, mu, n_samples=n_samples)
        else:
            X, y, w = self._generate(jet_es, mu, n_samples=n_samples)
        return X, y, w

    def _generate(self, jet_es, mu, n_samples=None):
        jet_es = self.tensor(jet_es, requires_grad=True, dtype=torch.float32)
        mu = self.tensor(mu, requires_grad=True, dtype=torch.float32)
        X, y, w = self._skew(jet_es, mu, n_samples=n_samples)
        return X, y, w

    def _skew(self, jet_es, mu, n_samples=None):
        missing_value = self.tensor(0.0, dtype=torch.float32)
        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)

        data = jet_energy_scale(data, scale=jet_es, missing_value=missing_value)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w

    def diff_generate(self, jet_es, mu, n_samples=None):
        """Generator for Tangent Propagation"""
        X, y, w = self._skew(jet_es, mu, n_samples=n_samples)
        return X, y, w.view(-1, 1)

    def split_generate(self, jet_es, mu, n_samples=None):
        """Generator for INFERNO"""
        # torch.autograd.set_detect_anomaly(True)
        X, y, w = self._skew(jet_es, mu, n_samples=n_samples)
        X_s = X[y==1]
        X_b = X[y==0]
        w_s = w[y==1]
        w_b = w[y==0]
        return X_s, w_s.view(-1, 1), X_b, w_b.view(-1, 1), y


class GeneratorTorchLes(BaseGeneratorTorch):
    def generate(self, lep_es, mu, n_samples=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                X, y, w = self._generate(lep_es, mu, n_samples=n_samples)
        else:
            X, y, w = self._generate(lep_es, mu, n_samples=n_samples)
        return X, y, w

    def _generate(self, lep_es, mu, n_samples=None):
        lep_es = self.tensor(lep_es, requires_grad=True, dtype=torch.float32)
        mu = self.tensor(mu, requires_grad=True, dtype=torch.float32)
        X, y, w = self._skew(lep_es, mu, n_samples=n_samples)
        return X, y, w

    def _skew(self, lep_es, mu, n_samples=None):
        missing_value = self.tensor(0.0, dtype=torch.float32)
        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)

        data = lep_energy_scale(data, scale=lep_es, missing_value=missing_value)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w

    def diff_generate(self, lep_es, mu, n_samples=None):
        """Generator for Tangent Propagation"""
        X, y, w = self._skew(lep_es, mu, n_samples=n_samples)
        return X, y, w.view(-1, 1)

    def split_generate(self, lep_es, mu, n_samples=None):
        """Generator for INFERNO"""
        # torch.autograd.set_detect_anomaly(True)
        X, y, w = self._skew(lep_es, mu, n_samples=n_samples)
        X_s = X[y==1]
        X_b = X[y==0]
        w_s = w[y==1]
        w_b = w[y==0]
        return X_s, w_s.view(-1, 1), X_b, w_b.view(-1, 1), y



class GeneratorTorchTesJesLes(BaseGeneratorTorch):
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
        X, y, w = self._skew(tau_es, jet_es, lep_es, mu, n_samples=n_samples)
        return X, y, w

    def _skew(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        missing_value = self.tensor(0.0, dtype=torch.float32)
        data = self.data_dict if (n_samples is None) else self.sample(n_samples)
        data = self._deep_copy_data(data)

        syst_effect(data, tes=tau_es, jes=jet_es, les=lep_es, missing_value=missing_value)
        normalize_weight(data, background_luminosity=self.background_luminosity, signal_luminosity=self.signal_luminosity)
        mu_reweighting(data, mu)
        X, y, w = split_data_label_weights(data, self.feature_names)
        return X, y, w

    def diff_generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        """Generator for Tangent Propagation"""
        X, y, w = self._skew(tau_es, jet_es, lep_es, mu, n_samples=n_samples)
        return X, y, w.view(-1, 1)

    def split_generate(self, tau_es, jet_es, lep_es, mu, n_samples=None):
        """Generator for INFERNO"""
        # torch.autograd.set_detect_anomaly(True)
        X, y, w = self._skew(tau_es, jet_es, lep_es, mu, n_samples=n_samples)
        X_s = X[y==1]
        X_b = X[y==0]
        w_s = w[y==1]
        w_b = w[y==0]
        return X_s, w_s.view(-1, 1), X_b, w_b.view(-1, 1), y



ALL_GENERATOR_DICT = {
    'Tes' : GeneratorTorchTes,
    'Jes' : GeneratorTorchJes,
    'Les' : GeneratorTorchLes,
    # 'TesJes' : GeneratorTorchTesJes,
    # 'TesLes' : GeneratorTorchTesLes,
    'TesJesLes' : GeneratorTorchTesJesLes,
}


def get_generator_class(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_GENERATOR_DICT :
        return ALL_GENERATOR_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")


class BaseHiggsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def constraints_nll(self, params):
        nll = 0.0
        for param_name, distrib in self.constraints_distrib.items():
            if param_name in params:
                nll = nll - distrib.log_prob(params[param_name])
        return nll

    def forward(self, input, target, params):
        """
        input is the total count, the summaries,
        target is the asimov, the expected
        param_list is the OrderedDict of tensor containing the parameters
                [mu, tes, jes, les]
        """
        poisson = torch.distributions.Poisson(input)
        nll = - torch.sum(poisson.log_prob(target)) + self.constraints_nll(params)
        param_list = params.values()
        h = hessian(nll, param_list, create_graph=True)
        h_inverse = torch.inverse(h)  # FIXME : may break, handle exception
        loss = h_inverse[0,0]
        return loss


class HiggsLossTes(BaseHiggsLoss):
    def __init__(self):
        super().__init__()
        config = HiggsConfigTes()
        tes_loc = torch.tensor(config.CALIBRATED.tes)
        tes_std = torch.tensor(config.CALIBRATED_ERROR.tes)
        self.tes_constraints = torch.distributions.Normal(tes_loc, tes_std)

        self.constraints_distrib = {'tes': self.tes_constraints,
                                   }


class HiggsLossJes(BaseHiggsLoss):
    def __init__(self):
        super().__init__()
        config = HiggsConfigJes()
        jes_loc = torch.tensor(config.CALIBRATED.jes)
        jes_std = torch.tensor(config.CALIBRATED_ERROR.jes)
        self.jes_constraints = torch.distributions.Normal(jes_loc, jes_std)

        self.constraints_distrib = {'jes': self.jes_constraints,
                                   }

class HiggsLossLes(BaseHiggsLoss):
    def __init__(self):
        super().__init__()
        config = HiggsConfigLes()
        les_loc = torch.tensor(config.CALIBRATED.les)
        les_std = torch.tensor(config.CALIBRATED_ERROR.les)
        self.les_constraints = torch.distributions.Normal(les_loc, les_std)

        self.constraints_distrib = {'les': self.les_constraints,
                                   }



class HiggsLossTesJesLes(BaseHiggsLoss):
    def __init__(self):
        super().__init__()
        config = HiggsConfigTesJesLes()
        tes_loc = torch.tensor(config.CALIBRATED.tes)
        tes_std = torch.tensor(config.CALIBRATED_ERROR.tes)
        self.tes_constraints = torch.distributions.Normal(tes_loc, tes_std)

        jes_loc = torch.tensor(config.CALIBRATED.jes)
        jes_std = torch.tensor(config.CALIBRATED_ERROR.jes)
        self.jes_constraints = torch.distributions.Normal(jes_loc, jes_std)

        les_loc = torch.tensor(config.CALIBRATED.les)
        les_std = torch.tensor(config.CALIBRATED_ERROR.les)
        self.les_constraints = torch.distributions.Normal(les_loc, les_std)
        self.constraints_distrib = {'tes': self.tes_constraints,
                                    'jes': self.jes_constraints,
                                    'les': self.les_constraints,
                                   }



ALL_HIGGSLOSS_DICT = {
    'Tes' : HiggsLossTes,
    'Jes' : HiggsLossJes,
    'Les' : HiggsLossLes,
    # 'TesJes' : HiggsLossTesJes,
    # 'TesLes' : HiggsLossTesLes,
    'TesJesLes' : HiggsLossTesJesLes,
}


def get_higgsloss_class(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_HIGGSLOSS_DICT :
        return ALL_HIGGSLOSS_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")



def get_generators_torch(seed, train_size=0.5, test_size=0.5, cuda=False, GeneratorClass=GeneratorTorchTesJesLes):
    data = load_data()
    data['origWeight'] = data['Weight'].copy()

    cv_train_other = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx_train, idx_other = next(cv_train_other.split(data, data['Label']))
    train_data = data.iloc[idx_train]
    train_generator = GeneratorClass(train_data, seed=seed, cuda=cuda, float32=True)
    other_data = data.iloc[idx_other]

    cv_valid_test = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    idx_valid, idx_test = next(cv_valid_test.split(other_data, other_data['Label']))
    valid_data = other_data.iloc[idx_valid]
    test_data = other_data.iloc[idx_test]
    valid_generator = GeneratorClass(valid_data, seed=seed+1, cuda=cuda)
    test_generator = GeneratorClass(test_data, seed=seed+2, cuda=cuda)

    return train_generator, valid_generator, test_generator



def get_balanced_generators_torch(seed, train_size=0.5, test_size=0.1, cuda=False, GeneratorClass=GeneratorTorchTesJesLes):
    train_generator, valid_generator, test_generator = get_generators_torch(seed, train_size=train_size,
                                    test_size=test_size, cuda=cuda, GeneratorClass=GeneratorClass)
    train_generator.background_luminosity = 1
    train_generator.signal_luminosity = 1

    valid_generator.background_luminosity = 1
    valid_generator.signal_luminosity = 1

    test_generator.background_luminosity = 1
    test_generator.signal_luminosity = 1

    return train_generator, valid_generator, test_generator


def get_easy_generators_torch(seed, train_size=0.5, test_size=0.1, cuda=False, GeneratorClass=GeneratorTorchTesJesLes):
    train_generator, valid_generator, test_generator = get_generators_torch(seed, train_size=train_size,
                                    test_size=test_size, cuda=cuda, GeneratorClass=GeneratorClass)
    train_generator.background_luminosity = 95
    train_generator.signal_luminosity = 5

    valid_generator.background_luminosity = 95
    valid_generator.signal_luminosity = 5

    test_generator.background_luminosity = 95
    test_generator.signal_luminosity = 5

    return train_generator, valid_generator, test_generator
