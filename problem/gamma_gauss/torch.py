# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
# import numpy as np
import torch.nn as nn

from collections import OrderedDict
from hessian import hessian
from torch.distributions import Gamma
from torch.distributions import Normal

from .config import GGConfig


class GeneratorTorch():
    def __init__(self, seed=None, gamma_k=2.0, gamma_loc=0.0, normal_mean=5.0, normal_sigma=0.5, cuda=False,
                     background_luminosity=1000, signal_luminosity=1000):
        self.seed = seed
        if cuda:
            self.cuda()
        else:
            self.cpu()
        config = GGConfig()
        self.rescale = self.tensor(config.CALIBRATED.rescale, requires_grad=True)
        self.mu = self.tensor(config.CALIBRATED.mu, requires_grad=True)
        self.nuisance_params = OrderedDict([
                                ('rescale', self.rescale),
                                ])
        # Define distributions
        self.gamma_k      = self.tensor(gamma_k)
        self.gamma_loc    = self.tensor(gamma_loc)
        self.gamma_rate   = self.tensor(1.0)

        self.normal_mean  = self.tensor(normal_mean)
        self.normal_sigma = self.tensor(normal_sigma)

        self.gamma = Gamma(self.gamma_k, self.gamma_rate)
        self.norm  = Normal(self.normal_mean, self.normal_sigma)

        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity
        self.n_expected_events = background_luminosity + signal_luminosity

    def cpu(self):
        self.cuda_flag = False
        self.device = 'cpu'

    def cuda(self):
        self.cuda_flag = True
        self.device = 'cuda'

    def tensor(self, data, requires_grad=False, dtype=None):
        return torch.tensor(data, requires_grad=requires_grad, device=self.device, dtype=dtype)

    def sample_event(self, rescale, mu, size=1):
        n_sig = int(mu * size)
        n_bkg = size - n_sig
        rescale = self.tensor(rescale, requires_grad=True)
        mu = self.tensor(mu, requires_grad=True)
        x = self._generate_vars(rescale, mu, n_bkg, n_sig)
        labels = self._generate_labels(n_bkg, n_sig)
        return x, labels

    def __call__(self, n_samples):
        return self.generate(n_samples)

    def generate(self, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
        x_s, w_s, x_b, w_b, y = self._generate(n_bkg=n_bkg, n_sig=n_sig)
        return x_s, w_s, x_b, w_b, y

    def _generate(self, n_bkg=1000, n_sig=50):
        """
        """
        x_s, x_b = self._generate_vars(n_bkg, n_sig)
        y = self._generate_labels(n_bkg, n_sig)
        w_s, w_b = self._generate_weights(n_bkg, n_sig, self.n_expected_events)
        return x_s, w_s, x_b, w_b, y

    def _generate_vars(self, n_bkg, n_sig):
        x_s = self.norm.rsample((n_sig, 1)) * self.rescale
        x_b = self.gamma.rsample((n_bkg, 1)) + self.gamma_loc
        x_b = x_b * self.rescale
        return x_s, x_b

    def _generate_labels(self, n_bkg, n_sig):
        y_b = torch.zeros(n_bkg, dtype=int)
        y_s = torch.ones(n_sig, dtype=int)
        y = torch.cat([y_b, y_s], axis=0)
        if self.cuda_flag:
            y = y.cuda()
        return y

    def _generate_weights(self, n_bkg, n_sig, n_expected_events):
        w_b = torch.ones(n_bkg) * (self.background_luminosity / n_bkg)
        w_s = torch.ones(n_sig) * (self.mu * self.signal_luminosity / n_sig)
        if self.cuda_flag:
            w_b = w_b.cuda()
            w_s = w_s.cuda()
        return w_s.view(-1, 1), w_b.view(-1, 1)

    def diff_generate(self, n_samples=None):
        """Generator for Tangent Propagation"""
        s, w_s, b, w_b, y = self.generate(n_samples=n_samples)
        X = torch.cat([s, b], axis=0)
        w = torch.cat([w_s, w_b], axis=0)
        return X, y, w

    def split_generate(self, n_samples=None):
        """Generator for INFERNO"""
        # torch.autograd.set_detect_anomaly(True)
        s, w_s, b, w_b, y = self.generate(n_samples=n_samples)
        return s, w_s, b, w_b, y

    def proba_density(self, x, rescale, mu):
        """
        Computes p(x | rescale, mu)
        """
        # assert_clean_rescale(rescale)
        # assert_clean_mu(mu)
        proba_gamma   = torch.exp(self.gamma.log_prob(x-self.gamma_loc))
        proba_normal  = torch.exp(self.norm.log_prob(x))
        total_luminosity = mu * self.signal_luminosity + self.background_luminosity
        signal_strength = mu * self.signal_luminosity / total_luminosity
        background_strength = self.background_luminosity / total_luminosity
        proba_density = signal_strength * proba_normal + background_strength * proba_gamma
        return proba_density

    def log_proba_density(self, x, rescale, mu):
        """
        Computes log p(x | rescale, mu)
        """
        proba_density = self.proba_density(x, rescale, mu)
        logproba_density = torch.log(proba_density)
        return logproba_density

    def nll(self, data, rescale, mu):
        """
        Computes the negative log likelihood of the data given y and rescale.
        """
        nll = - self.log_proba_density(data, rescale, mu).sum()
        return nll



class GGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        config = GGConfig()
        rescale_loc = torch.tensor(config.CALIBRATED.rescale)
        rescale_std = torch.tensor(config.CALIBRATED_ERROR.rescale)
        self.rescale_constraints = torch.distributions.Normal(rescale_loc, rescale_std)

        self.constraints_distrib = {'rescale': self.rescale_constraints,
                                   }
        self.i =  0

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
        """
        poisson = torch.distributions.Poisson(input)
        nll = - torch.sum(poisson.log_prob(target)) + self.constraints_nll(params)
        param_list = params.values()
        h = hessian(nll, param_list, create_graph=True)
        h_inverse = torch.inverse(h)  # FIXME : may break, handle exception
        loss = h_inverse[0,0]
        return loss


class GGHessian(nn.Module):
    def __init__(self):
        super().__init__()
        config = GGConfig()
        rescale_loc = torch.tensor(config.CALIBRATED.rescale)
        rescale_std = torch.tensor(config.CALIBRATED_ERROR.rescale)
        self.rescale_constraints = torch.distributions.Normal(rescale_loc, rescale_std)

        self.constraints_distrib = {'rescale': self.rescale_constraints,
                                   }

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
        """
        poisson = torch.distributions.Poisson(input)
        nll = - torch.sum(poisson.log_prob(target)) + self.constraints_nll(params)
        param_list = params.values()
        h = hessian(nll, param_list, create_graph=True)
        return h
