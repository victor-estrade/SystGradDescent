# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from hessian import hessian
from torch.distribution import Gamma
from torch.distribution import Normal


SEED = 42

class GeneratorTorch():
    def __init__(self, seed=None, gamma_k=2, gamma_loc=0, normal_mean=5, normal_sigma=0.5, cuda=False):
        self.seed = seed
        self.gamma_k = gamma_k
        self.gamma_loc = gamma_loc
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma
        self.n_expected_events = 2000
        if cuda:
            self.cuda()
        else:
            self.cpu()

    def cpu(self):
        self.device = 'cpu'

    def cuda(self):
        self.device = 'cuda'

    def tensor(self, data, requires_grad=False, dtype=None):
        return torch.tensor(data, requires_grad=requires_grad, device=self.device, dtype=None)


    def sample_event(self, rescale, mix, size=1):
        n_sig = int(mix * size)
        n_bkg = size - n_sig
        rescale = self.tensor(rescale, requires_grad=True)
        mix = self.tensor(mix, requires_grad=True)
        x = self._generate_vars(rescale, mix, n_bkg, n_sig)
        labels = self._generate_labels(n_bkg, n_sig)
        return x, labels

    def generate(self, rescale, mix, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
        rescale = self.tensor(rescale, requires_grad=True)
        mix = self.tensor(mix, requires_grad=True)
        X, y, w = self._generate(rescale, mix, n_bkg=n_bkg, n_sig=n_sig)
        return X, y, w

    def _generate(self, rescale, mix, n_bkg=1000, n_sig=50):
        """
        """
        X = self._generate_vars(rescale, n_bkg, n_sig)
        y = self._generate_labels(n_bkg, n_sig)
        w = self._generate_weights(mix, n_bkg, n_sig, self.n_expected_events)
        return X, y, w

    def _generate_vars(self, rescale, n_bkg, n_sig):
        gamma_k      = self.tensor(self.gamma_k)
        gamma_loc    = self.tensor(self.gamma_loc)
        gamma_rate   = self.tensor(1.0) / rescale
        normal_mean  = self.tensor(self.normal_mean) * rescale
        normal_sigma = self.tensor(self.normal_sigma) * rescale
        gamma = Gamma(gamma_k, gamma_rate)
        norm  = Normal(normal_mean, normal_sigma)
        x_b = gamma.rsample((n_bkg,)) + gamma_loc
        x_s = norm.rsample((n_sig,))
        x = torch.cat([x_b, x_s], axis=0)
        return x

    def _generate_labels(self, n_bkg, n_sig):
        y_b = torch.zeros(n_bkg)
        y_s = torch.ones(n_sig)
        y = torch.cat([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mix, n_bkg, n_sig, n_expected_events):
        w_b = torch.ones(n_bkg) * (1-mix) * n_expected_events/n_bkg
        w_s = torch.ones(n_sig) * mix * n_expected_events/n_sig
        w = torch.cat([w_b, w_s], axis=0)
        return w

    def proba_density(self, x, rescale, mix):
        """
        Computes p(x | rescale, mix)
        """
        # assert_clean_rescale(rescale)
        # assert_clean_mix(mix)
        gamma_k      = self.tensor(self.gamma_k)
        gamma_loc    = self.tensor(self.gamma_loc)
        gamma_rate   = self.tensor(1.0) / rescale
        normal_mean  = self.tensor(self.normal_mean) * rescale
        normal_sigma = self.tensor(self.normal_sigma) * rescale
        gamma = Gamma(gamma_k, gamma_rate)
        norm  = Normal(normal_mean, normal_sigma)
        proba_gamma   = torch.exp(gamma.log_prob(x-gamma_loc))
        proba_normal  = torch.exp(norm.log_prob(x))
        proba_density = mix * proba_normal + (1-mix) * proba_gamma
        return proba_density

    def log_proba_density(self, x, rescale, mix):
        """
        Computes log p(x | rescale, mix)
        """
        proba_density = self.proba_density(x, rescale, mix)
        logproba_density = torch.log(proba_density)
        return logproba_density

    def nll(self, data, rescale, mix):
        """
        Computes the negative log likelihood of the data given y and rescale.
        """
        nll = - self.log_proba_density(data, rescale, mix).sum()
        return nll



class GGLoss(nn.Module):
    # TODO : S3D2 to GG refactor
    def __init__(self):
        super().__init__()
        r_loc, r_std = 0.0, 0.4
        self.r_constraints = torch.distributions.Normal(r_loc, r_std)

        b_rate_loc, b_rate_std = 0.0, 0.4
        self.b_rate_constraints = torch.distributions.Normal(b_rate_loc, b_rate_std)
        self.constraints_distrib = {'r_dist': self.r_constraints,
                                    'b_rate': self.b_rate_constraints,
                                   }
        self.i =  0
    def constraints_nll(self, params):
        nll = 0
        for param_name, distrib in self.constraints_distrib.items():
            if param_name in params:
                nll = nll - distrib.log_prob(params['b_rate'])
        return nll

    def forward(self, input, target, params):
        """
        input is the total count, the summaries, 
        target is the asimov, the expected
        param_list is the OrderedDict of tensor containing the parameters
                [MU, R_DIST, B_RATE]
        """
        poisson = torch.distributions.Poisson(input)
        nll = - torch.sum(poisson.log_prob(target)) + self.constraints_nll(params)
        param_list = params.values()
        h = hessian(nll, param_list, create_graph=True)
        h_inverse = torch.inverse(h)  # FIXME : may break, handle exception
        loss = h_inverse[0,0]
        return loss

