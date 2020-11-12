import torch
import numpy as np
from collections import OrderedDict

import torch.nn as nn

from hessian import hessian

from .config import S3D2Config

SEED = 42

class GeneratorTorch():
    def __init__(self, seed=SEED, r_dist=2.0, b_rate=3.0, s_rate=2.0, ratio=50/(1000+50),
                        background_luminosity=1000, signal_luminosity=50,
                        reset_every=None, cuda=False):
        self.seed = seed
        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity
        if cuda:
            self.cuda()
        else:
            self.cpu()
        self.R_DIST = self.tensor(r_dist, requires_grad=True)
        self.B_RATE = self.tensor(b_rate, requires_grad=True)
        self.s_b_ratio = self.tensor(ratio, requires_grad=True)
        self.nuisance_params = OrderedDict([
                                ('r', self.R_DIST),
                                ('lam', self.B_RATE),
                                ])
        zero = torch.zeros(1)
        if self.cuda_flag:
            zero = zero.cuda()
        self.b_loc = torch.cat([self.R_DIST.view(-1), zero])
        self.b_cov = torch.from_numpy(np.array([[5., 0.], [0., 9.]], dtype=np.float32))
        if self.cuda_flag:
            self.b_cov = self.b_cov.cuda()
        self.b_01 = torch.distributions.MultivariateNormal(loc=self.b_loc, covariance_matrix=self.b_cov)
        self.b_2 = torch.distributions.Exponential(self.B_RATE)

        self.S_RATE = self.tensor(s_rate)
        self.s_loc =  torch.zeros(2)
        self.s_cov = torch.eye(2)
        if self.cuda_flag:
            self.s_loc = self.s_loc.cuda()
            self.s_cov = self.s_cov.cuda()
        self.s_01 = torch.distributions.MultivariateNormal(loc=self.s_loc, covariance_matrix=self.s_cov)
        self.s_2 = torch.distributions.Exponential(self.S_RATE)

        self.n_generated = 0
        self.reset_every = reset_every

    def cpu(self):
        self.cuda_flag = False
        self.device = 'cpu'

    def cuda(self):
        self.cuda_flag = True
        self.device = 'cuda'

    def tensor(self, data, requires_grad=False, dtype=None):
        return torch.tensor(data, requires_grad=requires_grad, device=self.device, dtype=dtype)

    def gen_bkg(self, n_samples):
        a = self.b_01.rsample((n_samples,))
        b = self.b_2.rsample((n_samples,)).type(a.type())
        return torch.cat((a, b.view(-1, 1)), dim=1)

    def gen_sig(self, n_samples):
        a = self.s_01.rsample((n_samples,))
        b = self.s_2.rsample((n_samples,)).type(a.type())
        return torch.cat((a, b.view(-1, 1)), dim=1)

    def _generate_labels(self, n_bkg, n_sig):
        y_b = torch.zeros(n_bkg, dtype=int)
        y_s = torch.ones(n_sig, dtype=int)
        y = torch.cat([y_b, y_s], axis=0)
        if self.cuda_flag:
            y = y.cuda()
        return y

    def _generate_weights(self, n_bkg, n_sig):
        w_b = torch.ones(n_bkg) * (self.background_luminosity / n_bkg)
        w_s = torch.ones(n_sig) * (self.signal_luminosity / n_sig)
        if self.cuda_flag:
            w_b = w_b.cuda()
            w_s = w_s.cuda()
        return w_s.view(-1, 1), w_b.view(-1, 1)

    def generate(self, n_samples):
        if self.reset_every is not None and self.reset_every < self.n_generated:
            self.reset()
            self.n_generated = 0
        n_signals = int(n_samples*self.s_b_ratio)
        n_backgrounds = int(n_samples*(1-self.s_b_ratio))
        s = self.gen_sig(n_signals)
        b = self.gen_bkg(n_backgrounds)
        self.n_generated += n_samples
        w_s, w_b = self._generate_weights(n_backgrounds, n_signals)
        y = self._generate_labels(n_backgrounds, n_signals)
        return s, w_s, b, w_b, y

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

    def reset(self):
        torch.manual_seed(self.seed)

    def __call__(self, n_samples):
        return self.generate(n_samples)


class S3D2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        config = S3D2Config()
        r_loc = torch.tensor(config.CALIBRATED.r)
        r_std = torch.tensor(config.CALIBRATED_ERROR.r)
        self.r_constraints = torch.distributions.Normal(r_loc, r_std)

        lam_loc = torch.tensor(config.CALIBRATED.lam)
        lam_std = torch.tensor(config.CALIBRATED_ERROR.lam)
        self.lam_constraints = torch.distributions.Normal(lam_loc, lam_std)
        self.constraints_distrib = {'r': self.r_constraints,
                                    'lam': self.lam_constraints,
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
                [MU, R_DIST, B_RATE]
        """
        poisson = torch.distributions.Poisson(input)
        nll = - torch.sum(poisson.log_prob(target)) + self.constraints_nll(params)
        param_list = params.values()
        h = hessian(nll, param_list, create_graph=True)
        h_inverse = torch.inverse(h)  # FIXME : may break, handle exception
        loss = h_inverse[0,0]
        return loss
