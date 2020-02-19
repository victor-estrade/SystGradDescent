import torch
import numpy as np
from collections import OrderedDict

import torch.nn as nn

from hessian import hessian

SEED = 42

class Synthetic3DGeneratoTorch():
    def __init__(self, seed=SEED, r_dist=2.0, b_rate=3.0, s_rate=2.0, ratio=50/(1000+50),
                        reset_every=None):
        self.seed = seed
        self.R_DIST = torch.tensor(r_dist, requires_grad=True)
        self.B_RATE = torch.tensor(b_rate, requires_grad=True)
        self.s_b_ratio = torch.tensor(ratio, requires_grad=True)
        self.nuisance_params = OrderedDict([
                                ('r_dist', self.R_DIST), 
                                ('b_rate', self.B_RATE), 
                                ])
        self.b_loc = torch.cat([self.R_DIST.view(-1), torch.zeros(1)])
        self.b_cov = torch.from_numpy(np.array([[5., 0.], [0., 9.]], dtype=np.float32))
        self.b_01 = torch.distributions.MultivariateNormal(loc=self.b_loc, covariance_matrix=self.b_cov)
        self.b_2 = torch.distributions.Exponential(self.B_RATE)
        
        self.S_RATE = s_rate
        self.s_loc =  torch.zeros(2)
        self.s_cov = torch.eye(2)
        self.s_01 = torch.distributions.MultivariateNormal(loc=self.s_loc, covariance_matrix=self.s_cov)
        self.s_2 = torch.distributions.Exponential(self.S_RATE)
        
        self.n_generated = 0
        self.reset_every = reset_every

    def gen_bkg(self, n_samples):
        a = self.b_01.rsample((n_samples,))
        b = self.b_2.rsample((n_samples,)).type(a.type())
        return torch.cat((a, b.view(-1, 1)), dim=1)
    
    def gen_sig(self, n_samples):
        a = self.s_01.rsample((n_samples,))
        b = self.s_2.rsample((n_samples,)).type(a.type())
        return torch.cat((a, b.view(-1, 1)), dim=1)

    def generate(self, n_samples):
        if self.reset_every is not None and self.reset_every < self.n_generated:
            self.reset()
            self.n_generated = 0
        s = self.gen_sig(int(n_samples*self.s_b_ratio))
        b = self.gen_bkg(int(n_samples*(1-self.s_b_ratio)))
        self.n_generated += n_samples
        return s, b

    def reset(self):
        torch.manual_seed(self.seed)

    def __call__(self, n_samples):
        return self.generate(n_samples)


class S3DLoss(nn.Module):
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

