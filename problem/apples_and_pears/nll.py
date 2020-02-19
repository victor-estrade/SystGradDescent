# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ..nll import poisson_nll
from .config import AP1Config

class AP1NLL():
    def __init__(self, compute_summaries, valid_generator, X_exp, w_exp):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_exp = X_exp
        self.w_exp = w_exp
    
    def __call__(self, apple_ratio):
        pb_config = AP1Config()
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(apple_ratio, n_samples=pb_config.N_VALIDATION_SAMPLES)
        valid_summaries = self.compute_summaries(X, w)
        test_summaries  = self.compute_summaries(self.X_exp, self.w_exp)

        # Compute NLL
        EPSILON = 1e-6  # avoid log(0)
        rate = valid_summaries# + EPSILON
        total_nll = np.sum(poisson_nll(test_summaries, rate))
        return total_nll
