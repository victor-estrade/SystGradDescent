# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from ..nll import gauss_nll
from ..nll import poisson_nll
from .config import GGConfig


class GGNLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        
    def __call__(self, alpha, mix):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$
        """
        pb_config = GGConfig()
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(alpha, mix, n_samples=pb_config.N_VALIDATION_SAMPLES)
        valid_summaries = self.compute_summaries(X, w)
        test_summaries = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        rate = valid_summaries
        data_nll = np.sum(poisson_nll(test_summaries, rate))
        alpha_constraint = gauss_nll(alpha, pb_config.CALIBRATED.alpha, pb_config.CALIBRATED_ERROR.alpha)
        total_nll = data_nll + alpha_constraint
        return total_nll

