# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ..nll import gauss_nll
from ..nll import poisson_nll
from .config import S3D2Config


class S3D2NLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config=None):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        self.config = S3D2Config() if config is None else config
        
    def __call__(self, r, lam, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""
        config = self.config
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(r, lam, mu, n_samples=config.N_VALIDATION_SAMPLES)
        EPSILON = 1e-5  # avoid log(0)
        valid_summaries = self.compute_summaries(X, w) + EPSILON
        test_summaries = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        rate = valid_summaries
        data_nll = np.sum(poisson_nll(test_summaries, rate))
        r_constraint = gauss_nll(r, config.CALIBRATED_R, config.CALIBRATED_R_ERROR)
        lam_constraint = gauss_nll(lam, config.CALIBRATED_LAMBDA, config.CALIBRATED_LAMBDA_ERROR)
        total_nll = data_nll + r_constraint + lam_constraint
        return total_nll

