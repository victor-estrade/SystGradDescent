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
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config=None):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        self.test_summaries = self.compute_summaries(self.X_test, self.w_test)
        self.config = GGConfig() if config is None else config

    def __call__(self, rescale, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$
        """
        config = self.config
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(rescale, mu, n_samples=config.N_VALIDATION_SAMPLES)
        valid_summaries = self.compute_summaries(X, w)
        # test_summaries = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        EPSILON = 1e-5  # avoid log(0)
        rate = valid_summaries + EPSILON
        data_nll = np.sum(poisson_nll(self.test_summaries, rate))
        rescale_constraint = gauss_nll(rescale, config.CALIBRATED.rescale, config.CALIBRATED_ERROR.rescale)
        total_nll = data_nll + rescale_constraint
        print(f"{rescale}, {mu}, {total_nll}")
        return total_nll
