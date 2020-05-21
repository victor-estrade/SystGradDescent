# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from .config import HiggsConfig

import numpy as np


def poisson_nll(n, rate):
    return rate - n * np.log(rate)

def gauss_nll(x, mean, std):
    return np.log(std) + np.square(x - mean) / (2 * np.square(std))


class HiggsNLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config=None):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        self.config = HiggsConfig() if config is None else config


    def get_s_b(self, tes, jes, les, nasty_bkg, sigma_soft, mu):
        # Systematic effects
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(mu, tes, jes, les, sigma_soft, nasty_bkg, n_samples=None)
        s = X[y==1]
        w_s = w[y==1]
        b = X[y==0]
        w_b = w[y==0]
        return s, w_s, b, w_b
        
    def __call__(self, tes, jes, les, nasty_bkg, sigma_soft, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""        
        s, w_s, b, w_b = self.get_s_b(tes, jes, les, nasty_bkg, sigma_soft, mu)
        s_histogram = self.compute_summaries(s, w_s)
        b_histogram = self.compute_summaries(b, w_b)
        xp_histogram = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        config = self.config
        EPSILON = 1e-5  # avoid log(0)
        rate = s_histogram + b_histogram + EPSILON
        mu_nll = np.sum(poisson_nll(xp_histogram, rate))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        nasty_bkg_constraint = gauss_nll(nasty_bkg, config.CALIBRATED.nasty_bkg, config.CALIBRATED_ERROR.nasty_bkg)
        if sigma_soft is not None:
            sigma_soft_constraint = gauss_nll(sigma_soft, config.CALIBRATED.sigma_soft, config.CALIBRATED_ERROR.sigma_soft)
        else:
            sigma_soft_constraint = 0
        total_nll = mu_nll + tes_constraint + jes_constraint + les_constraint + sigma_soft_constraint + nasty_bkg_constraint
        return total_nll



