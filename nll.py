# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from higgs_4v_pandas import tau_energy_scale

def poisson_nll(n, rate):
    return rate - n * np.log(rate)

def gauss_nll(x, mean, std):
    return np.log(std) + np.square(x - mean) / (2 * np.square(std))


class HiggsNLL():
    def __init__(self, model, X_test, y_test, W_test, X_xp, W_xp, N_BIN=2):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.W_test = W_test
        self.X_xp = X_xp
        self.W_xp = W_xp
        self.N_BIN = N_BIN
    
    def get_s_b(self, tau_es):
        # Systematic effects
        d = self.X_test.copy()
        tau_energy_scale(d, scale=tau_es)
        s = d.loc[self.y_test==1]
        w_s = self.W_test.loc[self.y_test==1]
        b = d.loc[self.y_test==0]
        w_b = self.W_test.loc[self.y_test==0]
        return s, w_s, b, w_b
        
    def __call__(self, mu, tau_es):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""        
        s, w_s, b, w_b = self.get_s_b(tau_es)
        s_histogram = self.model.compute_summaries(s, w_s)
        b_histogram = self.model.compute_summaries(b, w_b)
        xp_histogram = self.model.compute_summaries(self.X_xp, self.W_xp)

        # Compute NLL
        CALIBRATED_TAU_ENERGY_SCALE = 1
        CALIBRATED_TAU_ENERGY_SCALE_ERROR = 0.05
        EPSILON = 1e-4  # avoid log(0)
        rate = mu * s_histogram + b_histogram + EPSILON
        nll = np.sum(poisson_nll(xp_histogram, rate))
        constraints = gauss_nll(tau_es, CALIBRATED_TAU_ENERGY_SCALE, CALIBRATED_TAU_ENERGY_SCALE_ERROR)
        return nll + constraints

