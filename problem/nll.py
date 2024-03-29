# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import config

import numpy as np

from .higgs.higgs_geant import split_data_label_weights

from .higgs.higgs_4v_pandas import tau_energy_scale
from .higgs.higgs_4v_pandas import jet_energy_scale
from .higgs.higgs_4v_pandas import lep_energy_scale
from .higgs.higgs_4v_pandas import soft_term
from .higgs.higgs_4v_pandas import nasty_background

def poisson_nll(n, rate):
    return rate - n * np.log(rate)

def gauss_nll(x, mean, std):
    return np.log(std) + np.square(x - mean) / (2 * np.square(std))


class HiggsNLL():
    def __init__(self, model, test_data, X_xp, W_xp, N_BIN=2):
        self.model = model
        self.test_data = test_data
        self.X_xp = X_xp
        self.W_xp = W_xp
        self.N_BIN = N_BIN

    def get_s_b(self, tau_es, jet_es, lep_es, sigma_soft, nasty_bkg):
        # Systematic effects
        data = self.test_data.copy()
        tau_energy_scale(data, scale=tau_es)
        jet_energy_scale(data, scale=jet_es)
        lep_energy_scale(data, scale=lep_es)
        soft_term(data, sigma_soft)
        nasty_background(data, nasty_bkg)
        X_test, y_test, W_test = split_data_label_weights(data)
        s = X_test.loc[y_test==1]
        w_s = W_test.loc[y_test==1]
        b = X_test.loc[y_test==0]
        w_b = W_test.loc[y_test==0]
        return s, w_s, b, w_b

    def __call__(self, mu, tau_es, jet_es, lep_es, sigma_soft, nasty_bkg):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""
        s, w_s, b, w_b = self.get_s_b(tau_es, jet_es, lep_es, sigma_soft, nasty_bkg)
        s_histogram = self.model.compute_summaries(s, w_s)
        b_histogram = self.model.compute_summaries(b, w_b)
        xp_histogram = self.model.compute_summaries(self.X_xp, self.W_xp)

        # Compute NLL
        EPSILON = 1e-4  # avoid log(0)
        rate = mu * s_histogram + b_histogram + EPSILON
        mu_nll = np.sum(poisson_nll(xp_histogram, rate))
        tau_es_constraint = gauss_nll(tau_es, config.CALIBRATED_TAU_ENERGY_SCALE, config.CALIBRATED_TAU_ENERGY_SCALE_ERROR)
        jet_es_constraint = gauss_nll(jet_es, config.CALIBRATED_JET_ENERGY_SCALE, config.CALIBRATED_JET_ENERGY_SCALE_ERROR)
        lep_es_constraint = gauss_nll(lep_es, config.CALIBRATED_LEP_ENERGY_SCALE, config.CALIBRATED_LEP_ENERGY_SCALE_ERROR)
        sigma_soft_constraint = gauss_nll(sigma_soft, config.CALIBRATED_SIGMA_SOFT, config.CALIBRATED_SIGMA_SOFT_ERROR)
        nasty_bkg_constraint = gauss_nll(nasty_bkg, config.CALIBRATED_NASTY_BKG, config.CALIBRATED_NASTY_BKG_ERROR)
        total_nll = mu_nll + tau_es_constraint + jet_es_constraint + lep_es_constraint + sigma_soft_constraint + nasty_bkg_constraint
        return total_nll
