# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy import stats

SEED = 42

def assert_clean_rescale(rescale):
    assert rescale > 0, f"rescale should be > 0  {rescale} found"

def assert_clean_mix(mix):
    assert mix > 0 and mix < 1, f"mix is a mixture coef it should be in ]0, 1[  {mix} found"


class Generator():
    def __init__(self, seed=None, gamma_k=2, gamma_loc=0, normal_mean=5, normal_sigma=0.5):
        self.seed = seed
        self.gamma_k = gamma_k
        self.gamma_loc = gamma_loc
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma

    def sample_event(self, rescale, mix, size=1):
        assert_clean_rescale(rescale)
        assert_clean_mix(mix)
        n_sig = int(mix * size)
        n_bkg = size - n_sig
        x = self._generate_vars(rescale, mix, n_bkg, n_sig)
        labels = self._generate_labels(n_bkg, n_sig)
        return x, labels

    def generate(self, rescale, mix, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
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
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = rescale
        normal_mean  = self.normal_mean * rescale
        normal_sigma = self.normal_sigma * rescale
        x_b = stats.gamma.rvs(gamma_k, loc=gamma_loc, scale=gamma_scale, size=n_bkg, random_state=self.seed)
        x_s = stats.norm.rvs(loc=normal_mean, scale=normal_sigma, size=n_sig, random_state=self.seed)
        x = np.concatenate([x_b, x_s], axis=0)
        return x

    def _generate_labels(self, n_bkg, n_sig):
        y_b = np.zeros(n_bkg)
        y_s = np.ones(n_sig)
        y = np.concatenate([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mix, n_bkg, n_sig, n_expected_events):
        w_b = np.ones(n_bkg) * (1-mix) * n_expected_events/n_bkg
        w_s = np.ones(n_sig) * mix * n_expected_events/n_sig
        w = np.concatenate([w_b, w_s], axis=0)
        return w

    def proba_density(self, x, rescale, mix):
        """
        Computes p(x | rescale, mix)
        """
        # assert_clean_rescale(rescale)
        # assert_clean_mix(mix)
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = rescale
        normal_mean  = self.normal_mean * rescale
        normal_sigma = self.normal_sigma * rescale
        proba_gamma  = stats.gamma.pdf(x, gamma_k, loc=gamma_loc, scale=gamma_scale)
        proba_normal  = stats.norm.pdf(x, loc=normal_mean, scale=normal_sigma)
        proba_density = mix * proba_normal + (1-mix) * proba_gamma
        return proba_density

    def log_proba_density(self, x, rescale, mix):
        """
        Computes log p(x | rescale, mix)
        """
        proba_density = self.proba_density(x, rescale, mix)
        logproba_density = np.log(proba_density)
        return logproba_density

    def nll(self, data, rescale, mix):
        """
        Computes the negative log likelihood of the data given y and rescale.
        """
        nll = - self.log_proba_density(data, rescale, mix).sum()
        return nll

