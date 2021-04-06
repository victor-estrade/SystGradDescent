# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np


def poisson_nll(n, rate):
    return rate - n * np.log(rate)

def gauss_nll(x, mean, std):
    return np.log(std) + np.square(x - mean) / (2 * np.square(std))


class LabelNLL():
    def __init__(self, valid_generator, y_test, w_test, config=None):
        self.valid_generator = valid_generator
        self.y_test = y_test
        self.w_test = w_test
        self.config = HiggsConfig() if config is None else config

    def __call__(self, tes, jes, les, mu):
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, jes, les, mu, n_samples=None)
        # s = w[y==1].sum()
        # b = w[y==0].sum()
        # rate = s + b #+ EPSILON

        # EPSILON = 1e-5  # avoid log(0)
        rate = w.sum()
        n = self.w_test.sum()
        mu_nll = np.sum(poisson_nll(n, rate))
        config = self.config
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        total_nll = mu_nll + tes_constraint + jes_constraint + les_constraint
        return total_nll



class MonoHiggsNLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        EPSILON = 1e-6  # avoid log(0)
        self.xp_histogram = self.compute_summaries(self.X_test, self.w_test) + EPSILON
        self.config = config

    def __call__(self, tes, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON
        # xp_histogram = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        total_nll = mu_nll + tes_constraint
        return total_nll



class BaseHiggsNLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        EPSILON = 1e-6  # avoid log(0)
        self.xp_histogram = self.compute_summaries(self.X_test, self.w_test) + EPSILON
        self.config = config



class HiggsNLLTes(BaseHiggsNLL):
    def __call__(self, tes, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        total_nll = mu_nll + tes_constraint
        return total_nll


class HiggsNLLJes(BaseHiggsNLL):
    def __call__(self, jes, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(jes, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        total_nll = mu_nll + jes_constraint
        return total_nll


class HiggsNLLLes(BaseHiggsNLL):
    def __call__(self, les, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(les, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        total_nll = mu_nll + les_constraint
        return total_nll


class HiggsNLLTesJes(BaseHiggsNLL):
    def __call__(self, tes, jes, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, jes, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        total_nll = mu_nll + tes_constraint + jes_constraint
        return total_nll



class HiggsNLLTesLes(BaseHiggsNLL):
    def __call__(self, tes, les, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, les, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        total_nll = mu_nll + tes_constraint + les_constraint
        return total_nll


class HiggsNLLTesJesLes(BaseHiggsNLL):
    def __call__(self, tes, jes, les, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, jes, les, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        total_nll = mu_nll + tes_constraint + jes_constraint + les_constraint
        return total_nll


# For backward compatibility
class HiggsNLL(BaseHiggsNLL):
    def __call__(self, tes, jes, les, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate) + constraints$
         with $constraints = gaussian nll (\alpha | \alpha_{calibrated), \delta_\alpha )$
         with $rate = \mu s + b$
        """
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, jes, les, mu, n_samples=None, no_grad=True)
        EPSILON = 1e-6  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
        tes_constraint = gauss_nll(tes, config.CALIBRATED.tes, config.CALIBRATED_ERROR.tes)
        jes_constraint = gauss_nll(jes, config.CALIBRATED.jes, config.CALIBRATED_ERROR.jes)
        les_constraint = gauss_nll(les, config.CALIBRATED.les, config.CALIBRATED_ERROR.les)
        total_nll = mu_nll + tes_constraint + jes_constraint + les_constraint
        return total_nll



class FuturHiggsNLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test, config=None):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        EPSILON = 1e-6  # avoid log(0)
        self.xp_histogram = self.compute_summaries(self.X_test, self.w_test) + EPSILON
        self.config = config


    def __call__(self, tes, jes, les, nasty_bkg, sigma_soft, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(tes, jes, les, nasty_bkg, sigma_soft, mu, n_samples=None)
        EPSILON = 1e-5  # avoid log(0)
        rate_histogram = self.compute_summaries(X, w) + EPSILON

        # Compute NLL
        config = self.config
        EPSILON = 1e-5  # avoid log(0)
        mu_nll = np.sum(poisson_nll(self.xp_histogram, rate_histogram))
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


ALL_HIGGSNLL_DICT = {
    'Tes' : HiggsNLLTes,
    'Jes' : HiggsNLLJes,
    'Les' : HiggsNLLLes,
    'TesJes' : HiggsNLLTesJes,
    'TesLes' : HiggsNLLTesLes,
    'TesJesLes' : HiggsNLLTesJesLes,
}


def get_higgsnll_class(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_HIGGSNLL_DICT :
        return ALL_HIGGSNLL_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")
