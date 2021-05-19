# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from scipy import stats

SEED = 42


class Generator():
    def __init__(self, apple_center=1., apple_std=1.,
                    pear_center=2.5, pear_std=1.,
                    n_pears=500, n_apple=200, seed=SEED):
        self.apple_center = apple_center
        self.apple_std    = apple_std
        self.pear_center  = pear_center
        self.pear_std     = pear_std
        self.n_apple      = n_apple
        self.n_pears      = n_pears
        self.seed   = seed
        self.random = np.random.RandomState(seed=seed)

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)

    def generate(self, rescale, mu, n_samples=1000):
        n_apple = n_samples // 2
        n_pear = n_samples // 2
        X, y, w = self._generate(rescale, mu, n_apple=n_apple, n_pear=n_pear)
        return X, y, w

    def _generate(self, rescale, mu, n_apple=50, n_pear=50):
        x = self._generate_vars(rescale, n_apple, n_pear)
        y = self._generate_labels(n_apple, n_pear)
        w = self._generate_weights(mu, n_apple, n_pear)
        return x, y, w

    def _generate_vars(self, rescale, n_apple, n_pear):
        x_apple = self.random.normal(self.apple_center * rescale, self.apple_std * rescale, size=n_apple)
        x_pear  = self.random.normal(self.pear_center * rescale, self.pear_std * rescale, size=n_pear)
        x = np.concatenate([x_apple, x_pear]).reshape(-1, 1)
        return x

    def _generate_labels(self, n_apple, n_pear):
        y_apple = np.ones(n_apple)
        y_pear = np.zeros(n_pear)
        y = np.concatenate([y_apple, y_pear])
        return y

    def _generate_weights(self, mu, n_apple, n_pear):
        w_apple = np.ones(n_apple) * mu * self.n_apple / n_apple
        w_pear = np.ones(n_pear) * self.n_pears / n_pear
        w = np.concatenate([w_apple, w_pear], axis=0)
        return w


    def proba_density(self, x, rescale, mu):
        """
        Computes p(x | rescale, mu)
        """
        apple_center  = self.apple_center * rescale
        apple_std = self.apple_std * rescale
        proba_apple  = stats.norm.pdf(x, loc=apple_center, scale=apple_std)

        pear_center  = self.pear_center * rescale
        pear_std = self.pear_std * rescale
        proba_pears  = stats.norm.pdf(x, loc=pear_center, scale=pear_std)

        total_fruits = mu * self.n_apple + self.n_pears
        apple_strength = mu * self.n_apple / total_fruits
        pears_strength = self.n_pears / total_fruits
        proba_density = apple_strength * proba_apple + pears_strength * proba_pears
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
