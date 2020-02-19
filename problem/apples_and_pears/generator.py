# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

SEED = 42


class ApplePear():
    def __init__(self, apple_center=1., apple_std=1., 
                    pear_center=2.5, pear_std=1., 
                    n_total=500, seed=SEED):
        self.apple_center = apple_center
        self.apple_std    = apple_std
        self.pear_center  = pear_center
        self.pear_std     = pear_std
        self.n_total      = n_total 
        self.seed   = seed
        self.random = np.random.RandomState(seed=seed)

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)
        
    def generate(self, apple_ratio=0.5, n_apple=50, n_pear=50):
        x = self._generate_vars(n_apple, n_pear)
        y = self._generate_labels(n_apple, n_pear)
        w = self._generate_weights(apple_ratio, n_apple, n_pear)
        return x, y, w

    def _generate_vars(self, n_apple, n_pear):
        x_apple = self.random.normal(self.apple_center, self.apple_std, size=n_apple)
        x_pear  = self.random.normal(self.pear_center, self.pear_std, size=n_pear)
        x = np.concatenate([x_apple, x_pear]).reshape(-1, 1)
        return x

    def _generate_labels(self, n_apple, n_pear):
        y_apple = np.ones(n_apple)
        y_pear = np.zeros(n_pear)
        y = np.concatenate([y_apple, y_pear])
        return y

    def _generate_weights(self, apple_ratio, n_apple, n_pear):
        w_apple = np.ones(n_apple) * apple_ratio * self.n_total / n_apple
        w_pear = np.ones(n_pear) * (1 - apple_ratio) * self.n_total / n_pear
        w = np.concatenate([w_apple, w_pear], axis=0)
        return w


class AP1(ApplePear):
    def __init__(self, seed):
        apple_center = 1.
        apple_std    = 1.
        pear_center  = 2.5
        pear_std     = 1.
        n_total      = 500
        super().__init__(seed=seed, apple_center=apple_center, apple_std=apple_std, 
                    pear_center=pear_center, pear_std=pear_std, n_total=n_total)

    def generate(self, apple_ratio, n_samples=1000):
        n_apple = n_samples // 2
        n_pear = n_samples // 2
        X, y, w = super().generate(apple_ratio=apple_ratio, n_apple=n_apple, n_pear=n_pear)
        return X, y, w

