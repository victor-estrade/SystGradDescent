# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np


class DataAugmenter(object):
    def __init__(self):
        super().__init__()

    def augment(self, X, y, sample_weight=None):
        raise NotImplementedError

    def __call__(self, X, y, sample_weight=None):
        return self.augment(X, y, sample_weight=None)

    def sample_z(self, size):
        raise NotImplementedError


class NormalDataAugmenter(DataAugmenter):
    def __init__(self, skewing_function, width=1, center=0, n_augment=2):
        super().__init__()
        self.skewing_function = skewing_function
        self.width = width
        self.center = center
        self.n_augment = n_augment

    def augment(self, X, y, sample_weight=None):
        z_list = [self.sample_z( size=X.shape[0] ) for _ in range(self.n_augment)]
        X_aug = np.concatenate( [X, ] + [ self.skewing_function(X, z) for z in z_list ], axis=0)
        y_aug = np.concatenate( [y, ] + [y for _ in range(self.n_augment) ], axis=0)
        z_aug = np.concatenate( [np.zeros(X.shape[0]) + self.center, ] + z_list)
        w_aug = None
        if sample_weight is not None:
            w_aug = np.concatenate( [sample_weight, ] + [sample_weight for _ in range(self.n_augment) ], axis=0)
        return X_aug, y_aug, w_aug, z_aug

    def sample_z(self, size):
        z = np.random.normal( loc=self.center, scale=self.width, size=size )
        return z


class NormalDataPerturbator(object):
    def __init__(self, skewing_function, width=1, center=0):
        super().__init__()
        self.skewing_function = skewing_function
        self.width = width
        self.center = center

    def perturb(self, X):
        z = self.sample_z(size=X.shape[0])
        X = self.skewing_function(X, z)
        return X, z

    def __call__(self, X, y, sample_weight=None):
        return self.perturb(X)

    def sample_z(self, size):
        z = np.random.normal( loc=self.center, scale=self.width, size=size )
        return z
