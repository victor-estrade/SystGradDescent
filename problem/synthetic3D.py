# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

SEED = 42


class Synthetic3D():
    """
    $$
    f_b (x|r, \lambda) = \mathcal N \left ( (x_0, x_1) | (2+r, 0) 
    \begin{bmatrix} 5 & 0 \\ 0 & 9 \end{bmatrix} \right ) Exp((x_2| \lambda)
    $$


    $$
    f_s (x|r, \lambda) = \mathcal N \left ( (x_0, x_1) | (1, 1) 
    \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right ) Exp((x_2| 2)
    $$

    $$
    p(x | r, \lambda, \mu ) = (1-\mu) f_b(x|r, \lambda) + \mu f_s(x|r, \lambda)
    $$
    """
    N_BKG = 1000
    N_SIG = 50
    def __init__(self, seed=SEED, n_expected_events=1050):
        self.train = Synthetic3DGenerator(seed)
        self.test = Synthetic3DGenerator(seed+1)
        self.n_expected_events = n_expected_events
    
    def train_sample(self, r, lam, mu, n_samples=1000):
        n_bkg = n_samples//2
        n_sig = n_samples - n_bkg
        data = self.train.generate(r, lam, mu, n_bkg=n_bkg, n_sig=n_sig, 
                                n_expected_events=self.n_expected_events)
        return data

    def test_sample(self, r, lam, mu):
        data = self.test.generate(r, lam, mu, n_bkg=self.N_BKG, n_sig=self.N_SIG,
                                n_expected_events=self.n_expected_events)
        return data


class Synthetic3DGenerator():
    def __init__(self, seed=SEED):
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)

    def generate(self, r, lam, mu, n_bkg=1000, n_sig=50, n_expected_events=1050):
        """
        $$
        f_b (x|r, \lambda) = \mathcal N \left ( (x_0, x_1) | (2+r, 0) 
        \begin{bmatrix} 5 & 0 \\ 0 & 9 \end{bmatrix} \right ) Exp((x_2| \lambda)
        $$


        $$
        f_s (x|r, \lambda) = \mathcal N \left ( (x_0, x_1) | (1, 1) 
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right ) Exp((x_2| 2)
        $$

        $$
        p(x | r, \lambda, \mu ) = (1-\mu) f_b(x|r, \lambda) + \mu f_s(x|r, \lambda)
        $$
        """
        X = self._generate_vars(r, lam, mu, n_bkg, n_sig)
        y = self._generate_labels(n_bkg, n_sig)
        w = self._generate_weights(mu, n_bkg, n_sig, n_expected_events)
        data = pd.DataFrame(X, columns=("x1", "x2", "x3"))
        data['label'] = y
        data['weight'] = w
        return data
    
    def _generate_vars(self, r, lam, mu, n_bkg, n_sig):
        bkg_mean = np.array([2.+r, 0.])
        bkg_cov = np.array([[5., 0.], [0., 9.]])
        sig_mean = np.array([0., 0.])
        sig_cov = np.eye(2)
        X_b_12 = self.random.multivariate_normal(bkg_mean, bkg_cov, n_bkg)
        X_s_12 = self.random.multivariate_normal(sig_mean, sig_cov, n_sig)
        X_b_3 =  self.random.exponential(lam, n_bkg).reshape(-1, 1)
        X_s_3 =  self.random.exponential(2, n_sig).reshape(-1, 1)

        X_b = np.concatenate([X_b_12, X_b_3], axis=1)
        X_s = np.concatenate([X_s_12, X_s_3], axis=1)
        X = np.concatenate([X_b, X_s], axis=0)
        return X
        
    def _generate_labels(self, n_bkg, n_sig):
        y_b = np.zeros(n_bkg)
        y_s = np.ones(n_sig)
        y = np.concatenate([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mu, n_bkg, n_sig, n_expected_events):
        w_b = np.ones(n_bkg) * (1-mu) * n_expected_events/n_bkg
        w_s = np.ones(n_sig) * mu * n_expected_events/n_sig
        w = np.concatenate([w_b, w_s], axis=0)
        return w

