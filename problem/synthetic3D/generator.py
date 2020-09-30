# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as sts

SEED = 42


class S3D2():
    def __init__(self, seed, background_luminosity=1000, 
                              signal_luminosity=50):
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.sig_rate =  2
        self.feature_names = ["x1", "x2", "x3"]
        self.background_luminosity = background_luminosity
        self.signal_luminosity = signal_luminosity

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)

    def generate(self, r, lam, mu, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
        X, y, w = self._generate(r, lam, mu, n_bkg=n_bkg, n_sig=n_sig)
        return X, y, w

    def _generate(self, r, lam, mu, n_bkg=1000, n_sig=50):
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
        w = self._generate_weights(mu, n_bkg, n_sig)
        return X, y, w
    
    def _generate_vars(self, r, lam, mu, n_bkg, n_sig):
        bkg_mean = self.get_bkg_mean(r)
        bkg_cov  = self.get_bkg_cov()
        sig_mean = self.get_sig_mean()
        sig_cov  = self.get_sig_cov()
        X_b_12 = self.random.multivariate_normal(bkg_mean, bkg_cov, n_bkg)
        X_s_12 = self.random.multivariate_normal(sig_mean, sig_cov, n_sig)
        X_b_3  = self.random.exponential(scale=1./lam, size=n_bkg).reshape(-1, 1)
        X_s_3  = self.random.exponential(scale=1./self.sig_rate, size=n_sig).reshape(-1, 1)

        X_b = np.concatenate([X_b_12, X_b_3], axis=1)
        X_s = np.concatenate([X_s_12, X_s_3], axis=1)
        X = np.concatenate([X_b, X_s], axis=0)
        return X
        
    def _generate_labels(self, n_bkg, n_sig):
        y_b = np.zeros(n_bkg)
        y_s = np.ones(n_sig)
        y = np.concatenate([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mu, n_bkg, n_sig):
        w_b = np.ones(n_bkg) * self.background_luminosity / n_bkg
        w_s = np.ones(n_sig) * mu * self.signal_luminosity / n_sig
        w = np.concatenate([w_b, w_s], axis=0)
        return w

    def get_bkg_mean(self, r):
        return np.array([2.+r, 0.])

    def get_bkg_cov(self):
        return np.array([[5., 0.], [0., 9.]])

    def get_sig_mean(self):
        return np.array([0., 0.])

    def get_sig_cov(self):
        return np.eye(2)

    def sample_event(self, r, lam, mu, size=1):
        assert mu > 0, 'mu should be in ]0, +inf[ : {} found'.format(mu)
        n_sig = int(mu * size)
        n_bkg = size - n_sig
        X = self._generate_vars(r, lam, mu, n_bkg, n_sig)
        y = self._generate_labels(n_bkg, n_sig)
        return X, y


    def proba_density(self, x, r, lam, mu):
        bkg_mean = self.get_bkg_mean(r)
        bkg_cov  = self.get_bkg_cov()
        sig_mean = self.get_sig_mean()
        sig_cov  = self.get_sig_cov()
        x_1_2 = x[:, :2]
        x_3 = x[:, -1]
        p_bkg =  sts.multivariate_normal.pdf(x_1_2, bkg_mean, bkg_cov)
        p_sig =  sts.multivariate_normal.pdf(x_1_2, sig_mean, sig_cov)
        p_bkg = p_bkg * sts.expon.pdf(x_3, loc=0., scale=1./lam)
        p_sig = p_sig * sts.expon.pdf(x_3, loc=0., scale=1./self.sig_rate)
        proba_density = mu * p_sig + (1-mu) * p_bkg
        return proba_density

    def log_proba_density(self, x, r, lam, mu):
        """
        computes log p(x | y, alpha)
        """
        proba_density = self.proba_density(x, r, lam, mu)
        log_proba_density = np.log(proba_density)
        return log_proba_density

    def nll(self, x, r, lam, mu):
        """
        Computes the negative log likelihood of teh data given y and alpha.
        """
        nll = - self.log_proba_density(x, r, lam, mu).sum()
        return nll

Generator = S3D2 # alias
