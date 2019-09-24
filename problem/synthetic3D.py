# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
from .nll import gauss_nll
from .nll import poisson_nll

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
        self.test  = Synthetic3DGenerator(seed+1)
        self.final = Synthetic3DGenerator(seed+2)
        self.n_expected_events = n_expected_events
    
    def train_sample(self, r, lam, mu, n_samples=1000):
        n_bkg = n_samples//2
        n_sig = n_samples - n_bkg
        data = self.train.generate(r, lam, mu, n_bkg=n_bkg, n_sig=n_sig, 
                                n_expected_events=self.n_expected_events)
        return data

    def test_sample(self, r, lam, mu, reset=True):
        if reset:
            self.test.reset()
        data = self.test.generate(r, lam, mu, n_bkg=self.N_BKG, n_sig=self.N_SIG,
                                n_expected_events=self.n_expected_events)
        return data

    def final_sample(self, r, lam, mu, reset=True):
        if reset:
            self.final.reset()
        data = self.final.generate(r, lam, mu, n_bkg=self.N_BKG, n_sig=self.N_SIG,
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
        return X, y, w
    
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


def split_data_label_weights(data):
    X = data.drop(['label', 'weight'], axis=1)
    y = data['label']
    w = data['weight']
    return X, y, w


class Config():
    CALIBRATED_MU = 50/1050
    CALIBRATED_R = 0.0
    CALIBRATED_LAMBDA = 3.0

    CALIBRATED_MU_ERROR = 1.0  # minuit default
    CALIBRATED_R_ERROR = 0.4
    CALIBRATED_LAMBDA_ERROR = 1.0
    
    TRUE_MU = 50/1050
    TRUE_R = 0.0
    TRUE_LAMBDA = 3.0

    N_SIG = 5000
    N_BKG = 20000
    N_TRAINING_SAMPLES = 30000


class Synthetic3DNLL():
    def __init__(self, summary_computer, generator, X_final, w_final):
        self.summary_computer = summary_computer
        self.generator = generator
        self.X_final = X_final
        self.w_final = w_final

    def simulation(self, r, lam, mu):
        # Systematic effects
        test_data = self.generator.test_sample(r, lam, mu)
        X_test, y_test, w_test = split_data_label_weights(test_data)
        X_sig = X_test.loc[y_test==1]
        w_sig = w_test.loc[y_test==1]
        X_bkg = X_test.loc[y_test==0]
        w_bkg = w_test.loc[y_test==0]
        return X_sig, w_sig, X_bkg, w_bkg
        
    def __call__(self, r, lam, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""        
        X_sig, w_sig, X_bkg, w_bkg = self.simulation(r, lam, mu)
        s_histogram = self.summary_computer(X_sig, w_sig)
        b_histogram = self.summary_computer(X_bkg, w_bkg)
        n_histogram = self.summary_computer(self.X_final, self.w_final)

        # Compute NLL
        rate = s_histogram + b_histogram
        data_nll = np.sum(poisson_nll(n_histogram, rate))
        r_constraint = gauss_nll(r, 0, 0.4)
        lam_constraint = gauss_nll(lam, 3, 1.0)
        total_nll = data_nll + r_constraint + lam_constraint
        return total_nll

