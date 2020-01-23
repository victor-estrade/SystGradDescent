# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
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
    def __init__(self, n_expected_events=1050, seed=SEED):
        self.n_expected_events = n_expected_events
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)

    def generate(self, r, lam, mu, n_bkg=1000, n_sig=50):
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
        w = self._generate_weights(mu, n_bkg, n_sig, self.n_expected_events)
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


class S3D2():
    def __init__(self, seed):
        n_expected_events = 1050
        self.generator = Synthetic3DGenerator(n_expected_events=n_expected_events, seed=seed)

    def reset(self):
        self.generator.reset()

    def generate(self, r, lam, mu, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
        X, y, w = self.generator.generate(r, lam, mu, n_bkg=n_bkg, n_sig=n_sig)
        return X, y, w


def split_data_label_weights(data):
    X = data.drop(['label', 'weight'], axis=1)
    y = data['label']
    w = data['weight']
    return X, y, w


class S3D2Config():
    PARAM_NAMES = ['mu', 'r', 'lam']
    INTEREST_PARAM_NAME = 'mu'

    CALIBRATED_MU = 50/1050
    CALIBRATED_R = 0.0
    CALIBRATED_LAMBDA = 3.0

    CALIBRATED_MU_ERROR = 1.0  # minuit default
    CALIBRATED_R_ERROR = 0.4
    CALIBRATED_LAMBDA_ERROR = 1.0
    
    TRUE_MU = 150/1050
    TRUE_R = 0.1
    TRUE_LAMBDA = 2.7

    TRUE_MU_RANGE = np.arange(0, 0.3, 0.05)
    # TRUE_MU_RANGE = [100/1050, 150/1050, 200/1050]
    TRUE_R_RANGE = np.arange(-0.2, 0.2, 0.1)
    TRUE_LAMBDA_RANGE = np.arange(2.1, 3.5, 0.2)

    N_TRAINING_SAMPLES = 30000
    N_VALIDATION_SAMPLES = 30000
    N_TESTING_SAMPLES = 30000


class S3D2NLL():
    def __init__(self, compute_summaries, valid_generator, X_test, w_test):
        self.compute_summaries = compute_summaries
        self.valid_generator = valid_generator
        self.X_test = X_test
        self.w_test = w_test
        
    def __call__(self, r, lam, mu):
        """$\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$"""        
        pb_config = S3D2Config()
        self.valid_generator.reset()
        X, y, w = self.valid_generator.generate(r, lam, mu, n_samples=pb_config.N_VALIDATION_SAMPLES)
        valid_summaries = self.compute_summaries(X, w)
        test_summaries = self.compute_summaries(self.X_test, self.w_test)

        # Compute NLL
        rate = valid_summaries
        data_nll = np.sum(poisson_nll(test_summaries, rate))
        r_constraint = gauss_nll(r, pb_config.CALIBRATED_R, pb_config.CALIBRATED_R_ERROR)
        lam_constraint = gauss_nll(lam, pb_config.CALIBRATED_LAMBDA, pb_config.CALIBRATED_LAMBDA_ERROR)
        total_nll = data_nll + r_constraint + lam_constraint
        return total_nll

