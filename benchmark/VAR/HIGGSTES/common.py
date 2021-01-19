# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging

import numpy as np
import pandas as pd


class GeneratorCPU:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.n_samples = data_generator.size

    def generate(self, *params, n_samples=None, no_grad=False):
            X, y, w = self.data_generator.generate(*params, n_samples=n_samples, no_grad=no_grad)
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            return X, y, w

    def reset(self):
        self.data_generator.reset()


def measurement(model, i_cv, config, valid_generator, test_generator):
    logger = logging.getLogger()
    some_fisher = compute_fisher(*compute_bins(model, valid_generator, config, n_bins=3), config.TRUE.mu)
    some_fisher_bis = compute_fisher(*compute_bins(model, valid_generator, config, n_bins=3), config.TRUE.mu)

    assert some_fisher == some_fisher_bis, f"Fisher info should be deterministic but found : {some_fisher} =/= {some_fisher_bis}"

    # MEASUREMENT
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for n_bins in range(1, 60):
            result_row = {'i_cv': i_cv}
            gamma_array, beta_array = compute_bins(model, valid_generator, test_config, n_bins=n_bins)
            fisher = compute_fisher(gamma_array, beta_array, test_config.TRUE.mu)
            result_row.update({f'gamma_{i}' : gamma for i, gamma in enumerate(gamma_array, 1)})
            result_row.update({f'beta_{i}' : beta for i, beta in enumerate(beta_array, 1)})
            result_row.update(test_config.TRUE.to_dict(prefix='true_'))
            result_row['n_test_samples'] = test_config.N_TESTING_SAMPLES
            result_row['fisher'] = fisher
            result_row['n_bins'] = n_bins
            results.append(result_row.copy())
    results = pd.DataFrame(results)
    return results


def safe_division(numerator, denominator):
    div = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype="float"), where=denominator!=0)
    return div


def compute_fisher(gamma_array, beta_array, mu):
    EPSILON = 1e-7  # Avoid zero division
    fisher_k = np.sum( (gamma_array**2) / (mu * gamma_array + beta_array + EPSILON) )
    return fisher_k


def compute_bins(model, valid_generator, config, n_bins=10):
    valid_generator.reset()
    X, y, w = valid_generator.generate(*config.TRUE, n_samples=config.N_VALIDATION_SAMPLES)
    X_sig = X[y==1]
    w_sig = w[y==1]
    X_bkg = X[y==0]
    w_bkg = w[y==0]

    gamma_array = model.compute_summaries(X_sig, w_sig, n_bins=n_bins)
    beta_array = model.compute_summaries(X_bkg, w_bkg, n_bins=n_bins)
    return gamma_array, beta_array
