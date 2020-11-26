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


    # MEASUREMENT
def measurement(model, i_cv, config, valid_generator, test_generator):
    logger = logging.getLogger()
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for threshold in np.linspace(0, 1, 50):
            result_row = {'i_cv': i_cv}
            result_row['threshold'] = threshold
            result_row.update(test_config.TRUE.to_dict(prefix='true_'))
            result_row['n_test_samples'] = test_generator.n_samples

            X, y, w = valid_generator.generate(*test_config.TRUE, n_samples=config.N_VALIDATION_SAMPLES)
            proba = model.predict_proba(X)
            decision = proba[:, 1]
            bin_2 = decision > threshold
            bin_1 = np.logical_not(bin_2)

            result_row['beta_2'] = np.sum( w [ np.logical_and(bin_2, y == 0) ] )
            result_row['gamma_2'] = np.sum( w [ np.logical_and(bin_2, y == 1) ] )
            result_row['beta_1'] = np.sum( w [ np.logical_and(bin_1, y == 0) ] )
            result_row['gamma_1'] = np.sum( w [ np.logical_and(bin_1, y == 1) ] )
            result_row['beta_0'] = np.sum( w [ y == 0 ] )
            result_row['gamma_0'] = np.sum( w [ y == 1 ] )

            gamma_array = np.array(result_row['gamma_0'])
            beta_array = np.array(result_row['beta_0'])
            result_row['fisher_1'] = compute_fisher(gamma_array, beta_array, test_config.TRUE.mu)

            gamma_array = np.array([result_row['gamma_1'], result_row['gamma_2']])
            beta_array = np.array([result_row['beta_1'], result_row['beta_2']])
            result_row['fisher_2'] = compute_fisher(gamma_array, beta_array, test_config.TRUE.mu)

            result_row['g_sqrt_b_g'] = safe_division( result_row['gamma_2'], np.sqrt(result_row['gamma_2'] + result_row['beta_2']) )
            result_row['g_sqrt_b'] = safe_division( result_row['gamma_2'], np.sqrt(result_row['beta_2']) )

            # On TEST SET
            X, y, w = test_generator.generate(*test_config.TRUE, n_samples=config.N_VALIDATION_SAMPLES)
            proba = model.predict_proba(X)
            decision = proba[:, 1]
            bin_2 = decision > threshold
            n_bin_2 = np.sum(bin_2)
            bin_1 = np.logical_not(bin_2)

            result_row['b_2'] = np.sum( w [ np.logical_and(bin_2, y == 0) ] )
            result_row['s_2'] = np.sum( w [ np.logical_and(bin_2, y == 1) ] )
            result_row['b_1'] = np.sum( w [ np.logical_and(bin_1, y == 0) ] )
            result_row['s_1'] = np.sum( w [ np.logical_and(bin_1, y == 1) ] )
            result_row['b_0'] = np.sum( w [ y == 0 ] )
            result_row['s_0'] = np.sum( w [ y == 1 ] )

            result_row['s_sqrt_n'] = safe_division( result_row['s_2'], np.sqrt(result_row['s_2'] + result_row['b_2']) )
            result_row['s_sqrt_b'] = safe_division( result_row['s_2'], np.sqrt(result_row['b_2']) )

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
