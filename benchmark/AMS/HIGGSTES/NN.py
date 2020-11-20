#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.VAR.GG.NN

import os
import logging
from config import SEED
from config import _ERROR
from config import _TRUTH

import numpy as np
import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_classifier
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images

from visual.misc import plot_params

from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_generators_torch
from problem.higgs import GeneratorCPU
from problem.higgs import HiggsNLL as NLLComputer

from model.neural_network import NeuralNetClassifier
from archi.classic import L4 as ARCHI
from ...my_argparser import NET_parse_args

DATA_NAME = 'HIGGSTES'
BENCHMARK_NAME = 'VAR-'+DATA_NAME
N_ITER = 30


def build_model(args, i_cv):
    args.net = ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, NeuralNetClassifier)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = NET_parse_args(main_description="Training launcher for INFERNO on GG benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    os.makedirs(model.results_directory, exist_ok=True)
    # RUN
    logger.info(f'Running runs [{args.start_cv},{args.end_cv}[')
    results = [run(args, i_cv) for i_cv in range(args.start_cv, args.end_cv)]
    results = pd.concat(results, ignore_index=True)
    # EVALUATION
    results.to_csv(os.path.join(model.results_directory, 'threshold.csv'))
    print(results)
    print("DONE !")


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()


    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda)
    train_generator = GeneratorCPU(train_generator)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_classifier(model, train_generator, config.CALIBRATED, config.N_TRAINING_SAMPLES, retrain=args.retrain)

    # MEASUREMENT
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for threshold in np.linspace(0, 1, 50):
            result_row = {'i_cv': i_cv}
            result_row['threshold'] = threshold
            result_row.update(test_config.TRUE.to_dict(prefix='true_'))
            result_row['n_test_samples'] = test_config.N_TESTING_SAMPLES

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

            gamma_array = np.array(result_row['gamma_1'], result_row['gamma_2'])
            beta_array = np.array(result_row['beta_1'], result_row['beta_2'])
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
    print(results)
    return results

def safe_division(numerator, denominator):
    div = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype="float"), where=denominator!=0)
    return div


def compute_fisher(gamma_array, beta_array, mu):
    EPSILON = 1e-7  # Avoid zero division
    fisher_k = np.sum( (gamma_array**2) / (mu * gamma_array + beta_array + EPSILON) )
    return fisher_k


if __name__ == '__main__':
    main()
