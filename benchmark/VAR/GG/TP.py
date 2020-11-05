#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.VAR.GG.TP

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
from utils.model import train_or_load_neural_net
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss.torch import GeneratorTorch
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import Generator
from problem.gamma_gauss import param_generator
from problem.gamma_gauss import GGNLL as NLLComputer

from model.tangent_prop import TangentPropClassifier
from archi.classic import L4 as ARCHI
from ...my_argparser import TP_parse_args


DATA_NAME = 'GG'
BENCHMARK_NAME = 'VAR-'+DATA_NAME
N_ITER = 30


class TrainGenerator:
    def __init__(self, data_generator, cuda=False):
        self.data_generator = data_generator
        if cuda:
            self.data_generator.cuda()
        else:
            self.data_generator.cpu()
        self.nuisance_params = self.data_generator.nuisance_params


    def generate(self, n_samples=None):
            X, y, w = self.data_generator.diff_generate(n_samples=n_samples)
            return X, y, w

    def reset(self):
        self.data_generator.reset()

    def tensor(self, data, requires_grad=False, dtype=None):
        return self.data_generator.tensor(data, requires_grad=requires_grad, dtype=dtype)


def build_model(args, i_cv):
    args.net = ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, TangentPropClassifier)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = TP_parse_args(main_description="Training launcher for INFERNO on GG benchmark")
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
    results.to_csv(os.path.join(model.results_directory, 'fisher.csv'))
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
    train_generator = GeneratorTorch(seed, cuda=args.cuda)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    some_fisher = compute_fisher(*compute_bins(model, valid_generator, config, n_bins=3), config.TRUE.mu)
    some_fisher_bis = compute_fisher(*compute_bins(model, valid_generator, config, n_bins=3), config.TRUE.mu)

    assert some_fisher == some_fisher_bis, f"Fisher info should be deterministic but found : {some_fisher} =/= {some_fisher_bis}"

    # MEASUREMENT
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for n_bins in range(1, 30):
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
    print(results)
    return results


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


def compute_fisher(gamma_array, beta_array, mu):
    EPSILON = 1e-7  # Avoid zero division
    fisher_k = np.sum( (gamma_array**2) / (mu * gamma_array + beta_array + EPSILON) )
    return fisher_k



if __name__ == '__main__':
    main()
