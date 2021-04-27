#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.GB

import os
import logging
from config import SEED

import numpy as np
import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import train_or_load_classifier
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from visual.misc import plot_params

from problem.synthetic3D import S3D2Config as Config
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import Generator
from problem.synthetic3D import S3D2NLL as NLLComputer

from visual.special.synthetic3D import plot_nll_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


DATA_NAME = 'S3D2'
BENCHMARK_NAME = DATA_NAME+'-reboot'
N_ITER = 3

def build_model(args, i_cv):
    model = get_model(args, GradientBoostingModel)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


def poisson_nll(n, rate):
    return rate - n * np.log(rate)

def gauss_nll(x, mean, std):
    return np.log(std) + np.square(x - mean) / (2 * np.square(std))


class NLL():
    def __init__(self, X_test, w_test, i_cv, args, config=None, n_bins=10):
        self.X_test = X_test
        self.w_test = w_test
        self.args = args
        self.i_cv = i_cv
        self.config = Config() if config is None else config
        self.n_bins = n_bins

    def __call__(self, r, lam, mu):
        """
        $\sum_{i=0}^{n_{bin}} rate - n_i \log(rate)$ with $rate = \mu s + b$
        """
        config = self.config
        seed = SEED + self.i_cv * 5
        train_generator = Generator(seed)
        valid_generator = Generator(seed+1)

        classifier = build_model(self.args, self.i_cv)
        X_train, y_train, w_train = train_generator.generate(r, lam, config.CALIBRATED.mu, n_samples=config.N_TRAINING_SAMPLES)
        classifier.fit(X_train, y_train, w_train)

        X, y, w = valid_generator.generate(r, lam, mu, n_samples=config.N_VALIDATION_SAMPLES)
        valid_summaries = classifier.compute_summaries(X, w, n_bins=self.n_bins)
        test_summaries = classifier.compute_summaries(self.X_test, self.w_test, n_bins=self.n_bins)

        # Compute NLL
        EPSILON = 1e-5  # avoid log(0)
        rate = valid_summaries + EPSILON
        data_nll = np.sum(poisson_nll(test_summaries, rate))
        r_constraint = gauss_nll(r, config.CALIBRATED.r, config.CALIBRATED_ERROR.r)
        lam_constraint = gauss_nll(lam, config.CALIBRATED.lam, config.CALIBRATED_ERROR.lam)
        total_nll = data_nll + r_constraint + lam_constraint
        return total_nll


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    config = Config()
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(model.results_directory, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, results)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(model.results_directory, 'evaluation.csv'))
    gather_images(model.results_directory)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()

    result_row = {'i_cv': i_cv}

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    # train_generator = Generator(seed)
    # valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    # logger.info('Set up classifier')
    model = build_model(args, i_cv)
    # flush(logger)

    # TRAINING / LOADING
    # train_or_load_classifier(model, train_generator, config.CALIBRATED, config.N_TRAINING_SAMPLES, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    # X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)

    # result_row.update(evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid'))

    # MEASUREMENT
    N_BINS = 10
    # evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix='')
    result_table = [run_iter(model, result_row, i, i_cv, args, test_config, test_generator, n_bins=N_BINS)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.results_path, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)

    logger.info('DONE')
    return result_table


def run_iter(model, result_row, i_iter, i_cv, args, config, test_generator, n_bins=10):
    logger = logging.getLogger()
    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    suffix = f'-mu={config.TRUE.mu:1.2f}_r={config.TRUE.r}_lambda={config.TRUE.lam}'
    logger.info('Generate testing data')
    test_generator.reset()
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    # PLOT SUMMARIES
    # evaluate_summary_computer(model, X_test, y_test, w_test, n_bins=n_bins, prefix='', suffix=suffix, directory=iter_directory)

    # logger.info('Set up NLL computer')
    # compute_summaries = ClassifierSummaryComputer(model, n_bins=n_bins)
    # compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    compute_nll = NLL(X_test, w_test, i_cv, args, config=config, n_bins=n_bins)
    # NLL PLOTS
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    result_row.update(evaluate_minuit(minimizer, config.TRUE))
    return result_row.copy()

if __name__ == '__main__':
    main()
