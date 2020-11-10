#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.Likelihood_histogram-Prior

import os
import logging

import numpy as np
import pandas as pd

from config import SAVING_DIR
from config import SEED

from visual import set_plot_config
set_plot_config()
from visual.misc import plot_params

from utils.log import set_logger
from utils.log import print_line
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.summaries import HistogramSummaryComputer


# SEED = None
DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Likelihood_histogram")
N_ITER = 30

def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = Config()
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.TRUE.interest_parameters_names, results)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(DIRECTORY, 'evaluation.csv'))
    gather_images(DIRECTORY)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
    directory = os.path.join(DIRECTORY, f'cv_{i_cv}')
    os.makedirs(directory, exist_ok=True)

    config = Config()
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    N_BINS = 10
    X_train, y_train, w_train = train_generator.generate(*config.CALIBRATED, n_samples=config.N_TRAINING_SAMPLES)
    compute_summaries = HistogramSummaryComputer(n_bins=N_BINS).fit(X_train)

    result_table = [run_iter(compute_summaries, i_cv, i, test_config, valid_generator, test_generator, directory) for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(directory, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Likelihood fit', directory=directory)

    return result_table


def run_iter(compute_summaries, i_cv, i_iter, config, valid_generator, test_generator, directory):
    logger = logging.getLogger()
    result_row = dict(i_cv=i_cv, i=i_iter)
    iter_directory = os.path.join(directory, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)

    logger.info(f"True Parameters   = {config.TRUE}")
    suffix = f'-mix={config.TRUE.mix:1.2f}_rescale={config.TRUE.rescale}'
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    debug_label(y_test)

    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)

    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    result_row.update(evaluate_minuit(minimizer, config.TRUE))
    return result_row


def debug_label(label):
    logger = logging.getLogger()
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    logger.debug(f"nb of signal      = {n_sig}")
    logger.debug(f"nb of backgrounds = {n_bkg}")

if __name__ == '__main__':
    main()
