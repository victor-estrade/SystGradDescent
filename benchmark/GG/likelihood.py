#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.likelihood

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
from utils.evaluation import evaluate_config
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig
from problem.gamma_gauss import get_minimizer

from visual.special.gamma_gauss import plot_nll_around_min


# SEED = None
DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Likelihood")
N_ITER = 30

def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = GGConfig()
    config_table = evaluate_config(config)
    config_table.to_csv(os.path.join(DIRECTORY, 'config_table.csv'))
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(DIRECTORY, 'estimations.csv'))
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

    config = GGConfig()
    seed = SEED + i_cv * 5
    test_seed = seed + 2

    result_table = [run_iter(i_cv, i, test_config, test_seed, directory) for i, test_config in enumerate(config.iter_test_config_param_only())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(directory, 'estimations.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Likelihood fit', directory=directory)

    return result_table


def run_iter(i_cv, i_iter, config, seed, directory):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    result_row = dict(i_cv=i_cv, i=i_iter)
    iter_directory = os.path.join(directory, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)

    logger.info(f"True Parameters   = {config.TRUE}")
    suffix = f'-mu={config.TRUE.mu:1.2f}_rescale={config.TRUE.rescale}'
    generator  = Generator(seed)  # test_generator
    data, label = generator.sample_event(*config.TRUE, size=config.N_TESTING_SAMPLES)
    result_row['n_test_samples'] = config.N_TESTING_SAMPLES
    debug_label(label)

    compute_nll = lambda rescale, mu : generator.nll(data, rescale, mu)
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)

    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    minimizer.precision = None
    result_row.update(evaluate_minuit(minimizer, config.TRUE, iter_directory, suffix=suffix))
    return result_row


def debug_label(label):
    logger = logging.getLogger()
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    logger.debug(f"nb of signal      = {n_sig}")
    logger.debug(f"nb of backgrounds = {n_bkg}")

if __name__ == '__main__':
    main()
