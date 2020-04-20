#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

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
from utils.evaluation import estimate
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.synthetic3D import Generator
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import Parameter

from visual.special.synthetic3D import plot_nll_around_min


# SEED = None
BENCHMARK_NAME = "S3D2"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Likelihood")
N_ITER = 9

def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = S3D2Config()
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, results)
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

    config = S3D2Config()

    seed = SEED + i_cv * 5
    test_seed = seed + 2

    result_table = [run_iter(i_cv, i, test_config, test_seed, directory) for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(directory, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Likelihood fit', directory=directory)

    return result_table


def run_iter(i_cv, i_iter, config, seed, directory):
    logger = logging.getLogger()
    result_row = dict(i_cv=i_cv, i=i_iter)
    iter_directory = os.path.join(directory, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)

    logger.info(f"True Parameters   = {config.TRUE}")
    suffix = f'-mu={config.TRUE.mu:1.2f}_r={config.TRUE.r}_lambda={config.TRUE.lam}'
    generator  = Generator(seed)  # test_generator
    data, label = generator.sample_event(*config.TRUE, size=config.N_TESTING_SAMPLES)
    debug_label(label)

    compute_nll = lambda r, lam, mu : generator.nll(data, r, lam, mu)
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