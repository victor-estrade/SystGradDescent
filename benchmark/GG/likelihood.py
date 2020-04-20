#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging
import itertools

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

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import Parameter

from visual.special.gamma_gauss import plot_nll_around_min


# SEED = None
BENCHMARK_NAME = "GG"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Likelihood")
N_ITER = 9

def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = GGConfig()
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

    config = GGConfig()
    DATA_N_SAMPLES = 80_000

    seed = SEED + i_cv * 5
    generator  = Generator(seed+2)  # test_generator

    result_row = {'i_cv': i_cv}
    result_table = []
    for i, (true_rescale, true_mix) in enumerate(itertools.product(*config.RANGE)):
        iter_directory = os.path.join(directory, f'iter_{i}')
        os.makedirs(iter_directory, exist_ok=True)
        result_row['i'] = i
        true_params = Parameter(true_rescale, true_mix)
        logger.info(f"True Parameters   = {true_params}")
        suffix = f'-mix={true_params.mix:1.2f}_rescale={true_params.rescale}'
        data, label = generator.sample_event(*true_params, size=DATA_N_SAMPLES)
        n_sig = np.sum(label==1)
        n_bkg = np.sum(label==0)
        logger.info(f"nb of signal      = {n_sig}")
        logger.info(f"nb of backgrounds = {n_bkg}")

        compute_nll = lambda rescale, mix : generator.nll(data, rescale, mix)
        plot_nll_around_min(compute_nll, true_params, iter_directory, suffix)

        logger.info('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
        result_row.update(evaluate_minuit(minimizer, true_params))

        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(directory, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Likelihood fit', directory=directory)

    return result_table

if __name__ == '__main__':
    main()