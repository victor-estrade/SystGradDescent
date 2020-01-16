#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.AP1.GB

import os
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5

import pandas as pd
import numpy as np

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.log import print_params
from utils.model import get_model
from utils.model import save_model
from utils.plot import plot_valid_distrib
from utils.plot import plot_summaries
from utils.plot import plot_params
from utils.misc import gather_images
from utils.misc import register_params
from utils.misc import _ERROR
from utils.misc import _TRUTH

from problem.apples_and_pears import AP1
from problem.apples_and_pears import AP1NLL
from problem.apples_and_pears import AP1Config

from utils.AP1 import plot_apple_ratio_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


BENCHMARK_NAME = 'AP1'
N_ITER = 3


def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on AP1 benchmark")
    logger.info(args)
    flush(logger)
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    model = get_model(args, GradientBoostingModel)
    model.set_info(BENCHMARK_NAME, -1)
    pb_config = AP1Config()
    for name in pb_config.PARAM_NAMES:
        values = results[name]
        errors = results[name+_ERROR]
        truths = results[name+_TRUTH]
        values + errors + truths

    gather_images(model.directory)
    results.to_csv(os.path.join(model.directory, 'results.csv'))



def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
    
    result_row = {'i_cv': i_cv}
    result_table = []

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    pb_config = AP1Config()
    seed = config.SEED + i_cv * 5
    train_generator = AP1(seed)
    valid_generator = AP1(seed+1)
    test_generator  = AP1(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = get_model(args, GradientBoostingModel)
    model.set_info(BENCHMARK_NAME, i_cv)
    flush(logger)

    # TRAINING
    logger.info('Generate training data')
    X_train, y_train, w_train = train_generator.generate(
                                    apple_ratio=pb_config.CALIBRATED_APPLE_RATIO,
                                    n_samples=pb_config.N_TRAINING_SAMPLES)
    logger.info('Training {}'.format(model.get_name()))
    model.fit(X_train, y_train, w_train)
    logger.info('Training DONE')

    # SAVE MODEL
    save_model(model)


    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(
                                    apple_ratio=pb_config.CALIBRATED_APPLE_RATIO,
                                    n_samples=pb_config.N_VALIDATION_SAMPLES)

    logger.info('Plot distribution of the score')
    plot_valid_distrib(model, X_valid, y_valid, classes=("pears", "apples"))
    result_row['valid_accuracy'] = model.score(X_valid, y_valid)


    # MEASUREMENT
    n_bins = 10
    compute_summaries = ClassifierSummaryComputer(model, n_bins=n_bins)
    for mu in pb_config.TRUE_APPLE_RATIO_RANGE:
        pb_config.TRUE_APPLE_RATIO = mu
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(
                                        apple_ratio=pb_config.TRUE_APPLE_RATIO,
                                        n_samples=pb_config.N_TESTING_SAMPLES)
        
        logger.info('Set up NLL computer')
        compute_nll = AP1NLL(compute_summaries, valid_generator, X_test, w_test)


        logger.info('Plot summaries')
        extension = '-mu={:1.1f}'.format(pb_config.TRUE_APPLE_RATIO)
        plot_summaries( model, n_bins, extension,
                        X_valid, y_valid, w_valid,
                        X_test, w_test, classes=('pears', 'apples', 'fruits') )

        logger.info('Plot NLL around minimum')
        plot_apple_ratio_around_min(compute_nll, 
                                    pb_config.TRUE_APPLE_RATIO,
                                    model,
                                    extension)

        # MINIMIZE NLL
        logger.info('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll)
        fmin, params = estimate(minimizer)
        params_truth = [pb_config.TRUE_APPLE_RATIO]

        print_params(params, params_truth)
        register_params(params, params_truth, result_row)
        result_row['is_mingrad_valid'] = minimizer.migrad_ok()
        result_row.update(fmin)
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    param_names = pb_config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, model)

    logger.info('DONE')
    return result_table


def get_minimizer(compute_nll, pb_config=None):
    start_apple_ratio = 0.1
    error_apple_ratio = 1.
    minimizer = iminuit.Minuit(compute_nll,
                               errordef=ERRORDEF_NLL,
                               apple_ratio=start_apple_ratio,
                               error_apple_ratio=error_apple_ratio,
                               limit_apple_ratio=(0, 1),
                              )
    return minimizer


def estimate(minimizer):
    logger = logging.getLogger()

    if logger.getEffectiveLevel() <= logging.DEBUG:
        minimizer.print_param()
    logger.info('Mingrad()')
    fmin, params = minimizer.migrad()
    logger.info('Mingrad DONE')

    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        logger.info('Hesse()')
        params = minimizer.hesse()
        logger.info('Hesse DONE')
    else:
        logger.warning('Mingrad IS NOT VALID !')
    return fmin, params



if __name__ == '__main__':
    main()
