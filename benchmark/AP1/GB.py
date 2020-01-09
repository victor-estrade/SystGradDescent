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

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.log import print_params
from utils.model import get_model
from utils.model import get_model_id
from utils.model import get_model_path
from utils.model import save_model
from utils.plot import plot_valid_distrib
from utils.plot import plot_summaries
from utils.plot import plot_params
from utils.misc import gather_images

from problem.apples_and_pears import AP1
from problem.apples_and_pears import AP1NLL
from problem.apples_and_pears import AP1Config

from utils.AP1 import plot_apple_ratio_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


BENCHMARK_NAME = 'AP1'
N_ITER = 5


def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on AP1 benchmark")
    logger.info(args)
    flush(logger)
    for i_cv in range(N_ITER):
        run(args, i_cv)
    model = get_model(args, GradientBoostingModel)
    model.set_info(BENCHMARK_NAME, -1)
    gather_images(model.directory)



def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
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


    # MEASUREMENT
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(
                                    apple_ratio=pb_config.TRUE_APPLE_RATIO,
                                    n_samples=pb_config.N_TESTING_SAMPLES)
    
    logger.info('Set up NLL computer')
    compute_summaries = ClassifierSummaryComputer(model, n_bins=10)
    compute_nll = AP1NLL(compute_summaries, valid_generator, X_test, w_test)

    logger.info('Plot summaries')
    plot_summaries(compute_summaries, model,
                    X_valid, y_valid, w_valid,
                    X_test, w_test, classes=('pears', 'apples', 'fruits') )

    logger.info('Plot NLL around minimum')
    plot_apple_ratio_around_min(compute_nll, pb_config.TRUE_APPLE_RATIO, model)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    start_apple_ratio = 0.1
    error_apple_ratio = 1.
    minimizer = iminuit.Minuit(compute_nll,
                               errordef=ERRORDEF_NLL,
                               apple_ratio=start_apple_ratio,
                               error_apple_ratio=error_apple_ratio,
                               limit_apple_ratio=(0, 1),
                              )
    
    minimizer.print_param()
    logger.info('Mingrad()')
    fmin, params = minimizer.migrad()
    logger.info('Mingrad DONE')
    params_truth = [pb_config.TRUE_APPLE_RATIO]

    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        logger.info('Hesse()')
        params = minimizer.hesse()
        logger.info('Hesse DONE')
    else:
        logger.info('Mingrad IS NOT VALID !')

    logger.info('Plot params')
    print_params(params, params_truth)
    plot_params(params, params_truth, model, param_min=0, param_max=1)
    logger.info('DONE')


if __name__ == '__main__':
    main()
