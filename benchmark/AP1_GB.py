#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.AP1_GB

import os
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5

from utils import set_plot_config
set_plot_config()
from utils import set_logger
from utils import flush
from utils import print_line
from utils import get_model
from utils import print_params

from myplot import plot_valid_distrib
from myplot import plot_summaries
from myplot import plot_params

from problem.apples_and_pears import AP1
from problem.apples_and_pears import AP1NLL

from .AP1_utils import plot_apple_ratio_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from .my_argparser import GB_parse_args


BENCHMARK_NAME = 'AP1'
N_ITER = 2


def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on AP1 benchmark")
    logger.info(args)
    flush(logger)
    for i_cv in range(N_ITER):
        run(args, i_cv)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    seed = config.SEED + i_cv * 5
    train_generator = AP1(seed)
    valid_generator = AP1(seed+1)
    test_generator  = AP1(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = get_model(args, GradientBoostingModel)

    # TRAINING
    logger.info('Generate training data')
    X_train, y_train, w_train = train_generator.generate(apple_ratio=0.5, n_samples=2_000)
    logger.info('Training {}'.format(model.get_name()))
    model.fit(X_train, y_train, w_train)
    logger.info('Training DONE')

    # SAVE MODEL
    model_name = '{}-{}'.format(model.get_name(), i_cv)
    model_path = os.path.join(config.SAVING_DIR, BENCHMARK_NAME, model_name)
    logger.info("Saving in {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)


    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(apple_ratio=0.5, n_samples=20_000)

    logger.info('Plot distribution of the score')
    plot_valid_distrib(model, model_name, model_path, X_valid, y_valid, classes=("apples", "pears"))


    # MEASUREMENT
    logger.info('Generate testing data')
    true_apple_ratio = 0.8
    param_truth = [true_apple_ratio]
    X_test, y_test, w_test = test_generator.generate(apple_ratio=true_apple_ratio, n_samples=2_000)
    
    logger.info('Set up NLL computer')
    compute_summaries = ClassifierSummaryComputer(model, n_bins=10)
    compute_nll = AP1NLL(compute_summaries, valid_generator, X_test, w_test)

    logger.info('Plot summaries')
    plot_summaries(compute_summaries, model_name, model_path, 
                    X_valid, y_valid, w_valid,
                    X_test, w_test, classes=('apples', 'pears', 'fruits') )

    logger.info('Plot NLL around minimum')
    plot_apple_ratio_around_min(compute_nll, true_apple_ratio, model_path)

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

    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        print_params(params, params_truth)
        logger.info('Hesse()')
        params = minimizer.hesse()
        logger.info('Hesse DONE')

    logger.info('Plot params')
    plot_params(params, params_truth, model_name, model_path)
    logger.info('DONE')


if __name__ == '__main__':
    main()
