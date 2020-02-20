#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.AP1.REG

import os
import logging
import config

import pandas as pd

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import save_model
from utils.plot import plot_REG_losses
from utils.plot import plot_REG_log_mse
from utils.plot import plot_params
from utils.misc import gather_images
from utils.misc import _ERROR
from utils.misc import _TRUTH
from utils.misc import evaluate_estimator

from problem.apples_and_pears import AP1
from problem.apples_and_pears import AP1Config

from model.regressor import Regressor
# from archi.net import RegNet
from archi.net import F3R3

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'AP1'
N_ITER = 9


def param_generator():
    import numpy as np
    apple_ratio = np.random.uniform(0., 1.0)
    return (apple_ratio,)


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on AP1 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    args.net = F3R3(n_in=1, n_out=2)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor, quiet=False)
    model.set_info(BENCHMARK_NAME, -1)
    pb_config = AP1Config()
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(model.directory, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(pb_config.INTEREST_PARAM_NAME, results)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(model.directory, 'evaluation.csv'))
    gather_images(model.directory)


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
    logger.info('Set up rergessor')
    args.net = F3R3(n_in=1, n_out=2)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, i_cv)
    model.param_generator = param_generator
    flush(logger)

    # TRAINING / LOADING
    if not args.retrain:
        try:
            logger.info('loading from {}'.format(model.path))
            model.load(model.path)
        except Exception as e:
            logger.warning(e)
            args.retrain = True
    if args.retrain:
        logger.info('Training {}'.format(model.get_name()))
        model.fit(train_generator)
        logger.info('Training DONE')

        # SAVE MODEL
        save_model(model)

    # CHECK TRAINING
    logger.info('Plot losses')
    plot_REG_losses(model)
    plot_REG_log_mse(model)
    result_row['loss'] = model.losses[-1]
    result_row['mse_loss'] = model.mse_losses[-1]

    # MEASUREMENT
    for mu in pb_config.TRUE_APPLE_RATIO_RANGE:
        pb_config.TRUE_APPLE_RATIO = mu
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(apple_ratio=pb_config.TRUE_APPLE_RATIO,
                                                        n_samples=pb_config.N_TESTING_SAMPLES)

        pred, sigma = model.predict(X_test, w_test)
        name = pb_config.INTEREST_PARAM_NAME 
        result_row[name] = pred
        result_row[name+_ERROR] = sigma
        result_row[name+_TRUTH] = pb_config.TRUE_APPLE_RATIO

        logger.info('{} =vs= {} +/- {}'.format(pb_config.TRUE_APPLE_RATIO, pred, sigma))
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    param_names = pb_config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, model)

    logger.info('DONE')
    return result_table

if __name__ == '__main__':
    main()
	