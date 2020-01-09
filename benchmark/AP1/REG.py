#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.AP1.REG

import logging
import config

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_model_id
from utils.model import get_model_path
from utils.model import save_model
from utils.plot import plot_REG_losses
from utils.plot import plot_REG_log_mse
from utils.misc import gather_images

from problem.apples_and_pears import AP1

from model.regressor import Regressor
from archi.net import RegNet

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'AP1'
N_ITER = 5


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
    for i_cv in range(N_ITER):
        run(args, i_cv)
    model = get_model(args, Regressor)
    model_path = get_model_path(BENCHMARK_NAME, model)
    gather_images(model_path)


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
    logger.info('Set up rergessor')
    net = RegNet(n_in=1, n_out=2)
    args.net = net
    model = get_model(args, Regressor)
    model.param_generator = param_generator
    flush(logger)

    # TRAINING
    logger.info('Training {}'.format(model.get_name()))
    model.fit(train_generator)
    logger.info('Training DONE')

    # SAVE MODEL
    model_path = get_model_path(BENCHMARK_NAME, model, i_cv)
    save_model(model, model_path)

    # CHECK TRAINING
    model_id = get_model_id(model, i_cv)

    logger.info('Plot losses')
    plot_REG_losses(model, model_id, model_path)
    plot_REG_log_mse(model, model_id, model_path)

    # MEASUREMENT
    logger.info('Generate testing data')
    true_apple_ratio = 0.8
    param_truth = [true_apple_ratio]
    X_test, y_test, w_test = test_generator.generate(apple_ratio=true_apple_ratio, n_samples=2000)

    pred, sigma = model.predict(X_test, w_test)
    print(true_apple_ratio, '=vs=', pred, '+/-', sigma)




    logger.info('DONE')

if __name__ == '__main__':
    main()
	