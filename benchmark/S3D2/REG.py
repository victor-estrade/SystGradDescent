#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.S3D2.REG

import os
import logging
import config

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import save_model
from utils.plot import plot_REG_losses
from utils.plot import plot_REG_log_mse
from utils.misc import gather_images

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config

from model.regressor import Regressor
from archi.net import RegNet

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2'
N_ITER = 5


def param_generator():
    import numpy as np
    pb_config = S3D2Config()

    # r = np.random.normal(pb_config.CALIBRATED_R, pb_config.CALIBRATED_R_ERROR)
    # lam = -1
    # while lam <= 0:
    #     lam = np.random.normal(pb_config.CALIBRATED_LAMBDA, pb_config.CALIBRATED_LAMBDA_ERROR)
    
    r = pb_config.CALIBRATED_R
    lam = pb_config.CALIBRATED_LAMBDA
    
    mu = np.random.uniform(0, 1)
    return (r, lam, mu,)


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)

    for i_cv in range(N_ITER):
        run(args, i_cv)
    logger.info("Gathering sub plots")
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, -1)
    gather_images(model.directory)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    pb_config = S3D2Config()
    seed = config.SEED + i_cv * 5
    train_generator = S3D2(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    net = RegNet(n_in=3, n_out=2, n_extra=2)
    args.net = net
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, i_cv)
    model.param_generator = param_generator
    flush(logger)

    # TRAINING
    logger.info('Training {}'.format(model.get_name()))
    model.fit_batch(train_generator)
    logger.info('Training DONE')

    # SAVE MODEL
    save_model(model)

    # CHECK TRAINING

    logger.info('Plot losses')
    plot_REG_losses(model)
    plot_REG_log_mse(model)


    # MEASUREMENT
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(
                                     # pb_config.TRUE_R,
                                     # pb_config.TRUE_LAMBDA,
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.TRUE_MU,
                                     n_samples=pb_config.N_TESTING_SAMPLES)
    
    import numpy as np
    p_test = np.array( (pb_config.CALIBRATED_R, pb_config.CALIBRATED_LAMBDA) )
    pred, sigma = model.predict(X_test, w_test, p_test)
    print(pb_config.TRUE_MU, '=vs=', pred, '+/-', sigma)


    logger.info('DONE')

if __name__ == '__main__':
    main()
	