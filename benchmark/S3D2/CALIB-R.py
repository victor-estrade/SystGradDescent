#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.S3D2.CALIB-R

import os
import logging
import config

import pandas as pd

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.plot import plot_REG_losses
from utils.plot import plot_REG_log_mse
from utils.plot import plot_params
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import save_model
from utils.misc import _ERROR
from utils.misc import _TRUTH

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import param_generator

from model.regressor import Regressor

from archi.net import AR9R9 as ARCHI

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2'
CALIB = "Calib_r"


class Generator_r:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        r, lam, mu = self.param_generator()
        X, y, w = self.data_generator.generate(r, lam, mu, n_samples)
        return X, r, w, None


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)

    # Setup model
    logger.info("Setup model")
    args.net = ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB
    model.set_info(BENCHMARK_NAME, 0)

    # Setup data
    logger.info("Setup data")
    pb_config = S3D2Config()
    seed = config.SEED + 99999
    train_generator = Generator_r(param_generator, S3D2(seed))
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # TRAIN / LOAD
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
        # SAVE
        save_model(model)
    flush(logger)

    # CHECK TRAINING
    logger.info('Plot losses')
    plot_REG_losses(model)
    plot_REG_log_mse(model)

    result_row = {}
    result_row['loss'] = model.losses[-1]
    result_row['mse_loss'] = model.mse_losses[-1]
    result_table = []

    print_line()

    for r in pb_config.TRUE_R_RANGE:
        X_test, y_test, w_test = test_generator.generate(
                                         r,
                                         pb_config.TRUE_LAMBDA,
                                         pb_config.TRUE_MU,
                                         n_samples=pb_config.N_TESTING_SAMPLES)
        target, sigma = model.predict(X_test, w_test)
        logger.info('{} =vs= {} +/- {}'.format(r, target, sigma))

        name = "r"
        result_row[name] = target
        result_row[name+_ERROR] = sigma
        result_row[name+_TRUTH] = r
        result_table.append(result_row.copy())
    print_line()
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    param_names = ["r"]
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)

    logger.info('DONE')


if __name__ == '__main__':
    main()
