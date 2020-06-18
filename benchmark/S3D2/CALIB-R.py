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
from config import SEED
from config import _ERROR
from config import _TRUTH

import pandas as pd

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.evaluation import evaluate_neural_net
from utils.evaluation import evaluate_regressor
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_neural_net

from visual.misc import plot_params

from problem.synthetic3D import S3D2 as Generator
from problem.synthetic3D import S3D2Config as Config
from problem.synthetic3D import param_generator

from model.regressor import Regressor

# from archi.reducer import A3ML3 as ARCHI
from archi.reducer import A1AR8MR8L1 as ARCHI
# from archi.net import AR9R9 as ARCHI


from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2-Calib'
CALIB = "Calib_r"
CALIB_PARAM_NAME = "r"


class TrainGenerator:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        n_samples = Config().N_TRAINING_SAMPLES if n_samples is None else n_samples
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
    config = Config()
    seed = SEED + 99999
    train_generator = TrainGenerator(param_generator, Generator(seed))
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    i_cv = -1
    result_row = {'i_cv': i_cv}

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    result_row.update(evaluate_neural_net(model, prefix='valid'))
    evaluate_regressor(model, prefix='valid')
    print_line()

    result_table = [run_iter(model, result_row, i, test_config, valid_generator, test_generator)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.path, 'results.csv'))

    logger.info('Plot params')
    param_names = [CALIB_PARAM_NAME]
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)

    logger.info('DONE')


def run_iter(model, result_row, i_iter, config, valid_generator, test_generator):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    target, sigma = model.predict(X_test, w_test)
    logger.info('{} =vs= {} +/- {}'.format(config.TRUE.r, target, sigma))

    result_row.update(params_to_dict(config.TRUE, ext=_TRUTH ))
    name = CALIB_PARAM_NAME
    result_row[name] = target
    result_row[name+_ERROR] = sigma
    result_row[name+_TRUTH] = config.TRUE.r
    return result_row.copy()


def params_to_dict(params, ext=""):
    return {name+ext: value for name, value in zip(params._fields, params)}

if __name__ == '__main__':
    main()
