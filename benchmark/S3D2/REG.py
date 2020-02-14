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

import numpy as np
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

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config

from model.regressor import Regressor
# from archi.net import RegNetExtra
from archi.net import AR5R5E

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2'
N_ITER = 9
NCALL = 100

def param_generator():
    import numpy as np
    pb_config = S3D2Config()

    r = np.random.normal(pb_config.CALIBRATED_R, pb_config.CALIBRATED_R_ERROR)
    lam = -1
    while lam <= 0:
        lam = np.random.normal(pb_config.CALIBRATED_LAMBDA, pb_config.CALIBRATED_LAMBDA_ERROR)
    
    mu_min = min(pb_config.TRUE_MU_RANGE)
    mu_max = max(pb_config.TRUE_MU_RANGE)
    mu_range = mu_max - mu_min
    mu_min = max(0.0, mu_min - mu_range/10)
    mu_max = min(1.0, mu_max + mu_range/10)

    mu = np.random.uniform(0, 1)
    return (r, lam, mu,)


def monte_carlo(all_pred, all_sigma):
    pred  = np.mean(all_pred)
    s_squared = all_sigma ** 2
    mu_squared = all_pred ** 2
    sigma = np.mean(s_squared + mu_squared) - (pred ** 2)
    return pred, sigma


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    args.net = AR5R5E(n_in=3, n_out=2, n_extra=2)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, -1)
    pb_config = S3D2Config()

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
    pb_config = S3D2Config()
    seed = config.SEED + i_cv * 5
    train_generator = S3D2(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    args.net = AR5R5E(n_in=3, n_out=2, n_extra=2)
    args.optimizer = get_optimizer(args)
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
    result_row['loss'] = model.losses[-1]
    result_row['mse_loss'] = model.mse_losses[-1]

    result_row['nfcn'] = NCALL

    # MEASUREMENT
    for mu in pb_config.TRUE_MU_RANGE:
        pb_config.TRUE_MU = mu
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(
                                         pb_config.TRUE_R,
                                         pb_config.TRUE_LAMBDA,
                                         pb_config.TRUE_MU,
                                         n_samples=pb_config.N_TESTING_SAMPLES)
        # MONTE CARLO
        all_pred, all_sigma, all_nuisance_params = model.many_predict(X_test, w_test, param_generator, ncall=NCALL)
        all_pred            = np.array(all_pred).reshape(-1, 1)
        all_sigma           = np.array(all_sigma).reshape(-1, 1)
        all_nuisance_params = np.array(all_nuisance_params)
        monte_carlo = np.concatenate([all_pred, all_sigma, all_nuisance_params], axis=1)
        monte_carlo = pd.DataFrame(monte_carlo)
        monte_carlo.to_csv(os.path.join(model.path, 'monte_carlo.csv'))

        pred  = np.mean(all_pred)
        sigma = np.mean(all_sigma)
       
        name = pb_config.INTEREST_PARAM_NAME 
        result_row[name] = pred
        result_row[name+_ERROR] = sigma
        result_row[name+_TRUTH] = pb_config.TRUE_MU
        logger.info('{} =vs= {} +/- {}'.format(pb_config.TRUE_MU, pred, sigma) ) 
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    name = pb_config.INTEREST_PARAM_NAME 
    plot_params(name, result_table, model)


    logger.info('DONE')
    return result_table

if __name__ == '__main__':
    main()
	