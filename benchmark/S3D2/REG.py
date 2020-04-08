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
from problem.synthetic3D import Parameter

from model.regressor import Regressor
from model.monte_carlo import many_predict
from model.monte_carlo import monte_carlo_data
from model.monte_carlo import monte_carlo_infer
from model.monte_carlo import save_monte_carlo
# from archi.net import RegNetExtra
from archi.net import AR19R5E as ARCHI
from archi.net import AR5R5 as CALIB_ARCHI

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2'
CALIB_R = "Calib_r"
CALIB_LAM = "Calib_lam"
N_ITER = 9
NCALL = 100

def param_generator():
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
    return Parameter(r, lam, mu)


class Generator_mu:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        r, lam, mu = self.param_generator()
        X, y, w = self.data_generator.generate(r, lam, mu, n_samples)
        return X, mu, w, (r, lam)


def load_calib_r():
    args = lambda : None
    args.n_unit     = 80
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 5e-4
    args.n_samples  = 1000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_R
    model.set_info(BENCHMARK_NAME, 0)
    model.load(model.path)
    return model

def load_calib_lam():
    args = lambda : None
    args.n_unit     = 80
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 5e-4
    args.n_samples  = 1000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_LAM
    model.set_info(BENCHMARK_NAME, 0)
    model.load(model.path)
    return model

def calib_param_sampler(r_mean, r_sigma, lam_mean, lam_sigma):
    def param_sampler():
        r = np.random.normal(r_mean, r_sigma)
        lam = -1
        while lam <= 0:
            lam = np.random.normal(lam_mean, lam_sigma)
        
        mu = np.random.uniform(0, 1)
        return Parameter(r, lam, mu)
    return param_sampler

def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    args.net = ARCHI(n_in=3, n_out=2, n_extra=2, n_unit=args.n_unit)
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
    train_generator = Generator_mu(param_generator, S3D2(seed))
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    args.net = ARCHI(n_in=3, n_out=2, n_extra=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, i_cv)
    flush(logger)

    calib_r = load_calib_r()
    calib_lam = load_calib_lam()

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

    result_row['nfcn'] = NCALL

    # MEASUREMENT
    for mu in pb_config.TRUE_MU_RANGE:
        pb_config.TRUE_MU = mu
        print_line('-')
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(
                                         pb_config.TRUE_R,
                                         pb_config.TRUE_LAMBDA,
                                         pb_config.TRUE_MU,
                                         n_samples=pb_config.N_TESTING_SAMPLES)
        # CALIBRATION
        r_mean, r_sigma = calib_r.predict(X_test, w_test)
        lam_mean, lam_sigma = calib_lam.predict(X_test, w_test)
        param_sampler = calib_param_sampler(r_mean, r_sigma, lam_mean, lam_sigma)
        logger.info('r   = {} =vs= {} +/- {}'.format(pb_config.TRUE_R, r_mean, r_sigma) ) 
        logger.info('lam = {} =vs= {} +/- {}'.format(pb_config.TRUE_LAMBDA, lam_mean, lam_sigma) )
        result_row['r'] = r_mean
        result_row['r'+_ERROR] = r_sigma
        result_row['r'+_TRUTH] = pb_config.TRUE_R
        result_row['lam'] = lam_mean
        result_row['lam'+_ERROR] = lam_sigma
        result_row['lam'+_TRUTH] = pb_config.TRUE_LAMBDA

        # MONTE CARLO
        logger.info('Making {} predictions'.format(NCALL))
        all_pred, all_params = many_predict(model, X_test, w_test, param_sampler, ncall=NCALL)
        logger.info('Gathering it all')
        mc_data = monte_carlo_data(all_pred, all_params)
        save_monte_carlo(mc_data, model.path, ext='_mu={:1.2f}'.format(mu))
        target, sigma = monte_carlo_infer(mc_data)

        name = pb_config.INTEREST_PARAM_NAME 
        result_row[name] = target
        result_row[name+_ERROR] = sigma
        result_row[name+_TRUTH] = pb_config.TRUE_MU
        logger.info('mu  = {} =vs= {} +/- {}'.format(pb_config.TRUE_MU, target, sigma) ) 
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    name = pb_config.INTEREST_PARAM_NAME 
    plot_params(name, result_table, model)


    logger.info('DONE')
    return result_table

if __name__ == '__main__':
    main()
	