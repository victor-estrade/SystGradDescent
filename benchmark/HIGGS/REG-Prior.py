#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.S3D2.GB

import os
import logging
from config import SEED

import numpy as np
import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_neural_net
from utils.evaluation import evaluate_neural_net
from utils.evaluation import evaluate_regressor
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from config import _ERROR
from config import _TRUTH

from visual.misc import plot_params

from problem.higgs import HiggsConfig
from problem.higgs import get_generators
from problem.higgs import Generator
from problem.higgs import param_generator

from model.regressor import Regressor
from model.monte_carlo import many_predict
from model.monte_carlo import monte_carlo_data
from model.monte_carlo import monte_carlo_infer
from model.monte_carlo import save_monte_carlo

# from archi.net import RegNetExtra  as ARCHI
# from archi.net import AR19R5E as ARCHI
from archi.net import AR5R5E as ARCHI
# from archi.net import AR5R5 as CALIB_ARCHI

from ..my_argparser import REG_parse_args


BENCHMARK_NAME = 'Higgs-prior'
N_ITER = 3
NCALL = 100

class TrainGenerator:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        if n_samples is not None:
            tes, jes, les, nasty_bkg, sigma_soft, mu = self.param_generator()
            X, y, w = self.data_generator.generate(tes, jes, les, nasty_bkg, sigma_soft, mu, n_samples)
            return X, mu, w, (tes, jes, les, nasty_bkg, sigma_soft)
        else:
            X, y, w = self.data_generator.generate(1, 1, 1, 1, None, 1, n_samples=None)
            return X, y, w, 1


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    args.net = ARCHI(n_in=30, n_out=2, n_extra=5, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, -1)
    config = HiggsConfig()
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(model.directory, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, results)
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

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = HiggsConfig()
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators(seed)
    train_generator = TrainGenerator(param_generator, train_generator)

    # SET MODEL
    logger.info('Set up rergessor')
    args.net = ARCHI(n_in=30, n_out=2, n_extra=5, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(BENCHMARK_NAME, i_cv)
    flush(logger)
    
    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)
    
    result_row.update(evaluate_neural_net(model, prefix='valid'))
    evaluate_regressor(model, prefix='valid')

    # MEASUREMENT
    result_row['nfcn'] = NCALL
    result_table = [run_iter(model, result_row, i, test_config, valid_generator, test_generator)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.path, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)

    logger.info('DONE')
    return result_table


def run_iter(model, result_row, i_iter, config, valid_generator, test_generator):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    iter_directory = os.path.join(model.path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    suffix = f'-mu={config.TRUE.mu:1.2f}_tes={config.TRUE.tes}_jes={config.TRUE.jes}_les={config.TRUE.les}'
    suffix += f'_nasty_bkg={config.TRUE.nasty_bkg}_sigma_soft={config.TRUE.sigma_soft}'
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=None)
    # CALIBRATION
    # logger.info('r   = {} =vs= {} +/- {}'.format(config.TRUE_R, r_mean, r_sigma) ) 
    # logger.info('lam = {} =vs= {} +/- {}'.format(config.TRUE_LAMBDA, lam_mean, lam_sigma) )
    result_row['tes'] = config.CALIBRATED.tes
    result_row['tes'+_ERROR] = config.CALIBRATED_ERROR.tes
    result_row['tes'+_TRUTH] = config.TRUE.tes
    result_row['jes'] = config.CALIBRATED.jes
    result_row['jes'+_ERROR] = config.CALIBRATED_ERROR.jes
    result_row['jes'+_TRUTH] = config.TRUE.jes
    result_row['les'] = config.CALIBRATED.les
    result_row['les'+_ERROR] = config.CALIBRATED_ERROR.les
    result_row['les'+_TRUTH] = config.TRUE.les
    result_row['nasty_bkg'] = config.CALIBRATED.nasty_bkg
    result_row['nasty_bkg'+_ERROR] = config.CALIBRATED_ERROR.nasty_bkg
    result_row['nasty_bkg'+_TRUTH] = config.TRUE.nasty_bkg
    result_row['sigma_soft'] = config.CALIBRATED.sigma_soft
    result_row['sigma_soft'+_ERROR] = config.CALIBRATED_ERROR.sigma_soft
    result_row['sigma_soft'+_TRUTH] = config.TRUE.sigma_soft
    param_sampler = param_generator

    # MONTE CARLO
    logger.info('Making {} predictions'.format(NCALL))
    all_pred, all_params = many_predict(model, X_test, w_test, param_sampler, ncall=NCALL)
    logger.info('Gathering it all')
    mc_data = monte_carlo_data(all_pred, all_params)
    save_monte_carlo(mc_data, iter_directory, ext=suffix)
    target, sigma = monte_carlo_infer(mc_data)

    name = config.INTEREST_PARAM_NAME 
    result_row[name] = target
    result_row[name+_ERROR] = sigma
    result_row[name+_TRUTH] = config.TRUE.mu
    logger.info('mu  = {} =vs= {} +/- {}'.format(config.TRUE.mu, target, sigma) ) 
    return result_row.copy()

if __name__ == '__main__':
    main()
