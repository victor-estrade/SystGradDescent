#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.S3D2.REG-Calib

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

from problem.synthetic3D import S3D2 as Generator
from problem.synthetic3D import S3D2Config as Config
from problem.synthetic3D import Parameter
from problem.synthetic3D import param_generator

from model.regressor import Regressor
from model.monte_carlo import many_predict
from model.monte_carlo import monte_carlo_data
from model.monte_carlo import monte_carlo_infer
from model.monte_carlo import save_monte_carlo

from archi.reducer import EA3ML3 as ARCHI
# from archi.reducer import EA1AR8MR8L1 as ARCHI
from archi.reducer import A3ML3 as CALIB_ARCHI
# from archi.reducer import EA1AR8MR8L1 as CALIB_ARCHI

from ..my_argparser import REG_parse_args


DATA_NAME = 'S3D2'
BENCHMARK_NAME = DATA_NAME+'-calib'
CALIB_R = "Calib_r"
CALIB_LAM = "Calib_lam"
N_ITER = 9
NCALL = 100

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
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
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
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
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


class TrainGenerator:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        if n_samples is not None:
            params = self.param_generator()
            X, y, w = self.data_generator.generate(*params, n_samples)
            return X, params.interest_parameters, w, params.nuisance_parameters
        else:
            config = Config()
            X, y, w = self.data_generator.generate(*config.CALIBRATED, n_samples=config.N_TRAINING_SAMPLES)
            return X, y, w, 1


def build_model(args, i_cv):
    args.net = ARCHI(n_in=3, n_out=2, n_extra=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


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
    model = build_model(args, -1)
    config = Config()
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
    config = Config()
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    train_generator = TrainGenerator(param_generator, train_generator)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up regressor')
    model = build_model(args, i_cv)
    flush(logger)
    
    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)
    
    result_row.update(evaluate_neural_net(model, prefix='valid'))
    evaluate_regressor(model, prefix='valid')

    # MEASUREMENT
    calib_r = load_calib_r()
    calib_lam = load_calib_lam()
    result_row['nfcn'] = NCALL
    result_table = [run_iter(model, result_row, i, test_config, valid_generator, test_generator, calib_r, calib_lam)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.path, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)

    logger.info('DONE')
    return result_table


def run_iter(model, result_row, i_iter, config, valid_generator, test_generator, calib_r, calib_lam):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    iter_directory = os.path.join(model.path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    suffix = f'-mu={config.TRUE.mu:1.2f}_r={config.TRUE.r}_lambda={config.TRUE.lam}'
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=None)
    # CALIBRATION
    r_mean, r_sigma = calib_r.predict(X_test, w_test)
    lam_mean, lam_sigma = calib_lam.predict(X_test, w_test)
    param_sampler = calib_param_sampler(r_mean, r_sigma, lam_mean, lam_sigma)
    logger.info('r   = {} =vs= {} +/- {}'.format(config.TRUE_R, r_mean, r_sigma) ) 
    logger.info('lam = {} =vs= {} +/- {}'.format(config.TRUE_LAMBDA, lam_mean, lam_sigma) )
    result_row['r'] = r_mean
    result_row['r'+_ERROR] = r_sigma
    result_row['r'+_TRUTH] = config.TRUE_R
    result_row['lam'] = lam_mean
    result_row['lam'+_ERROR] = lam_sigma
    result_row['lam'+_TRUTH] = config.TRUE_LAMBDA

    # CHEATER :
    cheat_target, cheat_sigma = model.predict(X_test, w_test, np.array(config.TRUE.nuisance_parameters))
    result_row['cheat_mu'] = cheat_target
    result_row['cheat_sigma_mu'] = cheat_sigma

    param_sampler = param_generator

    # MONTE CARLO
    logger.info('Making {} predictions'.format(NCALL))
    all_pred, all_params = many_predict(model, X_test, w_test, param_sampler, ncall=NCALL)
    logger.info('Gathering it all')
    mc_data = monte_carlo_data(all_pred, all_params)
    save_monte_carlo(mc_data, iter_directory, ext=suffix)
    target, sigma = monte_carlo_infer(mc_data)


    result_row.update(params_to_dict(config.CALIBRATED))
    result_row.update(params_to_dict(config.CALIBRATED_ERROR, ext=_ERROR ))
    result_row.update(params_to_dict(config.TRUE, ext=_TRUTH ))
    name = config.INTEREST_PARAM_NAME 
    result_row[name] = target
    result_row[name+_ERROR] = sigma
    result_row[name+_TRUTH] = config.TRUE.interest_parameters
    logger.info('mu  = {} =vs= {} +/- {}'.format(config.TRUE.interest_parameters, target, sigma) ) 
    return result_row.copy()


def params_to_dict(params, ext=""):
    return {name+ext: value for name, value in zip(params._fields, params)}

if __name__ == '__main__':
    main()
