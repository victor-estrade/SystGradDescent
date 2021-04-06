#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.HIGGS.FREG-Calib

import os
import logging
from config import SEED
from config import _ERROR
from config import _TRUTH

import numpy as np
import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from ..common import load_estimations
from ..common import load_conditional_estimations
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_neural_net
from utils.evaluation import evaluate_neural_net
from utils.evaluation import evaluate_config
from utils.evaluation import evaluate_regressor
from utils.evaluation import evaluate_estimator
from utils.evaluation import evaluate_conditional_estimation
from utils.images import gather_images

from visual.misc import plot_params


from model.regressor import FilterRegressor
from model.monte_carlo import many_predict
from model.monte_carlo import monte_carlo_data
from model.monte_carlo import monte_carlo_infer
from model.monte_carlo import save_monte_carlo

from archi.reducer import EA3ML3 as ARCHI
# from archi.reducer import EA1AR8MR8L1 as ARCHI

from ..my_argparser import REG_parse_args

from .common import DATA_NAME
from .common import N_BINS
from .common import N_ITER
from .common import Config
from .common import get_minimizer
from .common import NLLComputer
from .common import GeneratorClass
from .common import param_generator
from .common import get_generators_torch
from .common import Parameter


BENCHMARK_NAME = DATA_NAME+'-calib'
NCALL = 100

from .common import GeneratorCPU
from .common import load_calib_tes
from .common import load_calib_jes
from .common import load_calib_les


class TrainGenerator:
    def __init__(self, param_generator, data_generator):
        self.param_generator = param_generator
        self.data_generator = data_generator

    def generate(self, n_samples):
        if n_samples is not None:
            params = self.param_generator()
            X, y, w = self.data_generator.generate(*params, n_samples=n_samples)
            return X, params.interest_parameters, w, params.nuisance_parameters
        else:
            config = Config()
            X, y, w = self.data_generator.generate(*config.CALIBRATED, n_samples=config.N_TRAINING_SAMPLES)
            return X, y, w, 1

    def clf_generate(self, n_samples):
            config = Config()
            X, y, w = self.data_generator.generate(*config.CALIBRATED, n_samples=30000)
            return X, y, w


def build_model(args, i_cv):
    args.net = ARCHI(n_in=29, n_out=2, n_extra=3, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, FilterRegressor)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Gradient boosting on HIGGS benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    os.makedirs(model.results_directory, exist_ok=True)
    config = Config()
    config_table = evaluate_config(config)
    config_table.to_csv(os.path.join(model.results_directory, 'config_table.csv'))
    # RUN
    if not args.conditional_only:
        eval_table = get_eval_table(args, model.results_directory)
    if not args.estimate_only:
        eval_conditional = get_eval_conditional(args, model.results_directory)
    if not args.estimate_only and not args.conditional_only:
        eval_table = pd.concat([eval_table, eval_conditional], axis=1)
        # EVALUATION
        print_line()
        print_line()
        print(eval_table)
        print_line()
        print_line()
        eval_table.to_csv(os.path.join(model.results_directory, 'evaluation.csv'))
    gather_images(model.results_directory)


def get_eval_table(args, results_directory):
    logger = logging.getLogger()
    if args.load_run:
        logger.info(f'Loading previous runs [{args.start_cv},{args.end_cv}[')
        estimations = load_estimations(results_directory, start_cv=args.start_cv, end_cv=args.end_cv)
    else:
        logger.info(f'Running runs [{args.start_cv},{args.end_cv}[')
        estimations = [run_estimation(args, i_cv) for i_cv in range(args.start_cv, args.end_cv)]
        estimations = pd.concat(estimations, ignore_index=True)
    estimations.to_csv(os.path.join(results_directory, 'estimations.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(Config.INTEREST_PARAM_NAME, estimations)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(results_directory, 'estimation_evaluation.csv'))
    return eval_table


def get_eval_conditional(args, results_directory):
    logger = logging.getLogger()
    if args.load_run:
        logger.info(f'Loading previous runs [{args.start_cv},{args.end_cv}[')
        conditional_estimations = load_conditional_estimations(results_directory, start_cv=args.start_cv, end_cv=args.end_cv)
    else:
        logger.info(f'Running runs [{args.start_cv},{args.end_cv}[')
        conditional_estimations = [run_conditional_estimation(args, i_cv) for i_cv in range(args.start_cv, args.end_cv)]
        conditional_estimations = pd.concat(conditional_estimations, ignore_index=True)
    conditional_estimations.to_csv(os.path.join(results_directory, 'conditional_estimations.csv'))
    # EVALUATION
    eval_conditional = evaluate_conditional_estimation(conditional_estimations, interest_param_name=Config.INTEREST_PARAM_NAME)
    print_line()
    print_line()
    print(eval_conditional)
    print_line()
    print_line()
    eval_conditional.to_csv(os.path.join(results_directory, 'conditional_evaluation.csv'))
    return eval_conditional


def run_estimation(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()

    result_row = {'i_cv': i_cv}

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda, GeneratorClass=GeneratorClass)
    train_generator = GeneratorCPU(train_generator)
    train_generator = TrainGenerator(param_generator, train_generator)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES, no_grad=True)

    result_row.update(evaluate_neural_net(model, prefix='valid'))
    evaluate_regressor(model, prefix='valid')

    # MEASUREMENT
    calib_tes = load_calib_tes(DATA_NAME, BENCHMARK_NAME)
    calib_jes = load_calib_jes(DATA_NAME, BENCHMARK_NAME)
    calib_les = load_calib_les(DATA_NAME, BENCHMARK_NAME)
    calibs = (calib_tes, calib_jes, calib_les)
    result_row['nfcn'] = NCALL
    iter_results = [run_estimation_iter(model, result_row, i, test_config, valid_generator, test_generator, calibs)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(iter_results)
    result_table.to_csv(os.path.join(model.results_path, 'estimations.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.results_path)

    logger.info('DONE')
    return result_table


def run_estimation_iter(model, result_row, i_iter, config, valid_generator, test_generator, calibs):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    flush(logger)

    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    result_row['n_test_samples'] = test_generator.n_samples
    suffix = f'-mu={config.TRUE.mu:1.2f}_tes={config.TRUE.tes}_jes={config.TRUE.jes}_les={config.TRUE.les}'

    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES, no_grad=True)

    # CHEATER :
    cheat_target, cheat_sigma = model.predict(X_test, w_test, np.array(config.TRUE.nuisance_parameters))
    result_row['cheat_mu'] = cheat_target
    result_row['cheat_sigma_mu'] = cheat_sigma

    # CALIBRATION
    calib_tes, calib_jes, calib_les = calibs
    tes_mean, tes_sigma = calib_tes.predict(X_test, w_test)
    jes_mean, jes_sigma = calib_jes.predict(X_test, w_test)
    les_mean, les_sigma = calib_les.predict(X_test, w_test)
    logger.info('tes = {} =vs= {} +/- {}'.format(config.TRUE.tes, tes_mean, tes_sigma) )
    logger.info('jes = {} =vs= {} +/- {}'.format(config.TRUE.jes, jes_mean, jes_sigma) )
    logger.info('les = {} =vs= {} +/- {}'.format(config.TRUE.les, les_mean, les_sigma) )
    config.CALIBRATED = Parameter(tes_mean, jes_mean, les_mean, config.CALIBRATED.interest_parameters)
    config.CALIBRATED_ERROR = Parameter(tes_sigma, jes_sigma, les_sigma, config.CALIBRATED_ERROR.interest_parameters)
    for name, value in config.CALIBRATED.items():
        result_row[name+"_calib"] = value
    for name, value in config.CALIBRATED_ERROR.items():
        result_row[name+"_calib_error"] = value

    param_sampler = lambda: param_generator(config)

    # MONTE CARLO
    logger.info('Making {} predictions'.format(NCALL))
    all_pred, all_params = many_predict(model, X_test, w_test, param_sampler, ncall=NCALL)
    logger.info('Gathering it all')
    mc_data = monte_carlo_data(all_pred, all_params)
    save_monte_carlo(mc_data, iter_directory, ext=suffix)
    target, sigma = monte_carlo_infer(mc_data)

    result_row.update(config.CALIBRATED.to_dict())
    result_row.update(config.CALIBRATED_ERROR.to_dict( suffix=_ERROR) )
    result_row.update(config.TRUE.to_dict(suffix=_TRUTH) )
    name = config.INTEREST_PARAM_NAME
    result_row[name] = target
    result_row[name+_ERROR] = sigma
    result_row[name+_TRUTH] = config.TRUE.interest_parameters
    logger.info('mu  = {} =vs= {} +/- {}'.format(config.TRUE.interest_parameters, target, sigma) )
    return result_row.copy()


def run_conditional_estimation(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()

    result_row = {'i_cv': i_cv}

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda, GeneratorClass=GeneratorClass)
    train_generator = GeneratorCPU(train_generator)
    train_generator = TrainGenerator(param_generator, train_generator)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES, no_grad=True)

    # MEASUREMENT
    result_row['nfcn'] = NCALL
    iter_results = [run_conditional_estimation_iter(model, result_row, i, test_config, valid_generator, test_generator)
                    for i, test_config in enumerate(config.iter_test_config())]

    conditional_estimate = pd.concat(iter_results)
    conditional_estimate['i_cv'] = i_cv
    fname = os.path.join(model.results_path, "conditional_estimations.csv")
    conditional_estimate.to_csv(fname)
    logger.info('DONE')
    return conditional_estimate


def run_conditional_estimation_iter(model, result_row, i_iter, config, valid_generator, test_generator):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    flush(logger)

    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)

    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES, no_grad=True)

    # MEASURE STAT/SYST VARIANCE
    logger.info('MEASURE STAT/SYST VARIANCE')
    conditional_results = make_conditional_estimation(model, X_test, w_test, config)
    fname = os.path.join(iter_directory, "no_nuisance.csv")
    conditional_estimate = pd.DataFrame(conditional_results)
    conditional_estimate['i'] = i_iter
    conditional_estimate.to_csv(fname)

    return conditional_estimate


def make_conditional_estimation(model, X_test, w_test, config):
    results = []
    interest_name = config.INTEREST_PARAM_NAME
    for j, nuisance_parameters in enumerate(config.iter_nuisance()):
        result_row = {}
        target, sigma = model.predict(X_test, w_test, np.array(nuisance_parameters) )
        result_row[interest_name] = target
        result_row[interest_name+_ERROR] = sigma
        result_row[interest_name+_TRUTH] = config.TRUE.interest_parameters
        result_row['j'] = j
        for nuisance_name, value in zip(config.CALIBRATED.nuisance_parameters_names, nuisance_parameters):
            result_row[nuisance_name] = value
            result_row[nuisance_name+_TRUTH] = config.TRUE[nuisance_name]
        results.append(result_row)
    return results


if __name__ == '__main__':
    main()
