#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.S3D2.TP-Prior

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
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_config
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.evaluation import evaluate_conditional_estimation
from utils.images import gather_images

from visual.misc import plot_params

from problem.synthetic3D.torch import GeneratorTorch
from problem.synthetic3D import S3D2Config as Config
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import get_minimizer_no_nuisance
from problem.synthetic3D import Generator
from problem.synthetic3D import Parameter
from problem.synthetic3D import S3D2NLL as NLLComputer

from visual.special.synthetic3D import plot_nll_around_min

from model.tangent_prop import TangentPropClassifier
from archi.classic import L4 as ARCHI

from ..my_argparser import TP_parse_args

from .common import N_BINS

DATA_NAME = 'S3D2'
BENCHMARK_NAME = DATA_NAME+'-prior'
N_ITER = 30


class TrainGenerator:
    def __init__(self, data_generator, cuda=False):
        self.data_generator = data_generator
        if cuda:
            self.data_generator.cuda()
        else:
            self.data_generator.cpu()
        self.nuisance_params = self.data_generator.nuisance_params


    def generate(self, n_samples=None):
            X, y, w = self.data_generator.diff_generate(n_samples=n_samples)
            return X, y, w

    def reset(self):
        self.data_generator.reset()

    def tensor(self, data, requires_grad=False, dtype=None):
        return self.data_generator.tensor(data, requires_grad=requires_grad, dtype=dtype)


def build_model(args, i_cv):
    args.net = ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, TangentPropClassifier)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model



# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = TP_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
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
    train_generator = GeneratorTorch(seed, cuda=args.cuda)
    train_generator = TrainGenerator(train_generator, cuda=args.cuda)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)

    result_row.update(evaluate_neural_net(model, prefix='valid'))
    result_row.update(evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid'))

    # MEASUREMENT
    evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix='')
    iter_results = [run_estimation_iter(model, result_row, i, test_config, valid_generator, test_generator, n_bins=N_BINS)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(iter_results)
    result_table.to_csv(os.path.join(model.results_path, 'estimations.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.results_path)

    logger.info('DONE')
    return result_table


def run_estimation_iter(model, result_row, i_iter, config, valid_generator, test_generator, n_bins=N_BINS):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    flush(logger)

    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    result_row['n_test_samples'] = config.N_TESTING_SAMPLES
    suffix = f'-mu={config.TRUE.mu:1.2f}_r={config.TRUE.r}_lambda={config.TRUE.lam}'

    logger.info('Generate testing data')
    test_generator.reset()
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    # PLOT SUMMARIES
    evaluate_summary_computer(model, X_test, y_test, w_test, n_bins=n_bins, prefix='', suffix=suffix, directory=iter_directory)

    logger.info('Set up NLL computer')
    compute_summaries = lambda X, w : model.compute_summaries(X, w, n_bins=n_bins)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    # NLL PLOTS
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    result_row.update(evaluate_minuit(minimizer, config.TRUE, iter_directory, suffix=suffix))
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
    train_generator = Generator(seed)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_classifier(model, train_generator, config.CALIBRATED, config.N_TRAINING_SAMPLES, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)

    result_row.update(evaluate_neural_net(model, prefix='valid'))
    result_row.update(evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid'))

    # MEASUREMENT
    evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix='')
    iter_results = [run_conditional_estimation_iter(model, result_row, i, test_config, valid_generator, test_generator, n_bins=N_BINS)
                    for i, test_config in enumerate(config.iter_test_config())]

    conditional_estimate = pd.concat(iter_results)
    conditional_estimate['i_cv'] = i_cv
    fname = os.path.join(model.results_path, "conditional_estimations.csv")
    conditional_estimate.to_csv(fname)
    logger.info('DONE')
    return conditional_estimate


def run_conditional_estimation_iter(model, result_row, i_iter, config, valid_generator, test_generator, n_bins=N_BINS):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    flush(logger)

    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)

    logger.info('Generate testing data')
    test_generator.reset()
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES, no_grad=True)
    # SUMMARIES
    logger.info('Set up NLL computer')
    compute_summaries = model.summary_computer(n_bins=n_bins)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)

    # MEASURE STAT/SYST VARIANCE
    logger.info('MEASURE STAT/SYST VARIANCE')
    conditional_results = make_conditional_estimation(compute_nll, config)
    fname = os.path.join(iter_directory, "no_nuisance.csv")
    conditional_estimate = pd.DataFrame(conditional_results)
    conditional_estimate['i'] = i_iter
    conditional_estimate.to_csv(fname)

    return conditional_estimate


def make_conditional_estimation(compute_nll, config):
    results = []
    for j, nuisance_parameters in enumerate(config.iter_nuisance()):
        compute_nll_no_nuisance = lambda mu : compute_nll(*nuisance_parameters, mu)
        minimizer = get_minimizer_no_nuisance(compute_nll_no_nuisance, config.CALIBRATED, config.CALIBRATED_ERROR)
        results_row = evaluate_minuit(minimizer, config.TRUE, do_hesse=False)
        results_row['j'] = j
        for name, value in zip(config.CALIBRATED.nuisance_parameters_names, nuisance_parameters):
            results_row[name] = value
            results_row[name+_TRUTH] = config.TRUE[name]
        results.append(results_row)
    return results


if __name__ == '__main__':
    main()
