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

import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_inferno
from utils.evaluation import evaluate_neural_net
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_config
from utils.evaluation import evaluate_minuit
from utils.evaluation import estimate
from utils.evaluation import evaluate_estimator
from utils.evaluation import evaluate_conditional_estimation
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss.torch import GeneratorTorch
from problem.gamma_gauss.torch import GGLoss
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import get_minimizer_no_nuisance
from problem.gamma_gauss import Generator
from problem.gamma_gauss import param_generator
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.synthetic3D import plot_nll_around_min

from model.inferno import Inferno
# from archi.net import RegNetExtra
from archi.net import F6 as ARCHI

from ..my_argparser import INFERNO_parse_args

DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
N_ITER = 30


def build_model(args, i_cv):
    args.net = ARCHI(n_in=1, n_out=args.n_bins)
    args.optimizer = get_optimizer(args)
    args.criterion = GGLoss()
    model = get_model(args, Inferno)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = INFERNO_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    os.makedirs(model.results_directory, exist_ok=True)
    config = Config()
    config_table = evaluate_config(config)
    config_table.to_csv(os.path.join(model.results_directory, 'config_table.csv'))
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    estimations = [e0 for e0, e1 in results]
    estimations = pd.concat(estimations, ignore_index=True)
    estimations.to_csv(os.path.join(model.results_directory, 'estimations.csv'))
    conditional_estimations = [e1 for e0, e1 in results]
    conditional_estimations = pd.concat(conditional_estimations)
    conditional_estimations.to_csv(os.path.join(model.results_directory, 'conditional_estimations.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, estimations)
    eval_conditional = evaluate_conditional_estimation(conditional_estimations)
    eval_table = pd.concat([eval_table, eval_conditional], axis=1)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(model.results_directory, 'evaluation.csv'))
    gather_images(model.results_directory)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()

    result_row = {'i_cv': i_cv}
    result_table = []

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    pb_config = Config()
    seed = config.SEED + i_cv * 5
    train_generator = Synthetic3DGeneratorTorch(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up inferno')
    model = build_model(args, i_cv)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_inferno(model, train_generator, retrain=args.retrain)

    # CHECK TRAINING
    result_row.update(evaluate_neural_net(model))

    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_VALIDATION_SAMPLES)


    # MEASUREMENT
    N_BINS = args.n_bins
    compute_summaries = model.compute_summaries
    for mu in pb_config.TRUE_MU_RANGE:
        true_params = Parameter(pb_config.TRUE.r, pb_config.TRUE.lam, mu)
        suffix = f'-mu={true_params.mu:1.2f}_r={true_params.r}_lambda={true_params.lam}'
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(*true_params, n_samples=pb_config.N_TESTING_SAMPLES)
        # PLOT SUMMARIES
        evaluate_summary_computer(model, X_valid, y_valid, w_valid, X_test, w_test, n_bins=N_BINS, prefix='', suffix=suffix)

        logger.info('Set up NLL computer')
        compute_nll = S3D2NLL(compute_summaries, valid_generator, X_test, w_test)
        # NLL PLOTS
        plot_nll_around_min(compute_nll, true_params, model.path, suffix)

        # MINIMIZE NLL
        logger.info('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, pb_config.CALIBRATED, pb_config.CALIBRATED_ERROR)
        fmin, params = estimate(minimizer)
        result_row.update(evaluate_minuit(minimizer, fmin, params, true_params))

        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    param_names = pb_config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.path)


    logger.info('DONE')
    return result_table


if __name__ == '__main__':
    main()
	