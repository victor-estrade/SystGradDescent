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
from utils.evaluation import evaluate_minuit
from utils.evaluation import estimate
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from visual.misc import plot_params

from problem.synthetic3D.torch import Synthetic3DGeneratorTorch
from problem.synthetic3D.torch import S3DLoss
from problem.synthetic3D import S3D2
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import S3D2NLL

from visual.special.synthetic3D import plot_nll_around_min

from model.inferno import Inferno
# from archi.net import RegNetExtra
from archi.net import F6

from ..my_argparser import INFERNO_parse_args

BENCHMARK_NAME = 'S3D2'
N_ITER = 3

def build_model(args, i_cv):
    logger = logging.getLogger()
    logger.info('Set up rergessor')
    args.net = F6(n_in=3, n_out=args.n_bins)
    args.optimizer = get_optimizer(args)
    args.criterion = S3DLoss()
    model = get_model(args, Inferno)
    model.set_info(BENCHMARK_NAME, i_cv)
    return model


def main():
    # BASIC SETUP
    logger = set_logger()
    args = INFERNO_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
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
    train_generator = Synthetic3DGeneratorTorch(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
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
        pb_config.TRUE_MU = mu
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(
                                         pb_config.TRUE_R,
                                         pb_config.TRUE_LAMBDA,
                                         pb_config.TRUE_MU,
                                         n_samples=pb_config.N_TESTING_SAMPLES)
        logger.info('Set up NLL computer')
        compute_nll = S3D2NLL(compute_summaries, valid_generator, X_test, w_test)

        suffix = '-mu={:1.2f}_r={}_lambda={}'.format(pb_config.TRUE_MU, 
                                    pb_config.TRUE_R, pb_config.TRUE_LAMBDA)
        evaluate_summary_computer(model, X_valid, y_valid, w_valid, X_test, w_test, n_bins=N_BINS, prefix='', suffix=suffix)

        # NLL PLOTS
        plot_nll_around_min(compute_nll, pb_config, model.path, suffix)

        # MINIMIZE NLL
        logger.info('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, pb_config)
        fmin, params = estimate(minimizer)
        params_truth = [pb_config.TRUE_R, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU]
        result_row.update(evaluate_minuit(minimizer, fmin, params, params_truth))

        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    param_names = pb_config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, model)


    logger.info('DONE')
    return result_table


if __name__ == '__main__':
    main()
	