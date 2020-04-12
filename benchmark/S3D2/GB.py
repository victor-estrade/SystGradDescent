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
import config

import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import train_or_load_classifier
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_minuit
from utils.evaluation import estimate
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from visual.misc import plot_params

from problem.synthetic3D import S3D2Config
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2NLL

from visual.special.synthetic3D import plot_nll_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


BENCHMARK_NAME = 'S3D2-prior'
N_ITER = 9


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = get_model(args, GradientBoostingModel)
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
    logger.info('Set up classifier')
    model = get_model(args, GradientBoostingModel)
    model.set_info(BENCHMARK_NAME, i_cv)
    flush(logger)
    
    # TRAINING / LOADING
    train_or_load_classifier(model, train_generator, pb_config.CALIBRATED, pb_config.N_TRAINING_SAMPLES, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*pb_config.CALIBRATED, n_samples=pb_config.N_VALIDATION_SAMPLES)
    
    result_row.update(evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid'))

    # MEASUREMENT
    N_BINS = 10
    compute_summaries = ClassifierSummaryComputer(model, n_bins=N_BINS)
    for mu in pb_config.TRUE_MU_RANGE:
        pb_config.TRUE_MU = mu
        suffix = f'-mu={pb_config.TRUE_MU:1.2f}_r={pb_config.TRUE_R}_lambda={pb_config.TRUE_LAMBDA}'
        logger.info('Generate testing data')
        X_test, y_test, w_test = test_generator.generate(
                                         pb_config.TRUE_R,
                                         pb_config.TRUE_LAMBDA,
                                         pb_config.TRUE_MU,
                                         n_samples=pb_config.N_TESTING_SAMPLES)
        # PLOT SUMMARIES
        evaluate_summary_computer(model, X_valid, y_valid, w_valid, X_test, w_test, n_bins=N_BINS, prefix='', suffix=suffix)

        logger.info('Set up NLL computer')
        compute_nll = S3D2NLL(compute_summaries, valid_generator, X_test, w_test)
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
