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
import iminuit
ERRORDEF_NLL = 0.5

import pandas as pd

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.log import print_params
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import save_model
from utils.plot import plot_summaries
from utils.plot import plot_params
from utils.plot import plot_INFERNO_losses
from utils.misc import gather_images
from utils.misc import estimate
from utils.misc import register_params
from utils.misc import evaluate_estimator

from problem.synthetic3D.torch import Synthetic3DGeneratorTorch
from problem.synthetic3D.torch import S3DLoss
from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import S3D2NLL

from utils.S3D2 import plot_R_around_min
from utils.S3D2 import plot_LAMBDA_around_min
from utils.S3D2 import plot_MU_around_min

from model.inferno import Inferno
# from archi.net import RegNetExtra
from archi.net import F6

from ..my_argparser import INFERNO_parse_args

BENCHMARK_NAME = 'S3D2'
N_ITER = 3


def main():
    # BASIC SETUP
    logger = set_logger()
    args = INFERNO_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    args.net = F6(n_in=3, n_out=args.n_bins)
    args.optimizer = get_optimizer(args)
    args.criterion = S3DLoss()
    model = get_model(args, Inferno)
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
    train_generator = Synthetic3DGeneratorTorch(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    args.net = F6(n_in=3, n_out=args.n_bins)
    args.optimizer = get_optimizer(args)
    args.criterion = S3DLoss()
    model = get_model(args, Inferno)
    model.set_info(BENCHMARK_NAME, i_cv)
    flush(logger)

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
    plot_INFERNO_losses(model)
    result_row['loss'] = model.loss_hook.losses[-1]

    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_VALIDATION_SAMPLES)


    # MEASUREMENT
    n_bins = args.n_bins
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

        logger.info('Plot summaries')
        extension = '-mu={:1.2f}_r={}_lambda={}'.format(pb_config.TRUE_MU, 
                                    pb_config.TRUE_R, pb_config.TRUE_LAMBDA)
        plot_summaries( model, n_bins, extension,
                        X_valid, y_valid, w_valid,
                        X_test, w_test, classes=('b', 's', 'n') )

        # NLL PLOTS
        logger.info('Plot NLL around minimum')
        plot_R_around_min(compute_nll, pb_config, model, extension)
        plot_LAMBDA_around_min(compute_nll, pb_config, model, extension)
        plot_MU_around_min(compute_nll, pb_config, model, extension)

        # MINIMIZE NLL
        logger.info('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, pb_config)
        fmin, params = estimate(minimizer)
        params_truth = [pb_config.TRUE_R, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU]

        print_params(params, params_truth)
        register_params(params, params_truth, result_row)
        result_row['is_mingrad_valid'] = minimizer.migrad_ok()
        result_row.update(fmin)
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)

    logger.info('Plot params')
    name = pb_config.INTEREST_PARAM_NAME 
    plot_params(name, result_table, model)


    logger.info('DONE')
    return result_table


def get_minimizer(compute_nll, pb_config):
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           r=pb_config.CALIBRATED_R,
                           error_r=pb_config.CALIBRATED_R_ERROR,
                           #limit_r=(0, None),
                           lam=pb_config.CALIBRATED_LAMBDA,
                           error_lam=pb_config.CALIBRATED_LAMBDA_ERROR,
                           limit_lam=(0, None),
                           mu=pb_config.CALIBRATED_MU,
                           error_mu=pb_config.CALIBRATED_MU_ERROR,
                           limit_mu=(0, 1),
                          )
    return minimizer

if __name__ == '__main__':
    main()
	