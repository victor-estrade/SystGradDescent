#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.GB_S3D 

import os
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5


from utils import set_plot_config
set_plot_config()
from utils import set_logger
from utils import flush
from utils import print_line
from utils import get_model
from utils import print_params

from myplot import plot_valid_distrib
from myplot import plot_summaries
from myplot import plot_params

from problem.synthetic3D import S3D2Config
from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2NLL

from .S3D2_utils import plot_R_around_min
from .S3D2_utils import plot_LAMBDA_around_min
from .S3D2_utils import plot_MU_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from .my_argparser import GB_parse_args


BENCHMARK_NAME = 'S3D2'
N_ITER = 1


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    for i_cv in range(N_ITER):
        run(args, i_cv)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
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
    
    # TRAINING
    logger.info('Generate training data')
    X_train, y_train, w_train = train_generator.generate(
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_TRAINING_SAMPLES)
    logger.info('Training {}'.format(model.get_name()))
    model.fit(X_train, y_train, w_train)
    logger.info('Training DONE')

    # SAVE MODEL
    model_name = '{}-{}'.format(model.get_name(), i_cv)
    model_path = os.path.join(config.SAVING_DIR, BENCHMARK_NAME, model_name)
    logger.info("Saving in {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)


    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_VALIDATION_SAMPLES)
    
    logger.info('Plot distribution of the score')
    plot_valid_distrib(model, model_name, model_path, X_valid, y_valid, classes=("b", "s"))
    

    # MEASUREMENT
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(
                                     pb_config.TRUE_R,
                                     pb_config.TRUE_LAMBDA,
                                     pb_config.TRUE_MU,
                                     n_samples=pb_config.N_TESTING_SAMPLES)

    logger.info('Set up NLL computer')
    compute_summaries = ClassifierSummaryComputer(model, n_bins=10)
    compute_nll = S3D2NLL(compute_summaries, valid_generator, X_test, w_test)

    logger.info('Plot summaries')
    plot_summaries(compute_summaries, model_name, model_path, 
                    X_valid, y_valid, w_valid,
                    X_test, w_test, classes=('b', 's', 'n') )


    # NLL PLOTS
    logger.info('Plot NLL around minimum')
    plot_R_around_min(compute_nll, pb_config, model_path)
    plot_LAMBDA_around_min(compute_nll, pb_config, model_path)
    plot_MU_around_min(compute_nll, pb_config, model_path)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
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
    minimizer.print_param()
    logger.info('Mingrad()')
    fmin, params = minimizer.migrad()
    logger.info('Mingrad DONE')
    params_truth = [pb_config.TRUE_R, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU]

    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        logger.info('Hesse()')
        params = minimizer.hesse()
        logger.info('Hesse DONE')
    else:
        logger.info('Mingrad IS NOT VALID !')

    logger.info('Plot params')
    print_params(params, params_truth)
    plot_params(params, params_truth, model_name, model_path)

    print(params[2]['value'] * 1050, 'signal events estimated')
    print(params[2]['error'] * 1050, 'error on # estimated sig event')
    print('Done.')


if __name__ == '__main__':
    main()
