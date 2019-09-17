#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.GB_S3D 

import os
import argparse
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set()
sns.set_style("whitegrid")
sns.set_context("poster")

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


from utils import set_logger
from utils import flush
from utils import get_model
from utils import print_params
from myplot import plot_test_distrib
from myplot import plot_param_around_min
from myplot import plot_params

from problem.synthetic3D import Synthetic3D
from problem.synthetic3D import Config
from problem.synthetic3D import split_data_label_weights
from problem.synthetic3D import Synthetic3DNLL

from net.gradient_boost import GradientBoostingModel

from models.classfier import ClassifierSummaryComputer


def parse_args():
    # TODO : more descriptive msg.
    parser = argparse.ArgumentParser(description="Training launcher")

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # MODEL HYPER PARAMETERS
    parser.add_argument('--n-estimators', help='number of estimators',
                        default=100, type=int)

    parser.add_argument('--max-depth', help='maximum depth of trees',
                        default=3, type=int)

    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-1, type=float)

    # OTHER
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    parser.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')

    args = parser.parse_args()
    return args


def plot_summaries(summary_computer, model_name, model_path, D_test, D_final):
    X_test, y_test, w_test = split_data_label_weights(D_test)
    X_final, y_final, w_final = split_data_label_weights(D_final) 
    logger = logging.getLogger()
    X_sig = X_test.loc[y_test==1]
    w_sig = w_test.loc[y_test==1]
    X_bkg = X_test.loc[y_test==0]
    w_bkg = w_test.loc[y_test==0]

    s_histogram = summary_computer(X_sig, w_sig)
    b_histogram = summary_computer(X_bkg, w_bkg)
    n_histogram = summary_computer(X_final, w_final)

    try:
        plt.bar(np.arange(10)+0.1, b_histogram, width=0.3, label='b')
        plt.bar(np.arange(10)+0.4, s_histogram, width=0.3, label='s')
        plt.bar(np.arange(10)+0.7, n_histogram, width=0.3, label='n')
        plt.title(model_name)
        plt.legend()
        plt.savefig(os.path.join(model_path, 'summaries.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))


def plot_R_around_min(compute_nll, pb_config, model_path):
    r_array = np.linspace(-1, 1, 50)
    nll_array = [compute_nll(r, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU) for r in r_array]
    plot_param_around_min(r_array, nll_array, pb_config.TRUE_R, 'r', model_path)


def plot_LAMBDA_around_min(compute_nll, pb_config, model_path):
    lam_array = np.linspace(0, 4, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, lam, pb_config.TRUE_MU) for lam in lam_array]
    plot_param_around_min(lam_array, nll_array, pb_config.TRUE_LAMBDA, 'lambda', model_path)


def plot_MU_around_min(compute_nll, pb_config, model_path):
    mu_array = np.linspace(0.0, 1.0, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, pb_config.TRUE_LAMBDA, mu) for mu in mu_array]
    plot_param_around_min(mu_array, nll_array, pb_config.TRUE_MU, 'mu', model_path)


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = parse_args()
    logger.info(args)
    flush(logger)

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    pb_config = Config()
    generator = Synthetic3D( seed=config.SEED,  n_expected_events=1050 )
    generator.N_SIG = pb_config.N_SIG
    generator.N_BKG = pb_config.N_BKG

    # SET MODEL
    logger.info('Set up classifier')
    model = get_model(args, GradientBoostingModel)
    
    # TRAINING
    logger.info('Generate training data')
    D_train = generator.train_sample(pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_TRAINING_SAMPLES)
    X_train, y_train, w_train = split_data_label_weights(D_train)
    logger.info('Training {}'.format(model.get_name()))
    model.fit(X_train, y_train, w_train)
    logger.info('Training DONE')

    # SAVE MODEL
    i = 2019
    model_name = '{}-{}'.format(model.get_name(), i)
    model_path = os.path.join(config.SAVING_DIR, model_name)
    logger.info("Saving in {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)

    # CHECK TRAINING
    logger.info('Generate testing data')
    D_test = generator.test_sample(pb_config.CALIBRATED_R,
                                   pb_config.CALIBRATED_LAMBDA,
                                   pb_config.CALIBRATED_MU)
    D_final = generator.final_sample(pb_config.TRUE_R,
                                   pb_config.TRUE_LAMBDA,
                                   pb_config.TRUE_MU)
    X_test, y_test, w_test = split_data_label_weights(D_test)
    X_final, y_final, w_final = split_data_label_weights(D_final) 
    
    logger.info('Plot distribution of the score')
    plot_test_distrib(model, model_name, model_path, X_test, y_test)

    logger.info('Plot summaries')
    summary_computer = ClassifierSummaryComputer(model)
    plot_summaries(summary_computer, model_name, model_path, D_test, D_final)

    # NLL
    logger.info('Building NLL function')
    compute_nll = Synthetic3DNLL(summary_computer, generator, X_final, w_final)

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
    fmin, param = minimizer.migrad()
    logger.info('Mingrad DONE')
    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        logger.info('Hesse()')
        param = minimizer.hesse()
        logger.info('Hesse DONE')

    logger.info('Plot params')
    params_truth = [pb_config.TRUE_R, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU]
    plot_params(param, params_truth, model_name, model_path)

    print_params(param, params_truth)

    print(param[2]['value'] * 1050, 'signal events estimated')
    print(param[2]['error'] * 1050, 'error on # estimated sig event')
    print('Done.')


if __name__ == '__main__':
    main()
