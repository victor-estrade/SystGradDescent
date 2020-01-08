#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


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


from utils.log import set_logger
from utils.log import flush
from utils.model import extract_model_args

from problem.synthetic3D import Synthetic3D
from problem.synthetic3D import Config
from problem.synthetic3D import split_data_label_weights
from problem.synthetic3D import Synthetic3DNLL

from model.gradient_boost import GradientBoostingModel

from model.summaries import compute_summaries


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



def get_model(args):
    logger = logging.getLogger()
    logger.info('Building model ...')
    model_class = GradientBoostingModel
    model_args = extract_model_args(args, model_class)
    logger.info( 'model_args :{}'.format(model_args) )
    model = model_class(**model_args)
    flush(logger)
    return model

def plot_test_distrib(model, model_name, model_path, X_test, y_test):
    logger = logging.getLogger()
    logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    proba = model.predict_proba(X_test)
    try:
        sns.distplot(proba[y_test==0, 1], label='b')
        sns.distplot(proba[y_test==1, 1], label='s')
        plt.title(model_name)
        plt.legend()
        plt.savefig(os.path.join(model_path, 'test_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_summaries(model, model_name, model_path, X_test, y_test, w_test):
    logger = logging.getLogger()
    X_sig = X_test.loc[y_test==1]
    w_sig = w_test.loc[y_test==1]
    X_bkg = X_test.loc[y_test==0]
    w_bkg = w_test.loc[y_test==0]

    s_histogram = compute_summaries(model, X_sig, w_sig, n_bins=10 )
    b_histogram = compute_summaries(model, X_bkg, w_bkg, n_bins=10 )

    try:
        plt.bar(np.arange(10)+0.1, b_histogram, width=0.4, label='b')
        plt.bar(np.arange(10)+0.5, s_histogram, width=0.4, label='s')
        plt.title(model_name)
        plt.legend()
        plt.savefig(os.path.join(model_path, 'summaries.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))


def plot_R_around_min(compute_nll, model_path):
    logger = logging.getLogger()
    pb_config = Config()
    
    r_list = np.linspace(-1, 1, 50)
    arr = [compute_nll(r, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU) for r in r_list]
    try:
        plt.plot(r_list, arr, label='r nll')
        plt.xlabel('r')
        plt.ylabel('nll')
        plt.title('NLL around min')
        plt.legend()
        plt.savefig(os.path.join(model_path, 'r_nll.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


def plot_LAMBDA_around_min(compute_nll, model_path):
    logger = logging.getLogger()
    pb_config = Config()
    
    lam_list = np.linspace(0, 4, 50)
    arr = [compute_nll(pb_config.TRUE_R, lam, pb_config.TRUE_MU) for lam in lam_list]
    try:
        plt.plot(lam_list, arr, label='lambda nll')
        plt.xlabel('lambda')
        plt.ylabel('nll')
        plt.title('NLL around min')
        plt.legend()
        plt.savefig(os.path.join(model_path, 'lambda_nll.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


def plot_MU_around_min(compute_nll, model_path):
    logger = logging.getLogger()
    pb_config = Config()
    
    mu_list = np.linspace(0.0, 1.0, 50)
    arr = [compute_nll(pb_config.TRUE_R, pb_config.TRUE_LAMBDA, mu) for mu in mu_list]
    try:
        plt.plot(mu_list, arr, label='mu nll')
        plt.xlabel('mu')
        plt.ylabel('nll')
        plt.title('NLL around min')
        plt.legend()
        plt.savefig(os.path.join(model_path, 'mu_nll.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = parse_args()
    logger.info(args)
    flush(logger)

    # SET MODEL
    model = get_model(args)

    # LOAD/GENERATE DATA
    logger.info('Generating data ...')
    pb_config = Config()
    generator = Synthetic3D( seed=config.SEED,  n_expected_events=1050 )
    generator.N_SIG = pb_config.N_SIG
    generator.N_BKG = pb_config.N_BKG
    D_train = generator.train_sample(pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.CALIBRATED_MU,
                                     n_samples=pb_config.N_TRAINING_SAMPLES)
    D_test = generator.test_sample(pb_config.CALIBRATED_R,
                                   pb_config.CALIBRATED_LAMBDA,
                                   pb_config.CALIBRATED_MU)
    X_train, y_train, w_train = split_data_label_weights(D_train)
    X_test, y_test, w_test = split_data_label_weights(D_test)
    
    # TRAINING
    model.fit(X_train, y_train, w_train)
    # SAVE MODEL
    i = 99
    model_name = '{}-{}'.format(model.get_name(), i)
    model_path = os.path.join(config.SAVING_DIR, model_name)
    logger.info("Saving in {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)

    # CHECK TRAINING
    plot_test_distrib(model, model_name, model_path, X_test, y_test)
    plot_summaries(model, model_name, model_path, X_test, y_test, w_test)

    # NLL
    summary_computer = lambda X, w : compute_summaries(model, X, w, n_bins=10 )
    D_final = generator.final_sample(pb_config.TRUE_R,
                                   pb_config.TRUE_LAMBDA,
                                   pb_config.TRUE_MU)
    X_final, y_final, w_final = split_data_label_weights(D_final) 
    compute_nll = Synthetic3DNLL(summary_computer, generator, X_final, w_final)

    # NLL PLOTS
    plot_R_around_min(compute_nll, model_path)
    plot_LAMBDA_around_min(compute_nll, model_path)
    plot_MU_around_min(compute_nll, model_path)

    # MINIMIZE NLL
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
    fmin, param = minimizer.migrad()
    param = minimizer.hesse()
    for name, (value, err) in {p['name']: (p['value'], p['error']) for p in param}.items():
        print('{name:3} = {value} ({err})'.format(**locals()))

    print('true_r', pb_config.TRUE_R)
    print('true_lam', pb_config.TRUE_LAMBDA)
    print('true_mu', pb_config.TRUE_MU)

    print(param[2]['value'] * 1050, 'signal events estimated')
    print(param[2]['error'] * 1050, 'error on # estimated sig event')
    print('Done.')


if __name__ == '__main__':
    main()
