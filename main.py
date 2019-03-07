#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import inspect
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# FIXME : change a module name is evil
import higgs_geant as problem

from sklearn.model_selection import ShuffleSplit
from higgs_geant import normalize_weight
from higgs_geant import split_train_test
from higgs_geant import split_data_label_weights
from higgs_4v_pandas import tau_energy_scale
from nll import HiggsNLL
from models import higgsml_models
from models import MODELS
ARG_MODELS = MODELS.keys()

def parse_args():
    # TODO : more descriptive msg.
    parser = argparse.ArgumentParser(description="Training launcher")

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # DATASET CHOICE
    # parser.add_argument('--data', help='chosen dataset',
                        # type=str, choices=PROBLEMS.keys(), default='mnist' )

    # MODEL CHOICE
    parser.add_argument('--model', help='model to train',
                        type=str, choices=ARG_MODELS )

    # MODEL HYPER PARAMETERS
    parser.add_argument('--n-estimators', help='number of estimators',
                        default=1000, type=int)

    parser.add_argument('--max-depth', help='maximum depth of trees',
                        default=3, type=int)

    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)

    parser.add_argument('--trade-off', help='trade-off for multi-objective models',
                        default=1.0, type=float)

    parser.add_argument('-w', '--width', help='width for the data augmentation sampling',
                        default=5, type=float)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=1024, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=10000, type=int)

    parser.add_argument('--n-augment', help='number of times the dataset is augmented',
                        default=2, type=int)

    parser.add_argument('--n-adv-pre-training-steps',
                        help='number of update steps for the pre-training',
                        default=3000, type=int)

    parser.add_argument('--n-clf-pre-training-steps',
                        help='number of update steps for the pre-training',
                        default=3000, type=int)

    parser.add_argument('--n-recovery-steps',
                        help='number of update steps for the catch training of auxiliary models',
                        default=5, type=int)

    parser.add_argument('--fraction-signal-to-keep',
                        help='fraction of signal to keep in Filters',
                        default=0.95, type=float)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')

    args = parser.parse_args()
    return args


# def get_model_class(data_name, model_name):
#     model_class = None
#     if data_name in MODELS:
#         model_class = MODELS[data_name](model_name)
#     else:
#         raise ValueError('Unrecognized dataset name : {}'
#                          'Expected one from {}'. format(data_name, MODELS.keys()))
#     return model_class


def extract_model_args(args, get_model):
    sig = inspect.signature(get_model)
    args_dict = vars(args)
    model_args = { k: args_dict[k] for k in sig.parameters.keys() if k in args_dict.keys() }
    return model_args



def get_cv_iter(X, y):
    cv = ShuffleSplit(n_splits=12, test_size=0.2, random_state=config.RANDOM_STATE)
    cv_iter = list(cv.split(X, y))
    return cv_iter

# =====================================================================
# MAIN
# =====================================================================
def main():
    # GET LOGGER
    #-----------
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.info('Hello')

    args = parse_args()
    logger.info(args)

    logger.handlers[0].flush()

    # GET CHOSEN MODEL
    #-----------------
    logger.info('Building model ...')
    logger.info( 'Model :{}'.format(args.model))
    model_class = higgsml_models(args.model)
    # args.skewing_function = problem.skew
    # args.tangent = problem.tangent
    model_args = extract_model_args(args, model_class)
    logger.info( 'model_args :{}'.format(model_args) )
    model = model_class(**model_args)
    logger.handlers[0].flush()

    logger.info('Loading data ...')
    data = problem.load_data()

    # SPLIT DATA
    #-----------
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    idx_sim, idx_xp = next(cv.split(data, data['Label']))
    data_sim, data_xp = split_train_test(data, idx_sim, idx_xp)

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    idx_train, idx_test = next(cv.split(data_sim, data_sim['Label']))
    data_train, data_test = split_train_test(data_sim, idx_train, idx_test)

    data_train = data_train.copy()
    data_test = data_test.copy()
    data_xp = data_xp.copy()
    data_train['Weight'] = normalize_weight(data_train['Weight'], data_train['Label'])
    data_test['Weight'] = normalize_weight(data_test['Weight'], data_test['Label'])
    data_xp['Weight'] = normalize_weight(data_xp['Weight'], data_xp['Label'])

    X_train, y_train, W_train = split_data_label_weights(data_train)
    X_test, y_test, W_test = split_data_label_weights(data_test)
    X_xp, y_xp, W_xp = split_data_label_weights(data_test)

    # TRAIN MODEL
    #------------
    logger.info('Start training submission : {}'.format(model.get_name()))
    model.fit(X_train, y_train, sample_weight=W_train)
    logger.info('End of training {}'.format(model.get_name()))

    # CHECK TRAINING RESULT
    #----------------------
    proba = model.predict_proba(X_test)
    sns.distplot(proba[y_test==0, 1], label='b')
    sns.distplot(proba[y_test==1, 1], label='s')
    plt.legend()
    # FIXME : name depend on model name and cv_iter
    plt.savefig('savings/test_distrib.png')

    X_infer = X_xp.copy()
    W_infer = W_xp.copy()
    TRUE_TAU_ES = 1.03
    tau_energy_scale(X_infer, scale=TRUE_TAU_ES)
    N_BIN = 20
    negative_log_likelihood = HiggsNLL(model, X_test, y_test, W_test, X_infer, W_infer, N_BIN=N_BIN)
    minimizer = iminuit.Minuit(negative_log_likelihood,
                    errordef=ERRORDEF_NLL,
                    mu=1, error_mu=0.1, limit_mu=(0, None),
                    tau_es=1, error_tau_es=0.1, limit_tau_es=(0, None),
                    )

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', message='.*arcsinh')
        fmin, param = minimizer.migrad()

    # What if mingrad failed ?
    valid = minimizer.migrad_ok()
    logger.info("Minigrad OK ? {}".format(valid) )

    # Compute hessian error
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', message='.*arcsinh')
        param = minimizer.hesse()
    logger.info("param = {} ".format(param) )

    # Stuff to save
    fitarg = minimizer.fitarg
    logger.info("fitarg = {} ".format(fitarg) )



if __name__ == '__main__':
    main()
