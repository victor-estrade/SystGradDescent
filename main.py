#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import argparse
import inspect
import logging
import json
import config
import iminuit
ERRORDEF_NLL = 0.5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from problem.higgs import higgs_geant

from sklearn.model_selection import ShuffleSplit
from higgs_geant import normalize_weight
from higgs_geant import split_train_test
from higgs_geant import split_data_label_weights

from higgs_4v_pandas import tau_energy_scale
from higgs_4v_pandas import jet_energy_scale
from higgs_4v_pandas import lep_energy_scale
from higgs_4v_pandas import soft_term
from higgs_4v_pandas import nasty_background

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
                        default=1, type=float)

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
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    parser.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')


    args = parser.parse_args()
    return args


def extract_model_args(args, get_model):
    sig = inspect.signature(get_model)
    args_dict = vars(args)
    model_args = { k: args_dict[k] for k in sig.parameters.keys() if k in args_dict }
    return model_args


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

    # PREPARE DATA AUGMENTATION/PERTURBATION/TANGENT
    #-----------------------------------------------
    def perturb(X, y, sample_weight=None):
        size = X.shape[0]
        z_tau_es = np.random.normal(loc=config.CALIBRATED_TAU_ENERGY_SCALE,
                                    scale=args.width * config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
                                    size=size, )
        z_jet_es = np.random.normal(loc=config.CALIBRATED_JET_ENERGY_SCALE,
                                    scale=args.width * config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
                                    size=size, )
        z_lep_es = np.random.normal(loc=config.CALIBRATED_LEP_ENERGY_SCALE,
                                    scale=args.width * config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
                                    size=size, )
        z_sigma_soft = np.random.normal(loc=config.CALIBRATED_SIGMA_SOFT,
                                    scale=args.width * config.CALIBRATED_SIGMA_SOFT_ERROR,
                                    size=size, )
        z_sigma_soft = np.abs(z_sigma_soft)
        z_nasty_bkg = np.random.normal(loc=config.CALIBRATED_NASTY_BKG,
                                    scale=args.width * config.CALIBRATED_NASTY_BKG_ERROR,
                                    size=size, )
        tau_energy_scale(X, scale=z_tau_es)
        jet_energy_scale(X, scale=z_jet_es)
        lep_energy_scale(X, scale=z_lep_es)
        soft_term(X, z_sigma_soft)
        X['Label'] = y
        X['Weight'] = sample_weight
        nasty_background(X, z_nasty_bkg)
        X, _, _ = split_data_label_weights(X)
        z =  np.concatenate([z_tau_es.reshape(-1, 1),
                         z_jet_es.reshape(-1, 1), 
                         z_lep_es.reshape(-1, 1)], axis=1)
        return X, y, sample_weight, z

    def augment(X, y, sample_weight=None):
        X = pd.concat([X for _ in range(args.n_augment)])
        y = pd.concat([y for _ in range(args.n_augment)])
        sample_weight = pd.concat([sample_weight for _ in range(args.n_augment)])
        X, y, sample_weight, z = perturb(X, y, sample_weight)
        return X, y, sample_weight, z

    def tangent_extractor(X, alpha=1e-2):
        X_plus = X.copy()
        tau_energy_scale(X_plus, scale=config.CALIBRATED_TAU_ENERGY_SCALE + alpha)
        jet_energy_scale(X_plus, scale=config.CALIBRATED_JET_ENERGY_SCALE + alpha)
        lep_energy_scale(X_plus, scale=config.CALIBRATED_LEP_ENERGY_SCALE + alpha)
        soft_term(       X_plus, sigma_met=config.CALIBRATED_SIGMA_SOFT + alpha)
        X_minus = X.copy()
        tau_energy_scale(X_minus, scale=config.CALIBRATED_TAU_ENERGY_SCALE - alpha)
        jet_energy_scale(X_minus, scale=config.CALIBRATED_JET_ENERGY_SCALE - alpha)
        lep_energy_scale(X_minus, scale=config.CALIBRATED_LEP_ENERGY_SCALE - alpha)
        soft_term(       X_minus, sigma_met=config.CALIBRATED_SIGMA_SOFT - alpha)
        T = ( X_plus - X_minus ) / ( 2 * alpha )
        return T

    args.augmenter = augment
    args.perturbator = perturb
    args.tangent_extractor = tangent_extractor

    # GET CHOSEN MODEL
    #-----------------
    logger.info('Building model ...')
    logger.info( 'Model :{}'.format(args.model))
    model_class = higgsml_models(args.model)
    model_args = extract_model_args(args, model_class)
    logger.info( 'model_args :{}'.format(model_args) )
    model = model_class(**model_args)
    logger.handlers[0].flush()

    logger.info('Loading data ...')
    data = higgs_geant.load_data()
    data = data.drop( ["DER_mass_MMC"], axis=1 )
    
    # CROSS VALIDATION
    #-----------------
    cv_sim_xp = ShuffleSplit(n_splits=config.N_CV, test_size=0.2, random_state=config.RANDOM_STATE)
    for i, (idx_sim, idx_xp) in enumerate(cv_sim_xp.split(data, data['Label'])):
        # SPLIT DATA
        #-----------
        data_sim, data_xp = split_train_test(data, idx_sim, idx_xp)
        cv_train_test = ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
        idx_train, idx_test = next(cv_train_test.split(data_sim, data_sim['Label']))
        data_train, data_test = split_train_test(data_sim, idx_train, idx_test)

        data_train = data_train.copy()
        data_test = data_test.copy()
        # data_train["origWeight"] = data_train["Weight"]
        # data_train['Weight'] = normalize_weight(data_train['Weight'], data_train['Label'])
        data_test["origWeight"] = data_test["Weight"]
        data_test['Weight'] = normalize_weight(data_test['Weight'], data_test['Label'])

        X_train, y_train, W_train = split_data_label_weights(data_train)
        X_test, y_test, W_test = split_data_label_weights(data_test)

        # TRAIN MODEL
        #------------
        model_name = '{}-{}'.format(model.get_name(), i)
        model_path = os.path.join(config.SAVING_DIR, model_name)

        if args.retrain or not os.path.exists(model_path):
            logger.info('Start training submission : {}'.format(model.get_name()))
            model.fit(X_train, y_train, sample_weight=W_train)
            logger.info('End of training {}'.format(model.get_name()))

            # SAVE MODEL
            #-----------
            logger.info('saving model {}/{}...'.format(i+1, config.N_CV))
            model_name = '{}-{}'.format(model.get_name(), i)
            model_path = os.path.join(config.SAVING_DIR, model_name)

            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
        else:
            logger.info('loading model {}/{}...'.format(i+1, config.N_CV))
            model.load(model_path)

        # CHECK TRAINING RESULT
        #----------------------
        logger.info( 'Accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
        proba = model.predict_proba(X_test)
        try:
            sns.distplot(proba[y_test==0, 1], label='b')
            sns.distplot(proba[y_test==1, 1], label='s')
            plt.title(model_name)
            plt.legend()
            plt.savefig(os.path.join(model_path, 'test_distrib.png'))
            plt.clf()
        except:
            pass

        if not args.skip_minuit:
            # PREPARE EXPERIMENTAL DATA
            #--------------------------
            logger.info("Preparing experimental data and NLL minimizer")
            data_infer = data_xp.copy()
            data_infer["origWeight"] = data_infer["Weight"]
            data_infer['Weight'] = normalize_weight(data_infer['Weight'], data_infer['Label'])
            tau_energy_scale(data_infer, scale=config.TRUE_TAU_ENERGY_SCALE)
            jet_energy_scale(data_infer, scale=config.TRUE_JET_ENERGY_SCALE)
            lep_energy_scale(data_infer, scale=config.TRUE_LEP_ENERGY_SCALE)
            soft_term(data, config.TRUE_SIGMA_SOFT)
            nasty_background(data, config.TRUE_NASTY_BKG)
            X_infer, y_infer, W_infer = split_data_label_weights(data_infer)
            
            # PREPARE NLL MINIZATION
            #-----------------------
            N_BIN = 20
            negative_log_likelihood = HiggsNLL(model, data_test, X_infer, W_infer, N_BIN=N_BIN)
            minimizer = iminuit.Minuit(negative_log_likelihood,
                            errordef=ERRORDEF_NLL,
                            mu=1, error_mu=0.1, limit_mu=(0, None),
                            tau_es=config.CALIBRATED_TAU_ENERGY_SCALE, 
                            error_tau_es=config.CALIBRATED_TAU_ENERGY_SCALE_ERROR, 
                            limit_tau_es=(0, None),
                            jet_es=config.CALIBRATED_JET_ENERGY_SCALE,
                            error_jet_es=config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
                            limit_jet_es=(0, None),
                            lep_es=config.CALIBRATED_LEP_ENERGY_SCALE,
                            error_lep_es=config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
                            limit_lep_es=(0, None),
                            sigma_soft=config.CALIBRATED_SIGMA_SOFT,
                            error_sigma_soft=config.CALIBRATED_SIGMA_SOFT_ERROR,
                            limit_sigma_soft=(0, None),
                            nasty_bkg=config.CALIBRATED_NASTY_BKG,
                            error_nasty_bkg=config.CALIBRATED_NASTY_BKG_ERROR,
                            limit_nasty_bkg=(0, None),
                            )

            # MINIZE NLL
            #-----------
            logger.info("minimizing NLL ...")
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', message='.*arcsinh')
                logger.info("Without systematics")
                minimizer.fixed['tau_es'] = True
                minimizer.fixed['jet_es'] = True
                minimizer.fixed['lep_es'] = True
                minimizer.fixed['sigma_soft'] = True
                minimizer.fixed['nasty_bkg'] = True
                fmin, param = minimizer.migrad(precision=config.PRECISION)
                logger.info("minimizing NLL END")

                # TODO : What if mingrad failed ?
                valid = minimizer.migrad_ok()
                logger.info("Minigrad OK ? {}".format(valid) )
                if valid:
                    logger.info("With systematics")
                    minimizer.fixed['tau_es'] = False
                    minimizer.fixed['jet_es'] = False
                    minimizer.fixed['lep_es'] = False
                    minimizer.fixed['sigma_soft'] = False
                    minimizer.fixed['nasty_bkg'] = False
                    fmin, param = minimizer.migrad(precision=config.PRECISION)
                                # TODO : What if mingrad failed ?
                    valid = minimizer.migrad_ok()
                    logger.info("Minigrad 2 OK ? {}".format(valid) )


            if valid:
                # COMPUTE HESSAIN ERROR
                #----------------------
                logger.info("Computing NLL Hessian ...")
                with np.warnings.catch_warnings():
                    np.warnings.filterwarnings('ignore', message='.*arcsinh')
                    param = minimizer.hesse()
                logger.info("Computing NLL Hessian END")
                logger.info("param = {} ".format(param) )

            # Stuff to save
            fitarg = minimizer.fitarg
            logger.info("fitarg = {} ".format(fitarg) )
            with open(os.path.join(model_path, 'fitarg.json'), 'w') as f:
                json.dump(fitarg, f)

            # PRINT ADDITIONNAL RESULTS
            #--------------------------
            mu_mle = fitarg['mu']
            print('mu MLE = {:2.3f} vs {:2.3f} = True mu'.format(mu_mle, config.TRUE_MU) )
            print('mu MLE offset = {}'.format(mu_mle - config.TRUE_MU))
            print('mu MLE errors = {}'.format(fitarg['error_mu']))
            print()
            tau_es_mle = fitarg['tau_es']
            print('tau_es MLE = {:2.3f} vs {:2.3f} = True tau_es'.format(tau_es_mle, config.TRUE_TAU_ENERGY_SCALE) )
            print('tau_es MLE offset = {}'.format(tau_es_mle - config.TRUE_TAU_ENERGY_SCALE))
            print('tau_es MLE errors = {}'.format(fitarg['error_tau_es']))
            print()
            jet_es_mle = fitarg['jet_es']
            print('jet_es MLE = {:2.3f} vs {:2.3f} = True jet_es'.format(jet_es_mle, config.TRUE_JET_ENERGY_SCALE) )
            print('jet_es MLE offset = {}'.format(jet_es_mle - config.TRUE_JET_ENERGY_SCALE))
            print('jet_es MLE errors = {}'.format(fitarg['error_jet_es']))
            print()
            lep_es_mle = fitarg['lep_es']
            print('lep_es MLE = {:2.3f} vs {:2.3f} = True lep_es'.format(lep_es_mle, config.TRUE_LEP_ENERGY_SCALE) )
            print('lep_es MLE offset = {}'.format(lep_es_mle - config.TRUE_LEP_ENERGY_SCALE))
            print('lep_es MLE errors = {}'.format(fitarg['error_lep_es']))
            print()
            sigma_soft_mle = fitarg['sigma_soft']
            print('sigma_soft MLE = {:2.3f} vs {:2.3f} = True sigma_soft'.format(sigma_soft_mle, config.TRUE_SIGMA_SOFT) )
            print('sigma_soft MLE offset = {}'.format(sigma_soft_mle - config.TRUE_SIGMA_SOFT))
            print('sigma_soft MLE errors = {}'.format(fitarg['error_sigma_soft']))
            print()
            nasty_bkg_mle = fitarg['nasty_bkg']
            print('nasty_bkg MLE = {:2.3f} vs {:2.3f} = True nasty_bkg'.format(nasty_bkg_mle, config.TRUE_NASTY_BKG) )
            print('nasty_bkg MLE offset = {}'.format(nasty_bkg_mle - config.TRUE_NASTY_BKG))
            print('nasty_bkg MLE errors = {}'.format(fitarg['error_nasty_bkg']))
            print()
            nll_true_params = negative_log_likelihood(config.TRUE_MU, 
                                        config.TRUE_TAU_ENERGY_SCALE,
                                        config.TRUE_JET_ENERGY_SCALE,
                                        config.TRUE_LEP_ENERGY_SCALE,
                                        config.TRUE_SIGMA_SOFT,
                                        config.TRUE_NASTY_BKG,
                                        )
            print('NLL of true params = {}'.format(nll_true_params))
            nll_MLE = negative_log_likelihood(mu_mle, tau_es_mle, jet_es_mle, lep_es_mle, sigma_soft_mle, nasty_bkg_mle)
            print('NLL of MLE  params = {}'.format(nll_MLE))

    logger.info("END.")

if __name__ == '__main__':
    main()
