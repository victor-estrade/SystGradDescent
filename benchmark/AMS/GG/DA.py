#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.VAR.GG.DA

import os
import logging
from config import SEED
from config import _ERROR
from config import _TRUTH

import numpy as np
import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.model import train_or_load_data_augmentation
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import Generator
from problem.gamma_gauss import param_generator
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.neural_network import NeuralNetClassifier
from model.summaries import ClassifierSummaryComputer
from ...my_argparser import NET_parse_args

from archi.classic import L4 as ARCHI


DATA_NAME = 'GG'
BENCHMARK_NAME = 'VAR-'+DATA_NAME
N_ITER = 30
N_AUGMENT = 5

def build_model(args, i_cv):
    args.net = ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, NeuralNetClassifier)
    model.base_name = "DataAugmentation"
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


class TrainGenerator:
    def __init__(self, param_generator, data_generator, n_bunch=1000):
        self.param_generator = param_generator
        self.data_generator = data_generator
        self.n_bunch = n_bunch

    def generate(self, n_samples):
        n_bunch_samples = n_samples // self.n_bunch
        params = [self.param_generator().clone_with(mu=0.5) for i in range(self.n_bunch)]
        data = [self.data_generator.generate(*parameters, n_samples=n_bunch_samples) for parameters in params]
        X = np.concatenate([X for X, y, w in data], axis=0)
        y = np.concatenate([y for X, y, w in data], axis=0)
        w = np.concatenate([w for X, y, w in data], axis=0)
        return X, y, w


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = NET_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    os.makedirs(model.results_directory, exist_ok=True)
    # RUN
    logger.info(f'Running runs [{args.start_cv},{args.end_cv}[')
    results = [run(args, i_cv) for i_cv in range(args.start_cv, args.end_cv)]
    results = pd.concat(results, ignore_index=True)
    # EVALUATION
    results.to_csv(os.path.join(model.results_directory, 'threshold.csv'))
    print(results)
    print("DONE !")


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()


    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    train_generator = TrainGenerator(param_generator, train_generator)
    valid_generator = Generator(seed+1)
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_data_augmentation(model, train_generator, config.N_TRAINING_SAMPLES*N_AUGMENT, retrain=args.retrain)


    # MEASUREMENT
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for threshold in np.linspace(0, 1, 500):
            result_row = {'i_cv': i_cv}
            result_row['threshold'] = threshold
            result_row.update(test_config.TRUE.to_dict(prefix='true_'))
            result_row['n_test_samples'] = test_config.N_TESTING_SAMPLES

            X, y, w = valid_generator.generate(*config.TRUE, n_samples=config.N_VALIDATION_SAMPLES)
            proba = model.predict_proba(X)
            decision = proba[:, 1]
            selected = decision > threshold
            beta = np.sum(y[selected] == 0)
            gamma = np.sum(y[selected] == 1)
            result_row['beta'] = beta
            result_row['gamma'] = gamma

            X, y, w = test_generator.generate(*config.TRUE, n_samples=config.N_VALIDATION_SAMPLES)
            proba = model.predict_proba(X)
            decision = proba[:, 1]
            selected = decision > threshold
            n_selected = np.sum(selected)
            n_selected_bkg = np.sum(y[selected] == 0)
            n_selected_sig = np.sum(y[selected] == 1)
            result_row['n'] = n_selected
            result_row['b'] = n_selected_bkg
            result_row['s'] = n_selected_sig
            result_row['s_sqrt_n'] = n_selected_sig / np.sqrt(n_selected)
            result_row['s_sqrt_b'] = n_selected_sig / np.sqrt(n_selected)
            results.append(result_row.copy())
    results = pd.DataFrame(results)
    print(results)
    return results




if __name__ == '__main__':
    main()
