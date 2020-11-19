#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.VAR.GG.TP

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
from utils.model import train_or_load_neural_net
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images

from visual.misc import plot_params

from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_generators_torch
from problem.higgs import GeneratorCPU
from problem.higgs import HiggsNLL as NLLComputer

from model.tangent_prop import TangentPropClassifier
from archi.classic import L4 as ARCHI
from ...my_argparser import TP_parse_args


DATA_NAME = 'GG'
BENCHMARK_NAME = 'VAR-'+DATA_NAME
N_ITER = 30


class TrainGenerator:
    def __init__(self, data_generator, cuda=False):
        self.data_generator = data_generator
        if cuda:
            self.data_generator.cuda()
        else:
            self.data_generator.cpu()
        self.nuisance_params = self.data_generator.nuisance_params


    def generate(self, n_samples=None):
            X, y, w = self.data_generator.diff_generate(n_samples=n_samples)
            return X, y, w

    def reset(self):
        self.data_generator.reset()

    def tensor(self, data, requires_grad=False, dtype=None):
        return self.data_generator.tensor(data, requires_grad=requires_grad, dtype=dtype)


def build_model(args, i_cv):
    args.net = ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, TangentPropClassifier)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = TP_parse_args(main_description="Training launcher for INFERNO on GG benchmark")
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
    results.to_csv(os.path.join(model.results_directory, 'fisher.csv'))
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
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda)
    train_generator = GeneratorTorch(seed, cuda=args.cuda)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_neural_net(model, train_generator, retrain=args.retrain)

    # MEASUREMENT
    result_row = {'i_cv': i_cv}
    results = []
    for test_config in config.iter_test_config():
        logger.info(f"Running test set : {test_config.TRUE}, {test_config.N_TESTING_SAMPLES} samples")
        for threshold in np.linspace(0, 1, 50):
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
            result_row['s_sqrt_n'] = safe_division( n_selected_sig, np.sqrt(n_selected) )
            result_row['s_sqrt_b'] = safe_division( n_selected_sig, np.sqrt(n_selected_bkg) )
            results.append(result_row.copy())
    results = pd.DataFrame(results)
    print(results)
    return results

def safe_division(numerator, denominator):
    div = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype="float"), where=denominator!=0)
    return div


if __name__ == '__main__':
    main()
