#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.VAR.GG.PIVOT

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
from utils.model import train_or_load_pivot
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images

from visual.misc import plot_params

from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_generators_torch
from problem.higgs import param_generator
from problem.higgs import GeneratorCPU
from problem.higgs import HiggsNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.pivot import PivotClassifier
from model.criterion.weighted_criterion import WeightedCrossEntropyLoss
from model.criterion.weighted_criterion import WeightedGaussEntropyLoss
from ...my_argparser import PIVOT_parse_args

from archi.classic import L4 as ARCHI

from .common import measurement


DATA_NAME = 'HIGGSTES'
BENCHMARK_NAME = 'VAR-'+DATA_NAME
N_ITER = 30
N_AUGMENT = 5

# net_criterion, adv_criterion, trade_off, net_optimizer, adv_optimizer,

def build_model(args, i_cv):
    args.net = ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.adv_net = ARCHI(n_in=2, n_out=2, n_unit=args.n_unit)
    args.net_optimizer = get_optimizer(args)
    args.adv_optimizer = get_optimizer(args, args.adv_net)
    args.net_criterion = WeightedCrossEntropyLoss()
    args.adv_criterion = WeightedGaussEntropyLoss()
    model = get_model(args, PivotClassifier)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


class TrainGenerator:
    def __init__(self, param_generator, data_generator, n_bunch=1000):
        self.param_generator = param_generator
        self.data_generator = data_generator
        self.n_bunch = n_bunch
        self.n_samples = self.data_generator.data_generator.size

    def generate(self, n_samples):
        n_bunch_samples = n_samples // self.n_bunch
        params = [self.param_generator().clone_with(mu=1.0) for i in range(self.n_bunch)]
        data = [self.data_generator.generate(*parameters, n_samples=n_bunch_samples) for parameters in params]
        X = np.concatenate([X for X, y, w in data], axis=0)
        y = np.concatenate([y for X, y, w in data], axis=0)
        w = np.concatenate([w for X, y, w in data], axis=0)
        z = np.array([p.nuisance_parameters for p in params])
        z = z.reshape(len(params), -1)
        z = np.repeat(z, n_bunch_samples, axis=0)
        return X, y, z, w


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = PIVOT_parse_args(main_description="Training launcher for PIVOT on GG benchmark")
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
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda)
    train_generator = GeneratorCPU(train_generator)
    train_generator = TrainGenerator(param_generator, train_generator)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)

    # TRAINING / LOADING
    train_or_load_pivot(model, train_generator, train_generator.n_samples*N_AUGMENT, retrain=args.retrain)

    # MEASUREMENT
    results = measurement(model, i_cv, config, valid_generator, test_generator)
    print(results)
    return results

if __name__ == '__main__':
    main()
