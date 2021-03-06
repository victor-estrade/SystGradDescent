#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.GB

import os
import logging
import json
from config import SEED

import iminuit
ERRORDEF_NLL = 0.5

import numpy as np
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
from utils.evaluation import register_params
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import Generator
from problem.gamma_gauss import Parameter
from problem.gamma_gauss import param_generator
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


DATA_NAME = 'GG-licorne'
BENCHMARK_NAME = DATA_NAME
N_ITER = 3

def build_model(args, i_cv):
    model = get_model(args, GradientBoostingModel)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


def get_minimizer(compute_nll):
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           mu=0.5,
                           error_mu=1.0,
                           limit_mu=(0, 1),
                          )
    return minimizer

def params_to_dict(params, suffix=''):
    return {name+suffix: value for name, value in zip(params.parameter_names, params)}


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # Config
    config = Config()
    config.TRUE = Parameter(rescale=0.9, mu=0.1)
    train_generator = Generator(SEED)
    valid_generator = Generator(SEED+1)
    test_generator  = Generator(SEED+2)
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)

    # for nuisance in p(nuisance | data)
    nuisance_param_sample = [param_generator().nuisance_parameters for _ in range(25)]
    average_list = []
    variance_list = []
    result_table = []
    for nuisance_params in nuisance_param_sample:
        logger.info(f"nuisance_params = {nuisance_params}")
        estimator_values = []
        for i_cv in range(N_ITER):
            clf = build_model(args, i_cv)
            parameters = Parameter(*nuisance_params, config.CALIBRATED.interest_parameters)
            print(parameters)
            n_samples = config.N_TRAINING_SAMPLES
            X_train, y_train, w_train = train_generator.generate(*parameters, n_samples=n_samples)
            logger.info(f"Training {clf.full_name}")
            clf.fit(X_train, y_train, w_train)
            compute_summaries = ClassifierSummaryComputer(clf, n_bins=10)
            nll_computer = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
            compute_nll = lambda mu : nll_computer(*nuisance_params, mu)
            minimizer = get_minimizer(compute_nll)
            results = evaluate_minuit(minimizer, [config.TRUE.interest_parameters])
            estimator_values.append(results['mu'])
            results['i_cv'] = i_cv
            results.update(params_to_dict(parameters, suffix='true'))
            result_table.append(results.copy())
        average_list.append(np.mean(estimator_values))
        variance_list.append(np.var(estimator_values))

    model = build_model(args, 0)
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    save_directory = model.results_path
    os.makedirs(save_directory, exist_ok=True)
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(save_directory, 'results.csv'))
    logger.info(f"average_list {average_list}")
    logger.info(f"variance_list {variance_list}")
    v_stat = np.mean(variance_list)
    v_syst = np.var(average_list)
    v_total = v_stat + v_syst
    logger.info(f"V_stat = {v_stat}")
    logger.info(f"V_syst = {v_syst}")
    logger.info(f"V_total = {v_total}")
    eval_dict = {"V_stat": v_stat
                ,"V_syst": v_syst
                ,"V_total": v_total
                     }
    eval_path = os.path.join(save_directory, 'info.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_dict, f)





if __name__ == '__main__':
    main()
