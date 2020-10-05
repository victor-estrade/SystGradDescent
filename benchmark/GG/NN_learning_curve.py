#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.GG.NN_learning_curve

import os
import logging
from config import SAVING_DIR
from config import DEFAULT_DIR
from config import SEED

import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_optimizer
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_summary_computer
from utils.images import gather_images


from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import Generator


from model.neural_network import NeuralNetClassifier
from ..my_argparser import NET_parse_args

from archi.classic import L4 as ARCHI


DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME
N_ITER = 30
N_TRAIN_RANGE = [100, 500, 1000, 2500, 5000, 8000, 10000, 12000, 15000, 17000, 20000]


def build_model(args, i_cv):
    args.net = ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, NeuralNetClassifier)
    model.set_info(DATA_NAME, f"{BENCHMARK_NAME}/learning_curve", i_cv)
    return model


def plot_auc(evaluation, model_name="GB", directory=DEFAULT_DIR):
    import matplotlib.pyplot as plt
    title = f"{model_name} AUC"
    x = []
    y = []
    y_err = []
    for n_train_samples, table in evaluation.groupby('n_train_samples'):
        x.append(n_train_samples)
        y.append(table['valid_auc'].mean())
        y_err.append(table['valid_auc'].std())
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label='AUC')
    fname = "auc.png"
    plt.xlabel('auc $\\pm$ std')
    plt.ylabel('# train samples')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(directory, fname))
    plt.clf()


def plot_accuracy(evaluation, model_name="GB", directory=DEFAULT_DIR):
    import matplotlib.pyplot as plt
    title = f"{model_name} AUC"
    x = []
    y = []
    y_err = []
    for n_train_samples, table in evaluation.groupby('n_train_samples'):
        x.append(n_train_samples)
        y.append(table['valid_accuracy'].mean())
        y_err.append(table['valid_accuracy'].std())
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label='accuracy')
    fname = "accuracy.png"
    plt.xlabel('accuracy $\\pm$ std')
    plt.ylabel('# train samples')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(directory, fname))
    plt.clf()

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
    # config = Config()
    # config_table = evaluate_config(config)
    # config_table.to_csv(os.path.join(model.results_directory, 'config_table.csv'))
    # RUN
    evaluation = [run(args, i_cv) for i_cv in range(N_ITER)]
    # EVALUATION
    evaluation = pd.concat(evaluation)
    evaluation.to_csv(os.path.join(model.results_directory, "evaluation.csv"))
    plot_auc(evaluation, model_name=model.base_name, directory=model.results_directory)
    plot_accuracy(evaluation, model_name=model.base_name, directory=model.results_directory)
    if False : # Temporary removed
        gather_images(model.results_directory)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()

    result_row = {'i_cv': i_cv}

    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    config = Config()
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    valid_generator = Generator(seed+1)
    # test_generator  = Generator(seed+2)

    results = []

    for n_train_samples in N_TRAIN_RANGE:
        result_row['n_train_samples'] = n_train_samples
        # SET MODEL
        logger.info('Set up classifier')
        model = build_model(args, i_cv)
        os.makedirs(model.results_path, exist_ok=True)
        flush(logger)

        # TRAINING / LOADING
        X_train, y_train, w_train = train_generator.generate(*config.CALIBRATED, n_samples=n_train_samples)
        model.fit(X_train, y_train, w_train)

        # CHECK TRAINING
        logger.info('Generate validation data')
        X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)
        
        some_eval = evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid', suffix=f'-{n_train_samples}')
        result_row['valid_auc'] = some_eval[f'valid_auc-{n_train_samples}']
        result_row['valid_accuracy'] = some_eval[f'valid_accuracy-{n_train_samples}']

        N_BINS = 10
        evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix=f'{n_train_samples}')

        results.append(result_row.copy())
    result_table = pd.DataFrame(results)

    return result_table



if __name__ == '__main__':
    main()
