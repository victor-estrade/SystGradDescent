#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.GG.GB-Prior

import os
import logging
from config import SEED
from config import _ERROR
from config import _TRUTH

import pandas as pd

from visual.misc import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import train_or_load_classifier
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_config
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.evaluation import evaluate_conditional_estimation
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import get_minimizer_no_nuisance
from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.gradient_boost import GradientBoostingModel
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args


DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
N_ITER = 30

def build_model(args, i_cv):
    model = get_model(args, GradientBoostingModel)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    # BASIC SETUP
    logger = set_logger()
    args = GB_parse_args(main_description="Training launcher for Gradient boosting on S3D2 benchmark")
    logger.info(args)
    flush(logger)
    # INFO
    model = build_model(args, -1)
    os.makedirs(model.results_directory, exist_ok=True)
    config = Config()
    config_table = evaluate_config(config)
    config_table.to_csv(os.path.join(model.results_directory, 'config_table.csv'))
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    estimations = [e0 for e0, e1 in results]
    estimations = pd.concat(estimations, ignore_index=True)
    estimations.to_csv(os.path.join(model.results_directory, 'estimations.csv'))
    conditional_estimations = [e1 for e0, e1 in results]
    conditional_estimations = pd.concat(conditional_estimations)
    conditional_estimations.to_csv(os.path.join(model.results_directory, 'conditional_estimations.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, estimations)
    eval_conditional = evaluate_conditional_estimation(conditional_estimations, interest_param_name=config.INTEREST_PARAM_NAME)
    eval_table = pd.concat([eval_table, eval_conditional], axis=1)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(model.results_directory, 'evaluation.csv'))
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
    test_generator  = Generator(seed+2)

    # SET MODEL
    logger.info('Set up classifier')
    model = build_model(args, i_cv)
    os.makedirs(model.results_path, exist_ok=True)
    flush(logger)
    
    # TRAINING / LOADING
    train_or_load_classifier(model, train_generator, config.CALIBRATED, config.N_TRAINING_SAMPLES, retrain=args.retrain)

    # CHECK TRAINING
    logger.info('Generate validation data')
    X_valid, y_valid, w_valid = valid_generator.generate(*config.CALIBRATED, n_samples=config.N_VALIDATION_SAMPLES)
    
    result_row.update(evaluate_classifier(model, X_valid, y_valid, w_valid, prefix='valid'))

    # MEASUREMENT
    N_BINS = 10
    evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix='')
    iter_results = [run_iter(model, result_row, i, test_config, valid_generator, test_generator, n_bins=N_BINS)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = [e0 for e0, e1 in iter_results]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.results_path, 'estimations.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.results_path)

    conditional_estimate = pd.concat([e1 for e0, e1 in iter_results])
    conditional_estimate['i_cv'] = i_cv
    logger.info('DONE')
    return result_table, conditional_estimate


def run_iter(model, result_row, i_iter, config, valid_generator, test_generator, n_bins=10):
    logger = logging.getLogger()
    logger.info('-'*45)
    logger.info(f'iter : {i_iter}')
    flush(logger)
    
    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    result_row['n_test_samples'] = config.N_TESTING_SAMPLES
    suffix = f'-mix={config.TRUE.mix:1.2f}_rescale={config.TRUE.rescale}'
    
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    # PLOT SUMMARIES
    evaluate_summary_computer(model, X_test, y_test, w_test, n_bins=n_bins, prefix='', suffix=suffix, directory=iter_directory)

    logger.info('Set up NLL computer')
    compute_summaries = ClassifierSummaryComputer(model, n_bins=n_bins)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    # NLL PLOTS
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)

    # MEASURE STAT/SYST VARIANCE
    logger.info('MEASURE STAT/SYST VARIANCE')
    conditional_results = make_conditional_estimation(compute_nll, config)
    fname = os.path.join(iter_directory, "no_nuisance.csv")
    conditional_estimate = pd.DataFrame(conditional_results)
    conditional_estimate['i'] = i_iter
    conditional_estimate.to_csv(fname)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    result_row.update(evaluate_minuit(minimizer, config.TRUE))
    return result_row.copy(), conditional_estimate



def make_conditional_estimation(compute_nll, config):
    results = []
    for j, nuisance_parameters in enumerate(config.iter_nuisance()):
        compute_nll_no_nuisance = lambda mix : compute_nll(*nuisance_parameters, mix)
        minimizer = get_minimizer_no_nuisance(compute_nll_no_nuisance, config.CALIBRATED, config.CALIBRATED_ERROR)
        results_row = evaluate_minuit(minimizer, config.TRUE, do_hesse=False)
        results_row['j'] = j
        for name, value in zip(config.CALIBRATED.nuisance_parameters_names, nuisance_parameters):
            results_row[name] = value
            results_row[name+_TRUTH] = config.TRUE[name]
        results.append(results_row)
    return results

if __name__ == '__main__':
    main()
