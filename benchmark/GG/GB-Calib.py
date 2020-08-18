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
from utils.model import train_or_load_classifier
from utils.evaluation import evaluate_classifier
from utils.evaluation import evaluate_summary_computer
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from visual.misc import plot_params

from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import get_minimizer_no_nuisance
from problem.gamma_gauss import Generator
from problem.gamma_gauss import Parameter
from problem.gamma_gauss import GGNLL as NLLComputer

from visual.special.gamma_gauss import plot_nll_around_min

from model.gradient_boost import GradientBoostingModel
from model.regressor import Regressor
from model.summaries import ClassifierSummaryComputer
from ..my_argparser import GB_parse_args

from archi.reducer import A3ML3 as CALIB_ARCHI
# from archi.reducer import EA1AR8MR8L1 as CALIB_ARCHI


DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-calib'
CALIB_RESCALE = "Calib_rescale"
N_ITER = 3

def build_model(args, i_cv):
    model = get_model(args, GradientBoostingModel)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    return model

def load_calib_rescale():
    args = lambda : None
    args.n_unit     = 80
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.n_samples  = 1000
    args.n_steps    = 1000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_RESCALE
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model



def calib_param_sampler(r_mean, r_sigma):
    def param_sampler():
        r = np.random.normal(r_mean, r_sigma)        
        mu = np.random.uniform(0, 1)
        return Parameter(r, mu)
    return param_sampler

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
    config = Config()
    # RUN
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(model.results_directory, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.INTEREST_PARAM_NAME, results)
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
    calib_rescale = load_calib_rescale()
    N_BINS = 10
    evaluate_summary_computer(model, X_valid, y_valid, w_valid, n_bins=N_BINS, prefix='valid_', suffix='')
    result_table = [run_iter(model, result_row, i, test_config, valid_generator, test_generator, calib_rescale, n_bins=N_BINS)
                    for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(model.results_path, 'results.csv'))
    logger.info('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title=model.full_name, directory=model.results_path)

    logger.info('DONE')
    return result_table


def run_iter(model, result_row, i_iter, config, valid_generator, test_generator, calib_rescale, n_bins=10):
    logger = logging.getLogger()
    iter_directory = os.path.join(model.results_path, f'iter_{i_iter}')
    os.makedirs(iter_directory, exist_ok=True)
    result_row['i'] = i_iter
    suffix = f'-mix={config.TRUE.mix:1.2f}_rescale={config.TRUE.rescale}'
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)
    # PLOT SUMMARIES
    evaluate_summary_computer(model, X_test, y_test, w_test, n_bins=n_bins, prefix='', suffix=suffix, directory=iter_directory)

    # CALIBRATION
    rescale_mean, rescale_sigma = calib_rescale.predict(X_test, w_test)
    logger.info('rescale  = {} =vs= {} +/- {}'.format(config.TRUE.rescale, rescale_mean, rescale_sigma) ) 
    config.CALIBRATED = Parameter(rescale_mean, config.CALIBRATED.interest_parameters)
    config.CALIBRATED_ERROR = Parameter(rescale_sigma, config.CALIBRATED_ERROR.interest_parameters)

    logger.info('Set up NLL computer')
    compute_summaries = ClassifierSummaryComputer(model, n_bins=n_bins)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    # NLL PLOTS
    plot_nll_around_min(compute_nll, config.TRUE, iter_directory, suffix)


    # MEASURE STAT/SYST VARIANCE
    param_sampler = calib_param_sampler(rescale_mean, rescale_sigma)
    some_results = sampling_hat_mu_wr_alpha(param_sampler, compute_nll, config)
    fname = os.path.join(iter_directory, "no_nuisance.csv")
    df = pd.DataFrame(some_results)
    df.to_csv(fname)
    mix_mean = df.mix.mean()
    mix_std = df.mix.std()
    result_row['mix_mean'] = mix_mean
    result_row['mix_std'] = mix_std

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    result_row.update(evaluate_minuit(minimizer, config.TRUE))
    return result_row.copy()



def sampling_hat_mu_wr_alpha(param_sampler, compute_nll, config):
    results = []
    for j in range(20):
        sampled_param = param_sampler()
        nuisance_parameters = sampled_param.nuisance_parameters
        compute_nll_no_nuisance = lambda mix : compute_nll(*nuisance_parameters, mix)
        minimizer = get_minimizer_no_nuisance(compute_nll_no_nuisance, config.CALIBRATED, config.CALIBRATED_ERROR)
        results_row = evaluate_minuit(minimizer, config.TRUE)
        results_row['j'] = j
        results_row['rescale'] = sampled_param.rescale
        results_row['rescale'+_TRUTH] = config.TRUE.rescale
        results.append(results_row)
    return results






if __name__ == '__main__':
    main()
