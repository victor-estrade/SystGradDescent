#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.edm_min


import os
import datetime
import argparse
import logging
import numpy as np
import pandas as pd

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

import matplotlib.pyplot as plt

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line

from ..common import N_BINS

from .load_model import load_some_GB
from .load_model import load_some_NN

from utils.evaluation import estimate

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import GGNLL as NLLComputer
from problem.gamma_gauss import get_minimizer


from scipy.optimize import approx_fprime
from scipy.optimize import fmin_bfgs

DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "gradient", )


def parse_args(main_description="Explore NLL shape"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")

    args, _ = parser.parse_known_args()
    return args

def main():
    logger = set_logger()
    root_directory = os.path.join(DIRECTORY, "nll_contour")
    os.makedirs(root_directory, exist_ok=True)
    args = parse_args()

    N_CV = 3
    # FIXME : remove lili and STEP to use all iteration !
    STEP = 1
    lili = list(Config().iter_test_config())[::STEP]
    N_ITER = len(lili)
    logger.info(f"{N_CV} cv and {N_ITER} iteractions ({N_ITER*N_CV} loops)")
    data = []
    for i_cv in range(N_CV):
        model = load_some_NN(i_cv=i_cv, cuda=args.cuda)
        model.to_double()
        # model = load_some_GB(i_cv=i_cv)
        for i_iter, config in enumerate(lili):
            i_iter = i_iter * STEP
            values = run_cv_iter(args, i_cv, i_iter, config, model, root_directory)
            data.append(values)
    data = pd.DataFrame(data)
    fname = os.path.join(root_directory, "data.csv")
    data.to_csv(fname)


def run_cv_iter(args, i_cv, i_iter, config, model, root_directory):
    logger = logging.getLogger()
    logger.info('='*135)
    logger.info(f'i_cv = {i_cv}, i_iter = {i_iter}, ')
    # Settings
    directory = os.path.join(root_directory, f"cv_{i_cv}", f"iter_{i_iter}")
    os.makedirs(directory, exist_ok=True)
    train_generator, valid_generator, test_generator = get_generators(i_cv)
    logger.info(f"{config.TRUE}, {config.N_TESTING_SAMPLES}")
    compute_nll = get_nll_computer(model, config, valid_generator, test_generator)

    # Results storage
    values = {}
    values['i_cv'] = i_cv
    values['i_iter'] = i_iter
    values['n_test_samples'] = config.N_TESTING_SAMPLES
    values['TRUE_rescale'] = config.TRUE.rescale
    values['TRUE_mu'] = config.TRUE.mu

    # compute Calibration NLL vs True value NLL
    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"true  nll = {nll}")
    values['TRUE_feval'] = nll

    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR, tolerance=args.tolerance)

    # minimizer.hesse()
    # print(minimizer.covariance)
    minimizer.scan()
    minimizer.migrad()
    logger.info(f"\n{minimizer}")
    # print(minimizer)
    logger.info(f"n grad calls = {minimizer.fmin.ngrad}")
    edm = minimizer.fmin.edm
    logger.info(f"edm before HESSE = {edm}")

    EPSILON = 1e-6
    x = np.array([minimizer.values[0], minimizer.values[1]])
    g_0 = minimizer.fcn(x + np.array([EPSILON, 0])) - minimizer.fcn(x - np.array([EPSILON, 0]))
    g_0 = g_0 / (EPSILON * 2)
    g_1 = minimizer.fcn(x + np.array([EPSILON, 0])) - minimizer.fcn(x - np.array([0, EPSILON]))
    g_1 = g_1 / (EPSILON * 2)
    logger.info(f"grad = {g_0}, {g_1}")
    grad = np.array([g_0, g_1])
    cov = np.array(minimizer.covariance)
    edm_bis = grad.T.dot(cov.dot(grad))
    logger.info(f"edm RECOMPUTED  = {edm_bis}")

    minimizer.hesse()
    # print(minimizer)
    logger.info(f"\n{minimizer}")

    # print(minimizer.values[0], minimizer.values[1])
    EPSILON = 1e-6
    # x = np.array(*config.TRUE)
    x = np.array([minimizer.values[0], minimizer.values[1]])
    g_0 = minimizer.fcn(x + np.array([EPSILON, 0])) - minimizer.fcn(x - np.array([EPSILON, 0]))
    g_0 = g_0 / (EPSILON * 2)
    g_1 = minimizer.fcn(x + np.array([EPSILON, 0])) - minimizer.fcn(x - np.array([0, EPSILON]))
    g_1 = g_1 / (EPSILON * 2)
    logger.info(f"grad = {g_0}, {g_1}")
    # logger.info(f"grad = {minimizer.grad(x)} (minuit.grad)")
    grad = np.array([g_0, g_1])
    cov = np.array(minimizer.covariance)

    edm = minimizer.fmin.edm
    logger.info(f"edm after HESSE = {edm}")
    edm_bis = grad.T.dot(cov.dot(grad))
    logger.info(f"edm RECOMPUTED  = {edm_bis}")

    minimizer.minos("mu")
    # print(minimizer)
    logger.info(f"\n{minimizer}")

    # raise 1

    # EDM at TRUE value of mu and alpha
    # x = np.array([config.TRUE.rescale, config.TRUE.mu])
    # g = minimizer.grad(x)
    # minimizer.values = x
    # minimizer.hesse()
    # edm = minimizer.fmin.edm
    # logger.info(f"TRUE edm after HESSE = {edm}")
    # logger.info(f"TRUE grad = {g} (minuit.grad)")
    # edm_bis = g.T.dot(cov.dot(g))
    # logger.info(f"TRUE edm RECOMPUTED  = {edm_bis}")


    return values



def get_generators(i_cv=0):
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    valid_generator = Generator(seed + 1)
    test_generator  = Generator(seed + 2)
    return train_generator, valid_generator, test_generator


def get_nll_computer(model, config, valid_generator, test_generator):
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)

    compute_summaries = model.summary_computer(n_bins=N_BINS)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    return compute_nll



if __name__ == '__main__':
    main()
