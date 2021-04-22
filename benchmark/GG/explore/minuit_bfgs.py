#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.minuit_bfgs


import os
import argparse
import logging
import numpy as np

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


from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import GGNLL as NLLComputer
from problem.gamma_gauss import get_minimizer


from scipy.optimize import approx_fprime
from scipy.optimize import fmin_bfgs

DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "gradient", )

SCALE = 1


def parse_args(main_description="Explore NLL shape"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")

    args, _ = parser.parse_known_args()
    return args


def main():
    logger = set_logger()
    directory = os.path.join(DIRECTORY, "nll_contour")
    os.makedirs(directory, exist_ok=True)
    args = parse_args()

    i_cv = 0
    train_generator, valid_generator, test_generator = get_generators(i_cv)
    config = Config()
    model = load_some_NN(i_cv=i_cv, cuda=args.cuda)
    compute_nll = get_nll_computer(model, config, valid_generator, test_generator)

    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"calib nll = {nll}")

    # run_grad(compute_nll, config)
    # out = run_scipy_bfgs(compute_nll, config)
    minimizer = run_minuit_migrad(compute_nll, config)
    # x, y = minimizer.parameters
    # minimizer.contour(x, y)
    # plt.show()



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


def run_grad(compute_nll, config):
    logger = logging.getLogger()
    f = lambda xk : compute_nll(*xk) * SCALE

    xk = np.array(list(config.TRUE))
    logger.info(f"starting point = {xk}")
    EPSILON = 1e-8
    epsilon = np.array([EPSILON]*2)
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"grad = {grad}, grad norm = {grad.dot(grad.T)}")


def run_scipy_bfgs(compute_nll, config):
    logger = logging.getLogger()
    f = lambda xk : compute_nll(*xk) * SCALE

    logger.info(f"Running BFGS on the NLL")
    x_0 = np.array(list(config.CALIBRATED))
    out = fmin_bfgs(f, x_0, full_output=True)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = out
    logger.info(f"xopt = {xopt} ")
    logger.info(f"fopt = {fopt} ")
    logger.info(f"gopt = {gopt} ")
    logger.info(f"Bopt = \n {Bopt} ")
    logger.info(f"func_calls = {func_calls} ")
    logger.info(f"grad_calls = {grad_calls} ")
    logger.info(f"warnflag = {warnflag} ")
    return out


def run_minuit_migrad(compute_nll, config):
    logger = logging.getLogger()
    logger.info(f"Running MIGRAD on the NLL")
    minimizer = get_minimizer(lambda rescale, mu : compute_nll(rescale, mu) * SCALE, config.CALIBRATED, config.CALIBRATED_ERROR)
    minimizer.migrad()
    logger.info(f"\n{minimizer}")
    cov = np.array(minimizer.covariance)
    print("last state", minimizer._last_state)
    print(minimizer.grad, minimizer.values)
    grad = minimizer.grad([1, 1])
    logger.info(f"cov = {cov}")
    logger.info(f"grad = {grad}")

    edm = grad.dot(cov.dot(grad.T))
    logger.info(f"edm = {edm}")
    return minimizer


if __name__ == '__main__':
    main()
