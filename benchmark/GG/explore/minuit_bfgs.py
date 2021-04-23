#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.minuit_bfgs


import os
import datetime
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
    i_iter = 7
    directory = os.path.join(directory, f"cv_{i_cv}")
    os.makedirs(directory, exist_ok=True)
    train_generator, valid_generator, test_generator = get_generators(i_cv)
    config = list(Config().iter_test_config())[i_iter]
    logger.info(f"{config.TRUE}, {config.N_TESTING_SAMPLES}")
    model = load_some_NN(i_cv=i_cv, cuda=args.cuda)
    compute_nll = get_nll_computer(model, config, valid_generator, test_generator)

    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"calib nll = {nll}")

    # run_grad(compute_nll, config)
    out = run_scipy_bfgs(compute_nll, config)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = out
    minimizer = run_minuit_migrad(compute_nll, config)
    # x, y = minimizer.parameters
    # minimizer.contour(x, y)
    # plt.show()

    #  I want grad and feval at minimum and grad at true value
    logger.info(f"Gradient !")
    f = lambda xk : compute_nll(*xk)
    EPSILON = 5e-6
    epsilon = np.array([EPSILON]*2)

    # @ true value
    xk = np.array(list(config.TRUE))
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"@ true value grad={grad}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"@ true value feval={nll}")

    # @ BFGS minimum
    xk = xopt
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"@ BFGS min grad={grad} ( =?= {gopt})")
    logger.info(f"@ BFGS min feval={fopt}")

    # @ Minuit minimum
    xk = np.array(list(minimizer.values))
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"@ Minuit min grad={grad}")
    cov = np.array(minimizer.covariance)
    edm = grad.dot(cov.dot(grad.T))
    logger.info(f"recomputed edm = {edm} ( =?= {minimizer.fmin.edm})")
    logger.info(f"@ Minuit min feval={minimizer.fmin.fval}")

    #  I want grad and feval at minimum and grad at true value
    logger.info(f"Contour plots !")
    ARRAY_SIZE = 10
    DELTA_mu = 0.1
    DELTA_alpha = 0.1
    mu_array = np.linspace(xopt[0]-DELTA_mu, xopt[0]+DELTA_mu, ARRAY_SIZE)
    alpha_array = np.linspace(xopt[0]-DELTA_alpha, xopt[0]-DELTA_alpha, ARRAY_SIZE)
    mu_mesh, alpha_mesh = np.meshgrid(mu_array, alpha_array)
    nll_mesh = np.array([compute_nll(mu, alpha) for mu, alpha in zip(mu_mesh.ravel(), alpha_mesh.ravel())]).reshape(mu_mesh.shape)
    plot_contour(mu_mesh, alpha_mesh, nll_mesh, directory, xlabel="mu", ylabel="alpha")
    print(nll_mesh)

    logger.info(f"Contour plots for gradients !")
    ARRAY_SIZE = 10
    mu_array = np.linspace(xopt[0]-DELTA_mu, xopt[0]+DELTA_mu, ARRAY_SIZE)
    alpha_array = np.linspace(xopt[0]-DELTA_alpha, xopt[0]-DELTA_alpha, ARRAY_SIZE)
    mu_mesh, alpha_mesh = np.meshgrid(mu_array, alpha_array)
    grad_mesh = np.array([np.linalg.norm(approx_fprime(np.array([mu, alpha]), f, epsilon))
                        for mu, alpha in zip(mu_mesh.ravel(), alpha_mesh.ravel())]
                        ).reshape(mu_mesh.shape)
    plot_contour(mu_mesh, alpha_mesh, grad_mesh, directory, xlabel="GRAD_mu", ylabel="alpha")
    print(grad_mesh)


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


def run_grad(compute_nll, config, scale=SCALE):
    logger = logging.getLogger()
    f = lambda xk : compute_nll(*xk) * scale

    xk = np.array(list(config.TRUE))
    logger.info(f"starting point = {xk}")
    EPSILON = 1e-8
    epsilon = np.array([EPSILON]*2)
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"grad = {grad}, grad norm = {grad.dot(grad.T)}")


def run_scipy_bfgs(compute_nll, config, scale=SCALE):
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
    logger.info(f" values = {list(minimizer.values)} ")
    # cov = np.array(minimizer.covariance)
    # print("last state", minimizer._last_state)
    # print(minimizer.grad, minimizer.values)
    # grad = minimizer.grad([1, 1])
    # logger.info(f"cov = {cov}")
    # logger.info(f"grad = {grad}")
    #
    # edm = grad.dot(cov.dot(grad.T))
    # logger.info(f"edm = {edm}")
    return minimizer


def plot_contour(x, y, z, directory, xlabel="mu", ylabel="alpha"):
    logger = logging.getLogger()
    fig, ax = plt.subplots()
    ax.grid(False)
    im = ax.imshow(z, interpolation='bilinear', origin='lower')
    levels = np.linspace(np.min(z), np.max(z), 10)
    # print(np.min(z), np.max(z), levels)
    CS = ax.contour(x, y, z, levels, origin='lower', cmap='flag', extend='both')
    # CB = fig.colorbar(CS, shrink=0.8)
    # ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    ax.set_title(now)
    fname = f"{xlabel}-{ylabel}_contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    plt.close(fig)
    logger.info(f"saved at {path}")


if __name__ == '__main__':
    main()
