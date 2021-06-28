#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.landscape


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

    N_CV = 2
    # FIXME : remove lili and STEP to use all iteration !
    STEP = 1
    lili = list(Config().iter_test_config())[::STEP]
    N_ITER = len(lili)
    logger.info(f"{N_CV} cv and {N_ITER} iteractions ({N_ITER*N_CV} loops)")
    data = []
    for i_cv in range(N_CV):
        model = load_some_NN(i_cv=i_cv, cuda=args.cuda)
        for i_iter, config in enumerate(lili):
            i_iter = i_iter * STEP
            values = run_cv_iter(args, model, i_cv, i_iter, config, root_directory)
            data.append(values)
    data = pd.DataFrame(data)
    fname = os.path.join(root_directory, "landscape.csv")
    data.to_csv(fname)



def run_cv_iter(args, model, i_cv, i_iter, config, root_directory):
    logger = logging.getLogger()
    logger.info(f"cv = {i_cv} iter = {i_iter}")
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


    logger.info(f"Contour plots !")
    ARRAY_SIZE = 150
    DELTA_alpha = 0.4
    DELTA_mu = 0.6
    ALPHA = 1.0
    MU = 1.0
    alpha_array = np.linspace(ALPHA-DELTA_alpha, ALPHA+DELTA_alpha, ARRAY_SIZE)
    mu_array = np.linspace(MU-DELTA_mu, MU+DELTA_mu, ARRAY_SIZE)
    # alpha_mesh, mu_mesh = np.meshgrid(alpha_array, mu_array)
    mu_mesh, alpha_mesh = np.meshgrid(mu_array, alpha_array)
    nll_mesh = np.array([compute_nll(alpha, mu) for alpha, mu in zip(alpha_mesh.ravel(), mu_mesh.ravel())]).reshape(mu_mesh.shape)
    # plot_contour(alpha_mesh, mu_mesh, nll_mesh, directory, xlabel="alpha", ylabel="mu", title=title)
    # plot_contour(mu_mesh, alpha_mesh, nll_mesh, directory, xlabel="mu", ylabel="alpha", title="landscape")

    xlabel = "mu"
    ylabel = "alpha"
    title  = "landscape"
    true_mu = config.TRUE.mu
    true_rescale = config.TRUE.rescale
    x = mu_mesh
    y = alpha_mesh
    z = nll_mesh

    fig, ax = plt.subplots()
    ax.grid(False)
    # im = ax.imshow(z, interpolation='bilinear', origin='lower')
    # levels = np.linspace(np.min(z), np.max(z), 10)
    # print(np.min(z), np.max(z), levels)
    CS = ax.contour(x, y, z, origin='lower', extend='both')
    CB = fig.colorbar(CS, shrink=0.8)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.scatter(true_mu, true_rescale, s=80, marker="x", label="True")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    ax.set_title(now)
    ax.legend()
    fname = f"{title}_contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    plt.close(fig)
    logger.info(f"saved at {path}")

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



def plot_contour(x, y, z, directory, xlabel="alpha", ylabel="mu", title=""):
    logger = logging.getLogger()
    fig, ax = plt.subplots()
    ax.grid(False)
    # im = ax.imshow(z, interpolation='bilinear', origin='lower')
    levels = np.linspace(np.min(z), np.max(z), 10)
    # print(np.min(z), np.max(z), levels)
    CS = ax.contour(x, y, z, origin='lower', extend='both')
    CB = fig.colorbar(CS, shrink=0.8)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    ax.set_title(now)
    fname = f"{title}_contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    plt.close(fig)
    logger.info(f"saved at {path}")


if __name__ == '__main__':
    main()
