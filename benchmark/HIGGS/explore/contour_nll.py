# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line

from utils.evaluation import evaluate_minuit

from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_minimizer
from problem.higgs import get_minimizer_no_nuisance
from problem.higgs import get_generators_torch
from problem.higgs import HiggsNLL as NLLComputer

from ..common import N_BINS

DATA_NAME = 'HIGGS'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "explore")


def do_iter(config, model, i_iter, valid_generator, test_generator, n_bins=N_BINS):
    logger = logging.getLogger()
    directory = os.path.join(DIRECTORY, "nll_contour", model.name, f"iter_{i_iter}")
    os.makedirs(directory, exist_ok=True)
    logger.info(f"saving dir = {directory}")

    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES, no_grad=True)

    logger.info('Set up NLL computer')
    compute_summaries = model.summary_computer(n_bins=n_bins)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)

    basic_check(compute_nll, config)
    basic_contourplot(compute_nll, config, directory)

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    some_dict =  evaluate_minuit(minimizer, config.TRUE, directory, suffix="")


def basic_check(compute_nll, config):
    logger = logging.getLogger()
    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"Calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"TRUE nll = {nll}")


def basic_contourplot(compute_nll, config, directory):
    logger = logging.getLogger()
    logger.info(f"basic mu-tes contour plot...")
    ARRAY_SIZE = 10
    # MESH NLL
    mu_array = np.linspace(0.5, 1.5, ARRAY_SIZE)
    tes_array = np.linspace(0.95, 1.05, ARRAY_SIZE)
    mu_mesh, tes_mesh = np.meshgrid(mu_array, tes_array)
    nll_func = lambda mu, tes : compute_nll(tes, config.TRUE.jes, config.TRUE.les, mu)
    nll_mesh = np.array([nll_func(mu, tes) for mu, tes in zip(mu_mesh.ravel(), tes_mesh.ravel())]).reshape(mu_mesh.shape)
    plot_contour(mu_mesh, tes_mesh, nll_mesh, directory, xlabel="mu", ylabel="tes")

    jes_array = np.linspace(0.95, 1.05, ARRAY_SIZE)
    mu_mesh, jes_mesh = np.meshgrid(mu_array, jes_array)
    nll_func = lambda mu, jes : compute_nll(config.TRUE.tes, jes, config.TRUE.les, mu)
    nll_mesh = np.array([nll_func(mu, jes) for mu, jes in zip(mu_mesh.ravel(), jes_mesh.ravel())]).reshape(mu_mesh.shape)
    plot_contour(mu_mesh, jes_mesh, nll_mesh, directory, xlabel="mu", ylabel="jes")

    les_array = np.linspace(0.95, 1.05, ARRAY_SIZE)
    mu_mesh, les_mesh = np.meshgrid(mu_array, les_array)
    nll_func = lambda mu, les : compute_nll(config.TRUE.tes, config.TRUE.jes, les, mu)
    nll_mesh = np.array([nll_func(mu, les) for mu, les in zip(mu_mesh.ravel(), les_mesh.ravel())]).reshape(mu_mesh.shape)
    plot_contour(mu_mesh, les_mesh, nll_mesh, directory, xlabel="mu", ylabel="les")


def plot_contour(x, y, z, directory, xlabel="mu", ylabel="tes"):
    logger = logging.getLogger()
    fig, ax = plt.subplots()
    CS = ax.contour(x, y, z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fname = f"{xlabel}-{ylabel}_contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    logger.info(f"saved at {path}")
