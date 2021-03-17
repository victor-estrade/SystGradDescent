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
    logger.info(f"basic contour plot...")
    # MESH NLL
    mu_array = np.linspace(0.5, 1.5, 8)
    tes_array = np.linspace(0.95, 1.05, 8)
    mu_mesh, tes_mesh = np.meshgrid(mu_array, tes_array)
    nll_func = lambda mu, tes : compute_nll(tes, config.TRUE.jes, config.TRUE.les, mu)
    nll_mesh = np.array([nll_func(mu, tes) for mu, tes in zip(mu_mesh.ravel(), tes_mesh.ravel())]).reshape(mu_mesh.shape)

    # plot NLL contour
    fig, ax = plt.subplots()
    CS = ax.contour(mu_mesh, tes_mesh, nll_mesh)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("mu")
    ax.set_ylabel("tes")
    fname = f"contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    logger.info(f"saved at {path}")
