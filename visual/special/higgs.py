# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from config import DEFAULT_DIR

from ..likelihood import plot_param_around_min

SKIP_NLL_PLOT = True


def plot_TES_around_min(compute_nll, true_params, directory, suffix):
    tes_array = np.linspace(0.8, 1.2, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(tes=tes)) for tes in tes_array]
    name = 'tes'
    plot_param_around_min(tes_array, nll_array, true_params.tes, name, suffix, directory, 'tes NLL')

def plot_JES_around_min(compute_nll, true_params, directory, suffix):
    jes_array = np.linspace(0.8, 1.2, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(jes=jes)) for jes in jes_array]
    name = 'jes'
    plot_param_around_min(jes_array, nll_array, true_params.jes, name, suffix, directory, 'jes NLL')

def plot_LES_around_min(compute_nll, true_params, directory, suffix):
    les_array = np.linspace(0.9, 1.1, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(les=les)) for les in les_array]
    name = 'les'
    plot_param_around_min(les_array, nll_array, true_params.les, name, suffix, directory, 'les NLL')


def plot_NASTY_BKG_around_min(compute_nll, true_params, directory, suffix):
    nasty_bkg_array = np.linspace(0.5, 1.5, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(nasty_bkg=nasty_bkg)) for nasty_bkg in nasty_bkg_array]
    name = 'nasty_bkg'
    plot_param_around_min(nasty_bkg_array, nll_array, true_params.nasty_bkg, name, suffix, directory, 'nasty bkg NLL')


def plot_SIGMA_SOFT_around_min(compute_nll, true_params, directory, suffix):
    sigma_soft_array = np.linspace(0.5, 3, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(sigma_soft=sigma_soft)) for sigma_soft in sigma_soft_array]
    name = 'sigma_soft'
    plot_param_around_min(sigma_soft_array, nll_array, true_params.sigma_soft, name, suffix, directory, 'sigma soft NLL')


def plot_MU_around_min(compute_nll, true_params, directory, suffix):
    mu_array = np.linspace(0.1, 3, 20)  # TODO : choose good range
    nll_array = [compute_nll(*true_params.clone_with(mu=mu)) for mu in mu_array]
    name = 'mu'
    plot_param_around_min(mu_array, nll_array, true_params.mu, name, suffix, directory, 'mu NLL')


def plot_nll_around_min(compute_nll, true_params, directory, suffix):
    logger = logging.getLogger()
    if SKIP_NLL_PLOT:
        logger.info('SKIPED Plot NLL around minimum')
    else:
        logger.info('Plot NLL around minimum')
        param_names = true_params.parameter_names
        if 'tes' in param_names : plot_TES_around_min(compute_nll, true_params, directory, suffix)
        if 'jes' in param_names : plot_JES_around_min(compute_nll, true_params, directory, suffix)
        if 'les' in param_names : plot_LES_around_min(compute_nll, true_params, directory, suffix)
        # TODO : FUTURE Reactivate
        # plot_NASTY_BKG_around_min(compute_nll, true_params, directory, suffix)
        # plot_SIGMA_SOFT_around_min(compute_nll, true_params, directory, suffix)
        plot_MU_around_min(compute_nll, true_params, directory, suffix)
