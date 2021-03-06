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

def plot_TES_around_min(compute_nll, true_params, directory, suffix):
    tes_array = np.linspace(0.8, 1.2, 20)  # TODO : choose good range
    # tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    # nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for tes in tes_array]
    tes, jes, les, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, mu) for tes in tes_array]
    name = 'tes'
    plot_param_around_min(tes_array, nll_array, true_params.tes, name, suffix, directory, 'tes NLL')

def plot_JES_around_min(compute_nll, true_params, directory, suffix):
    jes_array = np.linspace(0.8, 1.2, 20)  # TODO : choose good range
    # tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    # nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for jes in jes_array]
    tes, jes, les, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, mu) for jes in jes_array]
    name = 'jes'
    plot_param_around_min(jes_array, nll_array, true_params.jes, name, suffix, directory, 'jes NLL')

def plot_LES_around_min(compute_nll, true_params, directory, suffix):
    les_array = np.linspace(0.9, 1.1, 20)  # TODO : choose good range
    # tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    # nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for les in les_array]
    tes, jes, les, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, mu) for les in les_array]
    name = 'les'
    plot_param_around_min(les_array, nll_array, true_params.les, name, suffix, directory, 'les NLL')


def plot_NASTY_BKG_around_min(compute_nll, true_params, directory, suffix):
    nasty_bkg_array = np.linspace(0.5, 1.5, 20)  # TODO : choose good range
    tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for nasty_bkg in nasty_bkg_array]
    name = 'nasty_bkg'
    plot_param_around_min(nasty_bkg_array, nll_array, true_params.nasty_bkg, name, suffix, directory, 'nasty bkg NLL')


def plot_SIGMA_SOFT_around_min(compute_nll, true_params, directory, suffix):
    sigma_soft_array = np.linspace(0.5, 3, 20)  # TODO : choose good range
    tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for sigma_soft in sigma_soft_array]
    name = 'sigma_soft'
    plot_param_around_min(sigma_soft_array, nll_array, true_params.sigma_soft, name, suffix, directory, 'sigma soft NLL')


def plot_MU_around_min(compute_nll, true_params, directory, suffix):
    mu_array = np.linspace(0.1, 3, 20)  # TODO : choose good range
    # tes, jes, les, nasty_bkg, sigma_soft, mu =  true_params
    # nll_array = [compute_nll(tes, jes, les, nasty_bkg, sigma_soft, mu) for mu in mu_array]
    tes, jes, les, mu =  true_params
    nll_array = [compute_nll(tes, jes, les, mu) for mu in mu_array]
    name = 'mu'
    plot_param_around_min(mu_array, nll_array, true_params.mu, name, suffix, directory, 'mu NLL')


def plot_nll_around_min(compute_nll, true_params, directory, suffix):
    logger = logging.getLogger()
    logger.info('Plot NLL around minimum')
    plot_TES_around_min(compute_nll, true_params, directory, suffix)
    plot_JES_around_min(compute_nll, true_params, directory, suffix)
    plot_LES_around_min(compute_nll, true_params, directory, suffix)
    # TODO : FUTURE Reactivate
    # plot_NASTY_BKG_around_min(compute_nll, true_params, directory, suffix)
    # plot_SIGMA_SOFT_around_min(compute_nll, true_params, directory, suffix)
    plot_MU_around_min(compute_nll, true_params, directory, suffix)

