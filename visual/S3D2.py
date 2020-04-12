# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import numpy as np

from .likelihood import plot_param_around_min

def plot_R_around_min(compute_nll, pb_config, directory, suffix):
    r_array = np.linspace(-1, 1, 50)
    nll_array = [compute_nll(r, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU) for r in r_array]
    name = 'r'
    plot_param_around_min(r_array, nll_array, pb_config.TRUE_R, name, suffix, directory)


def plot_LAMBDA_around_min(compute_nll, pb_config, directory, suffix):
    lam_array = np.linspace(2, 4, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, lam, pb_config.TRUE_MU) for lam in lam_array]
    name = 'lambda'
    plot_param_around_min(lam_array, nll_array, pb_config.TRUE_LAMBDA, name, suffix, directory)


def plot_MU_around_min(compute_nll, pb_config, directory, suffix):
    mu_array = np.linspace(0.0, 1.0, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, pb_config.TRUE_LAMBDA, mu) for mu in mu_array]
    name = 'mu'
    plot_param_around_min(mu_array, nll_array, pb_config.TRUE_MU, name, suffix, directory)

def plot_nll_around_min(compute_nll, pb_config, directory, suffix):
    logger = logging.getLogger()
    logger.info('Plot NLL around minimum')
    plot_R_around_min(compute_nll, pb_config, directory, suffix)
    plot_LAMBDA_around_min(compute_nll, pb_config, directory, suffix)
    plot_MU_around_min(compute_nll, pb_config, directory, suffix)
