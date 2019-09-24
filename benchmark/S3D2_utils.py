# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from myplot import plot_param_around_min

def plot_R_around_min(compute_nll, pb_config, model_path):
    r_array = np.linspace(-1, 1, 50)
    nll_array = [compute_nll(r, pb_config.TRUE_LAMBDA, pb_config.TRUE_MU) for r in r_array]
    plot_param_around_min(r_array, nll_array, pb_config.TRUE_R, 'r', model_path)


def plot_LAMBDA_around_min(compute_nll, pb_config, model_path):
    lam_array = np.linspace(0, 4, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, lam, pb_config.TRUE_MU) for lam in lam_array]
    plot_param_around_min(lam_array, nll_array, pb_config.TRUE_LAMBDA, 'lambda', model_path)


def plot_MU_around_min(compute_nll, pb_config, model_path):
    mu_array = np.linspace(0.0, 1.0, 50)
    nll_array = [compute_nll(pb_config.TRUE_R, pb_config.TRUE_LAMBDA, mu) for mu in mu_array]
    plot_param_around_min(mu_array, nll_array, pb_config.TRUE_MU, 'mu', model_path)

