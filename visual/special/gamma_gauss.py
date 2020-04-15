# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import numpy as np

from ..likelihood import plot_param_around_min

def plot_RESCALE_around_min(compute_nll, true_params, directory, suffix):
    rescale_array = np.linspace(0.5, 3, 50)
    nll_array = [compute_nll(rescale, true_params.mix) for rescale in rescale_array]
    name = 'rescale'
    plot_param_around_min(rescale_array, nll_array, true_params.rescale, name, suffix, directory)

def plot_MIX_around_min(compute_nll, true_params, directory, suffix):
    mix_array = np.linspace(0.0, 1.0, 50)
    nll_array = [compute_nll(true_params.rescale, mix) for mix in mix_array]
    name = 'mix'
    plot_param_around_min(mix_array, nll_array, true_params.mix, name, suffix, directory)

def plot_nll_around_min(compute_nll, true_params, directory, suffix):
    logger = logging.getLogger()
    logger.info('Plot NLL around minimum')
    plot_RESCALE_around_min(compute_nll, true_params, directory, suffix)
    plot_MIX_around_min(compute_nll, true_params, directory, suffix)
