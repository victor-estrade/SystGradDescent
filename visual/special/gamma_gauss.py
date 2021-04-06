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
from ..misc import now_str

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


def plot_distrib(data, generator, true_params, expect_rescale, expect_mix,
                title="data distribution", directory=DEFAULT_DIR, fname='data_distrib.png'):
    logger = logging.getLogger()
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    true_proba = generator.proba_density(x_range, *true_params)
    infered_proba = generator.proba_density(x_range, expect_rescale, expect_mix)

    try:
        sns.distplot(data, label="data hist")
        plt.plot(x_range, true_proba, label="true proba")
        plt.plot(x_range, infered_proba, '--', label="infered proba")
        plt.title(now_str()+title)
        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()
        plt.savefig(os.path.join(directory, 'data_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning(f'Plot distrib failed')
        logger.warning(str(e))
