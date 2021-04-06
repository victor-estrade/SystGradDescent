# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib.pyplot as plt

from config import DEFAULT_DIR
from .misc import now_str

def plot_summaries(b_histogram, s_histogram,
                    title='no title', directory=DEFAULT_DIR, fname='summaries.png',
                    classes=('b', 's'),):
    logger = logging.getLogger()
    n_bins = len(b_histogram)
    x_ticks = np.arange(n_bins)
    try:
        plt.bar(x_ticks, b_histogram, label=classes[0])
        plt.bar(x_ticks, s_histogram, bottom=b_histogram, label=classes[1])
        plt.xlabel("bins")
        plt.ylabel("summary_value")
        plt.xticks(x_ticks)
        plt.title(now_str()+title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname), bbox_inches="tight")
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))


def plot_param_around_min(param_array, nll_array, true_value, param_name, suffix='', directory=DEFAULT_DIR, title=None):
    logger = logging.getLogger()
    try:
        plt.plot(param_array, nll_array, label=f'NLL {param_name}')
        plt.axvline(x=true_value, c='red', label='true value')
        vmin = param_array[np.argmin(nll_array)]
        plt.axvline(x=vmin, c='orange', label='min')
        plt.xlabel(param_name)
        plt.ylabel('nll')
        title = f'{param_name} ({suffix[1:]}) NLL around minimum' if title is None else title
        plt.title(now_str()+title)
        plt.legend()
        plt.savefig(os.path.join(directory, f'NLL_{param_name}{suffix}.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


def plot_all_contour(minimizer, params_truth, directory, suffix=''):
    for i in range( len(minimizer.params) - 1):
        plot_contour(-1, i, minimizer, params_truth, directory, suffix=suffix)


def plot_contour(x, y, minimizer, params_truth, directory, suffix=''):
    logger = logging.getLogger()
    interest_param = minimizer.params[x]
    nuisance_param = minimizer.params[y]
    true_mu = params_truth[interest_param.name]
    true_nuisance = params_truth[nuisance_param.name]
    logger.info(f'Plot contour for {interest_param.name}-{nuisance_param.name}')
    try:
        minimizer.draw_contour(interest_param.name, nuisance_param.name)
        plt.scatter(true_mu, true_nuisance, label="True value")
        fname = f"contour_{interest_param.name}_{nuisance_param.name}_{suffix}.png"
        plt.title(f"{now_str()}NLL_contour {interest_param.name} - {nuisance_param.name}")
        plt.legend()
        plt.savefig(os.path.join(directory, fname), bbox_inches="tight")
        plt.clf()
    except Exception as e:
        logger.warning('Plot contour around min failed')
        logger.warning(str(e))
