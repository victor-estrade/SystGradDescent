# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from config import DEFAULT_DIR



def plot_summaries(b_histogram, s_histogram, n_histogram, 
                    title='no title', directory=DEFAULT_DIR, fname='summaries.png', 
                    classes=('b', 's', 'n'),):
    logger = logging.getLogger()
    n_bins = len(b_histogram)
    x_ticks = np.arange(n_bins)
    try:
        plt.bar(x_ticks+0.1, b_histogram, width=0.3, label=classes[0])
        plt.bar(x_ticks+0.4, s_histogram, width=0.3, label=classes[1])
        plt.bar(x_ticks+0.7, n_histogram, width=0.3, label=classes[2])
        plt.xlabel("bins")
        plt.ylabel("summary_value")
        plt.xticks(x_ticks)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))


def plot_param_around_min(param_array, nll_array, true_value, param_name, suffix='', directory=DEFAULT_DIR):
    logger = logging.getLogger()
    try:
        plt.plot(param_array, nll_array, label=f'NLL {param_name}')
        plt.axvline(x=true_value, c='red', label='true value')
        vmin = param_array[np.argmin(nll_array)]
        plt.axvline(x=vmin, c='orange', label='min')
        plt.xlabel(param_name)
        plt.ylabel('nll')
        plt.title(f'{param_name} ({suffix[1:]}) NLL around minimum')
        plt.legend()
        plt.savefig(os.path.join(directory, f'NLL_{param_name}{suffix}.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))

