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



def plot_summaries(model, n_bins, extension,
                    X_valid, y_valid, w_valid,
                    X_test, w_test, classes=('b', 's', 'n') ):
    logger = logging.getLogger()
    X_sig = X_valid[y_valid==1]
    w_sig = w_valid[y_valid==1]
    X_bkg = X_valid[y_valid==0]
    w_bkg = w_valid[y_valid==0]

    s_histogram = model.compute_summaries(X_sig, w_sig, n_bins)
    b_histogram = model.compute_summaries(X_bkg, w_bkg, n_bins)
    n_histogram = model.compute_summaries(X_test, w_test, n_bins)

    try:
        x_ticks = np.arange(n_bins)
        plt.bar(x_ticks+0.1, b_histogram, width=0.3, label=classes[0])
        plt.bar(x_ticks+0.4, s_histogram, width=0.3, label=classes[1])
        plt.bar(x_ticks+0.7, n_histogram, width=0.3, label=classes[2])
        plt.xlabel("bins")
        plt.ylabel("summary_value")
        plt.xticks(x_ticks)
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'summaries{}.png'.format(extension)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))


def plot_param_around_min(param_array, nll_array, true_value, param_name, model):
    logger = logging.getLogger()
    try:
        plt.plot(param_array, nll_array, label='{} nll'.format(param_name))
        plt.axvline(x=true_value, c='red', label='true value')
        vmin = param_array[np.argmin(nll_array)]
        plt.axvline(x=vmin, c='orange', label='min')
        plt.xlabel(param_name)
        plt.ylabel('nll')
        plt.title('NLL around min')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'NLL_{}.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))

