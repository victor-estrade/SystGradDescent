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


def plot_test_distrib(model, model_name, model_path, X_test, y_test):
    logger = logging.getLogger()
    logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    proba = model.predict_proba(X_test)
    try:
        sns.distplot(proba[y_test==0, 1], label='b')
        sns.distplot(proba[y_test==1, 1], label='s')
        plt.title(model_name)
        plt.legend()
        plt.savefig(os.path.join(model_path, 'test_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_param_around_min(param_array, nll_array, true_value, param_name, model_path):
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
        plt.savefig(os.path.join(model_path, '{}_nll.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


def plot_params(param, params_truth, model_name, model_path):
    logger = logging.getLogger()
    params = [p['value'] for p in param]
    params_error = [p['error'] for p in param]
    params_names = [p['name'] for p in param]
    x = list(range(len(params)))
    try:
        plt.errorbar(x, params, yerr=params_error, fmt='o', capsize=20, capthick=2, label='infer')
        plt.scatter(x, params_truth, c='red', label='truth')
        plt.xticks(x, params_names)
        plt.yscale('log')
        plt.title(model_name)
        plt.legend()
        plt.savefig(os.path.join(model_path, 'params.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))
