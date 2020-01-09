# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_config():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("poster")

    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'


def plot_test_distrib(model, X_test, y_test):
    logger = logging.getLogger()
    logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    proba = model.predict_proba(X_test)
    try:
        sns.distplot(proba[y_test==0, 1], label='b')
        sns.distplot(proba[y_test==1, 1], label='s')
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'test_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_valid_distrib(model, X, y, classes=('b', 's')):
    logger = logging.getLogger()
    logger.info( 'Valid accuracy = {} %'.format(100 * model.score(X, y)) )
    proba = model.predict_proba(X)
    try:
        sns.distplot(proba[y==0, 1], label=classes[0])
        sns.distplot(proba[y==1, 1], label=classes[1])
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'valid_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot valid distrib failed')
        logger.warning(str(e))


def plot_REG_losses(model):
    logger = logging.getLogger()
    losses = model.losses
    mse_losses = model.mse_losses
    try:
        plt.plot(mse_losses, label='mse')
        plt.plot(losses, label='loss')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'losses.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot REG losses failed')
        logger.warning(str(e))


def plot_REG_log_mse(model):
    logger = logging.getLogger()
    mse_losses = model.mse_losses
    try:
        plt.plot(mse_losses, label='mse')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'log_mse_loss.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot REG losses failed')
        logger.warning(str(e))


def plot_summaries(summary_computer, model, 
                    X_valid, y_valid, w_valid,
                    X_test, w_test, classes=('b', 's', 'n') ):
    logger = logging.getLogger()
    X_sig = X_valid[y_valid==1]
    w_sig = w_valid[y_valid==1]
    X_bkg = X_valid[y_valid==0]
    w_bkg = w_valid[y_valid==0]

    s_histogram = summary_computer(X_sig, w_sig)
    b_histogram = summary_computer(X_bkg, w_bkg)
    n_histogram = summary_computer(X_test, w_test)
    n_summaries = len(n_histogram)

    try:
        plt.bar(np.arange(n_summaries)+0.1, b_histogram, width=0.3, label=classes[0])
        plt.bar(np.arange(n_summaries)+0.4, s_histogram, width=0.3, label=classes[1])
        plt.bar(np.arange(n_summaries)+0.7, n_histogram, width=0.3, label=classes[2])
        plt.xlabel("bins")
        plt.ylabel("summary_value")
        plt.xticks(list(range(n_summaries)))
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'summaries.png'))
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
        plt.savefig(os.path.join(model.path, '{}_nll.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot nll around min failed')
        logger.warning(str(e))


def plot_params(param, params_truth, model, param_max=None, param_min=None):
    logger = logging.getLogger()
    params = [p['value'] for p in param]
    params_error = [p['error'] for p in param]
    params_names = [p['name'] for p in param]
    x = list(range(len(params)))
    try:
        plt.errorbar(x, params, yerr=params_error, fmt='o', capsize=20, capthick=2, label='infer')
        plt.scatter(x, params_truth, c='red', label='truth')
        plt.xticks(x, params_names)
        # plt.yscale('log')
        plt.ylim(param_min, max([param_max] + [p+p_err+0.1 for p, p_err in zip(params, params_error)]) )
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'params.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot summaries failed')
        logger.warning(str(e))
