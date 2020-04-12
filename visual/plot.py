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

from config import DEFAULT_DIR

from .misc import _ERROR
from .misc import _TRUTH


def set_plot_config():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("poster")

    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'
    mpl.rcParams['lines.markersize'] = np.sqrt(20)


def old_plot_test_distrib(model, X_test, y_test):
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


def plot_test_distrib(y_proba, y_test, title="no title", 
                      directory=DEFAULT_DIR, fname='test_distrib.png', classes=('b', 's')):
    logger = logging.getLogger()
    # logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    try:
        sns.distplot(y_proba[y_test==0, 1], label=classes[0])
        sns.distplot(y_proba[y_test==1, 1], label=classes[1])
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
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
        plt.xlabel('classifier score')
        plt.ylabel('density')
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'valid_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot valid distrib failed')
        logger.warning(str(e))


def plot_ROC(fpr, tpr, title="no title", directory=DEFAULT_DIR, fname='roc.png'):
    from sklearn.metrics import auc
    logger = logging.getLogger()
    try:
        plt.plot(fpr, tpr, label='AUC {}'.format(auc(fpr, tpr)))
        plt.title(title)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot ROC failed')
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


def plot_INFERNO_losses(model):
    logger = logging.getLogger()
    losses = model.loss_hook.losses
    try:
        plt.plot(losses, label='loss')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'losses.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot INFERNO losses failed')
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
        logger.warning('Plot REG log losses failed')
        logger.warning(str(e))


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


def plot_params(param_name, result_table, model):
    logger = logging.getLogger()
    values = result_table[param_name]
    errors = result_table[param_name+_ERROR]
    truths = result_table[param_name+_TRUTH]
    xx = np.arange(len(values))
    if 'is_valid' in result_table:
        valid_values = values[result_table['is_valid']]
        valid_errors = errors[result_table['is_valid']]
        valid_x = xx[result_table['is_valid']]
        logger.debug("Plot_params valid lenght = {}, {}, {}".format(len(valid_x), len(valid_values), len(valid_errors)))
        values =  values[result_table['is_valid'] == False]
        errors =  errors[result_table['is_valid'] == False]
        x = xx[result_table['is_valid'] == False]
        logger.debug('Plot_params invalid lenght = {}, {}, {}'.format(len(x), len(values), len(errors)))
    try:
        if 'is_valid' in result_table:
            plt.errorbar(valid_x, valid_values, yerr=valid_errors, fmt='o', capsize=20, capthick=2, label='valid_infer')
            plt.errorbar(x, values, yerr=errors, fmt='o', capsize=20, capthick=2, label='invalid_infer')
        else:
            plt.errorbar(xx, values, yerr=errors, fmt='o', capsize=20, capthick=2, label='infer')
        plt.scatter(xx, truths, c='red', label='truth')
        plt.xticks(xx, map(lambda x: round(x, 3), truths))
        plt.xlabel('truth value')
        plt.ylabel(param_name)
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'estimate_{}.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot params failed')
        logger.warning(str(e))
