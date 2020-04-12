# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from .log import print_params

from visual.classifier import plot_test_distrib
from visual.classifier import plot_ROC
from visual.likelihood import plot_summaries

from config import _ERROR
from config import _TRUTH

def register_params(param, params_truth, measure_dict):
    for p, truth in zip(param, params_truth):
        name  = p['name']
        value = p['value']
        error = p['error']
        measure_dict[name] = value
        measure_dict[name+_ERROR] = error
        measure_dict[name+_TRUTH] = truth


def estimate(minimizer):
    import logging
    logger = logging.getLogger()

    if logger.getEffectiveLevel() <= logging.DEBUG:
        minimizer.print_param()
    logger.info('Mingrad()')
    fmin, params = minimizer.migrad()
    logger.info('Mingrad DONE')

    if minimizer.migrad_ok():
        logger.info('Mingrad is VALID !')
        logger.info('Hesse()')
        try:
            params = minimizer.hesse()
            logger.info('Hesse DONE')
        except Exception as e:
            logger.error('Exception during Hesse computation : {}'.format(e))
    else:
        logger.warning('Mingrad IS NOT VALID !')
    return fmin, params




def evaluate_classifier(model, X, y, w=None, prefix='test', suffix=''):
    logger = logging.getLogger()
    results = {}
    y_proba = model.predict_proba(X)
    y_decision = y_proba[:, 1]
    y_predict = model.predict(X)
    accuracy = np.mean(y_predict == y)

    results[f'{prefix}_accuracy'] = accuracy
    logger.info('Plot distribution of the decision')
    fname = f'{prefix}_distrib{suffix}.png'
    plot_test_distrib(y_proba, y, title=model.full_name, directory=model.path, fname=fname)

    logger.info('Plot ROC curve')
    fpr, tpr, thresholds = roc_curve(y, y_decision, pos_label=1)
    results[f"{prefix}_auc"] = auc(fpr, tpr)
    fname = f'{prefix}_roc{suffix}.png'
    plot_ROC(fpr, tpr, title=model.full_name, directory=model.path, fname=fname)

    return results

def evaluate_summary_computer(model, X_valid, y_valid, w_valid, X_test, w_test, n_bins=10, prefix='', suffix=''):
    logger = logging.getLogger()

    X_sig = X_valid[y_valid==1]
    w_sig = w_valid[y_valid==1]
    X_bkg = X_valid[y_valid==0]
    w_bkg = w_valid[y_valid==0]

    s_histogram = model.compute_summaries(X_sig, w_sig, n_bins)
    b_histogram = model.compute_summaries(X_bkg, w_bkg, n_bins)
    n_histogram = model.compute_summaries(X_test, w_test, n_bins)

    logger.info('Plot summaries')
    fname = f'{prefix}summaries{suffix}.png'
    plot_summaries(b_histogram, s_histogram, n_histogram, 
                    title=model.full_name, directory=model.path, fname=fname)

def evaluate_minuit(minimizer, fmin, params, params_truth):
    results = {}
    print_params(params, params_truth)
    register_params(params, params_truth, results)
    results['is_mingrad_valid'] = minimizer.migrad_ok()
    results.update(fmin)
    return results



def evaluate_neural_net(model, prefix='test', suffix=''):
    pass


def evaluate_regressor(model, prefix='test', suffix=''):
    pass


def evaluate_inferno(model, prefix='test', suffix=''):
    pass


def evaluate_likelihood(prefix='test', suffix=''):
    pass



def evaluate_estimator(name, results, valid_only=False):
    # TODO : evaluate mingrad's VALID only !
    truths = results[name+_TRUTH]
    eval_table = []
    for t in np.unique(truths):
        res = results[results[name+_TRUTH] == t]
        if valid_only:
            res = res[res['is_valid']]
        values = res[name]
        errors = res[name+_ERROR]
        row = evaluate_one_estimation(values, errors, t)
        eval_table.append(row)
    eval_table = pd.DataFrame(eval_table)
    return eval_table

def evaluate_one_estimation(values, errors, truth):
    row = dict(target_mean = np.mean(values)
          ,target_std = np.std(values)
          ,target_variance = np.var(values)
          ,sigma_mean = np.mean(errors)
          ,sigma_std = np.std(errors)
          ,sigma_variance = np.var(errors)
          )
    row['target_bias'] = row['target_mean'] - truth
    row['sigma_bias'] = row['sigma_mean'] - row['target_std']
    row['target_mse'] = row['target_bias']**2 + row['target_variance']
    row['target_rmse'] = np.sqrt(row['target_mse'])
    row['sigma_mse'] = row['sigma_bias']**2 + row['sigma_variance']
    row['truth'] = truth
    return row

