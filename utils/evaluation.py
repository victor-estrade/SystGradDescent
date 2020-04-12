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

from visual import plot_test_distrib
from visual import plot_ROC


_ERROR = '_error'
_TRUTH = '_truth'


def evaluate_classifier(model, X, y, w=None, prefix='test'):
    logger = logging.getLogger()
    results = {}
    y_proba = model.predict_proba(X)
    y_decision = y_proba[:, 1]
    y_predict = model.predict(X)
    accuracy = np.mean(y_predict == y)

    results[f'{prefix}_accuracy'] = accuracy
    logger.info('Plot distribution of the decision')
    plot_test_distrib(y_proba, y, title=model.full_name, directory=model.path, fname=f'{prefix}_distrib.png')

    logger.info('Plot ROC curve')
    fpr, tpr, thresholds = roc_curve(y, y_decision, pos_label=1)
    results[f"{prefix}_auc"] = auc(fpr, tpr)
    plot_ROC(fpr, tpr, title=model.full_name, directory=model.path, fname=f'{prefix}_roc.png')

    return results


def evaluate_neural_net(model):
    pass


def evaluate_regressor(model):
    pass


def evaluate_inferno(model):
    pass


def evaluate_likelihood():
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


