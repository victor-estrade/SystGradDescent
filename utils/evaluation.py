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
from visual.likelihood import plot_contour
from visual.neural_net import plot_losses
from visual.neural_net import plot_REG_log_mse

from config import _ERROR
from config import _TRUTH


def evaluate_config(config):
    table = []
    for i, test_config in enumerate(config.iter_test_config()):
        row = {"i" : i, "n_test_samples" : test_config.N_TESTING_SAMPLES }
        row["n_test_samples"] = test_config.N_TESTING_SAMPLES if test_config.N_TESTING_SAMPLES is not None else -1
        row.update({ 'true_'+k : v for k, v in test_config.TRUE.items()})
        table.append(row)
    table = pd.DataFrame(table)
    return table


def evaluate_classifier(model, X, y, w=None, prefix='test', suffix=''):
    logger = logging.getLogger()
    results = {}
    y_proba = model.predict_proba(X)
    y_decision = y_proba[:, 1]
    y_predict = model.predict(X)
    accuracy = np.mean(y_predict == y)

    results[f'{prefix}_accuracy{suffix}'] = accuracy
    logger.info('Plot distribution of the decision')
    fname = f'{prefix}_distrib{suffix}.png'
    os.makedirs(model.results_path, exist_ok=True)
    plot_test_distrib(y_proba, y, title=model.full_name, directory=model.results_path, fname=fname)

    logger.info('Plot ROC curve')
    fpr, tpr, thresholds = roc_curve(y, y_decision, pos_label=1)
    results[f"{prefix}_auc{suffix}"] = auc(fpr, tpr)
    fname = f'{prefix}_roc{suffix}.png'
    plot_ROC(fpr, tpr, title=model.full_name, directory=model.results_path, fname=fname)

    return results


def evaluate_summary_computer(model, X, y, w, n_bins=10, prefix='', suffix='', directory=None):
    logger = logging.getLogger()
    directory = model.results_path if directory is None else directory
    os.makedirs(directory, exist_ok=True)

    X_sig = X[y==1]
    w_sig = w[y==1]
    X_bkg = X[y==0]
    w_bkg = w[y==0]

    s_histogram = model.compute_summaries(X_sig, w_sig, n_bins)
    b_histogram = model.compute_summaries(X_bkg, w_bkg, n_bins)

    logger.info('Plot summaries')
    fname = f'{prefix}summaries{suffix}.png'
    plot_summaries(b_histogram, s_histogram,
                    title=model.full_name, directory=directory, fname=fname)


def evaluate_minuit(minimizer, params_truth, directory, do_hesse=True, suffix=''):
    results = {}
    estimate(minimizer, do_hesse=do_hesse)
    if not minimizer.valid :
        estimate_step_by_step(minimizer, do_hesse=do_hesse)
    params = minimizer.params
    fmin = minimizer.fmin
    print_params(params, params_truth)
    register_params(params, params_truth, results)
    results['is_mingrad_valid'] = minimizer.valid
    register_fmin(results, fmin)
    plot_contour(minimizer, params_truth, directory, suffix=suffix)
    return results


def estimate(minimizer, do_hesse=True):
    logger = logging.getLogger()

    if logger.getEffectiveLevel() <= logging.DEBUG:
        minimizer.print_param()
    _run_migrad(minimizer)

    if minimizer.valid:
        logger.info('Mingrad is VALID !')
    else:
        logger.warning('Mingrad IS NOT VALID !')
        _run_simplex_migrad(minimizer)
        if minimizer.valid:
            logger.info('Mingrad 2nd is  VALID !')
        else:
            logger.warning('Mingrad 2nd IS NOT VALID !')
    if do_hesse :
        _run_hesse(minimizer)


def estimate_step_by_step(minimizer, do_hesse=True):
    logger = logging.getLogger()
    logger.info("Param by param minimization ...")
    minimizer.fixed = True
    for i, param in enumerate(minimizer.params):
        logger.info(f"Unfixing {param.name}")
        minimizer.fixed[i] = False
        _run_simplex_migrad(minimizer)
    logger.info("Param by param minimization DONE")


def _run_migrad(minimizer):
    logger = logging.getLogger()
    logger.info('Mingrad()')
    minimizer.migrad()
    logger.info('Mingrad DONE')


def _run_simplex_migrad(minimizer):
    logger = logging.getLogger()
    logger.info('simplex()')
    minimizer.simplex()
    logger.info('simplex() DONE')
    logger.info('Mingrad() 2nd')
    minimizer.migrad()
    logger.info('Mingrad() 2nd DONE')


def _run_hesse(minimizer):
    logger = logging.getLogger()
    logger.info('Hesse()')
    try:
        params = minimizer.hesse()
        logger.info('Hesse DONE')
    except Exception as e:
        logger.error('Exception during Hesse computation : {}'.format(e))


def register_fmin(results, fmin):
    results["edm"] = fmin.edm
    results["edm_goal"] = fmin.edm_goal
    results["fval"] = fmin.fval
    results["has_parameters_at_limit"] = fmin.has_parameters_at_limit
    results["nfcn"] = fmin.nfcn
    results["ngrad"] = fmin.ngrad
    results["is_valid"] = fmin.is_valid
    results["has_valid_parameters"] = fmin.has_valid_parameters
    results["has_accurate_covar"] = fmin.has_accurate_covar
    results["has_posdef_covar"] = fmin.has_posdef_covar
    results["has_made_posdef_covar"] = fmin.has_made_posdef_covar
    results["hesse_failed"] = fmin.hesse_failed
    results["has_covariance"] = fmin.has_covariance
    results["is_above_max_edm"] = fmin.is_above_max_edm
    results["has_reached_call_limit"] = fmin.has_reached_call_limit
    results["errordef"] = fmin.errordef


def register_params(param, params_truth, measure_dict):
    for p in param:
        name  = p.name
        value = p.value
        error = p.error
        measure_dict[name] = value
        measure_dict[name+_ERROR] = error
        measure_dict[name+_TRUTH] = params_truth[name]


def evaluate_neural_net(model, prefix='', suffix=''):
    logger = logging.getLogger()
    logger.info('Plot losses')
    directory = model.results_path
    os.makedirs(directory, exist_ok=True)

    losses = model.get_losses()
    plot_losses(losses, title=model.full_name, directory=directory)
    plot_losses(losses, log=True, title=model.full_name, directory=directory, fname=f"log_losses.png")
    for loss_name, loss_values in losses.items():
        plot_losses({loss_name: loss_values}, title=model.full_name, directory=directory, fname=f"{loss_name}.png")
    results = {loss_name: loss_values[-1] for loss_name, loss_values in losses.items() if loss_values}

    return results



def evaluate_regressor(model, prefix='', suffix=''):
    os.makedirs(model.results_path, exist_ok=True)
    plot_REG_log_mse(model.mse_losses, title=model.full_name, directory=model.results_path)


def evaluate_inferno(model, prefix='test', suffix=''):
    pass


def evaluate_likelihood(prefix='test', suffix=''):
    pass



def evaluate_estimator(name, results, valid_only=False):
    eval_table = []
    if valid_only:
        results = results[results['is_valid']]
    for i, res in results.groupby('i'):
        values = res[name]
        errors = res[name+_ERROR]
        truth  = res[name+_TRUTH].iloc[0]

        row = dict(truth=truth
              ,target_mean = np.mean(values)
              ,target_std = np.std(values)
              ,target_variance = np.var(values)
              ,sigma_mean = np.mean(errors)
              ,sigma_std = np.std(errors)
              ,sigma_variance = np.var(errors)
              , i = i
              )
        row['target_bias'] = row['target_mean'] - truth
        row['sigma_bias'] = row['sigma_mean'] - row['target_std']
        row['target_mse'] = row['target_bias']**2 + row['target_variance']
        row['target_rmse'] = np.sqrt(row['target_mse'])
        row['sigma_mse'] = row['sigma_bias']**2 + row['sigma_variance']
        eval_table.append(row)
    eval_table = pd.DataFrame(eval_table)
    return eval_table


def evaluate_conditional_estimation(conditional_estimations, interest_param_name="mu"):
    mu_mean = conditional_estimations.groupby(['i', 'j'])[interest_param_name].mean()
    mu_var  = conditional_estimations.groupby(['i', 'j'])[interest_param_name].var()
    var_stat = mu_var.groupby('i').mean()
    var_syst = mu_mean.groupby('i').var()
    var_total = var_stat + var_syst
    evaluation = pd.concat([var_stat.to_frame(name='var_stat')
                            , var_syst.to_frame(name='var_syst')
                            , var_total.to_frame(name='var_total')]
                            , axis=1)
    return evaluation
