#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import os
import logging
import numpy as np
import pandas as pd

from config import SAVING_DIR
from config import SEED

from scipy.special import softmax
from scipy import stats

from visual import set_plot_config
set_plot_config()
from visual.misc import plot_params
from visual.proba import plot_infer

from utils.log import set_logger
from utils.log import print_line
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.synthetic3D import Generator
from problem.synthetic3D import S3D2Config

from model.bayes import expectancy
from model.bayes import variance
from model.bayes import stat_uncertainty
from model.bayes import syst_uncertainty
from model.bayes import get_iter_prod


BENCHMARK_NAME = "S3D2"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Bayes")
N_ITER = 3

from config import _ERROR
from config import _TRUTH


def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = S3D2Config()
    results = [run(args, i_cv) for i_cv in range(N_ITER)]
    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    # EVALUATION
    eval_table = evaluate_estimator(config.TRUE.interest_parameters_names, results)
    print_line()
    print_line()
    print(eval_table)
    print_line()
    print_line()
    eval_table.to_csv(os.path.join(DIRECTORY, 'evaluation.csv'))
    gather_images(DIRECTORY)

def run(args, i_cv):
    logger = logging.getLogger()
    config = S3D2Config()
    print_line()
    logger.info('Running CV iter n°{}'.format(i_cv))
    print_line()
    directory = os.path.join(DIRECTORY, f'cv_{i_cv}')
    os.makedirs(directory, exist_ok=True)

    seed = SEED + i_cv * 5
    test_seed = seed + 2
    result_table = [run_iter(i_cv, i, test_config, test_seed, directory) for i, test_config in enumerate(config.iter_test_config())]
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(directory, 'results.csv'))
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Bayes fit', directory=directory)
    return result_table


def run_iter(i_cv, i_iter, config, seed, directory):
    # Init
    logger = logging.getLogger()
    print_line()
    logger.info('running iter n°{}'.format(i_iter))
    directory = os.path.join(directory, f'iter_{i_iter}')
    os.makedirs(directory, exist_ok=True)
    results = dict(i_cv=i_cv, i=i_iter)

    # Config
    # DATA_N_SAMPLES = config.N_TESTING_SAMPLES
    DATA_N_SAMPLES = 9000

    R_MIN   = config.TRUE.r - 0.3
    R_MAX   = config.TRUE.r + 0.3 
    LAM_MIN = config.TRUE.lam - 1
    LAM_MAX = config.TRUE.lam + 1
    MU_MIN  = max(0, config.TRUE.mu - 0.1)
    MU_MAX  = min(1.0, config.TRUE.mu + 0.1)

    R_N_SAMPLES = 21
    LAM_N_SAMPLES = 22
    MU_N_SAMPLES = 23

    # Prior
    prior_r   = stats.uniform(loc=R_MIN, scale=R_MAX-R_MIN)
    prior_lam = stats.uniform(loc=LAM_MIN, scale=LAM_MAX-LAM_MIN)
    prior_mu  = stats.uniform(loc=MU_MIN, scale=MU_MAX-MU_MIN)

    # Param grid
    r_grid   = np.linspace(R_MIN, R_MAX, R_N_SAMPLES)
    lam_grid = np.linspace(LAM_MIN, LAM_MAX, LAM_N_SAMPLES)
    mu_grid  = np.linspace(MU_MIN, MU_MAX, MU_N_SAMPLES)

    # Data Generator
    generator = Generator(seed)
    data, label = generator.sample_event(*config.TRUE, size=DATA_N_SAMPLES)
    debug_label(label)

    # Compute likelihood
    shape = (R_N_SAMPLES, LAM_N_SAMPLES, MU_N_SAMPLES)
    n_elements = np.prod(shape)
    logger.info(f"3D grid has {n_elements} elements")
    log_likelihood = np.zeros(shape)
    log_prior_proba = np.zeros(shape)
    for i, j, k in get_iter_prod(R_N_SAMPLES, LAM_N_SAMPLES, MU_N_SAMPLES, progress_bar=True):
        log_likelihood[i, j, k] = generator.log_proba_density(data, r_grid[i], lam_grid[j], mu_grid[k]).sum()
        log_prior_proba[i, j, k] = prior_r.logpdf(r_grid[i]) \
                                    + prior_lam.logpdf(lam_grid[j]) \
                                    + prior_mu.logpdf(mu_grid[k])
    debug_log_proba(log_likelihood, log_prior_proba)

    # Normalization
    posterior_r_lam_mu = softmax(log_likelihood + log_prior_proba)
    debug_posterior(posterior_r_lam_mu)

    # Marginal posterior param proba
    marginal_r = posterior_r_lam_mu.sum(axis=2).sum(axis=1)
    marginal_lam = posterior_r_lam_mu.sum(axis=2).sum(axis=0)
    marginal_mu = posterior_r_lam_mu.sum(axis=1).sum(axis=0)
    marginal_r_lam = posterior_r_lam_mu.sum(axis=2)
    assert marginal_r.shape == r_grid.shape, "sum along the wrong axis for marginal r"
    assert marginal_lam.shape == lam_grid.shape, "sum along the wrong axis for marginal lam"
    assert marginal_mu.shape == mu_grid.shape, "sum along the wrong axis for marginal mu"
    assert marginal_r_lam.shape == (R_N_SAMPLES, LAM_N_SAMPLES), "sum along the wrong axis for marginal (r, lam)"
    debug_marginal(marginal_r, "r")
    debug_marginal(marginal_lam, "lam")
    debug_marginal(marginal_mu, "mu")
    debug_marginal(marginal_r_lam, "r_lam")

    # Conditional posterior 
    posterior_mu = np.divide(posterior_r_lam_mu, marginal_r_lam.reshape(R_N_SAMPLES, LAM_N_SAMPLES, 1),
        out=np.zeros_like(posterior_r_lam_mu), where=(posterior_r_lam_mu!=0))

    # Minor check
    logger.debug("probability densities should sum to one")
    debug_proba_sum_one(posterior_mu * marginal_r_lam.reshape(R_N_SAMPLES, LAM_N_SAMPLES, 1))
    debug_proba_sum_one(posterior_r_lam_mu)
    debug_proba_sum_one(marginal_r)
    debug_proba_sum_one(marginal_mu)

    # Compute estimator values
    sig_ratio = np.sum(label==1)/DATA_N_SAMPLES
    expect_mu = expectancy(mu_grid, marginal_mu)
    var_mu = variance(mu_grid, marginal_mu)
    std_mu = np.sqrt(var_mu)
    expect_r = expectancy(r_grid, marginal_r)
    var_r = variance(r_grid, marginal_r)
    std_r = np.sqrt(var_r)
    expect_lam = expectancy(lam_grid, marginal_lam)
    var_lam = variance(lam_grid, marginal_lam)
    std_lam = np.sqrt(var_lam)

    stat_err = stat_uncertainty(mu_grid, posterior_mu, marginal_r_lam, reshape=(1, 1, -1))
    syst_err = syst_uncertainty(mu_grid, posterior_mu, marginal_r_lam, reshape=(1, 1, -1))

    i_max, j_max, k_max = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    assert np.max(log_likelihood) == log_likelihood[i_max, j_max, k_max], "max and argmax should point to the same value"

    # Save estimator values
    results['mu'] = expect_mu
    results['mu'+_TRUTH] = config.TRUE.mu
    results['mu_std'] = std_mu
    results['mu'+_ERROR] = var_mu
    results['mu_stat'] = stat_err
    results['mu_syst'] = syst_err
    results['r'] = expect_r
    results['r'+_TRUTH] = config.TRUE.r
    results['r_std'] = std_r
    results['r'+_ERROR] = var_r
    results['lam'] = expect_lam
    results['lam'+_TRUTH] = config.TRUE.lam
    results['lam_std'] = std_lam
    results['lam'+_ERROR] = var_lam

    # Log estimator values
    logger.info(f"True mu value    = {config.TRUE.mu}")
    logger.info(f"Sig ratio        = {sig_ratio}")
    logger.info(f"E[mu|x]          = {expect_mu}")
    logger.info(f"Var[mu|x]        = {var_mu}")
    logger.info(f"sqrt(Var[mu|x])  = {std_mu}")
    logger.info(f"stat_uncertainty = {stat_err}")
    logger.info(f"syst_uncertainty = {syst_err}")
    logger.info(f"Var - stat       = {var_mu - stat_err}")
    logger.info(f"argmax_mu p(mu|x) = {mu_grid[np.argmax(marginal_mu)]}")
    logger.info(f"argmax_r_mu logp(x|r, mu) = {r_grid[i_max]} {mu_grid[j_max]}")

    # Minor checks
    debug_min_max(marginal_mu, 'p(mu | x)')
    debug_min_max(marginal_lam, 'p(lam | x)')
    debug_min_max(marginal_r, 'p(r | x)')
    debug_min_max(posterior_mu, 'p(mu | x, r)')
    debug_min_max(posterior_r_lam_mu, 'p(mu, r | x)')

    # Plots
    plot_infer(mu_grid, marginal_mu, expected_value=expect_mu,
                true_value=config.TRUE.mu, std=std_mu, name='mu',
                directory=directory, fname='marginal_mu.png')

    plot_infer(r_grid, marginal_r, expected_value=expect_r,
                true_value=config.TRUE.r, std=std_r, name='r',
                directory=directory, fname='marginal_r.png')

    plot_infer(lam_grid, marginal_lam, expected_value=expect_lam,
                true_value=config.TRUE.lam, std=std_lam, name='lam',
                directory=directory, fname='marginal_lam.png')

    # plot_distrib(data, generator, config.TRUE, expect_r, expect_mu,
    #             title="data distribution", directory=directory, fname='data_distrib.png')

    return results


def debug_label(label):
    logger = logging.getLogger()
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    logger.debug(f"nb of signal      = {n_sig}")
    logger.debug(f"nb of backgrounds = {n_bkg}")

def debug_log_proba(log_likelihood, log_prior_proba):
    logger = logging.getLogger()
    log_pdf = log_likelihood + log_prior_proba
    logger.debug(f"min logpdf = {log_pdf.min()}")
    logger.debug(f"max logpdf = {log_pdf.max()}")

def debug_posterior(posterior_r_lam_mu):
    logger = logging.getLogger()
    n_zeros = (posterior_r_lam_mu == 0).sum()
    n_elements = np.prod(posterior_r_lam_mu.shape)
    logger.info(f"number of zeros in posterior = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

def debug_marginal(marginal, name):
    logger = logging.getLogger()
    n_zeros = (marginal == 0).sum()
    n_elements = np.prod(marginal.shape)
    logger.info(f"number of zeros in marginal {name} = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

def debug_proba_sum_one(proba):
    logger = logging.getLogger()
    logger.debug(f" 1.0 =?= {proba.sum()}")


def debug_min_max(proba, name):
    logger = logging.getLogger()
    logger.debug(f"{name} in [{proba.min()}, {proba.max()}]")


if __name__ == '__main__':
    main()