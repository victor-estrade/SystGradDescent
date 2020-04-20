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
from visual.special.gamma_gauss import plot_distrib

from utils.log import set_logger
from utils.log import print_line
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig

from model.bayes import expectancy
from model.bayes import variance
from model.bayes import stat_uncertainty
from model.bayes import syst_uncertainty
from model.bayes import get_iter_prod


BENCHMARK_NAME = "GG"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Bayes")
N_ITER = 6

from config import _ERROR
from config import _TRUTH



def main():
    logger = set_logger()
    logger.info("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    args = None

    config = GGConfig()
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
    config = GGConfig()
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
    RESCALE_MIN = config.TRUE.rescale - 0.2
    RESCALE_MAX = config.TRUE.rescale + 0.2
    
    MIX_MIN = max(0, config.TRUE.mix - 0.1)
    MIX_MAX = min(1.0, config.TRUE.mix + 0.1)

    MIX_N_SAMPLES = 142
    RESCALE_N_SAMPLES = 145
    DATA_N_SAMPLES = 2000

    # Prior
    prior_rescale = stats.uniform(loc=RESCALE_MIN, scale=RESCALE_MAX-RESCALE_MIN)
    prior_mix     = stats.uniform(loc=MIX_MIN, scale=MIX_MAX-MIX_MIN)

    # Param grid
    rescale_grid   = np.linspace(RESCALE_MIN, RESCALE_MAX, RESCALE_N_SAMPLES)
    mix_grid       = np.linspace(MIX_MIN, MIX_MAX, MIX_N_SAMPLES)

    # Data Generator
    generator = Generator(seed)
    data, label = generator.sample_event(*config.TRUE, size=DATA_N_SAMPLES)
    debug_label(label)

    # Compute likelihood
    shape = (RESCALE_N_SAMPLES, MIX_N_SAMPLES)
    n_elements = np.prod(shape)
    logger.info(f"3D grid has {n_elements} elements")
    log_likelihood = np.zeros(shape)
    log_prior_proba = np.zeros(shape)
    for i, j in get_iter_prod(RESCALE_N_SAMPLES, MIX_N_SAMPLES, progress_bar=True):
        log_likelihood[i, j] = generator.log_proba_density(data, rescale_grid[i], mix_grid[j]).sum()
        log_prior_proba[i, j] = prior_rescale.logpdf(rescale_grid[i]) + prior_mix.logpdf(mix_grid[j])
    debug_log_proba(log_likelihood, log_prior_proba)

    # Normalization
    posterior_rescale_mix = softmax(log_likelihood + log_prior_proba)
    debug_posterior(posterior_rescale_mix)

    # Marginal posterior param proba
    marginal_rescale = posterior_rescale_mix.sum(axis=1)
    marginal_mix = posterior_rescale_mix.sum(axis=0)
    assert marginal_rescale.shape == rescale_grid.shape, "sum along the wrong axis for marginal rescale"
    assert marginal_mix.shape == mix_grid.shape, "sum along the wrong axis for marginal mix"
    debug_marginal(marginal_rescale, "rescale")
    debug_marginal(marginal_mix, "mix")

    # Conditional posterior 
    posterior_mix = np.divide(posterior_rescale_mix, marginal_rescale.reshape(RESCALE_N_SAMPLES, 1),
        out=np.zeros_like(posterior_rescale_mix), where=(posterior_rescale_mix!=0))

    # Minor check
    logger.debug("probability densities should sum to one")
    debug_proba_sum_one(posterior_mix*marginal_rescale.reshape(-1, 1))
    debug_proba_sum_one(posterior_rescale_mix)
    debug_proba_sum_one(marginal_rescale)
    debug_proba_sum_one(marginal_mix)

    # Compute estimator values
    sig_ratio = np.sum(label==1)/DATA_N_SAMPLES
    expect_mix = expectancy(mix_grid, marginal_mix)
    var_mix = variance(mix_grid, marginal_mix)
    std_mix = np.sqrt(var_mix)
    expect_rescale = expectancy(rescale_grid, marginal_rescale)
    var_rescale = variance(rescale_grid, marginal_rescale)
    std_rescale = np.sqrt(var_rescale)

    stat_err = stat_uncertainty(mix_grid, posterior_mix, marginal_rescale)
    syst_err = syst_uncertainty(mix_grid, posterior_mix, marginal_rescale)

    i_max, j_max = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    assert np.max(log_likelihood) == log_likelihood[i_max, j_max], "max and argmax should point to the same value"

    # Save estimator values
    results['mix'] = expect_mix
    results['mix'+_TRUTH] = config.TRUE.mix
    results['mix_std'] = std_mix
    results['mix'+_ERROR] = var_mix
    results['mix_stat'] = stat_err
    results['mix_syst'] = syst_err
    results['rescale'] = expect_rescale
    results['rescale'+_TRUTH] = config.TRUE.rescale
    results['rescale_std'] = std_rescale
    results['rescale'+_ERROR] = var_rescale

    # Log estimator values
    logger.info(f"True mix value    = {config.TRUE.mix}")
    logger.info(f"Sig ratio        = {sig_ratio}")
    logger.info(f"E[mix|x]          = {expect_mix}")
    logger.info(f"Var[mix|x]        = {var_mix}")
    logger.info(f"sqrt(Var[mix|x])  = {std_mix}")
    logger.info(f"stat_uncertainty = {stat_err}")
    logger.info(f"syst_uncertainty = {syst_err}")
    logger.info(f"Var - stat       = {var_mix - stat_err}")
    logger.info(f"argmax_mix p(mix|x) = {mix_grid[np.argmax(marginal_mix)]}")
    logger.info(f"argmax_rescale_mix logp(x|rescale, mix) = {rescale_grid[i_max]} {mix_grid[j_max]}")

    # Minor checks
    debug_min_max(marginal_mix, 'p(mix | x)')
    debug_min_max(marginal_rescale, 'p(rescale | x)')
    debug_min_max(posterior_mix, 'p(mix | x, rescale)')
    debug_min_max(posterior_rescale_mix, 'p(mix, rescale | x)')

    # Plots
    plot_infer(mix_grid, marginal_mix, expected_value=expect_mix,
                true_value=config.TRUE.mix, std=std_mix, name='mix',
                directory=directory, fname='marginal_mix.png')

    plot_infer(rescale_grid, marginal_rescale, expected_value=expect_rescale,
                true_value=config.TRUE.rescale, std=std_rescale, name='rescale',
                directory=directory, fname='marginal_rescale.png')

    plot_distrib(data, generator, config.TRUE, expect_rescale, expect_mix,
                title="data distribution", directory=directory, fname='data_distrib.png')

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

def debug_posterior(posterior_rescale_mix):
    logger = logging.getLogger()
    n_zeros = (posterior_rescale_mix == 0).sum()
    n_elements = np.prod(posterior_rescale_mix.shape)
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