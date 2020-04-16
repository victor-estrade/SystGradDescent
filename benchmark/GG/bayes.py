#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import os
import itertools
import logging
import numpy as np
import pandas as pd

from config import SAVING_DIR
# from config import SEED

from scipy.special import softmax
from scipy import stats

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from visual import set_plot_config
set_plot_config()
from visual.misc import plot_params

from utils.log import set_logger
from utils.log import print_line
from utils.evaluation import estimate
from utils.evaluation import evaluate_minuit
from utils.evaluation import evaluate_estimator
from utils.images import gather_images

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig
from problem.gamma_gauss import get_minimizer
from problem.gamma_gauss import Parameter

SEED = None
BENCHMARK_NAME = "GG"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Bayes")
N_ITER = 9

from config import _ERROR
from config import _TRUTH


def expectancy(values, probabilities, axis=None, keepdims=False):
    return np.sum(values * probabilities, axis=axis, keepdims=keepdims)

def variance(values, probabilities, axis=None):
    return np.sum(probabilities * np.square(values - expectancy(values, probabilities, axis=axis, keepdims=True)), axis=axis)

def variance_bis(values, probabilities, axis=None):
    return np.sum(values * values * probabilities, axis=axis) - np.square(expectancy(values, probabilities, axis=axis, keepdims=True))

def stat_uncertainty(values, posterior, marginal):
    v = variance(values.reshape(1, -1), posterior, axis=1)
    return expectancy(v.ravel(), marginal.ravel())

def syst_uncertainty(values, posterior, marginal):
    v = expectancy(values.reshape(1, -1), posterior, axis=1)
    return variance(v.ravel(), marginal.ravel())


def get_iter_prod(*sizes, progress_bar=False):
    generator = itertools.product(*(range(n) for n in sizes))
    if progress_bar:
        total = np.prod(sizes)
        return tqdm(generator, total=total)
    return generator


def main():
    config = GGConfig()
    results = [run(i, Parameter(rescale, mix)) for i, (rescale, mix) in enumerate(itertools.product(*config.RANGE))]
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(DIRECTORY, 'results.csv'))

        


def run(i_iter, true_params):
    directory = os.path.join(DIRECTORY, f'{i_iter}')
    os.makedirs(directory, exist_ok=True)
    RESCALE_MIN = true_params.rescale - 0.2
    RESCALE_MAX = true_params.rescale + 0.2
    
    MIX_MIN = max(0, true_params.mix - 0.1)
    MIX_MAX = min(1.0, true_params.mix + 0.1)

    MIX_N_SAMPLES = 142
    RESCALE_N_SAMPLES = 145
    DATA_N_SAMPLES = 10000
    TRUE = true_params

    results = {'i': i_iter}

    prior_rescale = stats.uniform(loc=RESCALE_MIN, scale=RESCALE_MAX-RESCALE_MIN)
    prior_mix     = stats.uniform(loc=MIX_MIN, scale=MIX_MAX-MIX_MIN)

    rescale_grid   = np.linspace(RESCALE_MIN, RESCALE_MAX, RESCALE_N_SAMPLES)
    mix_grid       = np.linspace(MIX_MIN, MIX_MAX, MIX_N_SAMPLES)

    generator = Generator(SEED)
    data, label = generator.sample_event(*TRUE, size=DATA_N_SAMPLES)
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    print(f"nb of signal      = {n_sig}")
    print(f"nb of backgrounds = {n_bkg}")

    shape = (RESCALE_N_SAMPLES, MIX_N_SAMPLES)
    n_elements = np.prod(shape)
    print(f"3D grid has {n_elements} elements")
    log_likelihood = np.zeros(shape)
    log_prior_proba = np.zeros(shape)
    for i, j in get_iter_prod(RESCALE_N_SAMPLES, MIX_N_SAMPLES, progress_bar=True):
        log_likelihood[i, j] = generator.log_proba_density(data, rescale_grid[i], mix_grid[j]).sum()
        log_prior_proba[i, j] = prior_rescale.logpdf(rescale_grid[i]) + prior_mix.logpdf(mix_grid[j])

    element_min = (log_likelihood + log_prior_proba).min()
    print("min logpdf = ", element_min)
    print("max logpdf = ", (log_likelihood + log_prior_proba).max())
    posterior_rescale_mix = softmax(log_likelihood + log_prior_proba)
    n_zeros = (posterior_rescale_mix == 0).sum()
    n_elements = np.prod(posterior_rescale_mix.shape)
    print()
    print(f"number of zeros in posterior = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    marginal_rescale = posterior_rescale_mix.sum(axis=1)
    marginal_mix = posterior_rescale_mix.sum(axis=0)
    assert marginal_rescale.shape == rescale_grid.shape, "sum along the wrong axis for marginal rescale"
    assert marginal_mix.shape == mix_grid.shape, "sum along the wrong axis for marginal mix"

    n_zeros = (marginal_rescale == 0).sum()
    n_elements = np.prod(marginal_rescale.shape)
    print(f"number of zeros in marginal rescale = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")
    n_zeros = (marginal_mix == 0).sum()
    n_elements = np.prod(marginal_mix.shape)
    print(f"number of zeros in marginal mix = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    posterior_mix = np.divide(posterior_rescale_mix, marginal_rescale.reshape(RESCALE_N_SAMPLES, 1),
        out=np.zeros_like(posterior_rescale_mix), where=(posterior_rescale_mix!=0))

    print("probability densities should sum to one")
    print(np.sum(posterior_mix*marginal_rescale.reshape(-1, 1)), np.sum(posterior_rescale_mix), np.sum(marginal_rescale), np.sum(marginal_mix))

    print()
    print("True mix value    =", TRUE.mix)
    results['mix'+_TRUTH] = TRUE.mix
    sig_ratio = n_sig/DATA_N_SAMPLES
    print("Sig ratio       =", sig_ratio)
    expect_mix = expectancy(mix_grid, marginal_mix)
    print("E[mix|x]          =", expect_mix)
    full_var = variance(mix_grid, marginal_mix)
    results['mix'+_ERROR] = full_var
    print("Var[mix|x]        =", full_var)
    std_mix = np.sqrt(full_var)
    results['mix_std'] = np.sqrt(std_mix)
    print("sqrt(Var[mix|x])  =", std_mix)
    print("argmax_mix p(mix|x) =", mix_grid[np.argmax(marginal_mix)])

    i_max, j_max = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    assert np.max(log_likelihood) == log_likelihood[i_max, j_max], "max and argmax should point to the same value"
    print("argmax_rescale_mix logp(x|rescale, mix) =", rescale_grid[i_max], mix_grid[j_max])
    stat_err = stat_uncertainty(mix_grid, posterior_mix, marginal_rescale)
    print("stat_uncertainty=", stat_err)
    syst_err = syst_uncertainty(mix_grid, posterior_mix, marginal_rescale)
    print("syst_uncertainty=", syst_err)
    print("Var - stat      =", full_var - stat_err)
    results['mix_stat'] = stat_err
    results['mix_syst'] = syst_err

    print()
    print("check marginals")
    print("mix    ", marginal_mix.min(), marginal_mix.max())
    print("rescale", marginal_rescale.min(), marginal_rescale.max())
    print("check posterior")
    print("p(y|x)  ", posterior_mix.min(), posterior_mix.max())
    print("p(y|x,a)", posterior_rescale_mix.min(), posterior_rescale_mix.max())


    # return None

    plt.axvline(TRUE.mix, c="orange", label="true mix")
    plt.axvline(TRUE.mix-std_mix, c="orange", label="true mix - std(mix)")
    plt.axvline(TRUE.mix+std_mix, c="orange", label="true mix + std(mix)")
    plt.axvline(sig_ratio, c="red", label="signal ratio")
    plt.axvline(expect_mix, c="green", label="E[mix|x]")
    plt.plot(mix_grid, marginal_mix, label="posterior")
    plt.xlabel("mix")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of mix vs mix values")
    plt.legend()
    plt.savefig(os.path.join(directory, 'marginal_mix.png'))
    plt.clf()


    plt.plot(rescale_grid, marginal_rescale, label="posterior")
    plt.axvline(TRUE.rescale, c="orange", label="true rescale")
    plt.xlabel("rescale")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of rescale vs rescale values")
    plt.legend()
    plt.savefig(os.path.join(directory, 'marginal_rescale.png'))
    plt.clf()

    sns.distplot(data, label="data hist")
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    p = generator.proba_density(x_range, *TRUE)
    plt.plot(x_range, p, label="true proba")
    p = generator.proba_density(x_range, expectancy(rescale_grid, marginal_rescale), expect_mix)
    plt.plot(x_range, p, '--', label="infered proba")
    plt.legend()
    plt.savefig(os.path.join(directory, 'data_distrib.png'))
    plt.clf()

    return results

if __name__ == '__main__':
    main()