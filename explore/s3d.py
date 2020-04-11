#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.special import softmax
from scipy import stats

from tqdm import tqdm

from utils.plot import set_plot_config

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import get_minimizer

from utils.misc import estimate
from utils.misc import register_params
# from utils.log import print_params



SEED = None
DIRECTORY = "/home/estrade/Bureau/PhD/SystML/SystGradDescent/savings/S3D2/Likelihood"


def expectancy(values, probabilities, axis=None, keepdims=False):
    return np.sum(values * probabilities, axis=axis, keepdims=keepdims)

def variance(values, probabilities, axis=None):
    return np.sum(probabilities * np.square(values - expectancy(values, probabilities, axis=axis, keepdims=True)), axis=axis)

def variance_bis(values, probabilities, axis=None):
    return np.sum(values * values * probabilities, axis=axis) - np.square(expectancy(values, probabilities, axis=axis, keepdims=True))


def stat_uncertainty(values, posterior, marginal):
    return sum([variance(values, posterior[i, j, :]) * marginal[i, j] 
                    for i, j in itertools.product(range(marginal.shape[0]), range(marginal.shape[1]))])

def stat_uncertainty2(values, posterior, marginal):
    v = np.array([variance(values, posterior[i, j, :]) 
        for i, j in itertools.product(range(posterior.shape[0]), range(posterior.shape[1]))])
    return expectancy(v.ravel(), marginal.ravel())

def stat_uncertainty3(values, posterior, marginal):
    v = variance(values.reshape(1, 1, -1), posterior, axis=2)
    return expectancy(v.ravel(), marginal.ravel())

def syst_uncertainty(values, posterior, marginal, marginal_posterior):
    E_y_x = expectancy(values, marginal_posterior)
    return sum([np.square(expectancy(values, posterior[i, j, :]) - E_y_x) * marginal[i, j] 
                    for i, j in itertools.product(range(marginal.shape[0]), range(marginal.shape[1]))])

def syst_uncertainty2(values, posterior, marginal):
    v = np.array([expectancy(values, posterior[i, j, :]) 
        for i, j in itertools.product(range(posterior.shape[0]), range(posterior.shape[1]))])
    return variance(v.ravel(), marginal.ravel())

def syst_uncertainty3(values, posterior, marginal):
    v = expectancy(values.reshape(1, 1, -1), posterior, axis=2)
    return variance(v.ravel(), marginal.ravel())



def explore():
    print("Hello master !")
    set_plot_config()
    config = S3D2Config()
    N_SAMPLES = 10_000
    R_MIN   = -0.3
    R_MAX   = 0.3 
    LAM_MIN = 2
    LAM_MAX = 4
    MU_MIN  = 0.0
    MU_MAX  = 1.0 

    generator = S3D2(SEED)
    X, label = generator.sample_event(config.TRUE.r, config.TRUE.lam, config.TRUE.mu, size=N_SAMPLES)
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    print(f"nb of signal      = {n_sig}")
    print(f"nb of backgrounds = {n_bkg}")


    df = pd.DataFrame(X, columns=["x1","x2","x3"])
    df['label'] = label
    g = sns.PairGrid(df, vars=["x1","x2","x3"], hue='label')
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(sns.kdeplot)
    g = g.map_lower(sns.kdeplot, n_levels=6)
    g = g.add_legend()
    # g = g.map_offdiag(sns.kdeplot, n_levels=6)
    g.savefig(os.path.join(DIRECTORY, 'pairgrid.png'))
    plt.clf()


    nll = generator.nll(X, config.TRUE.r, config.TRUE.lam, config.TRUE.mu)
    print(f"NLL = {nll}")

    R_RANGE = np.linspace(R_MIN, R_MAX, 30)
    nll = [generator.nll(X, r, config.TRUE.lam, config.TRUE.mu) for r in R_RANGE]
    min_nll = R_RANGE[np.argmin(nll)]
    plt.plot(R_RANGE, nll, label="nll(r)")
    plt.axvline(config.TRUE.r, c="orange", label="true r")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("r")
    plt.ylabel("NLL")
    plt.title("NLL according to r param")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_r.png'))
    plt.clf()


    LAM_RANGE = np.linspace(LAM_MIN, LAM_MAX, 30)
    nll = [generator.nll(X, config.TRUE.r, lam, config.TRUE.mu) for lam in LAM_RANGE]
    min_nll = LAM_RANGE[np.argmin(nll)]
    plt.plot(LAM_RANGE, nll, label="nll(lam)")
    plt.axvline(config.TRUE.lam, c="orange", label="true lam")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\lambda$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\lambda$ param")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_lambda.png'))
    plt.clf()

    MU_RANGE = np.linspace(MU_MIN, MU_MAX, 30)
    nll = [generator.nll(X, config.TRUE.r, config.TRUE.lam, mu) for mu in MU_RANGE]
    min_nll = MU_RANGE[np.argmin(nll)]
    plt.plot(MU_RANGE, nll, label="nll(mu)")
    plt.axvline(config.TRUE.mu, c="orange", label="true mu")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\mu$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\mu$ param")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_mu.png'))
    plt.clf()


def main():
    print("Hello world !")
    set_plot_config()
    config = S3D2Config()
    DATA_N_SAMPLES = 8_000
    R_MIN   = -0.3
    R_MAX   = 0.3 
    LAM_MIN = 2
    LAM_MAX = 4
    MU_MIN  = 0.1
    MU_MAX  = 0.3 

    R_N_SAMPLES = 101
    LAM_N_SAMPLES = 102
    MU_N_SAMPLES = 103

    prior_r   = stats.uniform(loc=R_MIN, scale=R_MAX-R_MIN)
    prior_lam = stats.uniform(loc=LAM_MIN, scale=LAM_MAX-LAM_MIN)
    prior_mu  = stats.uniform(loc=MU_MIN, scale=MU_MAX-MU_MIN)

    r_grid   = np.linspace(R_MIN, R_MAX, R_N_SAMPLES)
    lam_grid = np.linspace(LAM_MIN, LAM_MAX, LAM_N_SAMPLES)
    mu_grid  = np.linspace(MU_MIN, MU_MAX, MU_N_SAMPLES)

    data_generator = S3D2(SEED)
    data, label = data_generator.sample_event(config.TRUE.r, config.TRUE.lam, config.TRUE.mu, size=DATA_N_SAMPLES)
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    print(f"nb of signal      = {n_sig}")
    print(f"nb of backgrounds = {n_bkg}")


    shape = (R_N_SAMPLES, LAM_N_SAMPLES, MU_N_SAMPLES)
    n_elements = np.prod(shape)
    print(f"3D grid has {n_elements} elements")
    log_likelihood = np.zeros(shape)
    log_prior_proba = np.zeros(shape)
    for i, j, k in tqdm(itertools.product(range(R_N_SAMPLES), range(LAM_N_SAMPLES), range(MU_N_SAMPLES)), total=n_elements):
        log_likelihood[i, j, k] = data_generator.log_proba_density(data, r_grid[i], lam_grid[j], mu_grid[k]).sum()
        log_prior_proba[i, j, k] = prior_r.logpdf(r_grid[i]) \
                                    + prior_lam.logpdf(lam_grid[j]) \
                                    + prior_mu.logpdf(mu_grid[k])

    element_min = (log_likelihood + log_prior_proba).min()
    print("min element = ", element_min)
    posterior_r_lam_mu = softmax(log_likelihood + log_prior_proba)
    n_zeros = (posterior_r_lam_mu == 0).sum()
    n_elements = np.prod(posterior_r_lam_mu.shape)
    print()
    print(f"number of zeros in posterior = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    marginal_r = posterior_r_lam_mu.sum(axis=2).sum(axis=1)
    marginal_lam = posterior_r_lam_mu.sum(axis=2).sum(axis=0)
    marginal_mu = posterior_r_lam_mu.sum(axis=1).sum(axis=0)
    marginal_r_lam = posterior_r_lam_mu.sum(axis=2)
    assert marginal_r.shape == r_grid.shape, "sum along the wrong axis for marginal r"
    assert marginal_lam.shape == lam_grid.shape, "sum along the wrong axis for marginal lam"
    assert marginal_mu.shape == mu_grid.shape, "sum along the wrong axis for marginal mu"
    assert marginal_r_lam.shape == (R_N_SAMPLES, LAM_N_SAMPLES), "sum along the wrong axis for marginal (r, lam)"

    n_zeros = (marginal_r == 0).sum()
    n_elements = np.prod(marginal_r.shape)
    print(f"number of zeros in marginal r = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")
    n_zeros = (marginal_lam == 0).sum()
    n_elements = np.prod(marginal_lam.shape)
    print(f"number of zeros in marginal lam = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")
    n_zeros = (marginal_mu == 0).sum()
    n_elements = np.prod(marginal_mu.shape)
    print(f"number of zeros in marginal mu = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")
    n_zeros = (marginal_r_lam == 0).sum()
    n_elements = np.prod(marginal_r_lam.shape)
    print(f"number of zeros in marginal r_lam = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    posterior_mu = np.divide(posterior_r_lam_mu, marginal_r_lam.reshape(R_N_SAMPLES, LAM_N_SAMPLES, 1),
        out=np.zeros_like(posterior_r_lam_mu), where=(posterior_r_lam_mu!=0))

    print("probability densities should sum to one")
    # TODO : posterior_mu sum to SOME_N_SAMPLES. is it ok ?
    # TODO : with new division policy posterior_mu/ALPHA_N sums to 1-zero_ration in marginal_y
    #        ... It does not look good
    print(np.sum(posterior_mu)/n_elements, np.sum(posterior_r_lam_mu), np.sum(marginal_r), np.sum(marginal_lam))
    print(np.sum(marginal_r_lam))

    print()
    print("True mu value    =", config.TRUE.mu)
    sig_ratio = n_sig/DATA_N_SAMPLES
    print("Sig ratio       =", sig_ratio)
    expect_mu = expectancy(mu_grid, marginal_mu)
    print("E[mu|x]          =", expect_mu)
    full_var = variance(mu_grid, marginal_mu)
    print("Var[mu|x]        =", full_var)
    std_mu = np.sqrt(full_var)
    print("sqrt(Var[mu|x])  =", std_mu)
    print("argmax_mu p(mu|x) =", mu_grid[np.argmax(marginal_mu)])

    i_max, j_max, k_max = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    assert np.max(log_likelihood) == log_likelihood[i_max, j_max, k_max], "max and argmax should point to the same value"
    print("argmax_r_lam_mu logp(x|r, lam, mu) =", r_grid[i_max], lam_grid[j_max], mu_grid[k_max])
    stat_err = stat_uncertainty(mu_grid, posterior_mu, marginal_r_lam)
    print("stat_uncertainty=", stat_err)
    stat_err = stat_uncertainty2(mu_grid, posterior_mu, marginal_r_lam)
    print("stat_uncertainty=", stat_err)
    stat_err = stat_uncertainty3(mu_grid, posterior_mu, marginal_r_lam)
    print("stat_uncertainty=", stat_err)
    print("syst_uncertainty=", full_var - stat_err)
    syst_err = syst_uncertainty(mu_grid, posterior_mu, marginal_r_lam, marginal_mu)
    print("syst_uncertainty=", syst_err)
    syst_err = syst_uncertainty2(mu_grid, posterior_mu, marginal_r_lam)
    print("syst_uncertainty=", syst_err)
    syst_err = syst_uncertainty3(mu_grid, posterior_mu, marginal_r_lam)
    print("syst_uncertainty=", syst_err)

    print()
    print("check marginals")
    print("mu  ", marginal_mu.min(), marginal_mu.max())
    print("lam ", marginal_lam.min(), marginal_lam.max())
    print("r   ", marginal_r.min(), marginal_r.max())
    print("check posterior")
    print("p(y|x)  ", posterior_mu.min(), posterior_mu.max())
    print("p(y|x,a)", posterior_r_lam_mu.min(), posterior_r_lam_mu.max())


    # return None

    plt.axvline(config.TRUE.mu, c="orange", label="true mu")
    plt.axvline(config.TRUE.mu-std_mu, c="orange", label="true mu - std(mu)")
    plt.axvline(config.TRUE.mu+std_mu, c="orange", label="true mu + std(mu)")
    plt.axvline(sig_ratio, c="red", label="signal ratio")
    plt.axvline(expect_mu, c="green", label="E[mu|x]")
    plt.plot(mu_grid, marginal_mu, label="posterior")
    plt.xlabel("mu")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of mu vs mu values")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'marginal_mu.png'))
    plt.clf()


    plt.plot(lam_grid, marginal_lam, label="posterior")
    plt.axvline(config.TRUE.lam, c="orange", label="true lambda")
    plt.xlabel("lambda")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of lam vs lam values")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'marginal_lam.png'))
    plt.clf()

    plt.plot(r_grid, marginal_r, label="posterior")
    plt.axvline(config.TRUE.r, c="orange", label="true r")
    plt.xlabel("r")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of r vs r values")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'marginal_r.png'))
    plt.clf()

    # sns.distplot(data, label="data hist")
    # x_range = np.linspace(np.min(data), np.max(data), 1000)
    # p = data_generator.proba_density(x_range, Y_TRUE, ALPHA_TRUE)
    # plt.plot(x_range, p, label="true proba")
    # plt.legend()
    # plt.savefig(os.path.join(DIRECTORY, 'data_dstrib.png'))
    # plt.clf()

def likelihood_fit():
    print("Hello world !")
    set_plot_config()
    config = S3D2Config()
    DATA_N_SAMPLES = 80_000

    result_table = []

    for mu in config.TRUE_MU_RANGE[1:]:
        result_row = {}
        config.TRUE_MU = mu
        generator = S3D2(SEED)
        data, label = generator.sample_event(config.TRUE.r, config.TRUE.lam, config.TRUE_MU, size=DATA_N_SAMPLES)
        n_sig = np.sum(label==1)
        n_bkg = np.sum(label==0)
        print(f"nb of signal      = {n_sig}")
        print(f"nb of backgrounds = {n_bkg}")


        compute_nll = lambda r, lam, mu : generator.nll(data, r, lam, mu)

        print('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, config)
        fmin, params = estimate(minimizer)
        params_truth = [config.TRUE_R, config.TRUE_LAMBDA, config.TRUE_MU]
        my_print_params(params, params_truth)
        register_params(params, params_truth, result_row)
        result_row['is_mingrad_valid'] = minimizer.migrad_ok()
        result_row.update(fmin)
        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    print('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        my_plot_params(name, result_table)


def my_print_params(param, params_truth):
    for p, truth in zip(param, params_truth):
        name  = p['name']
        value = p['value']
        error = p['error']
        print('{name:4} = {truth} vs {value} +/- {error}'.format(**locals()))


def my_plot_params(param_name, result_table, directory=DIRECTORY):
    from utils.misc import _ERROR
    from utils.misc import _TRUTH

    values = result_table[param_name]
    errors = result_table[param_name+_ERROR]
    truths = result_table[param_name+_TRUTH]
    xx = np.arange(len(values))
    if 'is_valid' in result_table:
        valid_values = values[result_table['is_valid']]
        valid_errors = errors[result_table['is_valid']]
        valid_x = xx[result_table['is_valid']]
        print("Plot_params valid lenght = {}, {}, {}".format(len(valid_x), len(valid_values), len(valid_errors)))
        values =  values[result_table['is_valid'] == False]
        errors =  errors[result_table['is_valid'] == False]
        x = xx[result_table['is_valid'] == False]
        print('Plot_params invalid lenght = {}, {}, {}'.format(len(x), len(values), len(errors)))
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
        plt.title("Likelihood fit")
        plt.legend()
        plt.savefig(os.path.join(directory, 'estimate_{}.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        print('Plot params failed')
        print(str(e))

if __name__ == '__main__':
    # explore()
    main()
    # likelihood_fit()
    print('DONE !')