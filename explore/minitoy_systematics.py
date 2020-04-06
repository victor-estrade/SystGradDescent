#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals



"""
Exploring the possible minimal systematic error

The idea is to build a very simple but representative toy and to compute statistic and systematic error.
We can compute the likelihood for this mini toy.

x = value of an observable of an event.
y = target parameter = mixture coef of signal and background events
alpha = nuisance parameter = rescaling of the observable

"""

import itertools
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import scipy.stats as sts
from scipy.special import softmax

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


def assert_clean_alpha(alpha):
    assert alpha > 0, f"alpha should be > 0  {alpha} found"

def assert_clean_y(y):
    assert y > 0 and y < 1, f"y is a mixture coef it should be in ]0, 1[  {y} found"


class Generator():
    def __init__(self, seed=None, gamma_k=2, gamma_loc=0, normal_mean=5, normal_sigma=0.5):
        self.seed = seed
        self.gamma_k = gamma_k
        self.gamma_loc = gamma_loc
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma

    def sample_event(self, y, alpha, size=1):
        assert_clean_alpha(alpha)
        assert_clean_y(y)
        n_sig = int(y * size)
        n_bkg = size - n_sig
        x = self._generate_vars(y, alpha, n_bkg, n_sig)
        labels = self._generate_labels(n_bkg, n_sig)
        return x, labels

    def _generate_vars(self, y, alpha, n_bkg, n_sig):
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = alpha
        normal_mean  = self.normal_mean * alpha
        normal_sigma = self.normal_sigma * alpha
        x_b = sts.gamma.rvs(gamma_k, loc=gamma_loc, scale=gamma_scale, size=n_bkg, random_state=self.seed)
        x_s = sts.norm.rvs(loc=normal_mean, scale=normal_sigma, size=n_sig, random_state=self.seed)
        x = np.concatenate([x_b, x_s], axis=0)
        return x

    def _generate_labels(self, n_bkg, n_sig):
        y_b = np.zeros(n_bkg)
        y_s = np.ones(n_sig)
        y = np.concatenate([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mu, n_bkg, n_sig, n_expected_events):
        w_b = np.ones(n_bkg) * (1-mu) * n_expected_events/n_bkg
        w_s = np.ones(n_sig) * mu * n_expected_events/n_sig
        w = np.concatenate([w_b, w_s], axis=0)
        return w

    def proba_density(self, x, y, alpha):
        """
        computes p(x | y, alpha)
        """
        # assert_clean_alpha(alpha)
        # assert_clean_y(y)
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = alpha
        normal_mean  = self.normal_mean * alpha
        normal_sigma = self.normal_sigma * alpha
        proba_gamma  = sts.gamma.pdf(x, gamma_k, loc=gamma_loc, scale=gamma_scale)
        proba_normal  = sts.norm.pdf(x, loc=normal_mean, scale=normal_sigma)
        proba_density = y * proba_normal + (1-y) * proba_gamma
        return proba_density

    def log_proba_density(self, x, y, alpha):
        """
        computes log p(x | y, alpha)
        """
        proba_density = self.proba_density(x, y, alpha)
        logproba_density = np.log(proba_density)
        return logproba_density

    def nll(self, data, y, alpha):
        """
        Computes the negative log likelihood of teh data given y and alpha.
        """
        nll = - self.log_proba_density(data, y, alpha).sum()
        return nll


class UniformPrior():
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.scale = self.max_value - self.min_value

    def proba_density(self, y):
        p = sts.uniform.pdf(y, loc=self.min_value, scale=self.scale)
        return p

    def log_proba_density(self, y):
        p = sts.uniform.logpdf(y, loc=self.min_value, scale=self.scale)
        return p

    def grid(self, n_samples=10000):
        g = np.linspace(self.min_value, self.max_value, num=n_samples)
        return g


class PriorAlpha(UniformPrior):
    def __init__(self, alpha_min=0.7, alpha_max=1.3):
        super().__init__(alpha_min, alpha_max)

class PriorY(UniformPrior):
    def __init__(self, y_min=0.01, y_max=0.3):
        super().__init__(y_min, y_max)


def expectancy(values, probabilities):
    return np.sum(values * probabilities)

# def variance(values, probabilities):
#     return np.sum(values * values * probabilities ) - np.square(expectancy(values, probabilities))

def variance(values, probabilities):
    return np.sum(probabilities * np.square(values - expectancy(values, probabilities)))

def stat_uncertainty(values, posterior, marginal):
    return sum([variance(values, posterior[:, j]) * marginal[j] for j in range(marginal.shape[0])])

def small_test():
    print('hello world !')
    x_range = np.linspace(0, 10, 1000)
    generator = Generator()
    prior_alpha = PriorAlpha()
    prior_y = PriorY()
    alpha =  1
    y = 0.5
    p = generator.proba_density(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")
    data, idx = generator.sample_event(y, alpha, size=1000)
    sns.distplot(data, label="samples")

    alpha = 1.1
    p = generator.proba_density(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")
    plt.legend()
    plt.show()

    x_range = np.linspace(0, 2, 100)
    p = prior_alpha.proba_density(x_range)
    plt.plot(x_range, p, label=f"p_alpha")
    plt.legend()
    plt.show()


    x_range = np.linspace(0, 0.5, 100)
    p = prior_y.proba_density(x_range)
    plt.plot(x_range, p, label=f"p_y")
    plt.legend()
    plt.show()

    print('NLL arange 0, 10' )

    nll = generator.nll(data, 0.1, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.01, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.5, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.45, 1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.5, 1)
    print('best nll =', nll )


def main():
    set_plot_config()
    DIRECTORY = "/home/estrade/Bureau/PhD/SystML/SystGradDescent/savings/MINITOY"
    ALPHA_MIN = 0.9
    ALPHA_TRUE = 1.05
    ALPHA_MAX = 1.1
    Y_MIN = 0.1
    Y_TRUE = 0.15
    Y_MAX = 0.2
    ALPHA_N_SAMPLES = 310
    Y_N_SAMPLES = 290
    DATA_N_SAMPLES = 10000

    assert Y_MIN <= Y_TRUE and Y_TRUE <= Y_MAX, \
        f"truth value of y ({Y_TRUE}) should be inside its interval [{Y_MIN}, {Y_MAX}]"
    assert ALPHA_MIN <= ALPHA_TRUE and ALPHA_TRUE <= ALPHA_MAX, \
        f"truth value of y ({ALPHA_TRUE}) should be inside its interval [{ALPHA_MIN}, {ALPHA_MAX}]"

    prior_alpha = PriorAlpha(ALPHA_MIN, ALPHA_MAX)
    prior_y = PriorY(Y_MIN, Y_MAX)
    
    alpha_grid = prior_alpha.grid(ALPHA_N_SAMPLES)
    y_grid = prior_y.grid(Y_N_SAMPLES)
    
    data_generator = Generator()
    data, label = data_generator.sample_event(Y_TRUE, ALPHA_TRUE, size=DATA_N_SAMPLES)
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    print(f"nb of signal      = {n_sig}")
    print(f"nb of backgrounds = {n_bkg}")
    # Artificially inflate alpha uncertainty
    noise_lvl = 0.0
    if noise_lvl:
        print(f'apply noise ({noise_lvl*100}%) on alpha')
        noise = np.random.uniform(1-noise_lvl, 1+noise_lvl, size=DATA_N_SAMPLES)
        data = data * noise

    shape = (Y_N_SAMPLES, ALPHA_N_SAMPLES)
    log_likelihood = np.zeros(shape)
    log_prior_proba = np.zeros(shape)
    for i, j in itertools.product(range(Y_N_SAMPLES), range(ALPHA_N_SAMPLES)):
        log_likelihood[i, j] = data_generator.log_proba_density(data, y_grid[i], alpha_grid[j]).sum()
        log_prior_proba[i, j] = prior_y.log_proba_density(y_grid[i]) + prior_alpha.log_proba_density(alpha_grid[j])
    # log_likelihood = np.array([data_generator.log_proba_density(data, y_grid[i], alpha_grid[j]).sum()
    #                             for i, j in itertools.product(range(Y_N_SAMPLES), range(ALPHA_N_SAMPLES))]).reshape(shape)
    # log_prior_proba = np.array([prior_y.log_proba_density(y_grid[i]) + prior_alpha.log_proba_density(alpha_grid[j])
    #                             for i, j in itertools.product(range(Y_N_SAMPLES), range(ALPHA_N_SAMPLES))]).reshape(shape)
    
    element_min = (log_likelihood + log_prior_proba).min()
    print("min element = ", element_min)
    posterior_y_alpha = softmax(log_likelihood + log_prior_proba)
    n_zeros = (posterior_y_alpha == 0).sum()
    n_elements = np.prod(posterior_y_alpha.shape)
    print()
    print(f"number of zeros in posterior = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    marginal_y = posterior_y_alpha.sum(axis=1)
    marginal_alpha = posterior_y_alpha.sum(axis=0)
    assert marginal_y.shape == y_grid.shape, "sum along the wrong axis for marginal y"
    assert marginal_alpha.shape == alpha_grid.shape, "sum along the wrong axis for marginal alpha"

    n_zeros = (marginal_y == 0).sum()
    n_elements = np.prod(marginal_y.shape)
    print(f"number of zeros in marginal y = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")
    n_zeros = (marginal_alpha == 0).sum()
    n_elements = np.prod(marginal_alpha.shape)
    print(f"number of zeros in marginal alpha = {n_zeros}/{n_elements} ({n_zeros/n_elements*100:2.3f} %)")

    posterior_y = np.divide(posterior_y_alpha, marginal_alpha,
        out=np.zeros_like(posterior_y_alpha), where=(posterior_y_alpha!=0))

    print("probability densities should sum to one")
    # TODO : posterior_y sum to ALPHA_N_SAMPLES. is it ok ?
    # TODO : with new division policy posterior_y/ALPHA_N sums to 1-zero_ration in marginal_y
    #        ... It does not look good
    print(np.sum(posterior_y)/ALPHA_N_SAMPLES, np.sum(posterior_y_alpha), np.sum(marginal_y), np.sum(marginal_alpha))

    print()
    print("True y value    =", Y_TRUE)
    sig_ratio = n_sig/DATA_N_SAMPLES
    print("Sig ratio       =", sig_ratio)
    expect_y = expectancy(y_grid, marginal_y)
    print("E[y|x]          =", expect_y)
    full_var = variance(y_grid, marginal_y)
    print("Var[y|x]        =", full_var)
    std_y = np.sqrt(full_var)
    print("sqrt(Var[y|x])  =", std_y)
    print("argmax_y p(y|x) =", y_grid[np.argmax(marginal_y)])
    i_max = np.argmax(log_likelihood) // ALPHA_N_SAMPLES
    j_max = np.argmax(log_likelihood) % ALPHA_N_SAMPLES
    assert np.max(log_likelihood) == log_likelihood[i_max, j_max], "max and argmax should point to the same value"
    print("argmax_y_alpha logp(x|y, alpha) =", y_grid[i_max], alpha_grid[j_max])
    stat_err = stat_uncertainty(y_grid, posterior_y, marginal_alpha)
    print("stat_uncertainty=", stat_err)
    print("syst_uncertainty=", full_var - stat_err)

    print()
    print("check marginals")
    print("y    ", marginal_y.min(), marginal_y.max())
    print("alpha", marginal_alpha.min(), marginal_alpha.max())
    print("check posterior")
    print("p(y|x)  ", posterior_y.min(), posterior_y.max())
    print("p(y|x,a)", posterior_y_alpha.min(), posterior_y_alpha.max())


    # return None

    plt.axvline(Y_TRUE, c="orange", label="true y")
    plt.axvline(Y_TRUE-std_y, c="orange", label="true y - std(y)")
    plt.axvline(Y_TRUE+std_y, c="orange", label="true y + std(y)")
    plt.axvline(sig_ratio, c="red", label="signal ratio")
    plt.axvline(expect_y, c="green", label="E[y|x]")
    plt.plot(y_grid, marginal_y, label="posterior")
    plt.xlabel("y")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of y vs y values")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'marginal_y.png'))
    plt.clf()


    plt.plot(alpha_grid, marginal_alpha, label="posterior")
    plt.axvline(ALPHA_TRUE, c="orange", label="true alpha")
    plt.xlabel("alpha")
    plt.ylabel("proba density")
    plt.title("posterior marginal proba of alpha vs alpha values")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'marginal_alpha.png'))
    plt.clf()

    sns.distplot(data, label="data hist")
    x_range = np.linspace(np.min(data), np.max(data), 1000)
    p = data_generator.proba_density(x_range, Y_TRUE, ALPHA_TRUE)
    plt.plot(x_range, p, label="true proba")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'data_dstrib.png'))
    plt.clf()








if __name__ == '__main__':
    main()
