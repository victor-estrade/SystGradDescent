#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.some_likelihood

import os
import logging
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from visual import set_plot_config
set_plot_config()

from config import SEED
from config import SAVING_DIR
from config import DEFAULT_DIR

from problem.gamma_gauss.generator import Generator


from visual.misc import now_str


DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")


def main():
    print('hello')
    logger = logging.getLogger()


    TRUE_MU = 1.0
    directory = os.path.join(DIRECTORY, 'nll')
    os.makedirs(directory, exist_ok=True)
    suffix = f"_mu={TRUE_MU}"

    _make_rescale_plot(1.0, TRUE_MU)
    _make_rescale_plot(1.2, TRUE_MU)
    _make_rescale_plot(0.8, TRUE_MU)


    param_name = 'rescale'

    plt.xlabel(param_name)
    plt.ylabel('nll')
    title = f'{param_name} ({suffix[1:]}) NLL around minimum'
    plt.title(now_str()+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'NLL_{param_name}{suffix}.png'))
    plt.clf()



    TRUE_RESCALE = 1.0
    directory = os.path.join(DIRECTORY, 'nll')
    os.makedirs(directory, exist_ok=True)
    suffix = f"_mu={TRUE_MU}"

    _make_mu_plot(TRUE_RESCALE, 0.5)
    _make_mu_plot(TRUE_RESCALE, 1.0)
    _make_mu_plot(TRUE_RESCALE, 1.5)


    param_name = 'mu'

    plt.xlabel(param_name)
    plt.ylabel('nll')
    title = f'{param_name} ({suffix[1:]}) NLL around minimum'
    plt.title(now_str()+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'NLL_{param_name}{suffix}.png'))
    plt.clf()


def _make_rescale_plot(true_rescale, true_mu):
    generator = Generator(seed=SEED)
    X, y, w = generator.generate(true_rescale, true_mu)

    compute_nll = lambda rescale, mu : generator.nll(X, rescale, mu)
    rescale_array = np.linspace(0.5, 3, 50)
    nll_array = [compute_nll(rescale, true_mu) for rescale in rescale_array]
    param_name = 'rescale'
    p = plt.plot(rescale_array, nll_array, label=f'NLL {param_name}={true_rescale}')
    plt.axvline(x=true_rescale, linestyle='--', color=p[0].get_color(), label='true value')


def _make_mu_plot(true_rescale, true_mu):
    generator = Generator(seed=SEED)
    X, y = generator.sample_event(true_rescale, true_mu)

    compute_nll = lambda rescale, mu : generator.nll(X, rescale, mu)
    mu_array = np.linspace(0.1, 2, 50)
    nll_array = [compute_nll(true_rescale, mu) for mu in mu_array]
    param_name = 'mu'
    p = plt.plot(mu_array, nll_array, label=f'NLL {param_name}={true_mu}')
    plt.axvline(x=true_mu, linestyle='--', color=p[0].get_color(), label='true value')


def plot_data_distrib(X, y, w, generator, rescale, mu, directory, suffix=''):
    bkg = X[y==0].reshape(-1)
    sig = X[y==1].reshape(-1)
    w_bkg = w[y==0]
    w_sig = w[y==1]

    min_x = np.min(X) - 0.05
    max_x = np.max(X)
    x_range = np.linspace(min_x, max_x, 1000)
    proba = generator.proba_density(x_range, rescale, mu)

    plt.hist([bkg, sig], weights=[w_bkg, w_sig], bins=20, density=True, stacked=True, label=('b', 's'))
    plt.plot(x_range, proba, label=f"pdf")
    plt.title(f'Toy distribution ({suffix})')
    plt.ylabel('density')
    plt.xlabel('x')
    plt.legend()
    fname = os.path.join(directory, f'x_distrib{suffix}.png')
    plt.savefig(fname)
    plt.clf()




if __name__ == '__main__':
    main()
