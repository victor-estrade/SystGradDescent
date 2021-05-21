#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.AP1.explore

import os
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from visual import set_plot_config
set_plot_config()

from config import SAVING_DIR

from problem.apples_and_pears.generator import Generator

from visual.likelihood import plot_param_around_min


DATA_NAME = 'AP1'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")


def main():
    print('hello')
    generator = Generator(apple_center=7, pear_center=0, n_apple=100, n_pears=100)

    TRUE_RESCALE = 1.0
    TRUE_MU = 1.0
    X, y, w = generator.generate(TRUE_RESCALE, TRUE_MU)


    compute_nll = lambda rescale, mu : generator.nll(X, rescale, mu)

    directory = DIRECTORY
    os.makedirs(directory, exist_ok=True)
    suffix = f"_rescale={TRUE_RESCALE}_mu={TRUE_MU}"

    plot_data_distrib(X, y, w, generator, TRUE_RESCALE, TRUE_MU, directory, suffix=suffix)
    plot_rescale_around_min(compute_nll, TRUE_RESCALE, TRUE_MU, directory, suffix=suffix)





def plot_rescale_around_min(compute_nll, true_rescale, true_mu, directory, suffix=''):
    rescale_array = np.linspace(0.5, 6, 50)
    nll_array = [compute_nll(rescale, true_mu) for rescale in rescale_array]
    name = 'rescale'
    plot_param_around_min(rescale_array, nll_array, true_rescale, name, suffix, directory)



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
