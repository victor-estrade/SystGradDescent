#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from config import SAVING_DIR
from visual import set_plot_config
set_plot_config()

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config

BENCHMARK_NAME = "GG"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")

def main():
    print('hello world !')
    os.makedirs(DIRECTORY, exist_ok=True)

    explore_distribs()
    explore_links()


def explore_links():
    config = Config()
    generator = Generator()
    rescale_range = np.linspace(min(config.RANGE.rescale), max(config.RANGE.rescale), num=5)
    mix_range = np.linspace(min(config.RANGE.mix), max(config.RANGE.mix), num=15)
    for rescale in rescale_range:
        average_list = []
        target_list = []
        for mix in mix_range:
            data, label = generator.sample_event(rescale, mix, size=config.N_TESTING_SAMPLES)
            average_list.append(np.mean(data, axis=0))
            target_list.append(mix)
        plt.scatter(average_list, target_list, label=f'rescale={rescale}')

    plt.title('Link between mean(x) and mix')
    plt.ylabel('mix')
    plt.xlabel('mean(x)')
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'mean_link.png'))
    plt.clf()






def explore_distribs():
    RESCALE_MIN = 0.9
    RESCALE_TRUE = 1.05
    RESCALE_MAX = 1.1
    MIX_MIN = 0.1
    MIX_TRUE = 0.15
    MIX_MAX = 0.2
    config = Config()
    generator = Generator()
    data, label = generator.sample_event(*config.TRUE, size=config.N_TESTING_SAMPLES)
    
    prior_rescale = stats.uniform(loc=RESCALE_MIN, scale=RESCALE_MAX-RESCALE_MIN)
    prior_mix   = stats.uniform(loc=MIX_MIN, scale=MIX_MAX-MIX_MIN)

    plot_data_distrib(generator, config)
    plot_prior(prior_rescale, "rescale")
    plot_prior(prior_mix, "mix")



def plot_data_distrib(generator, config):
    data, label = generator.sample_event(*config.TRUE, size=config.N_TESTING_SAMPLES)
    bkg = data[label==0]
    sig = data[label==1]

    min_x = np.min(data) - 0.05
    max_x = np.max(data)
    x_range = np.linspace(min_x, max_x, 1000)
    p = generator.proba_density(x_range, *config.TRUE)

    plt.hist([bkg, sig], bins='auto', density=True, stacked=True, label=('b', 's'))
    plt.plot(x_range, p, label=f"pdf")
    plt.title('Toy distribution')
    plt.ylabel('density')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'x_distrib.png'))
    plt.clf()


def plot_prior(prior, name=''):
    x = np.linspace(prior.ppf(0.01), prior.ppf(0.99), 100)
    p = prior.pdf(x)
    plt.plot(x, p, label=name)
    plt.title(f'Prior {name}')
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, f'prior_{name}.png'))
    plt.clf()



if __name__ == '__main__':
    main()