# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.HIGGS.explore

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from visual import set_plot_config
set_plot_config()

from problem.higgs import Generator

from problem.higgs.higgs_geant import load_data
from problem.higgs.higgs_geant import split_data_label_weights
from problem.higgs import HiggsConfig as Config
from problem.higgs import param_generator
from problem.higgs import Parameter


SEED = None
BENCHMARK_NAME = "HIGGS"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")


def mu_vs_y_w(generator):
    print('comparing mu and labels and weights')
    N = 80
    N_SAMPLES = 20000
    mu_list = []
    y_w_list = []
    for i in range(N):
        params = param_generator()
        X, y, w = generator.generate(*params, n_samples=N_SAMPLES)
        mu_list.append(params.mu)
        y_w_list.append((y*w).sum() / w.sum())

    plt.scatter(mu_list, y_w_list, "S = f($\mu$)")
    plt.xlabel('mu')
    plt.ylabel('S = mean(labels, w)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'mu_vs_y_w.png'))
    plt.clf()






def noise_vs_mu_variance(generator):
    data = generator.data
    N = 100
    param_list = [param_generator() for _ in range(N)]
    mu_list = [p.mu for p in param_list]
    mu_std = np.std(mu_list)
    print('mu_std', mu_std)
    X, y, w = split_data_label_weights(data)
    S = w[y==1].sum()
    B = w[y==0].sum()
    print(S, B, S/(S+B))

    N_SAMPLES = 1000

    # FIXME : moyenne pondérée !
    base_param =  param_generator()
    lili = []
    for mu in mu_list:
        generator.reset()
        p = Parameter( *(base_param[:-1] + (mu,)) )
        X, y, w = generator.generate(*p, n_samples=N_SAMPLES)
        feature_sum = (X * w.reshape(-1, 1)).sum(axis=0)
        feature_mean = feature_sum / w.sum()
        lili.append(feature_mean)
    # print(np.array(lili))
    print(np.std(lili, axis=0))


    lulu = []
    for mu in mu_list:
        p = base_param
        X, y, w = generator.generate(*p, n_samples=N_SAMPLES)
        feature_sum = (X * w.reshape(-1, 1)).sum(axis=0)
        feature_mean = feature_sum / w.sum()
        lulu.append(feature_mean)
    # print(np.array(lulu))
    print(np.std(lulu, axis=0))



    lulu = []
    for mu in mu_list:
        generator.reset()
        p = base_param
        X, y, w = generator.generate(*p, n_samples=N_SAMPLES)
        feature_sum = (X * w.reshape(-1, 1)).sum(axis=0)
        feature_mean = feature_sum / w.sum()
        lulu.append(feature_mean)
    # print(np.array(lulu))
    print(np.std(lulu, axis=0))

    # TODO : ne faire varier que mu ! C'est ça qu'on veut voir. Si la variation des moyennes avec mu < variation des moyennes en général
    # Du coup il faut reset le generateur ?
    #  mu fixé + data variable vs data fixé + mu variable ?

    lili = []
    for mu in mu_list:
        generator.reset()
        p = Parameter( *(base_param[:-1] + (mu,)) )
        X, y, w = generator.generate(*p, n_samples=N_SAMPLES)
        feature_sum = (y * w).sum(axis=0)
        feature_mean = feature_sum / w.sum()
        lili.append(feature_mean)
    # print(np.array(lili))
    print(np.std(lili, axis=0))



    lulu = []
    for mu in mu_list:
        p = base_param
        X, y, w = generator.generate(*p, n_samples=N_SAMPLES)
        feature_sum = (y * w).sum(axis=0)
        feature_mean = feature_sum / w.sum()
        lulu.append(feature_mean)
    # print(np.array(lulu))
    print(np.std(lulu, axis=0))


def main():
    print('Hello world')
    os.makedirs(DIRECTORY, exist_ok=True)
    data = load_data()
    generator = Generator(data, seed=2)
    mu_vs_y_w(generator)
    # noise_vs_mu_variance(generator)


if __name__ == '__main__':
    main()