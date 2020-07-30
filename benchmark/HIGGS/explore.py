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
from config import SEED
from visual import set_plot_config
set_plot_config()

from problem.higgs import Generator
from problem.higgs import get_generators

from problem.higgs.higgs_geant import load_data
from problem.higgs.higgs_geant import split_data_label_weights
from problem.higgs import HiggsConfig as Config
from problem.higgs import param_generator
from problem.higgs import Parameter

from model.gradient_boost import GradientBoostingModel


DATA_NAME = 'HIGGS'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")


def main():
    print('Hello world')
    os.makedirs(DIRECTORY, exist_ok=True)
    data = load_data()
    generator = Generator(data, seed=2)

    dirname = os.path.join(DIRECTORY, 'link_standard')
    os.makedirs(dirname, exist_ok=True)
    explore_links(generator, dirname=dirname)
    dirname = os.path.join(DIRECTORY, 'link_balanced')
    os.makedirs(dirname, exist_ok=True)
    explore_links(generator,  background_luminosity=1, signal_luminosity=1, dirname=dirname)
    dirname = os.path.join(DIRECTORY, 'link_easy')
    os.makedirs(dirname, exist_ok=True)
    explore_links(generator,  background_luminosity=95, signal_luminosity=5, dirname=dirname)
    dirname = os.path.join(DIRECTORY, 'link_medium')
    os.makedirs(dirname, exist_ok=True)
    explore_links(generator,  background_luminosity=98, signal_luminosity=2, dirname=dirname)
    
    # mu_vs_y_w(generator)
    # noise_vs_mu_variance(generator)




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




def explore_links(full_generator, background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781, dirname=DIRECTORY):
    train_generator, valid_generator, test_generator = get_generators(SEED)
    generator = valid_generator
    # generator = full_generator
    generator.background_luminosity = background_luminosity
    generator.signal_luminosity = signal_luminosity

    config = Config()
    N_SAMPLES = 300_000
    feature_names = list(generator.feature_names) + ['Label', 'classifier', 'bin', 'log_p']
    mu_range = np.linspace(min(config.RANGE.mu), max(config.RANGE.mu), num=18)
    all_params = {"min": config.MIN, "true":config.TRUE, "max":config.MAX}
    # all_params = {"true":config.TRUE}

    clf = load_some_clf()
    all_average_df = {}
    for params_name, orig_params in all_params.items():
        print(f"computing link between X and mu using {params_name}...")
        average_list = []
        target_list = []
        for mu in mu_range:
            params = Parameter(*orig_params.nuisance_parameters, mu)
            data, label, weight = generator.generate(*params, n_samples=N_SAMPLES)
            sum_weight = np.sum(weight)
            average_array = np.sum(data*weight.reshape(-1, 1), axis=0) / sum_weight
            average_label = np.sum(label*weight, axis=0) / sum_weight
            proba = clf.predict_proba(data)
            decision = proba[:, 1]
            log_p = np.log(decision / (1 - decision))
            average_log_p = np.sum(log_p*weight, axis=0) / sum_weight
            average_clf = np.sum(decision*weight, axis=0) / sum_weight
            average_bin = np.sum((decision > 0.9)*weight, axis=0) / sum_weight
            average_array = np.hstack([average_array, average_label, average_clf, average_bin, average_log_p])
            average_list.append(average_array)
            target_list.append(mu)
        average_df = pd.DataFrame(np.array(average_list), columns=feature_names)
        all_average_df[params_name] = average_df

    for name in feature_names:
        for params_name, average_df in all_average_df.items(): 
            plt.scatter(average_df[name], target_list, label=params_name)
        plt.title(f'Link between weighted mean({name}) and mu')
        plt.ylabel('mu')
        plt.xlabel(f'weighted mean({name})')
        plt.legend()
        plt.savefig(os.path.join(dirname, f'link_{name}.png'))
        plt.clf()


def load_some_clf():
    model = GradientBoostingModel(learning_rate=0.1, n_estimators=300, max_depth=3)
    model.set_info(DATA_NAME, "HIGGS-prior", 0)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model


if __name__ == '__main__':
    main()