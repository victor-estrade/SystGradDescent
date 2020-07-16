#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from visual import set_plot_config
set_plot_config()

from problem.synthetic3D import Generator
from problem.synthetic3D import Parameter
from problem.synthetic3D import S3D2Config as Config

from model.gradient_boost import GradientBoostingModel

SEED = None
DATA_NAME = "S3D2"
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")


def load_some_clf():
    model = GradientBoostingModel(learning_rate=0.1, n_estimators=300, max_depth=3)
    model.set_info("S3D2", "S3D2-prior", 0)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model


def explore_links():
    generator = Generator(SEED)

    config = Config()
    N_SAMPLES = 30_000
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
        plt.savefig(os.path.join(DIRECTORY, f'link_{name}.png'))
        plt.clf()



def features():
    config = Config()
    N_SAMPLES = 10_000
    R_MIN   = -0.3
    R_MAX   = 0.3 
    LAM_MIN = 2
    LAM_MAX = 4
    MU_MIN  = 0.0
    MU_MAX  = 1.0 

    generator = Generator(SEED)
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

    R_RANGE = np.linspace(R_MIN, R_MAX, 100)
    nll = [generator.nll(X, r, config.TRUE.lam, config.TRUE.mu) for r in R_RANGE]
    min_nll = R_RANGE[np.argmin(nll)]
    plt.plot(R_RANGE, nll, label="nll(r)")
    plt.axvline(config.TRUE.r, c="orange", label="true r")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("r")
    plt.ylabel("NLL")
    plt.title("NLL according to r param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_r.png'))
    plt.clf()


    LAM_RANGE = np.linspace(LAM_MIN, LAM_MAX, 100)
    nll = [generator.nll(X, config.TRUE.r, lam, config.TRUE.mu) for lam in LAM_RANGE]
    min_nll = LAM_RANGE[np.argmin(nll)]
    plt.plot(LAM_RANGE, nll, label="nll(lam)")
    plt.axvline(config.TRUE.lam, c="orange", label="true lam")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\lambda$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\lambda$ param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_lambda.png'))
    plt.clf()

    MU_RANGE = np.linspace(MU_MIN, MU_MAX, 100)
    nll = [generator.nll(X, config.TRUE.r, config.TRUE.lam, mu) for mu in MU_RANGE]
    min_nll = MU_RANGE[np.argmin(nll)]
    plt.plot(MU_RANGE, nll, label="nll(mu)")
    plt.axvline(config.TRUE.mu, c="orange", label="true mu")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\mu$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\mu$ param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_mu.png'))
    plt.clf()


def main():
    print("Hello master !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()

    # features()

    explore_links()


if __name__ == '__main__':
    main()