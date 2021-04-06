#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging
import datetime

import pandas as pd



from utils.log import set_logger


from .loader import GBLoader
from .loader import NNLoader
from .loader import REGLoader


from visual.misc import set_plot_config
set_plot_config()

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from config import DEFAULT_DIR


def plot_eval_mse(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mu
        y = df.target_mse
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('$\\mu$')
    plt.ylabel("average MSE")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'eval_mse.png'))
    plt.clf()


def plot_eval_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mu
        y = df.var_stat
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('$\\mu$')
    plt.ylabel("V_stat")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'eval_v_stat.png'))
    plt.clf()


def plot_eval_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mu
        y = df.var_syst
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('$\\mu$')
    plt.ylabel("V_syst")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'eval_v_syst.png'))
    plt.clf()


def plot_eval_mu(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mu
        y = df.target_mean
        y_err = df.sigma_mean
        true = df.true_mu
        label = f"$\\alpha$ = {true_rescale}"
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500,)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'eval_mu.png'))
    plt.clf()


def plot_n_samples_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    true_mu = evaluation.true_mu.median()

    data = evaluation[ (evaluation.true_mu == true_mu)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.var_stat
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('#samples')
    plt.ylabel("V_stat")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'n_samples_v_stat.png'))
    plt.clf()



def plot_n_samples_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    true_mu = evaluation.true_mu.median()

    data = evaluation[ (evaluation.true_mu == true_mu)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.var_syst
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('#samples')
    plt.ylabel("V_syst")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend()
    plt.savefig(os.path.join(directory, f'n_samples_v_syst.png'))
    plt.clf()


def main():
    print("hello !")
    print(DEFAULT_DIR)
    os.makedirs(DEFAULT_DIR, exist_ok=True)

    my_loader = GBLoader('GG', 'GG-prior')
    # my_loader = NNLoader('GG', 'GG-prior', "L4")
    # my_loader = REGLoader('GG', 'GG-prior', "EA3ML3")
    evaluation = my_loader.load_evaluation_config()

    plot_eval_mu(evaluation, title=my_loader.model_full_name)
    plot_eval_mse(evaluation, title=my_loader.model_full_name)
    plot_eval_v_stat(evaluation, title=my_loader.model_full_name)
    plot_eval_v_syst(evaluation, title=my_loader.model_full_name)
    plot_n_samples_v_stat(evaluation, title=my_loader.model_full_name)
    plot_n_samples_v_syst(evaluation, title=my_loader.model_full_name)


    all_evaluations = []
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=3, n_estimators=100).load_evaluation_config() )
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=3, n_estimators=300).load_evaluation_config() )
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=3, n_estimators=1000).load_evaluation_config() )
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=5, n_estimators=100).load_evaluation_config() )
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=5, n_estimators=300).load_evaluation_config() )
    all_evaluations.append( GBLoader('GG', 'GG-prior', max_depth=5, n_estimators=1000).load_evaluation_config() )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        true_mu = evaluation.true_mu.median()

        data = evaluation[ (evaluation.true_mu == true_mu)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.n_test_samples
            y = df.var_stat
            label = f"$\\alpha$ = {true_rescale}"
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('#samples')
    plt.ylabel("V_stat")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+"many GB")
    plt.legend([f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(DEFAULT_DIR, f'many_n_samples_v_stat.png'))
    plt.clf()




    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()

        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.true_mu
            y = df.target_mean
            y_err = df.sigma_mean
            true = df.true_mu
            label = f"$\\alpha$ = {true_rescale}"
            plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500,)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+"many GB")
    plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(DEFAULT_DIR, f'many_eval_mu.png'))
    plt.clf()

if __name__ == '__main__':
    main()
