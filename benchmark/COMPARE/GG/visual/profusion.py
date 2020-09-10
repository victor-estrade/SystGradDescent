# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

from visual.misc import set_plot_config
set_plot_config()

# import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns

from config import DEFAULT_DIR

def n_samples_mse(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        true_mix = evaluation.true_mix.median()

        data = evaluation[ (evaluation.true_mix == true_mix)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.n_test_samples
            y = df.target_mse
            label = f"$\\alpha$ = {true_rescale}"
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("MSE $\\hat \\mu$")
    plt.title(title)
    plt.legend([f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_n_samples_mse.png'))
    plt.clf()


def n_samples_v_stat(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        true_mix = evaluation.true_mix.median()

        data = evaluation[ (evaluation.true_mix == true_mix)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.n_test_samples
            y = df.var_stat
            label = f"$\\alpha$ = {true_rescale}"
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("V_stat")
    plt.title(title)
    plt.legend([f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_n_samples_v_stat.png'))
    plt.clf()


def n_samples_v_syst(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        true_mix = evaluation.true_mix.median()

        data = evaluation[ (evaluation.true_mix == true_mix)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.n_test_samples
            y = df.var_syst
            label = f"$\\alpha$ = {true_rescale}"
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("V_syst")
    plt.title(title)
    plt.legend([f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_n_samples_v_syst.png'))
    plt.clf()



def n_samples_sigma_mean(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        true_mix = evaluation.true_mix.median()

        data = evaluation[ (evaluation.true_mix == true_mix)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.n_test_samples
            y = df.sigma_mean
            label = f"$\\alpha$ = {true_rescale}"
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("average $\\hat \\sigma_{\\hat \\mu}$")
    plt.title(title)
    plt.legend([f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_n_samples_sigma_mean.png'))
    plt.clf()



def true_mu_estimator(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()

        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.true_mix
            y = df.target_mean
            y_err = df.sigma_mean
            true = df.true_mix
            label = f"$\\alpha$ = {true_rescale}"
            plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500,)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\mu \\pm \\sigma$")
    plt.title(title)
    plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_true_mu_estimator.png'))
    plt.clf()


def true_mu_target_mean(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = all_evaluations[0].true_rescale.unique()

    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()

        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        for i, (true_rescale, df) in enumerate(data.groupby("true_rescale")):
            x = df.true_mix
            y = df.target_mean
            true = df.true_mix
            label = f"$\\alpha$ = {true_rescale}"
            plt.scatter(x, y, marker='o', label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500,)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\mu$")
    plt.title(title)
    plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_true_mu_target_mean.png'))
    plt.clf()