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
import seaborn as sns

from config import DEFAULT_DIR


def true_mu_mse(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.target_mse
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("MSE $\\hat \\mu$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_mse.png'), bbox_inches='tight')
    plt.clf()


def true_mu_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.var_stat
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("V_stat")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_v_stat.png'), bbox_inches='tight')
    plt.clf()


def true_mu_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.var_syst
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("V_syst")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_v_syst.png'), bbox_inches='tight')
    plt.clf()


def true_mu_estimator(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.target_mean
        y_err = df.sigma_mean
        true = df.true_mix
        label = f"$\\alpha$ = {true_rescale}"
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\hat \\mu \\pm \\sigma_{\\hat \\mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_estimator.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_mean_std(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.target_mean
        y_err = df.target_std
        true = df.true_mix
        label = f"$\\alpha$ = {true_rescale}"
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average $\\hat \\mu \\pm std(\\hat \\mu)$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_target_mean_std.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_mean(evaluation, title="No Title", directory=DEFAULT_DIR):
    max_n_test_samples = evaluation.n_test_samples.max()

    data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.true_mix
        y = df.target_mean
        true = df.true_mix
        label = f"$\\alpha$ = {true_rescale}"
        plt.scatter(x, y, marker='o', label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\hat \\mu \\pm \\sigma_{\\hat \\mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_target_mean.png'), bbox_inches='tight')
    plt.clf()


def n_samples_mse(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mix = evaluation.true_mix.median()

    data = evaluation[ (evaluation.true_mix == chosen_true_mix)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.target_mse
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("MSE $\\hat \\mu$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_mse.png'), bbox_inches='tight')
    plt.clf()


def box_n_samples_mse(evaluation, title="No Title", directory=DEFAULT_DIR):
    data = []
    x = []
    for i, (n_test_samples, df) in enumerate(evaluation.groupby("n_test_samples")):
        data.append(df.target_mse)
        x.append(n_test_samples)

    plt.boxplot(data, labels=x)
    plt.xlabel('# test samples')
    plt.ylabel("MSE $\\hat \\mu$")
    plt.title(title)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'box_n_samples_mse.png'), bbox_inches='tight')
    plt.clf()


def n_samples_sigma_mean(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mix = evaluation.true_mix.median()

    data = evaluation[ (evaluation.true_mix == chosen_true_mix)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.sigma_mean
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("avegrage $\\hat \\sigma_{\\hat \mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_sigma_mean.png'), bbox_inches='tight')
    plt.clf()


def n_samples_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mix = evaluation.true_mix.median()

    data = evaluation[ (evaluation.true_mix == chosen_true_mix)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.var_stat
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_stat")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_v_stat.png'), bbox_inches='tight')
    plt.clf()


def n_samples_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mix = evaluation.true_mix.median()

    data = evaluation[ (evaluation.true_mix == chosen_true_mix)]
    for true_rescale, df in data.groupby("true_rescale"):
        x = df.n_test_samples
        y = df.var_syst
        label = f"$\\alpha$ = {true_rescale}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_syst")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_v_syst.png'), bbox_inches='tight')
    plt.clf()


def nominal_fisher_n_bins(fisher_table, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mix = fisher_table.true_mix.median()
    chosen_true_rescale = fisher_table.true_rescale.median()
    data = fisher_table[ (fisher_table.true_mix == chosen_true_mix) & (fisher_table.true_rescale == chosen_true_rescale) ]
    data = data[ data.n_test_samples == 2000 ]

    data_mean = data.groupby('n_bins').mean()
    label = "nominal"
    x = data_mean.index
    y = data_mean.fisher
    data_std = data.groupby('n_bins').std()
    y_err = data_std.fisher
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    # plt.plot(x, y, label=label)
    plt.xlabel('# bins')
    plt.ylabel('mean( fisher info ) $\pm$ std( fisher info )')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'nominal_fisher_n_bins.png'), bbox_inches='tight')
    plt.clf()


def fisher_n_bins(data, title="No Title", directory=DEFAULT_DIR):
    NUM_COLORS = 15
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # cm = plt.get_cmap('gist_rainbow')
    # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    ax.set_prop_cycle(color=colors, linestyle=['solid', 'dashed', 'dashdot']*5)

    data = data[ data.n_test_samples == data.n_test_samples.max() ]
    for (true_mix, true_rescale), df in data.groupby(["true_mix", "true_rescale"]):
        df_mean = df.groupby('n_bins').mean()
        label = f"$\\mu = {true_mix}$, $\\alpha={true_rescale}$"
        x = df_mean.index
        y = df_mean.fisher
        ax.plot(x, y, label=label)
        # df_std = df.groupby('n_bins').std()
        # y_err = df_std.fisher
        # ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.xlabel('# bins')
    plt.ylabel('mean( fisher info ) $\pm$ std( fisher info )')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'fisher_n_bins.png'), bbox_inches='tight')
    plt.clf()
