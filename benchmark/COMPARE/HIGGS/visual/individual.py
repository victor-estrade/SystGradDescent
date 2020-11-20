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
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.target_mse
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("MSE $\\hat \\mu$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_mse.png'), bbox_inches='tight')
    plt.clf()


def true_mu_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.var_stat
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("V_stat")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_v_stat.png'), bbox_inches='tight')
    plt.clf()


def true_mu_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.var_syst
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("V_syst")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_v_syst.png'), bbox_inches='tight')
    plt.clf()


def true_mu_estimator(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.target_mean
        y_err = df.sigma_mean
        true = df.true_mu
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\hat \\mu \\pm \\hat \\sigma_{\\hat \\mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_estimator.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_mean_std(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.target_mean
        y_err = df.target_std
        true = df.true_mu
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average $\\hat \\mu \\pm std(\\hat \\mu)$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_target_mean_std.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_mean(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.target_mean
        true = df.true_mu
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.scatter(x, y, marker='o', label=label)
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\hat \\mu$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_target_mean.png'), bbox_inches='tight')
    plt.clf()


def true_mu_sigma_mean(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.sigma_mean
        true = df.true_mu
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.scatter(x, y, marker='o', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $\\hat \\sigma_{\\hat \\mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_sigma_mean.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_std(evaluation, title="No Title", directory=DEFAULT_DIR):
    # max_n_test_samples = evaluation.n_test_samples.max()
    # data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
    data = evaluation
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.true_mu
        y = df.target_std
        true = df.true_mu
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.scatter(x, y, marker='o', label=label)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average estimated $std(\\hat \\mu)$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'true_mu_target_std.png'), bbox_inches='tight')
    plt.clf()


def n_samples_mse(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mu = 1.0  # Nominal value of mu

    data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.n_test_samples
        y = df.target_mse
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
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
    chosen_true_mu = 1.0  # Nominal value of mu

    data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.n_test_samples
        y = df.sigma_mean
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("avegrage $\\hat \\sigma_{\\hat \mu}$")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_sigma_mean.png'), bbox_inches='tight')
    plt.clf()


def n_samples_v_stat(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mu = 1.0  # Nominal value of mu

    data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.n_test_samples
        y = df.var_stat
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_stat")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_v_stat.png'), bbox_inches='tight')
    plt.clf()



def n_samples_v_syst(evaluation, title="No Title", directory=DEFAULT_DIR):
    chosen_true_mu = 1.0  # Nominal value of mu

    data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
    for (true_tes, true_jes, true_les), df in data.groupby(["true_tes", "true_jes", "true_les"]):
        x = df.n_test_samples
        y = df.var_syst
        label = f"tes={true_tes}, jes={true_jes}, les={true_les}"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_syst")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'n_samples_v_syst.png'), bbox_inches='tight')
    plt.clf()



def threshold_s_sqrt_s_b(data, title="No Title", directory=DEFAULT_DIR):
    import numpy as np
    NUM_COLORS = 15
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # cm = plt.get_cmap('gist_rainbow')
    # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    ax.set_prop_cycle(color=colors, linestyle=['solid', 'dashed', 'dashdot']*5)

    print("AVANT !")
    print(data)
    data = data[ data.n_test_samples == data.n_test_samples.max() ]
    print(data)
    for (true_mu, true_tes, true_jes, true_les), df in data.groupby(["true_mu", "true_tes", "true_jes", "true_les"]):
        print(df)
        df_mean = df.groupby('threshold').mean()
        label = f"$\\mu = {true_mix}$, $\\alpha={true_rescale}$"
        x = df_mean.index
        y = df_mean.s_sqrt_n
        # ax.plot(x, y, label=label)
        df_std = df.groupby('n_bins').std()
        y_err = df_std.s_sqrt_n
        ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.xlabel('threshold')
    plt.ylabel('mean( $s / \sqrt{s+b} $ ) $\pm$ std( fisher info )')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'threshold_s_sqrt_s_b.png'), bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def threshold_fisher_diff(data, title="No Title", directory=DEFAULT_DIR):
    import numpy as np
    NUM_COLORS = 15
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # cm = plt.get_cmap('gist_rainbow')
    # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    ax.set_prop_cycle(color=colors, linestyle=['solid', 'dashed', 'dashdot']*5)

    data = data[ data.n_test_samples == data.n_test_samples.max() ]
    print(data)
    for (true_mu, true_tes, true_jes, true_les), df in data.groupby(["true_mu", "true_tes", "true_jes", "true_les"]):
        df['fisher_diff'] = df.fisher_2 - df.fisher_1
        df_mean = df.groupby('threshold').mean()
        label = f"$\\mu = {true_mix}$, $\\alpha={true_rescale}$"
        x = df_mean.index
        y = df_mean.fisher_diff
        ax.plot(x, y, label=label)
        # df_std = df.groupby('n_bins').std()
        # y_err = df_std.s_sqrt_n
        # ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label)
    plt.xlabel('threshold')
    plt.ylabel('mean( fisher_2 - fisher_2 )')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'threshold_fisher_diff.png'), bbox_inches='tight')
    plt.clf()
    plt.close(fig)
