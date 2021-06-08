# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import itertools
import datetime

from visual.misc import set_plot_config
set_plot_config()

# import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns

from config import DEFAULT_DIR

from collections import defaultdict

from .nuisance_param import detect_nuisance_param
from .nuisance_param import label_nuisance_param


def exctract_alpha_combinations(some_evaluation):
    nuisance_param_key = detect_nuisance_param(some_evaluation)
    unique_alphas = itertools.product(*[some_evaluation[key].unique() for key in nuisance_param_key])
    return list(unique_alphas)

def extract_nominal(evaluation):
    nuisance_param_key = detect_nuisance_param(evaluation)
    nominal_true_mu = 1.0  # Nominal value of mu
    evaluation = evaluation[(evaluation.true_mu == nominal_true_mu)]
    nominal_nuisance = 1.0  # Nominal value of all nuisance params
    for key in nuisance_param_key:
        evaluation = evaluation[(evaluation[key] == nominal_nuisance)]
    return evaluation


def n_samples_mse(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    for evaluation in all_evaluations:
        chosen_true_mu = evaluation.true_mu.median()
        data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.n_test_samples
            y = df.target_mse
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("MSE $\\hat \\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend([label_nuisance_param(nuisance_param_key, nuisance_param) for nuisance_param in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_n_samples_mse.png'), bbox_inches='tight')
    plt.clf()



def nominal_n_samples_mse(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    for evaluation in all_evaluations:
        df = extract_nominal(evaluation)
        x = df.n_test_samples
        y = df.target_mse
        label = f"nominal"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("MSE $\\hat \\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.savefig(os.path.join(directory, f'profusion_nominal_n_samples_mse.png'), bbox_inches='tight')
    plt.clf()


def n_samples_v_stat(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    for evaluation in all_evaluations:
        chosen_true_mu = evaluation.true_mu.median()
        data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.n_test_samples
            y = df.var_stat
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("V_stat")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend([label_nuisance_param(nuisance_param_key, nuisance_param) for nuisance_param in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_n_samples_v_stat.png'), bbox_inches='tight')
    plt.clf()


def nominal_n_samples_v_stat(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    for evaluation in all_evaluations:
        df = extract_nominal(evaluation)
        x = df.n_test_samples
        y = df.var_stat
        label = f"nominal"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_stat")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.savefig(os.path.join(directory, f'profusion_nominal_n_samples_v_stat.png'), bbox_inches='tight')
    plt.clf()



def n_samples_v_syst(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    for evaluation in all_evaluations:
        chosen_true_mu = evaluation.true_mu.median()
        data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.n_test_samples
            y = df.var_syst
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("V_syst")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend([label_nuisance_param(nuisance_param_key, nuisance_param) for nuisance_param in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_n_samples_v_syst.png'), bbox_inches='tight')
    plt.clf()


def nominal_n_samples_v_syst(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    for evaluation in all_evaluations:
        df = extract_nominal(evaluation)
        x = df.n_test_samples
        y = df.var_syst
        label = f"nominal"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("V_syst")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.savefig(os.path.join(directory, f'profusion_nominal_n_samples_v_syst.png'), bbox_inches='tight')
    plt.clf()



def n_samples_sigma_mean(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    for evaluation in all_evaluations:
        chosen_true_mu = evaluation.true_mu.median()
        data = evaluation[ (evaluation.true_mu == chosen_true_mu)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.n_test_samples
            y = df.sigma_mean
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.plot(x, y, 'o-', label=label, color=color_cycle[i%len(unique_alphas)])

    plt.xlabel('# test samples')
    plt.ylabel("average $\\hat \\sigma_{\\hat \\mu}$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend([label_nuisance_param(nuisance_param_key, nuisance_param) for nuisance_param in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_n_samples_sigma_mean.png'), bbox_inches='tight')
    plt.clf()


def nominal_n_samples_sigma_mean(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    for evaluation in all_evaluations:
        df = extract_nominal(evaluation)
        x = df.n_test_samples
        y = df.sigma_mean
        label = f"nominal"
        plt.plot(x, y, 'o-', label=label)

    plt.xlabel('# test samples')
    plt.ylabel("average $\\hat \\sigma_{\\hat \\mu}$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.savefig(os.path.join(directory, f'profusion_nominal_n_samples_sigma_mean.png'), bbox_inches='tight')
    plt.clf()



def true_mu_estimator(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()
        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.true_mu
            y = df.target_mean
            y_err = df.sigma_mean
            true = df.true_mu
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average $\\hat \\mu \\pm \\sigma_{\\hat \\mu}$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_true_mu_estimator.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_mean_std(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()
        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.true_mu
            y = df.target_mean
            y_err = df.target_std
            true = df.true_mu
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2, label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average $\\hat \\mu \\pm std(\\hat \\mu)$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ], bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(directory, f'profusion_true_mu_target_mean_std.png'), bbox_inches='tight')
    plt.clf()



def true_mu_target_mean(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    from matplotlib.lines import Line2D
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()
        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.true_mu
            y = df.target_mean
            true = df.true_mu
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.scatter(x, y, marker='o', label=label, color=color_cycle[i%len(unique_alphas)])
    plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("average $\\hat \\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    legend_elements = [Line2D([0], [0], marker='+', color='red', label='true', markersize=15, markeredgewidth=5)]
    legend_elements += [Line2D([0], [0], marker='o', color=color_cycle[i%len(unique_alphas)], label=f"$\\alpha$={a}")
                        for i, a in enumerate(unique_alphas)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_true_mu_target_mean.png'), bbox_inches='tight')
    plt.clf()


def true_mu_target_bias(all_evaluations, title="No Title", directory=DEFAULT_DIR):
    from matplotlib.lines import Line2D
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    unique_alphas = exctract_alpha_combinations(all_evaluations[0])
    n_alphas = len(unique_alphas)

    x, true = None, None  # Strange fix for 'x referenced before assignement' in plt.scatter(x, true, ...)
    for evaluation in all_evaluations:
        max_n_test_samples = evaluation.n_test_samples.max()
        data = evaluation[ (evaluation.n_test_samples == max_n_test_samples)]
        nuisance_param_key = detect_nuisance_param(data)
        for i, ( nuisance_param, df) in enumerate(data.groupby(nuisance_param_key)):
            x = df.true_mu
            y = df.target_bias
            true = df.true_mu
            label = label_nuisance_param(nuisance_param_key, nuisance_param)
            plt.scatter(x, y, marker='o', label=label, color=color_cycle[i%len(unique_alphas)])
    # plt.scatter(x, true, marker='+', c='red', label='truth', s=500, zorder=3)

    plt.xlabel('true $\\mu$')
    plt.ylabel("bias $\\hat \\mu$")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    plt.title(now+title)
    # legend_elements = [Line2D([0], [0], marker='+', color='red', label='true', markersize=15, markeredgewidth=5)]
    legend_elements = [Line2D([0], [0], marker='o', color=color_cycle[i%len(unique_alphas)], label=f"$\\alpha$={a}")
                        for i, a in enumerate(unique_alphas)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.legend(["true",] +[f"$\\alpha$={a}" for a in unique_alphas ])
    plt.savefig(os.path.join(directory, f'profusion_true_mu_target_bias.png'), bbox_inches='tight')
    plt.clf()


def mse_box_plot(all_evaluation, title="No Title", directory=DEFAULT_DIR):
    data = defaultdict(list)
    labels = defaultdict(list)
    for evaluation in all_evaluation:
        for i, (n_test_samples, df) in enumerate(evaluation.groupby("n_test_samples")):
            mse = df.target_mse
            data[n_test_samples].append(mse)
            hp_set = df.i_hp.iloc[0]
            labels[n_test_samples].append( hp_set )
    for n_test_samples in data.keys():
        plt.boxplot(data[n_test_samples], labels=labels[n_test_samples])
        plt.xlabel('hyper-parameter set')
        plt.ylabel("MSE $\\hat \\mu$")
        plot_title = f"{title}_N={n_test_samples}"
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
        plt.title(now+plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'), bbox_inches='tight')
        plt.clf()


def sigma_box_plot(all_evaluation, title="No Title", directory=DEFAULT_DIR):
    data = defaultdict(list)
    labels = defaultdict(list)
    for evaluation in all_evaluation:
        for i, (n_test_samples, df) in enumerate(evaluation.groupby("n_test_samples")):
            sigma = df.sigma_mean
            data[n_test_samples].append(sigma)
            hp_set = df.i_hp.iloc[0]
            labels[n_test_samples].append( hp_set )
    for n_test_samples in data.keys():
        plt.boxplot(data[n_test_samples], labels=labels[n_test_samples])
        plt.xlabel('hyper-parameter set')
        plt.ylabel("$\\hat \\sigma_{\\hat \\mu}$")
        plot_title = f"{title}_N={n_test_samples}"
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
        plt.title(now+plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_sigma_mean.png'), bbox_inches='tight')
        plt.clf()


def v_stat_box_plot(all_evaluation, title="No Title", directory=DEFAULT_DIR):
    data = defaultdict(list)
    labels = defaultdict(list)
    for evaluation in all_evaluation:
        for i, (n_test_samples, df) in enumerate(evaluation.groupby("n_test_samples")):
            v_stat = df.var_stat
            data[n_test_samples].append(v_stat)
            hp_set = df.i_hp.iloc[0]
            labels[n_test_samples].append( hp_set )
    for n_test_samples in data.keys():
        plt.boxplot(data[n_test_samples], labels=labels[n_test_samples])
        plt.xlabel('hyper-parameter set')
        plt.ylabel("V_stat")
        plot_title = f"{title}_N={n_test_samples}"
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
        plt.title(now+plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_stat.png'), bbox_inches='tight')
        plt.clf()


def v_syst_box_plot(all_evaluation, title="No Title", directory=DEFAULT_DIR):
    data = defaultdict(list)
    labels = defaultdict(list)
    for evaluation in all_evaluation:
        for i, (n_test_samples, df) in enumerate(evaluation.groupby("n_test_samples")):
            v_syst = df.var_syst
            data[n_test_samples].append(v_syst)
            hp_set = df.i_hp.iloc[0]
            labels[n_test_samples].append( hp_set )
    for n_test_samples in data.keys():
        plt.boxplot(data[n_test_samples], labels=labels[n_test_samples])
        plt.xlabel('hyper-parameter set')
        plt.ylabel("V_syst")
        plot_title = f"{title}_N={n_test_samples}"
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
        plt.title(now+plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_syst.png'), bbox_inches='tight')
        plt.clf()
