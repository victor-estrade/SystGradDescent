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


def mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        mse = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            mse.append(best_mse_evaluation.target_mse)
            methods.append(code_name)
        plt.boxplot(mse, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("MSE $\\hat \\mu$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'), bbox_inches="tight")
        plt.clf()


def mse_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        mse = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            mse.append(best_mse_evaluation.target_mse)
            methods.append(code_name)
        x = list( range(len(methods)) )
        y = [v.mean() for v in mse]
        y_err = [v.std() for v in mse]
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2)
        plt.xticks( ticks=x, labels=methods, rotation='vertical')
        plt.xlabel('method')
        plt.ylabel("Average MSE $\\hat \\mu \pm std $")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-errplot_mse.png'), bbox_inches="tight")
        plt.clf()


def v_stat_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        v_stat = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            v_stat.append(best_mse_evaluation.var_stat)
            methods.append(code_name)
        plt.boxplot(v_stat, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_stat")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_stat.png'), bbox_inches="tight")
        plt.clf()


def v_stat_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        v_stat = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            v_stat.append(best_mse_evaluation.var_stat)
            methods.append(code_name)
        x = list( range(len(methods)) )
        y = [v.mean() for v in v_stat]
        y_err = [v.std() for v in v_stat]
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2)
        plt.xticks( ticks=x, labels=methods, rotation='vertical')
        plt.xlabel('method')
        plt.ylabel("Average V_stat $\pm std$")
        plt.xlabel('method')
        plt.ylabel("Average V_stat $\pm std$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-errplot_v_stat.png'), bbox_inches="tight")
        plt.clf()


def v_syst_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        v_syst = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            v_syst.append(best_mse_evaluation.var_syst)
            methods.append(code_name)
        plt.boxplot(v_syst, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_syst")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_syst.png'), bbox_inches="tight")
        plt.clf()


def v_syst_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        v_syst = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            v_syst.append(best_mse_evaluation.var_syst)
            methods.append(code_name)
        x = list( range(len(methods)) )
        y = [v.mean() for v in v_syst]
        y_err = [v.std() for v in v_syst]
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2)
        plt.xticks( ticks=x, labels=methods, rotation='vertical')
        plt.xlabel('method')
        plt.ylabel("Average V_syst $\pm std$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-errplot_v_syst.png'), bbox_inches="tight")
        plt.clf()
