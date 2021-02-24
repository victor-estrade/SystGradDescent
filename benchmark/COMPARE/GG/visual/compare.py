# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import matplotlib.pyplot as plt
from config import DEFAULT_DIR
from config import SAVING_DIR

def min_target_mse_mean(all_evaluation):
    full_name, best_evaluation = min(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
    return full_name, best_evaluation

def min_target_mse_median(all_evaluation):
    full_name, best_evaluation = min(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
    return full_name, best_evaluation

def min_sigma_mean_mean(all_evaluation):
    full_name, best_evaluation = min(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].sigma_mean.mean())
    return full_name, best_evaluation

def min_sigma_mean_median(all_evaluation):
    full_name, best_evaluation = min(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].sigma_mean.median())
    return full_name, best_evaluation


def extract(df, value_name, criterion):
    values = []
    methods = []
    for code_name, all_evaluation in df.groupby("code_name"):
        full_name, best_evaluation = criterion(all_evaluation)
        values.append(best_evaluation[value_name])
        methods.append(code_name)
    return values, methods

def min_avg_mse_mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        mse, methods = extract(df, "target_mse", min_target_mse_mean)
        plt.boxplot(mse, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("MSE $\\hat \\mu$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'), bbox_inches="tight")
        plt.clf()


def min_avg_mse_mse_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        mse, methods = extract(df, "target_mse", min_target_mse_mean)
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


def min_avg_mse_sigma_mean_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        sigma, methods = extract(df, "sigma_mean", min_target_mse_mean)
        plt.boxplot(sigma, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("average $\\hat \\sigma_{\\hat \\mu}$")
        plt.ylim(top=0.15, bottom=0.0)
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_sigma_mean.png'), bbox_inches="tight")
        plt.clf()


def min_avg_mse_v_stat_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        v_stat, methods = extract(df, "var_stat", min_target_mse_mean)
        plt.boxplot(v_stat, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_stat")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_stat.png'), bbox_inches="tight")
        plt.clf()


def min_avg_mse_v_stat_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        v_stat, methods = extract(df, "var_stat", min_target_mse_mean)
        x = list( range(len(methods)) )
        y = [v.mean() for v in v_stat]
        y_err = [v.std() for v in v_stat]
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=15, capthick=2)
        plt.xticks( ticks=x, labels=methods, rotation='vertical')
        plt.xlabel('method')
        plt.ylabel("Average V_stat $\pm std$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-errplot_v_stat.png'), bbox_inches="tight")
        plt.clf()


def min_avg_mse_v_syst_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        v_syst, methods = extract(df, "var_syst", min_target_mse_mean)
        plt.boxplot(v_syst, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_syst")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_syst.png'), bbox_inches="tight")
        plt.clf()



def min_avg_mse_v_syst_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_avg_mse_N={n_test_samples}"
        v_syst, methods = extract(df, "var_syst", min_target_mse_mean)
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


def min_median_mse_mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        mse, methods = extract(df, "target_mse", min_target_mse_median)
        plt.boxplot(mse, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("MSE $\\hat \\mu$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'), bbox_inches="tight")
        plt.clf()


def min_median_mse_mse_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        mse, methods = extract(df, "target_mse", min_target_mse_median)
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


def min_median_mse_v_stat_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        v_stat, methods = extract(df, "var_stat", min_target_mse_median)
        plt.boxplot(v_stat, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_stat")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_stat.png'), bbox_inches="tight")
        plt.clf()


def min_median_mse_v_stat_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        v_stat, methods = extract(df, "var_stat", min_target_mse_median)
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


def min_median_mse_v_syst_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        v_syst, methods = extract(df, "var_syst", min_target_mse_median)
        plt.boxplot(v_syst, labels=methods)
        plt.xticks(rotation=90)
        plt.xlabel('method')
        plt.ylabel("V_syst")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_v_syst.png'), bbox_inches="tight")
        plt.clf()


def min_median_mse_v_syst_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_min_median_mse_N={n_test_samples}"
        v_syst, methods = extract(df, "var_syst", min_target_mse_median)
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
