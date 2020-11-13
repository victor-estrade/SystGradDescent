#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from .hyper_parameters import DA_HP
from .hyper_parameters import GB_HP
from .hyper_parameters import INF_HP
from .hyper_parameters import NN_HP
from .hyper_parameters import PIVOT_HP
from .hyper_parameters import REG_HP
from .hyper_parameters import REG_M_HP
from .hyper_parameters import TP_HP

from ..loader import DALoader
from ..loader import GBLoader
from ..loader import INFLoader
from ..loader import NNLoader
from ..loader import PIVOTLoader
from ..loader import REGLoader
from ..loader import TPLoader

from .visual.common import hp_kwargs_generator


import pandas as pd


import os

from visual.misc import set_plot_config
set_plot_config()

import matplotlib.pyplot as plt
from config import DEFAULT_DIR
from config import SAVING_DIR

BENCHMARK_NAME =  "COMPARE"


def best_average_mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        mse = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
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


def best_average_mse_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        mse = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
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


def best_average_v_stat_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        v_stat = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
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


def best_average_v_stat_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        v_stat = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
            v_stat.append(best_mse_evaluation.var_stat)
            methods.append(code_name)
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


def best_average_v_syst_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        v_syst = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
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



def best_average_v_syst_err_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_average_N={n_test_samples}"
        v_syst = []
        methods = []
        for code_name, all_evaluation in df.groupby("code_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
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


def best_median_mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
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


def best_median_mse_err_plot(data, title="No Title", directory=DEFAULT_DIR):
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


def best_median_v_stat_box_plot(data, title="No Title", directory=DEFAULT_DIR):
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


def best_median_v_stat_err_plot(data, title="No Title", directory=DEFAULT_DIR):
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


def best_median_v_syst_box_plot(data, title="No Title", directory=DEFAULT_DIR):
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


def best_median_v_syst_err_plot(data, title="No Title", directory=DEFAULT_DIR):
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



def load_all_estimation_evaluation(TheLoader, hp_args, data_name='HIGGSTES', benchmark_name='HIGGTES-marginal'):
    all_evaluation = []
    for kwargs in hp_kwargs_generator(hp_args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        try:
            config_table = loader.load_config_table()
            evaluation = loader.load_estimation_evaluation()
        except FileNotFoundError:
            print(f"Missing estimation results for {loader.model_full_name}")
        else:
            evaluation = evaluation.join(config_table, rsuffix='_')
            all_evaluation.append(evaluation)
    return all_evaluation


def load_all_conditional_evaluation(TheLoader, hp_args, data_name='HIGGSTES', benchmark_name='HIGGTES-marginal'):
    all_evaluation = []
    for kwargs in hp_kwargs_generator(hp_args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        try:
            config_table = loader.load_config_table()
            evaluation = loader.load_estimation_evaluation()
            conditional_evaluation = loader.load_conditional_evaluation()
        except FileNotFoundError:
            print(f"Missing conditional estimation results for {loader.model_full_name}")
        else:
            evaluation = evaluation.join(config_table, rsuffix='_')
            evaluation = evaluation.join(conditional_evaluation, rsuffix='__')
            all_evaluation.append(evaluation)
    return all_evaluation


def load_all_estimation_data(all_hp, all_loader_classes, all_code_names, data_name='HIGGSTES', benchmark_name='HIGGSTES-prior'):
    all_data = []
    for hp_args, TheLoader, name in zip(all_hp, all_loader_classes, all_code_names):
        all_evaluation = load_all_estimation_evaluation(TheLoader, hp_args, data_name=data_name, benchmark_name=benchmark_name)
        print(f" found {len(all_evaluation)} completed estimation runs for {name} in {benchmark_name}")
        if all_evaluation :
            all_evaluation = pd.concat(all_evaluation)
            all_evaluation['code_name'] = name
            all_data.append(all_evaluation)
    return all_data


def load_all_conditional_data(all_hp, all_loader_classes, all_code_names, data_name='HIGGSTES', benchmark_name='HIGGSTES-prior'):
    all_data = []
    for hp_args, TheLoader, name in zip(all_hp, all_loader_classes, all_code_names):
        all_evaluation = load_all_conditional_evaluation(TheLoader, hp_args, data_name=data_name, benchmark_name=benchmark_name)
        print(f" found {len(all_evaluation)} completed conditional runs for {name} in {benchmark_name}")
        if all_evaluation :
            all_evaluation = pd.concat(all_evaluation)
            all_evaluation['code_name'] = name
            all_data.append(all_evaluation)
    return all_data


def make_common_estimation_plots(data_and_marginal, benchmark_name):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    best_average_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_average_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_median_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)


def make_common_conditional_plots(data, benchmark_name):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    best_average_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    best_average_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    best_average_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    best_average_v_syst_err_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    best_median_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    best_median_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    best_median_v_syst_err_plot(data, title=benchmark_name, directory=directory)


def main():
    print("hello")
    os.makedirs(DEFAULT_DIR, exist_ok=True)

    ALL_HP = [
                DA_HP
                , GB_HP
                , INF_HP
                , NN_HP
                , PIVOT_HP
                , REG_HP
                , TP_HP
                ]
    ALL_LOADER = [
                DALoader
                , GBLoader
                , INFLoader
                , NNLoader
                , PIVOTLoader
                , REGLoader
                , TPLoader
                ]
    ALL_NAME = [
                "DA"
                , "GB"
                , "INF"
                , "NN"
                , "PIVOT"
                , "Param-REG"
                , "TP"
                ]

    data_name = 'HIGGSTES'

    # MARGINAL LOAD
    marginal_eval = load_all_estimation_evaluation(REGLoader, REG_M_HP, data_name=data_name, benchmark_name='HIGGTES-marginal')
    if marginal_eval :
        marginal_eval = pd.concat(marginal_eval, sort=False)
        marginal_eval['base_name'] = "Marginal"
        marginal_eval['code_name'] = "Blind-REG"
    else:
        marginal_eval = pd.DataFrame()

    # HIGGSTES CALIB PLOTS
    benchmark_name = 'HIGGSTES-calib'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)

    # HIGGSTES PRIOR PLOTS
    benchmark_name = 'HIGGSTES-prior'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)

    # EASYHIGGSTES CALIB PLOTS
    benchmark_name = 'EASYHIGGSTES-calib'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)

    # EASYHIGGSTES PRIOR PLOTS
    benchmark_name = 'EASYHIGGSTES-prior'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)

    # BALANCEDHIGGS CALIB PLOTS
    benchmark_name = 'BALANCEDHIGGS-calib'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)

    # BALANCEDHIGGS PRIOR PLOTS
    benchmark_name = 'BALANCEDHIGGS-prior'
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)


if __name__ == '__main__':
    main()
