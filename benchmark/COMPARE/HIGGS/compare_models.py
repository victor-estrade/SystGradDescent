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

from .visual import best_average
from .visual import best_median

import pandas as pd


import os

from visual.misc import set_plot_config
set_plot_config()

import matplotlib.pyplot as plt
from config import DEFAULT_DIR
from config import SAVING_DIR

BENCHMARK_NAME =  "COMPARE"



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
    best_average.mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_average.mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_average.true_mu_mse(data_and_marginal, title=benchmark_name, directory=directory)
    best_average.true_mu_sigma_mean(data_and_marginal, title=benchmark_name, directory=directory)
    best_average.true_mu_target_std(data_and_marginal, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median.mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_median.mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    best_median.true_mu_mse(data_and_marginal, title=benchmark_name, directory=directory)
    best_median.true_mu_sigma_mean(data_and_marginal, title=benchmark_name, directory=directory)
    best_median.true_mu_target_std(data_and_marginal, title=benchmark_name, directory=directory)


def make_common_conditional_plots(data, benchmark_name):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    best_average.v_stat_box_plot(data, title=benchmark_name, directory=directory)
    best_average.v_syst_box_plot(data, title=benchmark_name, directory=directory)
    best_average.v_stat_err_plot(data, title=benchmark_name, directory=directory)
    best_average.v_syst_err_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median.v_stat_box_plot(data, title=benchmark_name, directory=directory)
    best_median.v_syst_box_plot(data, title=benchmark_name, directory=directory)
    best_median.v_stat_err_plot(data, title=benchmark_name, directory=directory)
    best_median.v_syst_err_plot(data, title=benchmark_name, directory=directory)


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
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")

    # HIGGSTES PRIOR PLOTS
    benchmark_name = 'HIGGSTES-prior'
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")

    # EASYHIGGSTES CALIB PLOTS
    benchmark_name = 'EASYHIGGSTES-calib'
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")

    # EASYHIGGSTES PRIOR PLOTS
    benchmark_name = 'EASYHIGGSTES-prior'
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")

    # BALANCEDHIGGSTES CALIB PLOTS
    benchmark_name = 'BALANCEDHIGGSTES-calib'
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")

    # BALANCEDHIGGSTES PRIOR PLOTS
    benchmark_name = 'BALANCEDHIGGSTES-prior'
    print()
    print("="*15, benchmark_name, "="*15)
    all_estimation_data = load_all_estimation_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    all_conditional_data = load_all_conditional_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    if all_estimation_data :
        data_estimation_and_marginal = pd.concat(all_estimation_data+[marginal_eval], sort=False)
        make_common_estimation_plots(data_estimation_and_marginal, benchmark_name)
    else:
        print(f"WARNING : FOUND NO ESTIMATION FOR {benchmark_name}")
    if all_conditional_data:
        data_conditional = pd.concat(all_conditional_data, sort=False)
        make_common_conditional_plots(data_conditional, benchmark_name)
    else:
        print(f"WARNING : FOUND NO CONDITIONAL ESTIMATION FOR {benchmark_name}")


if __name__ == '__main__':
    main()