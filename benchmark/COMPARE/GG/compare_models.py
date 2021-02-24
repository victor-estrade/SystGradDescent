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

from .visual import compare

BENCHMARK_NAME =  "COMPARE"




def load_all_evaluation(TheLoader, hp_args, data_name='GG', benchmark_name='GG-marginal'):
    all_evaluation = []
    for kwargs in hp_kwargs_generator(hp_args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        try:
            evaluation = loader.load_evaluation_config()
        except FileNotFoundError:
            print(f"Missing results for {loader.model_full_name}")
        else:
            all_evaluation.append(evaluation)
    return all_evaluation



def load_all_data(all_hp, all_loader_classes, all_code_names, data_name='GG', benchmark_name='GG-calib'):
    all_data = []
    for hp_args, TheLoader, name in zip(all_hp, all_loader_classes, all_code_names):
        all_evaluation = load_all_evaluation(TheLoader, hp_args, data_name=data_name, benchmark_name=benchmark_name)
        print(f" found {len(all_evaluation)} completed runs for {name} in {benchmark_name}")
        if all_evaluation :
            all_evaluation = pd.concat(all_evaluation)
            all_evaluation['code_name'] = name
            all_data.append(all_evaluation)
    return all_data




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

    data_name = 'GG'

    # CALIB PLOTS

    marginal_eval = load_all_evaluation(REGLoader, REG_M_HP, data_name=data_name, benchmark_name='GG-marginal')
    if marginal_eval :
        marginal_eval = pd.concat(marginal_eval, sort=False)
        marginal_eval['base_name'] = "Marginal"
        marginal_eval['code_name'] = "Blind-REG"

    benchmark_name = 'GG-calib'
    all_data = load_all_data(ALL_HP, ALL_LOADER, ALL_NAME, benchmark_name=benchmark_name)
    data = pd.concat(all_data, sort=False)
    data_and_marginal = pd.concat(all_data+[marginal_eval], sort=False)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    compare.best_average_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_average_sigma_mean_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_average_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_syst_err_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    compare.best_median_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_median_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_median_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_v_syst_err_plot(data, title=benchmark_name, directory=directory)

    # PRIOR PLOTS

    benchmark_name = 'GG-prior'
    all_data = load_all_data(ALL_HP, ALL_LOADER, ALL_NAME, data_name=data_name, benchmark_name=benchmark_name)
    data = pd.concat(all_data, sort=False)
    data_and_marginal = pd.concat(all_data+[marginal_eval], sort=False)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    compare.best_average_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_average_sigma_mean_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_average_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    compare.best_average_v_syst_err_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    compare.best_median_mse_box_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_median_v_stat_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_v_syst_box_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_mse_err_plot(data_and_marginal, title=benchmark_name, directory=directory)
    compare.best_median_v_stat_err_plot(data, title=benchmark_name, directory=directory)
    compare.best_median_v_syst_err_plot(data, title=benchmark_name, directory=directory)



if __name__ == '__main__':
    main()
