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
        for base_name, all_evaluation in df.groupby("base_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.mean())
            mse.append(best_mse_evaluation.target_mse)    
            methods.append(base_name)
        plt.boxplot(mse, labels=methods)
        plt.xlabel('method')
        plt.ylabel("MSE $\\hat \\mu$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'))
        plt.clf()


def best_median_mse_box_plot(data, title="No Title", directory=DEFAULT_DIR):
    for n_test_samples, df in data.groupby("n_test_samples"):
        plot_title = f"{title}_best_median_N={n_test_samples}"
        mse = []
        methods = []
        for base_name, all_evaluation in df.groupby("base_name"):
            full_name, best_mse_evaluation = max(all_evaluation.groupby('model_full_name'), key=lambda t : t[1].target_mse.median())
            mse.append(best_mse_evaluation.target_mse)    
            methods.append(base_name)
        plt.boxplot(mse, labels=methods)
        plt.xlabel('method')
        plt.ylabel("MSE $\\hat \\mu$")
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(os.path.join(directory, f'{plot_title}-boxplot_mse.png'))
        plt.clf()


def main():
    print("hello")
    os.makedirs(DEFAULT_DIR, exist_ok=True)

    ALL_HP = [DA_HP
                , GB_HP
                # , INF_HP
                , NN_HP
                # , PIVOT_HP
                , REG_HP
                # , TP_HP
                ]
    ALL_LOADER = [DALoader
                , GBLoader
                # , INFLoader
                , NNLoader
                # , PIVOTLoader
                , REGLoader
                # , TPLoader
                ]
    ALL_NAME = ["DA"
                , "GB"
                # , "INF"
                , "NN"
                # , "PIVOT"
                , "REG"
                # , "TP"
                ]
    data_name = 'GG'
    
    benchmark_name = 'GG-calib'
    all_data = []
    for hp_args, TheLoader, name in zip(ALL_HP, ALL_LOADER, ALL_NAME):
        all_loader = [TheLoader(data_name, benchmark_name, **kwargs) for kwargs in hp_kwargs_generator(hp_args)]
        all_evaluation = [loader.load_evaluation_config() for loader in all_loader]
        all_data.append(pd.concat(all_evaluation))

    data = pd.concat(all_data, sort=False)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    best_average_mse_box_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median_mse_box_plot(data, title=benchmark_name, directory=directory)


    benchmark_name = 'GG-prior'
    all_data = []
    for hp_args, TheLoader, name in zip(ALL_HP, ALL_LOADER, ALL_NAME):
        all_loader = [TheLoader(data_name, benchmark_name, **kwargs) for kwargs in hp_kwargs_generator(hp_args)]
        all_evaluation = [loader.load_evaluation_config() for loader in all_loader]
        all_data.append(pd.concat(all_evaluation))

    data = pd.concat(all_data, sort=False)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MSE")
    os.makedirs(directory, exist_ok=True)
    best_average_mse_box_plot(data, title=benchmark_name, directory=directory)

    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, benchmark_name, "BEST_MEDIAN")
    os.makedirs(directory, exist_ok=True)
    best_median_mse_box_plot(data, title=benchmark_name, directory=directory)
            




if __name__ == '__main__':
    main()
