# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import pandas as pd

from . import individual
from . import profusion

from itertools import product

from config import SAVING_DIR
BENCHMARK_NAME =  "COMPARE"


def hp_kwargs_generator(args):
    hp_names = args.keys()
    for a in product(*args.values()):
        yield {k: v for k, v in zip(hp_names, a)}



def make_individual_estimation_plots(evaluation, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, loader.model_full_name)
    os.makedirs(directory, exist_ok=True)

    individual.true_mu_mse(evaluation, title=loader.model_full_name, directory=directory)
    individual.true_mu_estimator(evaluation, title=loader.model_full_name, directory=directory)
    individual.true_mu_target_mean(evaluation, title=loader.model_full_name, directory=directory)
    individual.true_mu_target_mean_std(evaluation, title=loader.model_full_name, directory=directory)
    individual.n_samples_mse(evaluation, title=loader.model_full_name, directory=directory)
    individual.n_samples_sigma_mean(evaluation, title=loader.model_full_name, directory=directory)
    individual.box_n_samples_mse(evaluation, title=loader.model_full_name, directory=directory)


def make_profusion_estimation_plots(all_evaluations, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, "PROFUSION")
    os.makedirs(directory, exist_ok=True)
    title = f"{loader.benchmark_name}-{loader.base_name}"

    profusion.n_samples_mse(all_evaluations, title=title, directory=directory)
    profusion.n_samples_sigma_mean(all_evaluations, title=title, directory=directory)
    profusion.true_mu_estimator(all_evaluations, title=title, directory=directory)
    profusion.true_mu_target_mean(all_evaluations, title=title, directory=directory)
    profusion.true_mu_target_mean_std(all_evaluations, title=title, directory=directory)
    profusion.nominal_n_samples_mse(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.nominal_n_samples_sigma_mean(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.mse_box_plot(all_evaluations, title=title, directory=directory)


def make_individual_conditional_plots(evaluation, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, loader.model_full_name)
    os.makedirs(directory, exist_ok=True)

    individual.true_mu_v_stat(evaluation, title=loader.model_full_name, directory=directory)
    individual.true_mu_v_syst(evaluation, title=loader.model_full_name, directory=directory)
    individual.n_samples_v_stat(evaluation, title=loader.model_full_name, directory=directory)
    individual.n_samples_v_syst(evaluation, title=loader.model_full_name, directory=directory)
    

def make_profusion_conditional_plots(all_evaluations, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, "PROFUSION")
    os.makedirs(directory, exist_ok=True)
    title = f"{loader.benchmark_name}-{loader.base_name}"

    profusion.n_samples_v_stat(all_evaluations, title=title, directory=directory)
    profusion.n_samples_v_syst(all_evaluations, title=title, directory=directory)
    profusion.nominal_n_samples_v_stat(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.nominal_n_samples_v_syst(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.v_stat_box_plot(all_evaluations, title=title, directory=directory)
    profusion.v_syst_box_plot(all_evaluations, title=title, directory=directory)



def make_common_plots(data_name, benchmark_name, args, TheLoader):
    """
    make all the common individual plots and profusion plots.
    """
    all_evaluations = []
    all_loaders = []
    for kwargs in hp_kwargs_generator(args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        try:
            config_table = loader.load_config_table()
            evaluation = loader.load_estimation_evaluation()
        except FileNotFoundError:
            print(f"Missing estimation results for {loader.model_full_name}")
        else:
            evaluation = evaluation.join(config_table, rsuffix='_')
            all_evaluations.append(evaluation)
            all_loaders.append(loader)
            make_individual_estimation_plots(evaluation, loader)
    if all_evaluations:
        make_profusion_estimation_plots(all_evaluations, loader)

    for kwargs in hp_kwargs_generator(args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        try:
            config_table = loader.load_config_table()
            evaluation = loader.load_conditional_evaluation()
        except FileNotFoundError:
            print(f"Missing conditional estimation results for {loader.model_full_name}")
        else:
            evaluation = evaluation.join(config_table, rsuffix='_')
            all_evaluations.append(evaluation)
            all_loaders.append(loader)
            make_individual_conditional_plots(evaluation, loader)
    if all_evaluations:
        make_profusion_conditional_plots(all_evaluations, loader)




def make_hp_table(data_name, benchmark_name, args, TheLoader):
    some_hp = next(hp_kwargs_generator(args))
    loader = TheLoader(data_name, benchmark_name, **some_hp)
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "hp_table.csv")
    pd.DataFrame(hp_kwargs_generator(args)).to_csv(path)


def _stuff(args):
    print(list(filter(lambda kwargs : kwargs['max_depth']==3, hp_kwargs_generator(args))))


