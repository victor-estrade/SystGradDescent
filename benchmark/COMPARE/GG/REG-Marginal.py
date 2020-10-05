#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

from .visual import individual
from .visual import profusion
from .visual.common import hp_kwargs_generator
from .visual.common import make_hp_table
from ..loader import REGLoader
from .hyper_parameters import REG_M_HP

from config import SAVING_DIR
BENCHMARK_NAME =  "COMPARE"

def main():
    print("hello")
    data_name = 'GG'
    benchmark_name = 'GG-marginal'
    make_hp_table(data_name, benchmark_name, REG_M_HP, REGLoader)
    all_evaluations = []
    all_loaders = []
    for kwargs in hp_kwargs_generator(REG_M_HP):
        loader = REGLoader(data_name, benchmark_name, **kwargs)
        evaluation = loader.load_evaluation_config()
        
        all_evaluations.append(evaluation)
        all_loaders.append(loader)

        directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, loader.model_full_name)
        os.makedirs(directory, exist_ok=True)
        individual.true_mu_mse(evaluation, title=loader.model_full_name, directory=directory)
        individual.true_mu_estimator(evaluation, title=loader.model_full_name, directory=directory)
        individual.true_mu_target_mean(evaluation, title=loader.model_full_name, directory=directory)
        individual.true_mu_target_mean_std(evaluation, title=loader.model_full_name, directory=directory)
        individual.n_samples_mse(evaluation, title=loader.model_full_name, directory=directory)
        individual.n_samples_sigma_mean(evaluation, title=loader.model_full_name, directory=directory)
        individual.box_n_samples_mse(evaluation, title=loader.model_full_name, directory=directory)
        
    title = f"{loader.benchmark_name}-{loader.base_name}"
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, "PROFUSION")
    os.makedirs(directory, exist_ok=True)
    profusion.n_samples_mse(all_evaluations, title=title, directory=directory)
    profusion.n_samples_sigma_mean(all_evaluations, title=title, directory=directory)
    profusion.true_mu_estimator(all_evaluations, title=title, directory=directory)
    profusion.true_mu_target_mean(all_evaluations, title=title, directory=directory)
    profusion.true_mu_target_mean_std(all_evaluations, title=title, directory=directory)
    profusion.nominal_n_samples_mse(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.nominal_n_samples_sigma_mean(all_evaluations, title=f"Nominal {title}", directory=directory)
    profusion.mse_box_plot(all_evaluations, title=title, directory=directory)






if __name__ == '__main__':
    main()
