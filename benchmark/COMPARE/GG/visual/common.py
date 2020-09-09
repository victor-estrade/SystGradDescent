# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from . import individual
from . import profusion

from itertools import product

from config import SAVING_DIR
BENCHMARK_NAME =  "COMPARE"


def hp_kwargs_generator(args):
    hp_names = args.keys()
    for a in product(*args.values()):
        yield {k: v for k, v in zip(hp_names, a)}



def make_individual_plots(evaluation, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, loader.model_full_name)
    os.makedirs(directory, exist_ok=True)

    individual.plot_eval_mse(evaluation, title=loader.model_full_name, directory=directory)
    individual.plot_eval_mu(evaluation, title=loader.model_full_name, directory=directory)
    individual.plot_eval_v_stat(evaluation, title=loader.model_full_name, directory=directory)
    individual.plot_eval_v_syst(evaluation, title=loader.model_full_name, directory=directory)
    individual.plot_n_samples_v_stat(evaluation, title=loader.model_full_name, directory=directory)
    individual.plot_n_samples_v_syst(evaluation, title=loader.model_full_name, directory=directory)


def make_profusion_plots(all_evaluations, loader):
    directory = os.path.join(SAVING_DIR, BENCHMARK_NAME, loader.benchmark_name, loader.base_name, "PROFUSION")
    os.makedirs(directory, exist_ok=True)
    title = f"{loader.benchmark_name}-{loader.base_name}"

    profusion.n_samples_v_stat(all_evaluations, title=title, directory=directory)
    profusion.n_samples_v_syst(all_evaluations, title=title, directory=directory)
    profusion.n_samples_mse(all_evaluations, title=title, directory=directory)
    profusion.true_mu_estimator(all_evaluations, title=title, directory=directory)
    


def make_common_plots(data_name, benchmark_name, args, TheLoader):
    """
    make all the common individual plots and profusion plots.
    """
    all_evaluations = []
    for kwargs in hp_kwargs_generator(args):
        loader = TheLoader(data_name, benchmark_name, **kwargs)
        evaluation = loader.load_evaluation_config()
        all_evaluations.append(evaluation)
        make_individual_plots(evaluation, loader)
    make_profusion_plots(all_evaluations, loader)


def _stuff(args):
    print(list(filter(lambda kwargs : kwargs['max_depth']==3, hp_kwargs_generator(args))))


