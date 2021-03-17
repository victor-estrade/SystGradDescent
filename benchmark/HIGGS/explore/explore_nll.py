# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.HIGGS.explore.explore_nll

import os
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line

from utils.evaluation import evaluate_minuit

from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_minimizer
from problem.higgs import get_minimizer_no_nuisance
from problem.higgs import get_generators_torch
from problem.higgs import param_generator
from problem.higgs import Generator
from problem.higgs import Parameter
from problem.higgs import HiggsNLL as NLLComputer

from model.gradient_boost import GradientBoostingModel

from .common import N_BINS

DATA_NAME = 'HIGGS'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "explore")

import argparse

def parse_args(main_description="Explore NLL shape"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')

    args = parser.parse_args()
    return args


def main():
    logger = set_logger()
    directory = os.path.join(DIRECTORY, "nll_contour")
    os.makedirs(directory, exist_ok=True)
    args = parse_args()
    i_cv = 0
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda)

    config = Config()
    model = load_some_model()
    for i_iter, test_config in enumerate(config.iter_test_config()):
        do_iter(config, model, i_iter, valid_generator, test_generator)



def do_iter(config, model, i_iter, valid_generator, test_generator):
    logger = logging.getLogger()
    directory = os.path.join(DIRECTORY, "nll_contour", f"iter_{i_iter}")

    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES, no_grad=True)

    logger.info('Set up NLL computer')
    compute_summaries = model.summary_computer(n_bins=N_BINS)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)

    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"Calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"TRUE nll = {nll}")

    # MINIMIZE NLL
    logger.info('Prepare minuit minimizer')
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
    some_dict =  evaluate_minuit(minimizer, config.TRUE, directory, suffix="")

    # MESH NLL
    mu_array = np.linspace(0.5, 1.5, 8)
    tes_array = np.linspace(0.95, 1.05, 8)
    mu_mesh, tes_mesh = np.meshgrid(mu_array, tes_array)
    nll_func = lambda mu, tes : compute_nll(tes, config.TRUE.jes, config.TRUE.les, mu)
    nll_mesh = np.array([nll_func(mu, tes) for mu, tes in zip(mu_mesh.ravel(), tes_mesh.ravel())]).reshape(mu_mesh.shape)

    # plot NLL contour
    fig, ax = plt.subplots()
    CS = ax.contour(mu_mesh, tes_mesh, nll_mesh)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("mu")
    ax.set_ylabel("tes")
    fname = f"contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    logger.info(f"saved at {path}")



def load_some_model():
    from model.gradient_boost import GradientBoostingModel
    i_cv = 0
    model = GradientBoostingModel(learning_rate=0.1, n_estimators=300, max_depth=3)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model

def load_some_NN():
    from model.neural_network import NeuralNetClassifier
    from archi.classic import L4 as ARCHI
    import torch.optim as optim
    i_cv = 0
    n_unit = 200
    net = ARCHI(n_in=29, n_out=2, n_unit=n_unit)
    optimizer = optim.Adam(lr=1e-4)
    model = NeuralNetClassifier(net, optimizer, n_steps=5000, batch_size=20, cuda=False)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)


if __name__ == '__main__':
    main()
