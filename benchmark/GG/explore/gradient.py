#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.gradient


import os
import argparse
import numpy as np

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line

from ..common import N_BINS

from .load_model import load_some_GB
from .load_model import load_some_NN


from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import GGNLL as NLLComputer


from scipy.optimize import approx_fprime
from scipy.optimize import fmin_bfgs

DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "gradient", )

def parse_args(main_description="Explore NLL shape"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")

    args, _ = parser.parse_known_args()
    return args



def main():
    logger = set_logger()
    directory = os.path.join(DIRECTORY, "nll_contour")
    os.makedirs(directory, exist_ok=True)
    args = parse_args()

    train_generator, valid_generator, test_generator = get_generators()

    config = Config()
    model = load_some_NN(cuda=args.cuda)
    compute_nll = get_nll_computer(model, config, valid_generator, test_generator)

    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"calib nll = {nll}")


    f = lambda xk : compute_nll(*xk)
    xk = np.array(list(config.TRUE))
    print(xk)
    EPSILON = 1e-8
    epsilon = np.array([EPSILON]*2)
    grad = approx_fprime(xk, f, epsilon)
    print(grad, grad.dot(grad.T))

    logger.info(f"Running BFGS on the NLL")
    x_0 = np.array(list(config.CALIBRATED))
    print(fmin_bfgs(f, x_0))




def get_generators(i_cv=0):
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    valid_generator = Generator(seed + 1)
    test_generator  = Generator(seed + 2)
    return train_generator, valid_generator, test_generator


def get_nll_computer(model, config, valid_generator, test_generator):
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)

    compute_summaries = model.summary_computer(n_bins=N_BINS)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    return compute_nll





if __name__ == '__main__':
    main()
