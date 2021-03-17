# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.MonoHIGGS.explore.NN_nll

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
from utils.evaluation import evaluate_config

from problem.higgs import MonoHiggsConfig as Config
from problem.higgs import get_mono_minimizer as get_minimizer
from problem.higgs import get_generators_torch
from problem.higgs import MonoHiggsNLL as NLLComputer
from problem.higgs import MonoGeneratorTorch

from ..common import N_BINS
from ..common import GeneratorCPU

from .load_model import load_some_GB
from .load_model import load_some_NN

from .contour_nll import do_iter

DATA_NAME = 'HIGGS'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "explore_mono")

import argparse

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
    i_cv = 0
    seed = SEED + i_cv * 5
    train_generator, valid_generator, test_generator = get_generators_torch(seed, cuda=args.cuda, GeneratorClass=MonoGeneratorTorch)
    train_generator = GeneratorCPU(train_generator)
    valid_generator = GeneratorCPU(valid_generator)
    test_generator = GeneratorCPU(test_generator)

    model = load_some_NN()

    config = Config()
    config_table = evaluate_config(config)
    os.makedirs(os.path.join(directory, model.name), exist_ok=True)
    config_table.to_csv(os.path.join(directory, model.name, 'config_table.csv'))
    for i_iter, test_config in enumerate(config.iter_test_config()):
        do_iter(test_config, model, i_iter, valid_generator, test_generator)




if __name__ == '__main__':
    main()