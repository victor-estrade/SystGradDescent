# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.HIGGS.explore.tau_effect


import os
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

from problem.higgs.higgs_geant import load_data
from problem.higgs.higgs_geant import split_data_label_weights

from problem.higgs import get_parameter_class
from problem.higgs import get_higgsnll_class
from problem.higgs import get_config_class
from problem.higgs import get_generator_class
from problem.higgs import get_higgsloss_class
from problem.higgs import get_parameter_generator
TES =  True
JES =  False
LES =  False
Parameter = get_parameter_class(TES, JES, LES)
NLLComputer = get_higgsnll_class(TES, JES, LES)
Config = get_config_class(TES, JES, LES)
GeneratorClass = get_generator_class(TES, JES, LES)
HiggsLoss = get_higgsloss_class(TES, JES, LES)
param_generator = get_parameter_generator(TES, JES, LES)


DATA_NAME = 'HIGGS'
BENCHMARK_NAME = DATA_NAME
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")



def main():
    print('Hello world')
    os.makedirs(DIRECTORY, exist_ok=True)
    data = load_data()
    generator = GeneratorClass(data, seed=2)

    dirname = os.path.join(DIRECTORY, 'tes_minibatch')
    os.makedirs(dirname, exist_ok=True)
    minibatchsize(generator, dirname=dirname)


def minibatchsize(generator, dirname=DIRECTORY):
    DELTA = 0.03
    config = Config()
    nominal_param = config.CALIBRATED
    up_param = nominal_param.clone_with(tes=nominal_param.tes + DELTA)
    down_param = nominal_param.clone_with(tes=nominal_param.tes - DELTA)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")

    mean_values = get_mean_pri_tau_pt_means(generator, nominal_param)
    for k, v in mean_values.items():
        print(f'{k} : {np.mean(v)} +/- {np.std(v)}')

    generator.reset()
    mean_values = get_mean_pri_tau_pt_means(generator, up_param)
    for k, v in mean_values.items():
        print(f'{k} : {np.mean(v)} +/- {np.std(v)}')

    generator.reset()
    mean_values = get_mean_pri_tau_pt_means(generator, down_param)
    for k, v in mean_values.items():
        print(f'{k} : {np.mean(v)} +/- {np.std(v)}')

def get_mean_pri_tau_pt_means(generator, param):
    print(param, *param)
    N_SAMPLES = 1
    SAMPLE_SIZE = np.arange(10_000, 100_000, 10_000)
    pri_tau_pt_idx = 12
    print(generator.feature_names[pri_tau_pt_idx])
    mean_values = {}
    for sample_size in SAMPLE_SIZE:
        print(f'processing with {sample_size} events ...')
        mean_values[sample_size] = []
        for i in range(N_SAMPLES):
            X, y, w = generator.generate(*param, n_samples=sample_size, no_grad=True)
            pri_tau_pt = X[:, pri_tau_pt_idx]
            print(pri_tau_pt)
            pri_tau_pt_mean = (pri_tau_pt * w).sum() / w.sum()
            mean_values[sample_size].append( pri_tau_pt_mean.detach().numpy() )
    return mean_values



if __name__ == '__main__':
    main()
