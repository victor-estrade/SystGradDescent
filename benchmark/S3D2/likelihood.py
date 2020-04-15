#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd


from config import SAVING_DIR

from visual import set_plot_config
set_plot_config()
from visual.misc import plot_params

from utils.log import print_params
from utils.evaluation import estimate
from utils.evaluation import register_params
from utils.evaluation import evaluate_minuit

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config
from problem.synthetic3D import get_minimizer
from problem.synthetic3D import Parameter


SEED = None
BENCHMARK_NAME = "S3D2"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "Likelihood")

def main():
    print("Hello world !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    config = S3D2Config()
    DATA_N_SAMPLES = 80_000

    result_table = []

    for true_mu in config.TRUE_MU_RANGE:
        result_row = {}
        true_params = Parameter(config.TRUE.r, config.TRUE.lam, true_mu)
        generator = S3D2(SEED)
        data, label = generator.sample_event(*true_params, size=DATA_N_SAMPLES)
        n_sig = np.sum(label==1)
        n_bkg = np.sum(label==0)
        print(f"nb of signal      = {n_sig}")
        print(f"nb of backgrounds = {n_bkg}")

        compute_nll = lambda r, lam, mu : generator.nll(data, r, lam, mu)

        print('Prepare minuit minimizer')
        minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR)
        fmin, params = estimate(minimizer)
        result_row.update(evaluate_minuit(minimizer, fmin, params, true_params))

        result_table.append(result_row.copy())
    result_table = pd.DataFrame(result_table)
    result_table.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    print('Plot params')
    param_names = config.PARAM_NAMES
    for name in param_names:
        plot_params(name, result_table, title='Likelihood fit', directory=DIRECTORY)

if __name__ == '__main__':
    main()