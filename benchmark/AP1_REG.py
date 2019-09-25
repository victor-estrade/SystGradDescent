#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.AP1_GB

import os
import logging
import config
import iminuit
ERRORDEF_NLL = 0.5

from utils import set_plot_config
set_plot_config()
from utils import set_logger
from utils import flush
from utils import print_line
from utils import get_model
from utils import print_params

from myplot import plot_params

from problem.apples_and_pears import AP1

from model.regressor import Regressor
from archi.net import RegNet

from .my_argparser import REG_parse_args

BENCHMARK_NAME = 'AP1'
N_ITER = 3


def param_generator():
    import numpy as np
    apple_ratio = np.random.uniform(0., 1.0)
    return (apple_ratio,)


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on AP1 benchmark")
    logger.info(args)
    flush(logger)
    for i_cv in range(N_ITER):
        run(args, i_cv)


def run(args, i_cv):
    logger = logging.getLogger()
    print_line()
    logger.info('Running iter n°{}'.format(i_cv))
    print_line()
    # LOAD/GENERATE DATA
    logger.info('Set up data generator')
    seed = config.SEED + i_cv * 5
    train_generator = AP1(seed)
    valid_generator = AP1(seed+1)
    test_generator  = AP1(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    net = RegNet(n_in=1, n_out=2)
    args.net = net
    model = get_model(args, Regressor)
    model.param_generator = param_generator
    logger.info('Training {}'.format(model.get_name()))
    model.fit(train_generator)
    logger.info('Training DONE')

    # SAVE MODEL
    model_name = '{}-{}'.format(model.get_name(), i_cv)
    model_path = os.path.join(config.SAVING_DIR, BENCHMARK_NAME, model_name)
    logger.info("Saving in {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)

    # CHECK TRAINING
    # import numpy as np
    import matplotlib.pyplot as plt
    # import seaborn as sns

    losses = model.losses
    mse_losses = model.mse_losses
    
    plt.plot(losses, label='loss')
    plt.plot(mse_losses, label='mse')
    plt.title(model_name)
    plt.xlabel('# iter')
    plt.ylabel('Loss/MSE')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'losses.png'))
    plt.clf()




    # MEASUREMENT
    logger.info('Generate testing data')
    true_apple_ratio = 0.8
    param_truth = [true_apple_ratio]
    X_test, y_test, w_test = test_generator.generate(apple_ratio=true_apple_ratio, n_samples=2_000)

    pred, sigma = model.predict(X_test, w_test)
    print(true_apple_ratio, '=vs=', pred, '+/-', sigma)




    logger.info('DONE')

if __name__ == '__main__':
    main()
	