#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.S3D2.REG

import os
import logging
import config

from utils.plot import set_plot_config
set_plot_config()
from utils.log import set_logger
from utils.log import flush
from utils.log import print_line
from utils.model import get_model
from utils.model import get_model_id
from utils.model import get_model_path
from utils.model import save_model


from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config

from model.regressor import Regressor
from archi.net import RegNet

from ..my_argparser import REG_parse_args

BENCHMARK_NAME = 'S3D2'
N_ITER = 5


def param_generator():
    import numpy as np
    pb_config = S3D2Config()

    # r = np.random.normal(pb_config.CALIBRATED_R, pb_config.CALIBRATED_R_ERROR)
    # lam = -1
    # while lam <= 0:
    #     lam = np.random.normal(pb_config.CALIBRATED_LAMBDA, pb_config.CALIBRATED_LAMBDA_ERROR)
    
    r = pb_config.CALIBRATED_R
    lam = pb_config.CALIBRATED_LAMBDA
    
    mu = np.random.uniform(0, 1)
    return (r, lam, mu,)


def main():
    # BASIC SETUP
    logger = set_logger()
    args = REG_parse_args(main_description="Training launcher for Regressor on S3D2 benchmark")
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
    pb_config = S3D2Config()
    seed = config.SEED + i_cv * 5
    train_generator = S3D2(seed)
    valid_generator = S3D2(seed+1)
    test_generator  = S3D2(seed+2)

    # SET MODEL
    logger.info('Set up rergessor')
    net = RegNet(n_in=3, n_out=2, n_extra=2)
    args.net = net
    model = get_model(args, Regressor)
    model.param_generator = param_generator
    logger.info('Training {}'.format(model.get_name()))
    model.fit_batch(train_generator)
    logger.info('Training DONE')

    # SAVE MODEL
    model_path = get_model_path(BENCHMARK_NAME, model, i_cv)
    save_model(model, model_path)

    # CHECK TRAINING
    # import numpy as np
    import matplotlib.pyplot as plt
    # import seaborn as sns
    model_id = get_model_id(model, i_cv)

    losses = model.losses
    mse_losses = model.mse_losses
    
    plt.plot(losses, label='loss')
    plt.plot(mse_losses, label='mse')
    plt.title(model_id)
    plt.xlabel('# iter')
    plt.ylabel('Loss/MSE')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'losses.png'))
    plt.clf()

    plt.plot(mse_losses, label='mse')
    plt.title(model_id)
    plt.xlabel('# iter')
    plt.ylabel('Loss/MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'mse_loss.png'))
    plt.clf()


    # MEASUREMENT
    logger.info('Generate testing data')
    X_test, y_test, w_test = test_generator.generate(
                                     # pb_config.TRUE_R,
                                     # pb_config.TRUE_LAMBDA,
                                     pb_config.CALIBRATED_R,
                                     pb_config.CALIBRATED_LAMBDA,
                                     pb_config.TRUE_MU,
                                     n_samples=pb_config.N_TESTING_SAMPLES)
    import numpy as np
    p_test = np.array( (pb_config.CALIBRATED_R, pb_config.CALIBRATED_LAMBDA) )
    pred, sigma = model.predict(X_test, w_test, p_test)
    print(pb_config.TRUE_MU, '=vs=', pred, '+/-', sigma)
    
    target = np.sum(w_test[y_test==0]) / np.sum(w_test)
    print(target, '=vs=', pred, '+/-', sigma)



    logger.info('DONE')

if __name__ == '__main__':
    main()
	