# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import logging

from utils.model import get_model
from utils.model import get_optimizer

from problem.gamma_gauss import Parameter

from model.regressor import Regressor
from archi.reducer import A3ML3 as CALIB_ARCHI
# from archi.reducer import EA1AR8MR8L1 as CALIB_ARCHI

N_BINS = 10
N_ITER = 30
CALIB_RESCALE = "Calib_rescale"

def load_calib_rescale(DATA_NAME, BENCHMARK_NAME):
    args = lambda : None
    args.n_unit     = 80
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.n_samples  = 1000
    args.n_steps    = 1000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=1, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_RESCALE
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def calibrates(calib_rescale, config, X_test, w_test):
    logger = logging.getLogger()
    rescale_mean, rescale_sigma = calib_rescale.predict(X_test, w_test)
    logger.info('FITTED rescale = {} =vs= {} +/- {}'.format(config.TRUE.rescale, rescale_mean, rescale_sigma) )
    config.FITTED = Parameter(rescale_mean, config.CALIBRATED.interest_parameters)
    config.FITTED_ERROR = Parameter(rescale_sigma, config.CALIBRATED_ERROR.interest_parameters)
    return config
