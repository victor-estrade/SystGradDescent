# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


from utils.model import get_model
from utils.model import get_optimizer


N_BINS = 10


def load_calib_r(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 200
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.n_samples  = 1000
    args.n_steps    = 2000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_R
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model

def load_calib_lam(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 200
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.n_samples  = 1000
    args.n_steps    = 2000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=3, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = CALIB_LAM
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def calibrates(calib_r, calib_lam, config, X_test, w_test):
    logger = logging.getLogger()

    r_mean, r_sigma = calib_r.predict(X_test, w_test)
    logger.info('r   = {} =vs= {} +/- {}'.format(config.TRUE_R, r_mean, r_sigma) )

    lam_mean, lam_sigma = calib_lam.predict(X_test, w_test)
    logger.info('lam = {} =vs= {} +/- {}'.format(config.TRUE_LAMBDA, lam_mean, lam_sigma) )

    config.FITTED = Parameter(r_mean, lam_mean, config.CALIBRATED.interest_parameters)
    config.FITTED_ERROR = Parameter(r_sigma, lam_sigma, config.CALIBRATED_ERROR.interest_parameters)
    return config
