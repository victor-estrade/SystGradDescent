# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging

from utils.model import get_model
from utils.model import get_optimizer


from problem.higgs import HiggsConfigTesOnly as Config
from problem.higgs import get_minimizer

from problem.higgs import get_generators_torch
DATA_NAME = 'HIGGS'
# from problem.higgs import get_easy_generators_torch as get_generators_torch
# DATA_NAME = 'EASY_HIGGS'
# from problem.higgs import get_balanced_generators_torch as get_generators_torch
# DATA_NAME = 'BALANCED_HIGGS'

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

if TES : DATA_NAME += "TES"
if JES : DATA_NAME += "JES"
if LES : DATA_NAME += "LES"

N_BINS = 30
N_ITER = 30



class GeneratorCPU:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.n_samples = data_generator.size

    def generate(self, *params, n_samples=None, no_grad=False):
            X, y, w = self.data_generator.generate(*params, n_samples=n_samples, no_grad=no_grad)
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            return X, y, w

    def reset(self):
        self.data_generator.reset()




def load_calib_tes(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_tes"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def load_calib_jes(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_jes"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def load_calib_les(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_les"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model



def calibrates(calibs, config, X_test, w_test):
    logger = logging.getLogger()
    if TES:
        calib_tes = calibs['tes']
        tes_mean, tes_sigma = calib_tes.predict(X_test, w_test)
        logger.info('tes = {} =vs= {} +/- {}'.format(config.TRUE.tes, tes_mean, tes_sigma) )
        config.CALIBRATED = config.CALIBRATED.clone_with(tes=tes_mean)
        config.CALIBRATED_ERROR = config.CALIBRATED_ERROR.clone_with(tes=tes_sigma)
    if JES:
        calib_jes = calibs['jes']
        jes_mean, jes_sigma = calib_jes.predict(X_test, w_test)
        logger.info('jes = {} =vs= {} +/- {}'.format(config.TRUE.jes, jes_mean, jes_sigma) )
        config.CALIBRATED = config.CALIBRATED.clone_with(jes=jes_mean)
        config.CALIBRATED_ERROR = config.CALIBRATED_ERROR.clone_with(jes=jes_sigma)
    if LES:
        calib_les = calibs['les']
        les_mean, les_sigma = calib_les.predict(X_test, w_test)
        logger.info('les = {} =vs= {} +/- {}'.format(config.TRUE.les, les_mean, les_sigma) )
        config.CALIBRATED = config.CALIBRATED.clone_with(les=les_mean)
        config.CALIBRATED_ERROR = config.CALIBRATED_ERROR.clone_with(les=les_sigma)
    return config
