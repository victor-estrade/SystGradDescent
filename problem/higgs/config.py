# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools
import numpy as np
from .parameter import ParameterTes
from .parameter import Parameter
from .parameter import FuturParameter


# TODO : Changer pour CALIBRATED ERROR = 1/2 sigma !!!


class MonoHiggsConfig():
    CALIBRATED = ParameterTes(tes=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterTes(tes=0.03, mu=1.0)
    TRUE = ParameterTes(tes=1.0, mu=1.0)
    RANGE = ParameterTes(tes=np.linspace(0.97, 1.03, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterTes(tes=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))

    MIN = ParameterTes(tes=0.9, mu=0.1)
    MAX = ParameterTes(tes=1.1, mu=2.2)
    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = TRUE.interest_parameters_names

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None
    RANGE_N_TEST = [None,]

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ HiggsConfig.RANGE_N_TEST ]
        for tes, mu, n_test_samples in itertools.product(*param_lists):
            new_config = MonoHiggsConfig()
            new_config.TRUE = ParameterTes(tes, mu)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance


class HiggsConfig():
    CALIBRATED = Parameter(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    CALIBRATED_ERROR = Parameter(tes=0.03, jes=0.03, les=0.02, mu=1.0)
    TRUE = Parameter(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    RANGE = Parameter(tes=np.linspace(0.97, 1.03, 3),
                        jes=np.linspace(0.97, 1.03, 3),
                        les=np.linspace(0.98, 1.02, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = Parameter(tes=np.linspace(0.9, 1.1, 5),
                        jes=np.linspace(0.9, 1.1, 5),
                        les=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))

    MIN = Parameter(tes=0.9, jes=0.95, les=0.98, mu=0.1)
    MAX = Parameter(tes=1.1, jes=1.05, les=1.02, mu=2.2)
    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = TRUE.interest_parameters_names

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None
    RANGE_N_TEST = [None,]

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ HiggsConfig.RANGE_N_TEST ]
        for tes, jes, les, mu, n_test_samples in itertools.product(*param_lists):
            new_config = HiggsConfig()
            new_config.TRUE = Parameter(tes, jes, les, mu)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance


class HiggsConfigTesOnly():
    CALIBRATED = Parameter(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    CALIBRATED_ERROR = Parameter(tes=0.03, jes=0.03, les=0.02, mu=1.0)
    TRUE = Parameter(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    RANGE = Parameter(tes=np.linspace(0.97, 1.03, 3),
                        jes=[1.0],
                        les=[1.0],
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = Parameter(tes=np.linspace(0.9, 1.1, 15),
                        jes=[1.0],
                        les=[1.0],
                        mu=[1.0])

    MIN = Parameter(tes=0.9, jes=0.95, les=0.98, mu=0.1)
    MAX = Parameter(tes=1.1, jes=1.05, les=1.02, mu=2.2)
    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = TRUE.interest_parameters_names

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None
    RANGE_N_TEST = [None,]

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ HiggsConfig.RANGE_N_TEST ]
        for tes, jes, les, mu, n_test_samples in itertools.product(*param_lists):
            new_config = HiggsConfig()
            new_config.TRUE = Parameter(tes, jes, les, mu)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance




class FuturHiggsConfig():
    CALIBRATED = FuturParameter(tes=1.0, jes=1.0, les=1.0, nasty_bkg=1.0, sigma_soft=0.0, mu=1.0)
    CALIBRATED_ERROR = FuturParameter(tes=0.03, jes=0.03, les=0.02, nasty_bkg=0.5, sigma_soft=1.0, mu=1.0)
    TRUE = FuturParameter(tes=1.0, jes=1.0, les=1.0, nasty_bkg=1.0, sigma_soft=1.0, mu=1.0)
    RANGE = FuturParameter(tes=np.linspace(0.97, 1.03, 3),
                        jes=[1.0],
                        les=[1.0],
                        nasty_bkg=[1.0],
                        sigma_soft=[3.0],
                        mu=[0.5,1,2])

    MIN = FuturParameter(tes=0.9, jes=0.95, les=0.98, nasty_bkg=1, sigma_soft=2, mu=0.3)
    MAX = FuturParameter(tes=1.1, jes=1.05, les=1.02, nasty_bkg=1, sigma_soft=4, mu=2.2)
    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = 'mu'

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None

    def iter_test_config(self):
        for true_params in itertools.product(*self.RANGE):
            new_config = HiggsConfig()
            new_config.TRUE = FuturParameter(*true_params)
            yield new_config
