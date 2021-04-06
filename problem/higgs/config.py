# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools
import copy
import numpy as np
from .parameter import ParameterTes
from .parameter import ParameterJes
from .parameter import ParameterLes
from .parameter import ParameterTesJes
from .parameter import ParameterTesLes
from .parameter import ParameterTesJesLes
from .parameter import Parameter
from .parameter import FuturParameter


# TODO : Changer pour CALIBRATED ERROR = 1/2 sigma !!!

class BaseHiggsConfig():

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None
    RANGE_N_TEST = [None,]

    @property
    def PARAM_NAMES(self):
        return self.TRUE.parameter_names

    @property
    def INTEREST_PARAM_NAMES(self):
        return self.TRUE.interest_parameter_names

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ HiggsConfig.RANGE_N_TEST ]
        for iter_params in itertools.product(*param_lists):
            params = iter_params[:-1]
            n_test_samples = iter_params[-1]
            new_config = copy.copy(self)
            new_config.TRUE = self.ParameterClass(*params)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance



class HiggsConfigTes(BaseHiggsConfig):
    ParameterClass = ParameterTes
    CALIBRATED = ParameterTes(tes=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterTes(tes=0.03, mu=1.0)
    TRUE = ParameterTes(tes=1.0, mu=1.0)
    RANGE = ParameterTes(tes=np.linspace(0.97, 1.03, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterTes(tes=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))



class HiggsConfigJes(BaseHiggsConfig):
    ParameterClass = ParameterJes
    CALIBRATED = ParameterJes(jes=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterJes(jes=0.03, mu=1.0)
    TRUE = ParameterJes(jes=1.0, mu=1.0)
    RANGE = ParameterJes(jes=np.linspace(0.97, 1.03, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterJes(jes=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))




class HiggsConfigLes(BaseHiggsConfig):
    ParameterClass = ParameterLes
    CALIBRATED = ParameterLes(les=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterLes(les=0.02, mu=1.0)
    TRUE = ParameterLes(les=1.0, mu=1.0)
    RANGE = ParameterLes(les=np.linspace(0.98, 1.02, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterLes(les=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))


class HiggsConfigTesJes(BaseHiggsConfig):
    ParameterClass = ParameterTesJes
    CALIBRATED = ParameterTesJes(tes=1.0, jes=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterTesJes(tes=0.03, jes=0.03, mu=1.0)
    TRUE = ParameterTesJes(tes=1.0, jes=1.0, mu=1.0)
    RANGE = ParameterTesJes(tes=np.linspace(0.97, 1.03, 3),
                        jes=np.linspace(0.97, 1.03, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterTesJes(tes=np.linspace(0.9, 1.1, 5),
                        jes=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))


class HiggsConfigTesLes(BaseHiggsConfig):
    ParameterClass = ParameterTesLes
    CALIBRATED = ParameterTesLes(tes=1.0, les=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterTesLes(tes=0.03, les=0.02, mu=1.0)
    TRUE = ParameterTesLes(tes=1.0, les=1.0, mu=1.0)
    RANGE = ParameterTesLes(tes=np.linspace(0.97, 1.03, 3),
                        les=np.linspace(0.98, 1.02, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterTesLes(tes=np.linspace(0.9, 1.1, 5),
                        les=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))


class HiggsConfigTesJesLes(BaseHiggsConfig):
    ParameterClass = ParameterTesJesLes
    CALIBRATED = ParameterTesJesLes(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    CALIBRATED_ERROR = ParameterTesJesLes(tes=0.03, jes=0.03, les=0.02, mu=1.0)
    TRUE = ParameterTesJesLes(tes=1.0, jes=1.0, les=1.0, mu=1.0)
    RANGE = ParameterTesJesLes(tes=np.linspace(0.97, 1.03, 3),
                        jes=np.linspace(0.97, 1.03, 3),
                        les=np.linspace(0.98, 1.02, 3),
                        mu=np.linspace(0.5, 2, 4))

    FINE_RANGE = ParameterTesJesLes(tes=np.linspace(0.9, 1.1, 5),
                        jes=np.linspace(0.9, 1.1, 5),
                        les=np.linspace(0.9, 1.1, 5),
                        mu=np.linspace(0.5, 2, 5))


class HiggsConfig(BaseHiggsConfig):
    ParameterClass = Parameter
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



class HiggsConfigTesOnly(BaseHiggsConfig):
    ParameterClass = Parameter
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



class FuturHiggsConfig(BaseHiggsConfig):
    CALIBRATED = FuturParameter(tes=1.0, jes=1.0, les=1.0, nasty_bkg=1.0, sigma_soft=0.0, mu=1.0)
    CALIBRATED_ERROR = FuturParameter(tes=0.03, jes=0.03, les=0.02, nasty_bkg=0.5, sigma_soft=1.0, mu=1.0)
    TRUE = FuturParameter(tes=1.0, jes=1.0, les=1.0, nasty_bkg=1.0, sigma_soft=1.0, mu=1.0)
    RANGE = FuturParameter(tes=np.linspace(0.97, 1.03, 3),
                        jes=[1.0],
                        les=[1.0],
                        nasty_bkg=[1.0],
                        sigma_soft=[3.0],
                        mu=[0.5,1,2])


ALL_CONFIG_DICT = {
    'Tes' : HiggsConfigTes,
    'Jes' : HiggsConfigJes,
    'Les' : HiggsConfigLes,
    'TesJes' : HiggsConfigTesJes,
    'TesLes' : HiggsConfigTesLes,
    'TesJesLes' : HiggsConfigTesJesLes,
}


def get_config_class(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_CONFIG_DICT :
        return ALL_CONFIG_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")
