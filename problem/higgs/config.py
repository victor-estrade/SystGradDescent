# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools
import numpy as np
from .parameter import Parameter

class HiggsConfig():
    CALIBRATED = Parameter(tes=1, jes=1, les=1, nasty_bkg=1, sigma_soft=0, mu=1)
    CALIBRATED_ERROR = Parameter(tes=0.1, jes=0.1, les=0.05, nasty_bkg=0.5, sigma_soft=1, mu=1)
    TRUE = Parameter(tes=1, jes=1, les=1, nasty_bkg=1, sigma_soft=1, mu=1)
    RANGE = Parameter(tes=np.linspace(0.9, 1.1, 3), 
                        jes=[1],
                        les=[1],
                        nasty_bkg=[1],
                        sigma_soft=[None],
                        mu=[0.5,1,2])

    MIN = Parameter(tes=0.9, jes=0.95, les=0.98, nasty_bkg=1, sigma_soft=1, mu=0.3)
    MAX = Parameter(tes=1.1, jes=1.05, les=1.02, nasty_bkg=1, sigma_soft=1, mu=2.2)
    PARAM_NAMES = TRUE._fields
    INTEREST_PARAM_NAME = 'mu'

    N_TRAINING_SAMPLES = None
    N_VALIDATION_SAMPLES = None
    N_TESTING_SAMPLES = None

    def iter_test_config(self):
        for true_params in itertools.product(*self.RANGE):
            new_config = HiggsConfig()
            new_config.TRUE = Parameter(*true_params)
            yield new_config
