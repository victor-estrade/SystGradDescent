# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
from .parameter import Parameter

class GGConfig():
    CALIBRATED = Parameter(rescale=1.0, mix=0.5)
    CALIBRATED_ERROR = Parameter(rescale=0.5, mix=0.5)
    TRUE = Parameter(rescale=1.2, mix=0.2)
    RANGE = Parameter(rescale=np.linspace(0.8, 1.2, 3), 
                        mix=np.linspace(0.1, 0.9, 5))

    FINE_RANGE = Parameter(rescale=np.linspace(0.5, 1.5, 31), 
                        mix=np.linspace(0.1, 0.9, 31))


    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = 'mix'

    N_TRAINING_SAMPLES = 10000
    N_VALIDATION_SAMPLES = 5000
    N_TESTING_SAMPLES = 5000
    RANGE_N_TEST = [50, 100, 300, 500, 2000]

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ self.RANGE_N_TEST ]
        for true_rescale, true_mix, n_test_samples in itertools.product(*param_lists):
            new_config = GGConfig()
            new_config.TRUE = Parameter(true_rescale, true_mix)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance




class GGConfigPlus(GGConfig):
    RANGE_N_TEST = [5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000]
