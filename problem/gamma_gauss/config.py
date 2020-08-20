# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
from .parameter import Parameter

class GGConfig():
    CALIBRATED = Parameter(rescale=1, mix=0.5)
    CALIBRATED_ERROR = Parameter(rescale=0.5, mix=0.5)
    TRUE = Parameter(rescale=1.2, mix=0.2)
    RANGE = Parameter(rescale=np.linspace(0.8, 1.2, 3), 
                        mix=np.linspace(0.1, 0.9, 3))

    FINE_RANGE = Parameter(rescale=np.linspace(0.5, 1.5, 11), 
                        mix=np.linspace(0.1, 0.9, 11))


    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = 'mix'

    N_TRAINING_SAMPLES = 2000
    N_VALIDATION_SAMPLES = 2000
    N_TESTING_SAMPLES = 2000

    def iter_test_config(self):
        for true_rescale, true_mix in itertools.product(*self.RANGE):
            new_config = GGConfig()
            new_config.TRUE = Parameter(true_rescale, true_mix)
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance


