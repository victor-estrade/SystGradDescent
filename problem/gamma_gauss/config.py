# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
from .parameter import Parameter

class GGConfig():
    CALIBRATED = Parameter(rescale=1, mix=0.1)
    CALIBRATED_ERROR = Parameter(rescale=2, mix=1)
    TRUE = Parameter(rescale=1.2, mix=0.2)
    RANGE = Parameter(rescale=np.linspace(0.8, 2, 4), 
                        mix=np.linspace(0.1, 0.9, 4))

    PARAM_NAMES = TRUE._fields

    N_TRAINING_SAMPLES = 2000
    N_VALIDATION_SAMPLES = 2000
    N_TESTING_SAMPLES = 2000

    def iter_test_config(self):
        for true_rescale, true_mix in itertools.product(*self.RANGE):
            new_config = GGConfig()
            new_config.TRUE = Parameter(true_rescale, true_mix)
            yield new_config

