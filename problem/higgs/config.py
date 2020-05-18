# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools
import numpy as np
from .parameter import Parameter

class HiggsConfig():
    CALIBRATED = Parameter(tes=1, jes=1, les=1, nasty_bkg=1, soft_term=None)
    CALIBRATED_ERROR = Parameter(tes=1, jes=1, les=1, nasty_bkg=1, soft_term=1)
    TRUE = Parameter(tes=1, jes=1, les=1, nasty_bkg=1, soft_term=1)
    RANGE = Parameter(tes=np.linspace(0.9, 1.1, 3), 
                        jes=[1],
                        les=[1],
                        nasty_bkg=[1],
                        soft_term=[None])

    PARAM_NAMES = TRUE._fields

    N_TRAINING_SAMPLES = 80000
    N_VALIDATION_SAMPLES = 80000
    N_TESTING_SAMPLES = 80000

    def iter_test_config(self):
        for true_params in itertools.product(*self.RANGE):
            new_config = HiggsConfig()
            new_config.TRUE = Parameter(*true_params)
            yield new_config
