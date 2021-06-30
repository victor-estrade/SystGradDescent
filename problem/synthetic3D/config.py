# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
from .parameter import Parameter

MINUIT_DEFAULT_ERROR = 1.0

class S3D2Config():
    CALIBRATED = Parameter(r=0.0,
                        lam=3.0,
                        mu=1.0)
    CALIBRATED_ERROR = Parameter(r=0.4,
                        lam=1.0,
                        mu=MINUIT_DEFAULT_ERROR)

    TRUE = Parameter(r=0.1, lam=2.7, mu=150/1050)

    MIN = Parameter(r=-0.5, lam=2, mu=0.1)
    MAX = Parameter(r=0.5, lam=4, mu=0.9)

    RANGE = Parameter(r=np.linspace(-0.5, 0.5, 3),
                    lam=np.linspace(2, 4, 3),
                    mu=np.linspace(0.5, 1.5, 3),)

    FINE_RANGE = Parameter(r=np.linspace(-1, 1, 7),
                    lam=np.linspace(1, 5, 7),
                    mu=np.linspace(0.5, 2, 7),)

    PARAM_NAMES = TRUE.parameter_names
    INTEREST_PARAM_NAME = TRUE.interest_parameters_names

    N_TRAINING_SAMPLES = 30000
    N_VALIDATION_SAMPLES = 30000
    N_TESTING_SAMPLES = 30000
    RANGE_N_TEST = [30000,]

    def iter_test_config(self):
        param_lists = [*self.RANGE ] + [ S3D2Config.RANGE_N_TEST ]
        for true_r, true_lam, true_mu, n_test_samples in itertools.product(*param_lists):
            new_config = S3D2Config()
            new_config.TRUE = Parameter(true_r, true_lam, true_mu)
            new_config.N_TESTING_SAMPLES = n_test_samples
            yield new_config

    def iter_nuisance(self):
        for nuisance in itertools.product(*self.FINE_RANGE.nuisance_parameters):
            yield nuisance
