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
                        mu=50/1050)
    CALIBRATED_ERROR = Parameter(r=0.4,
                        lam=1.0,
                        mu=MINUIT_DEFAULT_ERROR)

    TRUE = Parameter(r=0.1, lam=2.7, mu=150/1050)

    RANGE = Parameter(r=np.linspace(0.2, 2, 2), 
                    lam=np.linspace(2, 4, 1),
                    mu=np.linspace(0.01, 0.99, 2),)

    PARAM_NAMES = ['mu', 'r', 'lam']
    INTEREST_PARAM_NAME = TRUE.interest_parameters_names

    CALIBRATED_MU = CALIBRATED.mu
    CALIBRATED_R = CALIBRATED.r
    CALIBRATED_LAMBDA = CALIBRATED.lam

    CALIBRATED_MU_ERROR = CALIBRATED_ERROR.mu
    CALIBRATED_R_ERROR = CALIBRATED_ERROR.r
    CALIBRATED_LAMBDA_ERROR = CALIBRATED_ERROR.lam
    
    TRUE_MU = TRUE.mu
    TRUE_R = TRUE.r
    TRUE_LAMBDA = TRUE.lam

    TRUE_MU_RANGE = np.arange(0, 0.3, 0.05)[1:]
    # TRUE_MU_RANGE = [100/1050, 150/1050, 200/1050]
    TRUE_R_RANGE = np.arange(-0.2, 0.2, 0.1)
    TRUE_LAMBDA_RANGE = np.arange(2.1, 3.5, 0.2)

    N_TRAINING_SAMPLES = 30000
    N_VALIDATION_SAMPLES = 30000
    N_TESTING_SAMPLES = 30000

    def iter_test_config(self):
        for true_r, true_lam, true_mu in itertools.product(*self.RANGE):
            new_config = S3D2Config()
            new_config.TRUE = Parameter(true_r, true_lam, true_mu)
            yield new_config

