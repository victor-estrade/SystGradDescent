# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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

    PARAM_NAMES = ['mu', 'r', 'lam']
    INTEREST_PARAM_NAME = 'mu'

    CALIBRATED_MU = CALIBRATED.mu
    CALIBRATED_R = CALIBRATED.r
    CALIBRATED_LAMBDA = CALIBRATED.lam

    CALIBRATED_MU_ERROR = CALIBRATED_ERROR.mu
    CALIBRATED_R_ERROR = CALIBRATED_ERROR.r
    CALIBRATED_LAMBDA_ERROR = CALIBRATED_ERROR.lam
    
    TRUE_MU = TRUE.mu
    TRUE_R = TRUE.r
    TRUE_LAMBDA = TRUE.lam

    TRUE_MU_RANGE = np.arange(0, 0.3, 0.05)
    # TRUE_MU_RANGE = [100/1050, 150/1050, 200/1050]
    TRUE_R_RANGE = np.arange(-0.2, 0.2, 0.1)
    TRUE_LAMBDA_RANGE = np.arange(2.1, 3.5, 0.2)

    N_TRAINING_SAMPLES = 30000
    N_VALIDATION_SAMPLES = 30000
    N_TESTING_SAMPLES = 30000

