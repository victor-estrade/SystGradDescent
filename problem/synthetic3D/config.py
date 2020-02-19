# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


class S3D2Config():
    PARAM_NAMES = ['mu', 'r', 'lam']
    INTEREST_PARAM_NAME = 'mu'

    CALIBRATED_MU = 50/1050
    CALIBRATED_R = 0.0
    CALIBRATED_LAMBDA = 3.0

    CALIBRATED_MU_ERROR = 1.0  # minuit default
    CALIBRATED_R_ERROR = 0.4
    CALIBRATED_LAMBDA_ERROR = 1.0
    
    TRUE_MU = 150/1050
    TRUE_R = 0.1
    TRUE_LAMBDA = 2.7

    TRUE_MU_RANGE = np.arange(0, 0.3, 0.05)
    # TRUE_MU_RANGE = [100/1050, 150/1050, 200/1050]
    TRUE_R_RANGE = np.arange(-0.2, 0.2, 0.1)
    TRUE_LAMBDA_RANGE = np.arange(2.1, 3.5, 0.2)

    N_TRAINING_SAMPLES = 30000
    N_VALIDATION_SAMPLES = 30000
    N_TESTING_SAMPLES = 30000

