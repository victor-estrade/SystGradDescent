# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class AP1Config():
    PARAM_NAMES = ['apple_ratio']
    INTEREST_PARAM_NAME = 'apple_ratio'

    CALIBRATED_APPLE_RATIO = 0.5
    CALIBRATED_APPLE_RATIO_ERROR = 1  # minuit default
    TRUE_APPLE_RATIO = 0.8
    TRUE_APPLE_RATIO_RANGE = np.arange(0, 1, 0.1)

    N_TRAINING_SAMPLES = 2000
    N_VALIDATION_SAMPLES = 2000
    N_TESTING_SAMPLES = 2000
