# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from .parameter import Parameter

class GGConfig():
    CALIBRATED = Parameter(scale=1, mix=0.1)
    CALIBRATED_ERROR = Parameter(scale=2, mix=1)
    TRUE = Parameter(scale=1.2, mix=0.2)

    N_TRAINING_SAMPLES = 2000
    N_VALIDATION_SAMPLES = 2000
    N_TESTING_SAMPLES = 2000
