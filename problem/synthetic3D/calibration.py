# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from .parameter import Parameter
from .config import S3D2Config

def param_generator():
    pb_config = S3D2Config()

    r = np.random.normal(pb_config.CALIBRATED_R, pb_config.CALIBRATED_R_ERROR)
    lam = -1
    while lam <= 0:
        lam = np.random.normal(pb_config.CALIBRATED_LAMBDA, pb_config.CALIBRATED_LAMBDA_ERROR)
    
    mu_min = min(pb_config.TRUE_MU_RANGE)
    mu_max = max(pb_config.TRUE_MU_RANGE)
    mu_range = mu_max - mu_min
    mu_min = max(0.0, mu_min - mu_range/10)
    mu_max = min(1.0, mu_max + mu_range/10)

    mu = np.random.uniform(0, 1)
    return Parameter(r, lam, mu)


