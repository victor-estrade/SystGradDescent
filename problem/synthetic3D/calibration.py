# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from .parameter import Parameter
from .config import S3D2Config

def param_generator():
    pb_config = S3D2Config()

    r = np.random.normal(pb_config.CALIBRATED.r, pb_config.CALIBRATED_ERROR.r)
    lam = -1
    while lam <= 0:
        lam = np.random.normal(pb_config.CALIBRATED.lam, pb_config.CALIBRATED_ERROR.lam)
    
    mu = np.random.uniform(0, 1)
    return Parameter(r, lam, mu)


def calib_param_sampler(r_mean, r_sigma, lam_mean, lam_sigma):
    def param_sampler():
        r = np.random.normal(r_mean, r_sigma)
        lam = -1
        while lam <= 0:
            lam = np.random.normal(lam_mean, lam_sigma)
        
        mu = np.random.uniform(0, 1)
        return Parameter(r, lam, mu)
    return param_sampler
