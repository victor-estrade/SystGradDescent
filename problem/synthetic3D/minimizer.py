# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, pb_config):
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           r=pb_config.CALIBRATED_R,
                           error_r=pb_config.CALIBRATED_R_ERROR,
                           #limit_r=(0, None),
                           lam=pb_config.CALIBRATED_LAMBDA,
                           error_lam=pb_config.CALIBRATED_LAMBDA_ERROR,
                           limit_lam=(0, None),
                           mu=pb_config.CALIBRATED_MU,
                           error_mu=pb_config.CALIBRATED_MU_ERROR,
                           limit_mu=(0, 1),
                          )
    return minimizer
