# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, calibrated_param, calibrated_param_error):
    MIN_VALUE = 0.01
    MAX_VALUE = 10 
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           r=calibrated_param.r,
                           error_r=calibrated_param_error.r,
                           #limit_r=(0, None),
                           lam=calibrated_param.lam,
                           error_lam=calibrated_param_error.lam,
                           limit_lam=(MIN_VALUE, None),
                           mu=calibrated_param.mu,
                           error_mu=calibrated_param_error.mu,
                           limit_mu=(MIN_VALUE, MAX_VALUE),
                          )
    return minimizer

def get_minimizer_no_nuisance(compute_nll, calibrated_param, calibrated_param_error):
    MIN_VALUE = 0.01
    MAX_VALUE = 10 
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           mu=calibrated_param.mu,
                           error_mu=calibrated_param_error.mu,
                           limit_mu=(MIN_VALUE, MAX_VALUE),
                           print_level=0,
                          )
    return minimizer
