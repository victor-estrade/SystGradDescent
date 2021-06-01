# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, calibrated_param, calibrated_param_error, tolerance=0.1):
    MIN_VALUE = 0.0001
    MAX_VALUE = None
    minimizer = iminuit.Minuit(compute_nll,
                           r=calibrated_param.r,
                           lam=calibrated_param.lam,
                           mu=calibrated_param.mu,
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(None, None), (MIN_VALUE, None), (MIN_VALUE, MAX_VALUE)]
    minimizer.errors = [calibrated_param_error.r
                        ,calibrated_param_error.lam
                        ,calibrated_param_error.mu
                        ]
    minimizer.tol = tolerance  # Should I increase tolerance to help ???? (default is 0.1 according to doc)
    minimizer.throw_nan = True
    minimizer.precision = 1e-6
    return minimizer


def get_minimizer_no_nuisance(compute_nll, calibrated_param, calibrated_param_error):
    MIN_VALUE = 0.0
    MAX_VALUE = None
    minimizer = iminuit.Minuit(compute_nll,
                           mu=calibrated_param.mu,
                           print_level=0,
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(MIN_VALUE, MAX_VALUE)]
    minimizer.errors = [calibrated_param_error.mu]
    return minimizer
