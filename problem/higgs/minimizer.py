# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, calibrated_param, calibrated_param_error, tolerance=0.1):
    minimizer = iminuit.Minuit(compute_nll, **calibrated_param.to_dict())
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    MIN_VALUE = 0.0
    MAX_VALUE = None
    minimizer.limits = [(MIN_VALUE, MAX_VALUE) for _ in calibrated_param]
    minimizer.errors = list(calibrated_param_error)
    minimizer.tol = tolerance  # Should I increase tolerance to help ???? (default is 0.1 according to doc)
    minimizer.throw_nan = True
    # minimizer.precision = 1e-6
    # minimizer.print_level = 2
    return minimizer


def get_minimizer_no_nuisance(compute_nll, calibrated_param, calibrated_param_error):
    MIN_VALUE = 0.01
    MAX_VALUE = 10
    minimizer = iminuit.Minuit(compute_nll,
                           mu=calibrated_param.mu,
                           # error_mu=calibrated_param_error.mu,
                           # limit_mu=(MIN_VALUE, MAX_VALUE),
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(MIN_VALUE, MAX_VALUE), ]
    minimizer.errors = [ calibrated_param_error.mu ]
    return minimizer


def futur_get_minimizer(compute_nll, calibrated_param, calibrated_param_error):
    MIN_VALUE = 0.01
    MAX_VALUE = 10
    minimizer = iminuit.Minuit(compute_nll,
                           tes=calibrated_param.tes,
                           # error_tes=calibrated_param_error.tes,
                           # limit_tes=(MIN_VALUE, MAX_VALUE),
                           jes=calibrated_param.jes,
                           # error_jes=calibrated_param_error.jes,
                           # limit_jes=(MIN_VALUE, MAX_VALUE),
                           les=calibrated_param.les,
                           # error_les=calibrated_param_error.les,
                           # limit_les=(MIN_VALUE, MAX_VALUE),
                           nasty_bkg=calibrated_param.nasty_bkg,
                           # error_nasty_bkg=calibrated_param_error.nasty_bkg,
                           # limit_nasty_bkg=(MIN_VALUE, MAX_VALUE),
                           sigma_soft=calibrated_param.sigma_soft,
                           # error_sigma_soft=calibrated_param_error.sigma_soft,
                           # limit_sigma_soft=(0, MAX_VALUE),
                           mu=calibrated_param.mu,
                           # error_mu=calibrated_param_error.mu,
                           # limit_mu=(MIN_VALUE, MAX_VALUE),
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(MIN_VALUE, MAX_VALUE)
                        , (MIN_VALUE, MAX_VALUE)
                        , (MIN_VALUE, MAX_VALUE)
                        , (MIN_VALUE, MAX_VALUE)
                        , (MIN_VALUE, MAX_VALUE)
                        , (MIN_VALUE, MAX_VALUE)]
    minimizer.errors = [calibrated_param_error.tes
                        ,calibrated_param_error.jes
                        ,calibrated_param_error.les
                        ,calibrated_param_error.nasty_bkg
                        ,calibrated_param_error.sigma_soft
                        ,calibrated_param_error.mu
                        ]
    minimizer.tol = 0.5  # Should I increase tolerance to help ???? (default is 0.1 according to doc)
    return minimizer
