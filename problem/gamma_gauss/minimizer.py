# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, calibrated_param, calibrated_param_error):
    minimizer = iminuit.Minuit(compute_nll,
                           rescale=calibrated_param.rescale,
                           # error_rescale=calibrated_param_error.rescale,
                           # limit_rescale=(0, 100),
                           mu=calibrated_param.mu,
                           # error_mu=calibrated_param_error.mu,
                           # limit_mu=(0, 1),
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(0.001, None), (0.001, None)]
    minimizer.errors = [calibrated_param_error.rescale
                        ,calibrated_param_error.mu
                        ]
    return minimizer


def get_minimizer_no_nuisance(compute_nll, calibrated_param, calibrated_param_error):
    minimizer = iminuit.Minuit(compute_nll,
                           mu=calibrated_param.mu,
                           # error_mu=calibrated_param_error.mu,
                           # limit_mu=(0, 1),
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(0.001, None)]
    minimizer.errors = [calibrated_param_error.mu]
    return minimizer
