# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import iminuit
ERRORDEF_NLL = 0.5

def get_minimizer(compute_nll, calibrated_param, calibrated_param_error):
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           rescale=calibrated_param.rescale,
                           error_rescale=calibrated_param_error.rescale,
                           limit_rescale=(0, None),
                           mix=calibrated_param.mix,
                           error_mix=calibrated_param_error.mix,
                           limit_mix=(0, 1),
                          )
    return minimizer


def get_minimizer_no_nuisance(compute_nll, calibrated_param, calibrated_param_error):
    minimizer = iminuit.Minuit(compute_nll,
                           errordef=ERRORDEF_NLL,
                           mix=calibrated_param.mix,
                           error_mix=calibrated_param_error.mix,
                           limit_mix=(0, 1),
                          )
    return minimizer
