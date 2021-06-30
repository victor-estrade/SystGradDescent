# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import iminuit
ERRORDEF_NLL = 0.5

from .nll import RESCALE_NORMALIZATION
from .nll import MU_NORMALIZATION

def grad_factory(compute_nll, epsilon=1e-6):
    def grad_function(rescale, mu):
        print('my grad fun being used !')
        lower = compute_nll(rescale-epsilon, mu)
        upper = compute_nll(rescale+epsilon, mu)
        grad_rescale = (upper - lower) / (2*epsilon)

        lower = compute_nll(rescale, mu-epsilon)
        upper = compute_nll(rescale, mu+epsilon)
        grad_mu = (upper - lower) / (2*epsilon)

        return (grad_rescale, grad_mu)
    return grad_function



def get_minimizer(compute_nll, calibrated_param, calibrated_param_error, tolerance=0.1):
    grad_fun = grad_factory(compute_nll)
    minimizer = iminuit.Minuit(compute_nll,
                           rescale=calibrated_param.rescale / RESCALE_NORMALIZATION,
                           mu=calibrated_param.mu / RESCALE_NORMALIZATION,
                           # grad=grad_fun
                          )
    minimizer.errordef = iminuit.Minuit.LIKELIHOOD
    minimizer.limits = [(0.001, None), (0.001, None)]
    minimizer.errors = [calibrated_param_error.rescale / RESCALE_NORMALIZATION
                        ,calibrated_param_error.mu / RESCALE_NORMALIZATION
                        ]
    minimizer.tol = tolerance  # Should I increase tolerance to help ???? (default is 0.1 according to doc)
    minimizer.throw_nan = True
    # minimizer.precision = 1.2e-7
    minimizer.print_level = 2
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
