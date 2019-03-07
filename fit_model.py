# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import iminuit

import numpy as np

from nll import HiggsNLL

ERRORDEF_NLL = 0.5
ERRORDEF_LEAST_SQUARE = 1.0


# TODO : finish writing
def fit_model(model, test_data, xp_data):
    
    negative_log_likelihood = HiggsNLL(model)
    minimizer = iminuit.Minuit(negative_log_likelihood,
                    errordef=ERRORDEF_NLL,
                    )

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', message='.*arcsinh')
        fmin, param = minimizer.migrad()

    # What if mingrad failed ?
    valid = minimizer.migrad_ok()

    # Compute hessian error
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', message='.*arcsinh')
        param = minimizer.hesse()

    # Stuff to save
    fitarg = minimizer.fitarg


