# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd


def many_predict(model, X, w, param_generator, ncall=100):
    all_pred = []
    all_params = []
    for _ in range(ncall):
        params = param_generator()
        nuisance_params = params.nuisance_parameters
        p = np.array(nuisance_params)
        pred = model.predict(X, w, p)
        all_pred.append(pred)
        all_params.append(params)
    
    return all_pred, all_params


def monte_carlo_data(all_pred, all_params):
    all_target = [target for target, sigma in all_pred]
    all_sigma  = [sigma for target, sigma in all_pred]
    data = dict(target=all_target, sigma=all_sigma)
    nuisance_params_names = all_params[0].nuisance_parameters_names
    data.update({name: [p[name] for p in all_params]
                    for name in nuisance_params_names})
    data = pd.DataFrame(data)
    return data


def monte_carlo_infer(data):
    # FIXME :  Check that I am using the right formulas ! 
    all_target = data.target
    target  = np.mean(all_target)

    all_sigma = data.sigma
    sigma_squared = all_sigma ** 2
    target_squared = all_target ** 2
    variance = np.mean(sigma_squared + target_squared) - (target ** 2)
    sigma = np.sqrt(variance)
    return target, sigma


def save_monte_carlo(data, model_path, ext=''):
    data.to_csv(os.path.join(model_path, 'monte_carlo{}.csv'.format(ext)))

