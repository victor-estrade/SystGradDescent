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
    all_nuisance_params = []
    for _ in range(ncall):
        params = param_generator()
        nuisance_params = np.array(params[:-1])
        pred = model.predict(X, w, nuisance_params)
        all_pred.append(pred)
        all_nuisance_params.append(nuisance_params)
    
    return all_pred, all_nuisance_params


def monte_carlo_data(all_pred, all_nuisance_params):
    all_target = [target for target, sigma in all_pred] 
    all_sigma  = [sigma for target, sigma in all_pred] 
    all_target          = np.array(all_target).reshape(-1, 1)
    all_sigma           = np.array(all_sigma).reshape(-1, 1)
    all_nuisance_params = np.array(all_nuisance_params)
    data = dict(target=all_target, sigma=all_sigma)
    data = np.concatenate([all_pred, all_sigma, all_nuisance_params], axis=1)
    data = pd.DataFrame(data)
    return data

def monte_carlo_infer(data):
    all_target = data.target
    all_sigma = data.sigma
    value  = np.mean(all_target)
    sigma = np.mean(all_sigma)
    return value, sigma



def save_monte_carlo(data, model_path):
    data.to_csv(os.path.join(model_path, 'monte_carlo.csv'))

