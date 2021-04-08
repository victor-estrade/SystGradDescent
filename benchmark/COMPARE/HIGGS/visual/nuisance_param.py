# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals



def detect_nuisance_param(data):
    nuisance_param_key = []
    if "true_tes" in data:
        nuisance_param_key.append("true_tes")
    if "true_jes" in data:
        nuisance_param_key.append("true_jes")
    if "true_les" in data:
        nuisance_param_key.append("true_les")
    return nuisance_param_key


def label_nuisance_param(nuisance_param_key, nuisance_param):
    try:
        label = ', '.join([f"{key[5:]}={value}" for key, value in zip(nuisance_param_key, nuisance_param)])
    except TypeError:
        label = f"{nuisance_param_key[0][5:]}={nuisance_param}"
    return label
