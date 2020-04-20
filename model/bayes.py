#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import itertools

import numpy as np

from tqdm import tqdm

def expectancy(values, probabilities, axis=None, keepdims=False):
    return np.sum(values * probabilities, axis=axis, keepdims=keepdims)

def variance(values, probabilities, axis=None):
    return np.sum(probabilities * np.square(values - expectancy(values, probabilities, axis=axis, keepdims=True)), axis=axis)

def variance_bis(values, probabilities, axis=None):
    return np.sum(values * values * probabilities, axis=axis) - np.square(expectancy(values, probabilities, axis=axis, keepdims=True))

def stat_uncertainty(values, posterior, marginal, reshape=(1, -1), axis=-1):
    v = variance(values.reshape(reshape), posterior, axis=axis)
    return expectancy(v.ravel(), marginal.ravel())

def syst_uncertainty(values, posterior, marginal, reshape=(1, -1), axis=-1):
    v = expectancy(values.reshape(reshape), posterior, axis=axis)
    return variance(v.ravel(), marginal.ravel())


def get_iter_prod(*sizes, progress_bar=False):
    generator = itertools.product(*(range(n) for n in sizes))
    if progress_bar:
        total = np.prod(sizes)
        return tqdm(generator, total=total)
    return generator
