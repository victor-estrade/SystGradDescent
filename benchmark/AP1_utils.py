# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from myplot import plot_param_around_min

def plot_apple_ratio_around_min(compute_nll, true_apple_ratio, model_path):
    apple_ratio_array = np.linspace(0.0, 1.0, 50)
    nll_array = [compute_nll(apple_ratio) for apple_ratio in apple_ratio_array]
    plot_param_around_min(apple_ratio_array, nll_array, true_apple_ratio, 'apple_ratio', model_path)

