#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals



"""
Exploring the possible minimal systematic error

The idea is to build a very simple but representative toy and to compute statistic and systematic error.
We can compute the likelihood for this mini toy.

x = value of an observable of an event.
y = target parameter = mixture coef of signal and background events
alpha = nuisance parameter = rescaling of the observable

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.stats as sts


def compute_proba_densitity(x, y, alpha):
    assert y > 0 and y < 1, f"y is a mixture coef it should be in ]0, 1[  {y} found"
    gamma_k      = 2
    gamma_loc    = 0
    gamma_scale  = alpha
    normal_mean  = 5 * alpha
    normal_sigma = 0.5 * alpha
    proba_gamma  = sts.gamma.pdf(x, gamma_k, loc=gamma_loc, scale=gamma_scale)
    proba_normal  = sts.norm.pdf(x, loc=normal_mean, scale=normal_sigma)
    proba_density = y * proba_normal + (1-y) * proba_gamma
    return proba_density


def compute_logproba_densitity(x, y, alpha):
    proba_density = compute_proba_densitity(x, y, alpha)
    logproba_density = np.log(proba_density)
    return logproba_density


def compute_prior_alpha(alpha):
    alpha_min = 0.7
    alpha_max = 1.3
    scale = alpha_max - alpha_min
    p = sts.uniform.pdf(alpha, loc=alpha_min, scale=scale)
    return p


def compute_prior_y(y):
    y_min = 0.01
    y_max = 0.3
    scale = y_max - y_min
    p = sts.uniform.pdf(y, loc=y_min, scale=scale)
    return p



def main():
    print('hello world !')
    x_range = np.linspace(0, 10, 1000)
    alpha =  1
    y = 0.05
    p = compute_proba_densitity(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")

    alpha = 1.1
    p = compute_proba_densitity(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")
    plt.legend()
    plt.show()

    p = compute_prior_alpha(x_range)
    plt.plot(x_range, p, label=f"p_alpha")
    plt.legend()
    plt.show()


    p = compute_prior_y(x_range)
    plt.plot(x_range, p, label=f"p_y")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
