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


def assert_clean_alpha(alpha):
    assert alpha > 0, f"alpha should be > 0  {alpha} found"

def assert_clean_y(y):
    assert y > 0 and y < 1, f"y is a mixture coef it should be in ]0, 1[  {y} found"


class Generator():
    def __init__(self, gamma_k=2, gamma_loc=0, normal_mean=5, normal_sigma=0.5):
        self.gamma_k = gamma_k
        self.gamma_loc = gamma_loc
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma

    def sample_event(self, y, alpha, size=1):
        assert_clean_alpha(alpha)
        assert_clean_y(y)
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = alpha
        normal_mean  = self.normal_mean * alpha
        normal_sigma = self.normal_sigma * alpha
        x_b = sts.gamma.rvs(gamma_k, loc=gamma_loc, scale=gamma_scale, size=size)
        x_s  = sts.norm.rvs(loc=normal_mean, scale=normal_sigma, size=size)
        idx = np.random.random(size=size) < y
        x_b[idx] = x_s[idx]
        return x_b, idx

    def proba_density(self, x, y, alpha):
        """
        computes p(x | y, alpha)
        """
        assert_clean_alpha(alpha)
        assert_clean_y(y)
        gamma_k      = self.gamma_k
        gamma_loc    = self.gamma_loc
        gamma_scale  = alpha
        normal_mean  = self.normal_mean * alpha
        normal_sigma = self.normal_sigma * alpha
        proba_gamma  = sts.gamma.pdf(x, gamma_k, loc=gamma_loc, scale=gamma_scale)
        proba_normal  = sts.norm.pdf(x, loc=normal_mean, scale=normal_sigma)
        proba_density = y * proba_normal + (1-y) * proba_gamma
        return proba_density

    def log_proba_density(self, x, y, alpha):
        """
        computes log p(x | y, alpha)
        """
        proba_density = self.proba_density(x, y, alpha)
        logproba_density = np.log(proba_density)
        return logproba_density

    def nll(self, data, y, alpha):
        """
        Computes the negative log likelihood of teh data given y and alpha.
        """
        nll = - self.log_proba_density(data, y, alpha).sum()
        return nll



class PriorAlpha():
    def __init__(self, alpha_min=0.7, alpha_max=1.3):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def proba_density(self, alpha):
        scale = self.alpha_max - self.alpha_min
        p = sts.uniform.pdf(alpha, loc=self.alpha_min, scale=scale)
        return p

    def log_proba_density(self, alpha):
        scale = self.alpha_max - self.alpha_min
        p = sts.uniform.pdf(alpha, loc=self.alpha_min, scale=scale)
        return p


class PriorY():
    def __init__(self, y_min=0.01, y_max=0.3):
        self.y_min = y_min
        self.y_max = y_max

    def proba_density(self, y):
        scale = self.y_max - self.y_min
        p = sts.uniform.pdf(y, loc=self.y_min, scale=scale)
        return p

    def log_proba_density(self, y):
        scale = self.y_max - self.y_min
        p = sts.uniform.pdf(y, loc=self.y_min, scale=scale)
        return p




def main():
    print('hello world !')
    x_range = np.linspace(0, 10, 1000)
    generator = Generator()
    prior_alpha = PriorAlpha()
    prior_y = PriorY()
    alpha =  1
    y = 0.5
    p = generator.proba_density(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")
    data, idx = generator.sample_event(y, alpha, size=1000)
    sns.distplot(data, label="samples")

    alpha = 1.1
    p = generator.proba_density(x_range, y, alpha)
    plt.plot(x_range, p, label=f"{alpha}")
    plt.legend()
    plt.show()

    x_range = np.linspace(0, 2, 100)
    p = prior_alpha.proba_density(x_range)
    plt.plot(x_range, p, label=f"p_alpha")
    plt.legend()
    plt.show()


    x_range = np.linspace(0, 0.5, 100)
    p = prior_y.proba_density(x_range)
    plt.plot(x_range, p, label=f"p_y")
    plt.legend()
    plt.show()

    print('NLL arange 0, 10' )

    nll = generator.nll(data, 0.1, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.01, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.5, 1.1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.45, 1)
    print('nll =', nll )
    
    nll = generator.nll(data, 0.5, 1)
    print('best nll =', nll )



if __name__ == '__main__':
    main()
