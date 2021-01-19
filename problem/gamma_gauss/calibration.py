# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from scipy import stats

from .parameter import Parameter
from .config import GGConfig


def param_generator(random_state=None):
    config = GGConfig()
    offset = - config.CALIBRATED.rescale / config.CALIBRATED_ERROR.rescale
    prior_rescale = stats.truncnorm(offset, 5, loc=config.CALIBRATED.rescale, scale=config.CALIBRATED_ERROR.rescale)
    prior_mu   = stats.uniform(loc=0, scale=1)
    rescale = prior_rescale.rvs(random_state=random_state)
    mu   = prior_mu.rvs(random_state=random_state)
    return Parameter(rescale, mu)


def calib_param_sampler(rescale_mean, rescale_sigma, random_state=42):
    def param_sampler():
        offset = - rescale_mean / rescale_sigma
        prior_rescale = stats.truncnorm(offset, 5, loc=rescale_mean, scale=rescale_sigma)
        prior_mu   = stats.uniform(loc=0, scale=1)
        rescale = prior_rescale.rvs(random_state=random_state)
        mu   = prior_mu.rvs(random_state=random_state)
        return Parameter(rescale, mu)
    return param_sampler
