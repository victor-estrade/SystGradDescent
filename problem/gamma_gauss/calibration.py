# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from scipy import stats

from .parameter import Parameter
from .config import GGConfig


def param_generator():
    config = GGConfig()
    offset = - config.CALIBRATED.rescale / config.CALIBRATED_ERROR.rescale
    prior_rescale = stats.truncnorm(offset, 5, loc=config.CALIBRATED.rescale, scale=config.CALIBRATED_ERROR.rescale)
    prior_mix   = stats.uniform(loc=0, scale=1)
    rescale = prior_rescale.rvs()
    mix   = prior_mix.rvs()
    return Parameter(rescale, mix)

