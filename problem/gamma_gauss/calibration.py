# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from scipy import stats

from .parameter import Parameter
from .config import GGConfig


def param_generator():
    pb_config = GGConfig()
    prior_alpha = stats.norm(loc=pb_config.CALIBRATED.alpha, scale=pb_config.CALIBRATED_ERROR.alpha)
    prior_mix   = stats.uniform(loc=0, scale=1)
    alpha = prior_alpha.rvs()
    mix   = prior_mix.rvs()
    return Parameter(alpha, mix)

