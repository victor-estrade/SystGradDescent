# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy import stats

from .parameter import Parameter
from .parameter import FuturParameter
from .config import HiggsConfig

def param_generator(config=HiggsConfig()):
    offset = - config.CALIBRATED.tes / config.CALIBRATED_ERROR.tes
    prior_tes = stats.truncnorm(offset, 10, loc=config.CALIBRATED.tes, scale=config.CALIBRATED_ERROR.tes)

    offset = - config.CALIBRATED.jes / config.CALIBRATED_ERROR.jes
    prior_jes = stats.truncnorm(offset, 10, loc=config.CALIBRATED.jes, scale=config.CALIBRATED_ERROR.jes)

    offset = - config.CALIBRATED.les / config.CALIBRATED_ERROR.les
    prior_les = stats.truncnorm(offset, 10, loc=config.CALIBRATED.les, scale=config.CALIBRATED_ERROR.les)


    tes = prior_tes.rvs()
    jes = prior_jes.rvs()
    les = prior_les.rvs()
    mu = np.random.uniform(config.MIN.mu, config.MAX.mu)
    return Parameter(tes, jes, les, mu)



def futur_param_generator(config=HiggsConfig()):
    offset = - config.CALIBRATED.tes / config.CALIBRATED_ERROR.tes
    prior_tes = stats.truncnorm(offset, 10, loc=config.CALIBRATED.tes, scale=config.CALIBRATED_ERROR.tes)

    offset = - config.CALIBRATED.jes / config.CALIBRATED_ERROR.jes
    prior_jes = stats.truncnorm(offset, 10, loc=config.CALIBRATED.jes, scale=config.CALIBRATED_ERROR.jes)

    offset = - config.CALIBRATED.les / config.CALIBRATED_ERROR.les
    prior_les = stats.truncnorm(offset, 10, loc=config.CALIBRATED.les, scale=config.CALIBRATED_ERROR.les)

    offset = - config.CALIBRATED.nasty_bkg / config.CALIBRATED_ERROR.nasty_bkg
    prior_nasty_bkg = stats.truncnorm(offset, 10, loc=config.CALIBRATED.nasty_bkg, scale=config.CALIBRATED_ERROR.nasty_bkg)

    offset = - config.CALIBRATED.sigma_soft / config.CALIBRATED_ERROR.sigma_soft
    prior_sigma_soft = stats.truncnorm(offset, 10, loc=config.CALIBRATED.sigma_soft, scale=config.CALIBRATED_ERROR.sigma_soft)

    tes = prior_tes.rvs()
    jes = prior_jes.rvs()
    les = prior_les.rvs()
    nasty_bkg = prior_nasty_bkg.rvs()
    sigma_soft = prior_sigma_soft.rvs()
    mu = np.random.uniform(config.MIN.mu, config.MAX.mu)
    return FuturParameter(tes, jes, les, nasty_bkg, sigma_soft, mu)


