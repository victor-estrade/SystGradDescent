# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from .parameter import Parameter
from .config import HiggsConfig

def param_generator(config=HiggsConfig()):
    tes = np.random.normal(config.CALIBRATED.tes, config.CALIBRATED.tes)
    while tes < 0:
        tes = np.random.normal(config.CALIBRATED.tes, config.CALIBRATED.tes)

    jes = np.random.normal(config.CALIBRATED.jes, config.CALIBRATED.jes)
    while jes < 0:
        jes = np.random.normal(config.CALIBRATED.jes, config.CALIBRATED.jes)

    les = np.random.normal(config.CALIBRATED.les, config.CALIBRATED.les)
    while les < 0:
        les = np.random.normal(config.CALIBRATED.les, config.CALIBRATED.les)

    nasty_bkg = np.random.normal(config.CALIBRATED.nasty_bkg, config.CALIBRATED.nasty_bkg)
    while nasty_bkg < 0:
        nasty_bkg = np.random.normal(config.CALIBRATED.nasty_bkg, config.CALIBRATED.nasty_bkg)

    sigma_soft = np.random.normal(config.CALIBRATED.sigma_soft, config.CALIBRATED.sigma_soft)
    
    mu = np.random.uniform(config.MIN.mu, config.MAX.mu)
    return Parameter(tes, jes, les, nasty_bkg, sigma_soft, mu)


