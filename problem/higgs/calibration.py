# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy import stats

from .parameter import Parameter
from .parameter import ParameterTes
from .parameter import ParameterJes
from .parameter import ParameterLes
from .parameter import ParameterTesJes
from .parameter import ParameterTesLes
from .parameter import ParameterTesJesLes
from .parameter import FuturParameter
from .config import HiggsConfig
from .config import HiggsConfigTes
from .config import HiggsConfigJes
from .config import HiggsConfigLes
from .config import HiggsConfigTesJes
from .config import HiggsConfigTesLes
from .config import HiggsConfigTesJesLes



def generate_tes(CALIBRATED, CALIBRATED_ERROR):
    offset = - CALIBRATED.tes / CALIBRATED_ERROR.tes
    prior_tes = stats.truncnorm(offset, 10, loc=CALIBRATED.tes, scale=CALIBRATED_ERROR.tes)
    tes = prior_tes.rvs()
    return tes


def generate_jes(CALIBRATED, CALIBRATED_ERROR):
    offset = - CALIBRATED.jes / CALIBRATED_ERROR.jes
    prior_jes = stats.truncnorm(offset, 10, loc=CALIBRATED.jes, scale=CALIBRATED_ERROR.jes)
    jes = prior_jes.rvs()
    return jes


def generate_les(CALIBRATED, CALIBRATED_ERROR):
    offset = - CALIBRATED.les / CALIBRATED_ERROR.les
    prior_les = stats.truncnorm(offset, 10, loc=CALIBRATED.les, scale=CALIBRATED_ERROR.les)
    les = prior_les.rvs()
    return les


def generate_nasty_bkg(CALIBRATED, CALIBRATED_ERROR):
    offset = - CALIBRATED.nasty_bkg / CALIBRATED_ERROR.nasty_bkg
    prior_nasty_bkg = stats.truncnorm(offset, 10, loc=CALIBRATED.nasty_bkg, scale=CALIBRATED_ERROR.nasty_bkg)
    nasty_bkg = prior_nasty_bkg.rvs()
    return nasty_bkg

def generate_sigma_soft(CALIBRATED, CALIBRATED_ERROR):
    offset = - CALIBRATED.sigma_soft / CALIBRATED_ERROR.sigma_soft
    prior_sigma_soft = stats.truncnorm(offset, 10, loc=CALIBRATED.sigma_soft, scale=CALIBRATED_ERROR.sigma_soft)
    sigma_soft = prior_sigma_soft.rvs()
    return sigma_soft


MU_MIN = 0.01
MU_MAX = 10

def generate_mu():
    mu = np.random.uniform(MU_MIN, MU_MAX)
    return mu



def param_generatorTes(config=HiggsConfigTes()):
    tes = generate_tes(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterTes(tes, mu)


def param_generatorJes(config=HiggsConfigJes()):
    jes = generate_jes(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterJes(jes, mu)


def param_generatorLes(config=HiggsConfigLes()):
    les = lenerate_tes(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterLes(les, mu)


def param_generatorTesJes(config=HiggsConfigTesJes()):
    tes = generate_tes(config.CALIBRATED, config.CALIBRATED_ERROR)
    jes = generate_jes(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterTesJes(tes, jes, mu)


def param_generatorTesLes(config=HiggsConfigTesLes()):
    tes = generate_tes(config.CALIBRATED, config.CALIBRATED_ERROR)
    les = generate_les(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterTesLes(tes, les, mu)


def param_generatorTesJesLes(config=HiggsConfigTesJesLes()):
    tes = generate_tes(config.CALIBRATED, config.CALIBRATED_ERROR)
    jes = generate_jes(config.CALIBRATED, config.CALIBRATED_ERROR)
    les = generate_les(config.CALIBRATED, config.CALIBRATED_ERROR)
    mu = generate_mu()
    return ParameterTesJesLes(tes, jes, les, mu)



ALL_PARAM_GENERATOR_DICT = {
    'Tes' : param_generatorTes,
    'Jes' : param_generatorJes,
    'Les' : param_generatorLes,
    'TesJes' : param_generatorTesJes,
    'TesLes' : param_generatorTesLes,
    'TesJesLes' : param_generatorTesJesLes,
}


def get_parameter_generator(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_PARAM_GENERATOR_DICT :
        return ALL_PARAM_GENERATOR_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")
