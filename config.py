# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

RANDOM_STATE = 42
SAVING_DIR = 'savings'



# FIT PARAMETERS
#---------------
# TODO : one day maybe put all these values as input arguments 
#       and correctly sort experiments according to the paramters
#       Maybe when nvidia-docker supports Python 3.7 and dataclass !

# Parameter of interest
CALIBRATED_MU = 1.0
TRUE_MU = 1.0  # FIXME maybe I should try TRUE != CALIBRATED ?

# Nuisance parameters
CALIBRATED_TAU_ENERGY_SCALE = 1.0
CALIBRATED_TAU_ENERGY_SCALE_ERROR = 0.05
TRUE_TAU_ENERGY_SCALE = 1.03

CALIBRATED_JET_ENERGY_SCALE = 1.0
CALIBRATED_JET_ENERGY_SCALE_ERROR = 0.05
TRUE_JET_ENERGY_SCALE = 0.97

CALIBRATED_LEP_ENERGY_SCALE = 1.0
CALIBRATED_LEP_ENERGY_SCALE_ERROR = 0.01
TRUE_LEP_ENERGY_SCALE = 1.001

# FIXME What are the usual values for soft term ?
CALIBRATED_SOFT_TERM = 1.0
CALIBRATED_SOFT_TERM_ERROR = 0.05
TRUE_SOFT_TERM = 1.03
