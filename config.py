# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os

RANDOM_STATE = 42
SEED = 42
SAVING_DIR = 'OUTPUT'
DEFAULT_DIR = os.path.join(SAVING_DIR, "Default")
MODEL_SAVING_DIR = os.path.join(SAVING_DIR, "MODELS")

PRECISION = 1e-3
N_CV = 6

_ERROR = '_error'
_MINOS_UP = '_minos_up'
_MINOS_LOW = '_minos_low'
_TRUTH = '_truth'

PLOT_CONTOUR = False

# FIT PARAMETERS
#---------------
# TODO : one day maybe put all these values as input arguments
#       and correctly sort experiments according to the paramters

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
CALIBRATED_SIGMA_SOFT = 2.7
CALIBRATED_SIGMA_SOFT_ERROR = 0.5
TRUE_SIGMA_SOFT = 3.3

# FIXME What are the usual values for nasty bkg ?
CALIBRATED_NASTY_BKG = 1.0
CALIBRATED_NASTY_BKG_ERROR = 0.5
TRUE_NASTY_BKG = 1.5
