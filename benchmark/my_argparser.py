# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import argparse




def GB_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # MODEL HYPER PARAMETERS
    parser.add_argument('--n-estimators', help='number of estimators',
                        default=100, type=int)

    parser.add_argument('--max-depth', help='maximum depth of trees',
                        default=3, type=int)

    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-1, type=float)

    # OTHER
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    parser.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')

    args = parser.parse_args()
    return args
