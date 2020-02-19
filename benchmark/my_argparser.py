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


def REG_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.5, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=20, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    # parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
    #                     action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args



def INFERNO_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--temperature', help='control initial softmax steepness',
                        default=1.0, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--n-bins', help='number of output bins',
                        default=10, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=20, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    # parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
    #                     action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args

def NET_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbosity", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=20, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    # parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
    #                     action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args
