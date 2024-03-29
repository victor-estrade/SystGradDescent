# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse


def parse_args_tolerance():
    parser = argparse.ArgumentParser(description='just for tolerance')
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    args, _ = parser.parse_known_args()
    return args.tolerance

def GB_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--n-estimators', help='number of estimators',
                        default=100, type=int)

    parser.add_argument('--max-depth', help='maximum depth of trees',
                        default=3, type=int)

    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-1, type=float)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    parser.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')

    args = parser.parse_args()
    return args


def REG_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-4, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.5, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--n-unit', help='Number of units in layers. Controls NN width.',
                        default=200, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=20, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args



def INFERNO_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--temperature', help='control initial softmax steepness',
                        default=1.0, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.5, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--n-unit', help='Number of units in layers. Controls NN width.',
                        default=200, type=int)

    parser.add_argument('--n-bins', help='number of output bins',
                        default=10, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=20, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args



def NET_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

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

    parser.add_argument('--n-unit', help='Number of units in layers. Controls NN width.',
                        default=200, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=1000, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args

def TP_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--trade-off', help='trade-off between classic loss and adversarial loss',
                        default=1.0, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--n-unit', help='Number of units in layers. Controls NN width.',
                        default=200, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=1000, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args



def PIVOT_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--learning-rate', '--lr', help='learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--trade-off', help='trade-off between classic loss and adversarial loss',
                        default=1.0, type=float)

    parser.add_argument('--beta1', help='beta 1 for Adam',
                        default=0.9, type=float)
    parser.add_argument('--beta2', help='beta 2 for Adam',
                        default=0.999, type=float)
    parser.add_argument('--weight-decay', help='weight decay for SGD',
                        default=0.0, type=float)

    parser.add_argument('--optimizer', help='optimizer name', dest='optimizer_name',
                        default='Adam', type=str, choices=('Adam', 'SGD', 'ADAM', 'sgd', 'adam'))

    parser.add_argument('--n-unit', help='Number of units in layers. Controls NN width.',
                        default=200, type=int)

    parser.add_argument('--sample-size', help='data sample size',
                        default=1000, type=int)

    parser.add_argument('--batch-size', help='mini-batch size',
                        default=1000, type=int)

    parser.add_argument('--n-steps', help='number of update steps',
                        default=1000, type=int)
    parser.add_argument('--n-net-pre-training-steps', help='number of update steps for pretraining the classifier',
                        default=1000, type=int)
    parser.add_argument('--n-adv-pre-training-steps', help='number of update steps for pretraining the adversarial',
                        default=1000, type=int)
    parser.add_argument('--n-recovery-steps', help='number of update steps for adversarial recovery',
                        default=1, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')

    args = parser.parse_args()
    return args


def FF_parse_args(main_description="Training launcher"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")
    parser.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    parser.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    parser.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    # MODEL HYPER PARAMETERS
    parser.add_argument('--feature-id', help='feature index to filter on',
                        default=0, type=int)

    # OTHER
    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    parser.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')

    args = parser.parse_args()
    return args
