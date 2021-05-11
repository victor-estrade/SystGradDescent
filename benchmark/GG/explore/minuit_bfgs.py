#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.GG.explore.minuit_bfgs


import os
import datetime
import argparse
import logging
import numpy as np
import pandas as pd

from config import SAVING_DIR
from config import SEED
from visual import set_plot_config
set_plot_config()

import matplotlib.pyplot as plt

from utils.log import set_logger
from utils.log import flush
from utils.log import print_line

from ..common import N_BINS

from .load_model import load_some_GB
from .load_model import load_some_NN

from utils.evaluation import estimate

from problem.gamma_gauss import Generator
from problem.gamma_gauss import GGConfig as Config
from problem.gamma_gauss import GGNLL as NLLComputer
from problem.gamma_gauss import get_minimizer


from scipy.optimize import approx_fprime
from scipy.optimize import fmin_bfgs

DATA_NAME = 'GG'
BENCHMARK_NAME = DATA_NAME+'-prior'
DIRECTORY = os.path.join(SAVING_DIR, DATA_NAME, "gradient", )


def parse_args(main_description="Explore NLL shape"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    parser.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    parser.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    parser.add_argument("--tolerance", type=float,
                        default=0.1, help="tolerance value for Minuit migrad and simplex minimization")

    args, _ = parser.parse_known_args()
    return args


def main():
    logger = set_logger()
    root_directory = os.path.join(DIRECTORY, "nll_contour")
    os.makedirs(root_directory, exist_ok=True)
    args = parse_args()

    N_CV = 3
    # FIXME : remove lili and STEP to use all iteration !
    STEP = 1
    lili = list(Config().iter_test_config())[::STEP]
    N_ITER = len(lili)
    logger.info(f"{N_CV} cv and {N_ITER} iteractions ({N_ITER*N_CV} loops)")
    data = []
    for i_cv in range(N_CV):
        for i_iter, config in enumerate(lili):
            i_iter = i_iter * STEP
            values = run_cv_iter(args, i_cv, i_iter, config, root_directory)
            data.append(values)
    data = pd.DataFrame(data)
    fname = os.path.join(root_directory, "data.csv")
    data.to_csv(fname)



def run_cv_iter(args, i_cv, i_iter, config, root_directory):
    logger = logging.getLogger()
    # Settings
    directory = os.path.join(root_directory, f"cv_{i_cv}", f"iter_{i_iter}")
    os.makedirs(directory, exist_ok=True)
    train_generator, valid_generator, test_generator = get_generators(i_cv)
    logger.info(f"{config.TRUE}, {config.N_TESTING_SAMPLES}")
    model = load_some_NN(i_cv=i_cv, cuda=args.cuda)
    compute_nll = get_nll_computer(model, config, valid_generator, test_generator)

    for epsilon in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        rescale, mu = config.TRUE
        v_00 = compute_nll(rescale, mu)
        v_10 = compute_nll(rescale+epsilon, mu)
        v_01 = compute_nll(rescale, mu+epsilon)
        logger.info(f"e = {epsilon}  00 = {v_00}  10 = {v_10}   01 = {v_01}")

    # Results storage
    values = {}
    values['i_cv'] = i_cv
    values['i_iter'] = i_iter
    values['n_test_samples'] = config.N_TESTING_SAMPLES
    values['TRUE_rescale'] = config.TRUE.rescale
    values['TRUE_mu'] = config.TRUE.mu

    # compute Calibration NLL vs True value NLL
    nll = compute_nll(*config.CALIBRATED)
    logger.info(f"calib nll = {nll}")
    nll = compute_nll(*config.TRUE)
    logger.info(f"true  nll = {nll}")
    values['TRUE_feval'] = nll

    # run minimization with scipy BFGS
    # out = run_scipy_bfgs(compute_nll, config)
    # values.update(scipy_bfgs_to_values_dict(out))

    # run minimization with Minuit.MIGRAD
    minimizer = run_minuit_estimate(compute_nll, config, args.tolerance)
    values.update(minuit_migrad_to_values_dict(minimizer))

    #  I want grad and feval at minimum and grad at true value
    logger.info(f"Gradient !")
    f = lambda xk : compute_nll(*xk)
    EPSILON = 5e-6
    epsilon = np.array([EPSILON]*2)

    # @ true value
    # approx_gradient_at_true_value(f, config, epsilon)

    # @ BFGS minimum
    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = out
    # approx_gradient_at_scipy_minimum(f, xopt, epsilon)

    # @ Minuit minimum
    # approx_gradient_at_minuit_minimum(f, minimizer, epsilon)

    #  I want grad and feval contour plot at minimum and grad at true value
    # plot_feval_contour(xopt, compute_nll, epsilon, directory, title="feval_around_scipy_minimum")
    # plot_grad_contour(xopt, f, epsilon, directory, title="grad_around_scipy_minimum")
    # plot_feval_contour(minimizer.values, compute_nll, epsilon, directory, title="feval_around_minuit_minimum")
    # plot_grad_contour(minimizer.values, f, epsilon, directory, title="grad_around_minuit_minimum")

    # plot_feval_contour([1.0, 0.17], compute_nll, epsilon, directory, title="feval_around_minuit_minimum")

    return values


def get_generators(i_cv=0):
    seed = SEED + i_cv * 5
    train_generator = Generator(seed)
    valid_generator = Generator(seed + 1)
    test_generator  = Generator(seed + 2)
    return train_generator, valid_generator, test_generator


def get_nll_computer(model, config, valid_generator, test_generator):
    X_test, y_test, w_test = test_generator.generate(*config.TRUE, n_samples=config.N_TESTING_SAMPLES)

    compute_summaries = model.summary_computer(n_bins=N_BINS)
    compute_nll = NLLComputer(compute_summaries, valid_generator, X_test, w_test, config=config)
    return compute_nll


def run_grad(compute_nll, config):
    logger = logging.getLogger()
    f = lambda xk : compute_nll(*xk)

    xk = np.array(list(config.TRUE))
    logger.info(f"starting point = {xk}")
    EPSILON = 1e-8
    epsilon = np.array([EPSILON]*2)
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"grad = {grad}, grad norm = {grad.dot(grad.T)}")



#  Or maybe softplus !!
#  softplus(x) = ln( 1 +  exp (x) )
#  softplus_inv(x) = ln( exp(x) - 1 )
def transform_p_int(p_int, a, b):
     value = a + ( (b-a) / 2 ) * ( np.sin(p_int) + 1 )
     return value

def transform_p_ext(p_ext, a, b):
     value = np.arcsin(2 * (p_ext - a) / (b - a) - 1  )
     return value

# https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/
def softplus(x):
    return np.log1p(np.exp(x))

    #  softplus_inv(x) check = ln( exp( ln ( 1 + exp(x))  ) - 1 )
    #  softplus_inv(x) check = ln(           1 + exp(x)     - 1 )
    #  softplus_inv(x) check = ln(               exp(x)         )
    #  softplus_inv(x) check =                       x
def softplusinv(x):
    return np.log(np.expm1(x))

def lower_transform_p_int(x, a=0):
    value = a - 1 + np.sqrt(np.square(x) + 1)
    return value

def lower_transform_p_ext(x, a=0):
    value = np.sqrt(np.square(x - a + 1) - 1)
    return value


_transform = lambda x : softplus(x)
_transform = lambda x : lower_transform_p_int(x, a=0)
# _transform = lambda x : transform_p_int(x, 0, 10)

def run_scipy_bfgs(compute_nll, config):
    logger = logging.getLogger()
    f = lambda xk : compute_nll(*_transform(xk))

    logger.info(f"Running BFGS on the NLL")
    x_0 = np.array(list(config.CALIBRATED))
    x_0 = lower_transform_p_ext(x_0, 0)
    out = fmin_bfgs(f, x_0, full_output=True)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = out
    logger.info(f"xopt = {xopt} ({_transform(xopt)})")
    logger.info(f"fopt = {fopt} ")
    logger.info(f"gopt = {gopt} ")
    logger.info(f"Bopt = \n {Bopt} ")
    logger.info(f"func_calls = {func_calls} ")
    logger.info(f"grad_calls = {grad_calls} ")
    logger.info(f"warnflag = {warnflag} ")
    return _transform(xopt), fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def scipy_bfgs_to_values_dict(out):
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = out
    updates = dict(rescale = xopt[0]
                    , mu = xopt[1]
                    , feval = fopt
                    , g_rescale = gopt[0]
                    , g_mu = gopt[1]
                    , H_00 = Bopt[0][0]
                    , H_01 = Bopt[0][1]
                    , H_10 = Bopt[1][0]
                    , H_11 = Bopt[1][1]
                    , edm_estimated = gopt.T.dot(Bopt).dot(gopt)
                    , n_calls = func_calls
                    , n_grad_calls = grad_calls
                    , warnflag = warnflag
                    )
    updates = {f"BFGS_{key}": values for key, values in updates.items()}
    return updates


def run_minuit_migrad(compute_nll, config, tolerance):
    logger = logging.getLogger()
    logger.info(f"Running MIGRAD on the NLL")
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR, tolerance=tolerance)
    minimizer.migrad()
    logger.info(f"\n{minimizer}")
    logger.info(f" values = {list(minimizer.values)} ")
    # cov = np.array(minimizer.covariance)
    # print("last state", minimizer._last_state)
    # print(minimizer.grad, minimizer.values)
    # grad = minimizer.grad([1, 1])
    # logger.info(f"cov = {cov}")
    # logger.info(f"grad = {grad}")
    #
    # edm = grad.dot(cov.dot(grad.T))
    # logger.info(f"edm = {edm}")
    return minimizer


def run_minuit_estimate(compute_nll, config, tolerance):
    logger = logging.getLogger()
    logger.info(f"Running MIGRAD on the NLL")
    minimizer = get_minimizer(compute_nll, config.CALIBRATED, config.CALIBRATED_ERROR, tolerance=tolerance)
    estimate(minimizer, do_hesse=True)
    logger.info(f"\n{minimizer}")
    logger.info(f" values = {list(minimizer.values)} ")
    return minimizer



def minuit_migrad_to_values_dict(minimizer):
    updates = dict(rescale = minimizer.values[0]
                    , mu = minimizer.values[1]
                    , feval = minimizer.fmin.fval
                    , edm = minimizer.fmin.edm
                    , edm_goal = minimizer.fmin.edm_goal
                    , H_00 = minimizer.covariance[0][0]
                    , H_01 = minimizer.covariance[0][1]
                    , H_10 = minimizer.covariance[1][0]
                    , H_11 = minimizer.covariance[1][1]
                    , n_calls = minimizer.nfcn
                    , n_grad_calls = minimizer.ngrad
                    , has_accurate_covar = minimizer.fmin.has_accurate_covar
                    , has_covariance = minimizer.fmin.has_covariance
                    , has_made_posdef_covar = minimizer.fmin.has_made_posdef_covar
                    , has_parameters_at_limit = minimizer.fmin.has_parameters_at_limit
                    , has_posdef_covar = minimizer.fmin.has_posdef_covar
                    , has_reached_call_limit = minimizer.fmin.has_reached_call_limit
                    , has_valid_parameters = minimizer.fmin.has_valid_parameters
                    , hesse_failed = minimizer.fmin.hesse_failed
                    , is_above_max_edm = minimizer.fmin.is_above_max_edm
                    , is_valid = minimizer.fmin.is_valid
                    )
    updates = {f"MIGRAD_{key}": values for key, values in updates.items()}
    return updates


def approx_gradient_at_true_value(f, config, epsilon):
    logger = logging.getLogger()
    # @ true value
    xk = np.array(list(config.TRUE))
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"@ true value grad={grad}")
    return grad


def approx_gradient_at_scipy_minimum(f, xopt, epsilon):
    logger = logging.getLogger()
    grad = approx_fprime(xopt, f, epsilon)
    logger.info(f"@ BFGS min grad={grad}")
    return grad


def approx_gradient_at_minuit_minimum(f, minimizer, epsilon):
    logger = logging.getLogger()
    xk = np.array(list(minimizer.values))
    grad = approx_fprime(xk, f, epsilon)
    logger.info(f"@ Minuit min grad={grad}")
    cov = np.array(minimizer.covariance)
    edm = grad.dot(cov.dot(grad.T))
    logger.info(f"recomputed edm = {edm} ( =?= {minimizer.fmin.edm})")
    logger.info(f"@ Minuit min feval={minimizer.fmin.fval}")
    return grad


def plot_feval_contour(xopt, compute_nll, epsilon, directory, title="feval_around_minimum"):
    logger = logging.getLogger()
    logger.info(f"Contour plots !")
    ARRAY_SIZE = 40
    DELTA_alpha = 0.2
    DELTA_mu = 0.2
    alpha_array = np.linspace(xopt[0]-DELTA_alpha, xopt[0]+DELTA_alpha, ARRAY_SIZE)
    mu_array = np.linspace(xopt[1]-DELTA_mu, xopt[1]+DELTA_mu, ARRAY_SIZE)
    # alpha_mesh, mu_mesh = np.meshgrid(alpha_array, mu_array)
    mu_mesh, alpha_mesh = np.meshgrid(mu_array, alpha_array)
    nll_mesh = np.array([compute_nll(alpha, mu) for alpha, mu in zip(alpha_mesh.ravel(), mu_mesh.ravel())]).reshape(mu_mesh.shape)
    # plot_contour(alpha_mesh, mu_mesh, nll_mesh, directory, xlabel="alpha", ylabel="mu", title=title)
    plot_contour(mu_mesh, alpha_mesh, nll_mesh, directory, xlabel="mu", ylabel="alpha", title=title)


def plot_grad_contour(xopt, f, epsilon, directory, title="gradient_around_minimum"):
    logger = logging.getLogger()
    logger.info(f"Contour plots for gradients !")
    ARRAY_SIZE = 20
    DELTA_alpha = 0.1
    DELTA_mu = 0.1
    alpha_array = np.linspace(xopt[0]-DELTA_alpha, xopt[0]+DELTA_alpha, ARRAY_SIZE)
    mu_array = np.linspace(xopt[1]-DELTA_mu, xopt[1]+DELTA_mu, ARRAY_SIZE)
    alpha_mesh, mu_mesh = np.meshgrid(alpha_array, mu_array)
    grad_mesh = np.array([np.linalg.norm(approx_fprime(np.array([alpha, mu]), f, epsilon))
                        for alpha, mu in zip(alpha_mesh.ravel(), mu_mesh.ravel())]
                        ).reshape(mu_mesh.shape)
    plot_contour(alpha_mesh, mu_mesh, grad_mesh, directory, xlabel="alpha", ylabel="mu", title=title)


def plot_contour(x, y, z, directory, xlabel="alpha", ylabel="mu", title=""):
    logger = logging.getLogger()
    fig, ax = plt.subplots()
    ax.grid(False)
    # im = ax.imshow(z, interpolation='bilinear', origin='lower')
    levels = np.linspace(np.min(z), np.max(z), 10)
    # print(np.min(z), np.max(z), levels)
    CS = ax.contour(x, y, z, origin='lower', extend='both')
    CB = fig.colorbar(CS, shrink=0.8)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S\n")
    ax.set_title(now)
    fname = f"{title}_contour_plot.png"
    path = os.path.join(directory, fname)
    plt.savefig(path)
    plt.clf()
    plt.close(fig)
    logger.info(f"saved at {path}")


if __name__ == '__main__':
    main()
