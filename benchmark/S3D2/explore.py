#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from config import SAVING_DIR
from visual import set_plot_config
set_plot_config()

from problem.synthetic3D import S3D2
from problem.synthetic3D import S3D2Config

SEED = None
BENCHMARK_NAME = "S3D2"
DIRECTORY = os.path.join(SAVING_DIR, BENCHMARK_NAME, "explore")

def main():
    print("Hello master !")
    os.makedirs(DIRECTORY, exist_ok=True)
    set_plot_config()
    config = S3D2Config()
    N_SAMPLES = 10_000
    R_MIN   = -0.3
    R_MAX   = 0.3 
    LAM_MIN = 2
    LAM_MAX = 4
    MU_MIN  = 0.0
    MU_MAX  = 1.0 

    generator = S3D2(SEED)
    X, label = generator.sample_event(config.TRUE.r, config.TRUE.lam, config.TRUE.mu, size=N_SAMPLES)
    n_sig = np.sum(label==1)
    n_bkg = np.sum(label==0)
    print(f"nb of signal      = {n_sig}")
    print(f"nb of backgrounds = {n_bkg}")


    df = pd.DataFrame(X, columns=["x1","x2","x3"])
    df['label'] = label
    g = sns.PairGrid(df, vars=["x1","x2","x3"], hue='label')
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(sns.kdeplot)
    g = g.map_lower(sns.kdeplot, n_levels=6)
    g = g.add_legend()
    # g = g.map_offdiag(sns.kdeplot, n_levels=6)
    g.savefig(os.path.join(DIRECTORY, 'pairgrid.png'))
    plt.clf()


    nll = generator.nll(X, config.TRUE.r, config.TRUE.lam, config.TRUE.mu)
    print(f"NLL = {nll}")

    R_RANGE = np.linspace(R_MIN, R_MAX, 100)
    nll = [generator.nll(X, r, config.TRUE.lam, config.TRUE.mu) for r in R_RANGE]
    min_nll = R_RANGE[np.argmin(nll)]
    plt.plot(R_RANGE, nll, label="nll(r)")
    plt.axvline(config.TRUE.r, c="orange", label="true r")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("r")
    plt.ylabel("NLL")
    plt.title("NLL according to r param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_r.png'))
    plt.clf()


    LAM_RANGE = np.linspace(LAM_MIN, LAM_MAX, 100)
    nll = [generator.nll(X, config.TRUE.r, lam, config.TRUE.mu) for lam in LAM_RANGE]
    min_nll = LAM_RANGE[np.argmin(nll)]
    plt.plot(LAM_RANGE, nll, label="nll(lam)")
    plt.axvline(config.TRUE.lam, c="orange", label="true lam")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\lambda$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\lambda$ param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_lambda.png'))
    plt.clf()

    MU_RANGE = np.linspace(MU_MIN, MU_MAX, 100)
    nll = [generator.nll(X, config.TRUE.r, config.TRUE.lam, mu) for mu in MU_RANGE]
    min_nll = MU_RANGE[np.argmin(nll)]
    plt.plot(MU_RANGE, nll, label="nll(mu)")
    plt.axvline(config.TRUE.mu, c="orange", label="true mu")
    plt.axvline(min_nll, c="red", label="min nll")
    plt.xlabel("$\mu$")
    plt.ylabel("NLL")
    plt.title("NLL according to $\mu$ param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORY, 'NLL_mu.png'))
    plt.clf()


if __name__ == '__main__':
    main()